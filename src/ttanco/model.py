"""Permutation-equivariant edge policy, instance adapters, and safe checkpoints."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch import Tensor, nn

CHECKPOINT_SCHEMA_VERSION = "ttanco-edge-policy-v1"
FEATURE_SCHEMA_VERSION = "normalized-euclidean-node-edge-v1"


@dataclass(frozen=True, slots=True)
class PolicyConfig:
    hidden_dim: int = 64
    message_layers: int = 2
    mlp_layers: int = 2

    def __post_init__(self) -> None:
        if self.hidden_dim <= 0 or self.message_layers < 0 or self.mlp_layers <= 0:
            raise ValueError("policy dimensions must be positive")


def _mlp(input_dim: int, hidden_dim: int, layers: int, output_dim: int) -> nn.Sequential:
    modules: list[nn.Module] = []
    width = input_dim
    for _ in range(layers):
        modules.extend((nn.Linear(width, hidden_dim), nn.SiLU()))
        width = hidden_dim
    modules.append(nn.Linear(width, output_dim))
    return nn.Sequential(*modules)


def normalize_coordinates(coordinates: Tensor) -> Tensor:
    """Center and RMS-scale coordinates without changing TSP ordering semantics."""

    if coordinates.ndim != 2 or coordinates.shape[1] != 2 or coordinates.shape[0] < 4:
        raise ValueError("coordinates must have shape [nodes, 2] with at least four nodes")
    if not torch.all(torch.isfinite(coordinates)):
        raise ValueError("coordinates must be finite")
    centered = coordinates - torch.mean(coordinates, dim=0, keepdim=True)
    scale = torch.sqrt(torch.mean(torch.sum(centered * centered, dim=1)))
    if not torch.isfinite(scale) or float(scale.detach().cpu()) <= 1e-12:
        raise ValueError("coordinate scale is degenerate")
    return centered / scale


def node_features(coordinates: Tensor) -> tuple[Tensor, Tensor]:
    """Build equivariant node features and an invariant normalized distance matrix."""

    normalized = normalize_coordinates(coordinates)
    distances = torch.cdist(normalized, normalized, p=2)
    n = normalized.shape[0]
    diagonal_mask = torch.eye(n, dtype=torch.bool, device=normalized.device)
    off_diagonal = distances.masked_fill(diagonal_mask, float("inf"))
    nearest = torch.min(off_diagonal, dim=1).values
    mean_distance = torch.sum(distances, dim=1) / float(n - 1)
    radius = torch.linalg.vector_norm(normalized, dim=1)
    x = normalized[:, 0]
    y = normalized[:, 1]
    features = torch.stack(
        (
            x,
            y,
            radius,
            x * x - y * y,
            2.0 * x * y,
            mean_distance,
            nearest,
            torch.full_like(x, 1.0 / float(n)),
        ),
        dim=1,
    )
    if not torch.all(torch.isfinite(features)):
        raise RuntimeError("node feature construction produced non-finite values")
    return features, distances


class EdgePolicy(nn.Module):
    """Static symmetric edge-score model used by an autoregressive tour policy."""

    node_encoder: nn.Sequential
    message_updates: nn.ModuleList
    message_norms: nn.ModuleList
    edge_scorer: nn.Sequential

    def __init__(self, config: PolicyConfig | None = None) -> None:
        super().__init__()
        self.config = config or PolicyConfig()
        hidden = self.config.hidden_dim
        self.node_encoder = _mlp(8, hidden, self.config.mlp_layers, hidden)
        self.message_updates = nn.ModuleList(
            _mlp(3 * hidden, hidden, 1, hidden) for _ in range(self.config.message_layers)
        )
        self.message_norms = nn.ModuleList(
            nn.LayerNorm(hidden) for _ in range(self.config.message_layers)
        )
        self.edge_scorer = _mlp(2 * hidden + 1, hidden, self.config.mlp_layers, 1)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def encode(self, coordinates: Tensor) -> tuple[Tensor, Tensor]:
        features, distances = node_features(coordinates)
        hidden = cast(Tensor, self.node_encoder(features))
        for update, normalization in zip(
            self.message_updates,
            self.message_norms,
            strict=True,
        ):
            mean_pool = torch.mean(hidden, dim=0, keepdim=True).expand_as(hidden)
            max_pool = torch.max(hidden, dim=0, keepdim=True).values.expand_as(hidden)
            residual = cast(Tensor, update(torch.cat((hidden, mean_pool, max_pool), dim=1)))
            hidden = cast(Tensor, normalization(hidden + residual))
        if not torch.all(torch.isfinite(hidden)):
            raise RuntimeError("node encoder produced non-finite embeddings")
        return hidden, distances

    def score_embeddings(self, embeddings: Tensor, distances: Tensor) -> Tensor:
        if embeddings.ndim != 2 or embeddings.shape[1] != self.config.hidden_dim:
            raise ValueError("embeddings have an incompatible shape")
        n = embeddings.shape[0]
        if distances.shape != (n, n):
            raise ValueError("distance matrix has an incompatible shape")
        left = embeddings[:, None, :].expand(n, n, -1)
        right = embeddings[None, :, :].expand(n, n, -1)
        edge_features = torch.cat(
            (left + right, torch.abs(left - right), distances[..., None]), dim=2
        )
        logits = cast(Tensor, self.edge_scorer(edge_features)).squeeze(-1)
        logits = 0.5 * (logits + logits.transpose(0, 1))
        diagonal = torch.eye(n, dtype=torch.bool, device=logits.device)
        logits = logits.masked_fill(diagonal, -1.0e9)
        if not torch.all(torch.isfinite(logits)):
            raise RuntimeError("edge scorer produced non-finite logits")
        return logits

    def forward(self, coordinates: Tensor) -> Tensor:
        embeddings, distances = self.encode(coordinates)
        return self.score_embeddings(embeddings, distances)


class EmbeddingAdapter(nn.Module):
    """Per-instance residual embedding and node-bias adapter for efficient active search."""

    def __init__(self, node_count: int, hidden_dim: int) -> None:
        super().__init__()
        if node_count < 4 or hidden_dim <= 0:
            raise ValueError("adapter dimensions are invalid")
        self.node_count = node_count
        self.hidden_dim = hidden_dim
        self.delta = nn.Parameter(torch.zeros(node_count, hidden_dim))
        self.node_bias = nn.Parameter(torch.zeros(node_count))
        self.log_temperature = nn.Parameter(torch.zeros(()))

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @property
    def squared_norm(self) -> Tensor:
        return torch.sum(self.delta * self.delta) + torch.sum(self.node_bias * self.node_bias)

    def adapted_logits(
        self,
        model: EdgePolicy,
        base_embeddings: Tensor,
        distances: Tensor,
    ) -> Tensor:
        if base_embeddings.shape != (self.node_count, self.hidden_dim):
            raise ValueError("adapter and base embedding dimensions do not match")
        adapted = base_embeddings + self.delta
        logits = model.score_embeddings(adapted, distances)
        logits = logits + self.node_bias[:, None] + self.node_bias[None, :]
        temperature = torch.exp(torch.clamp(self.log_temperature, min=-2.0, max=2.0))
        logits = logits / temperature
        diagonal = torch.eye(self.node_count, dtype=torch.bool, device=logits.device)
        return logits.masked_fill(diagonal, -1.0e9)


def clone_policy(model: EdgePolicy) -> EdgePolicy:
    clone = EdgePolicy(model.config)
    clone.load_state_dict(model.state_dict(), strict=True)
    clone.to(model.device)
    return clone


def _checkpoint_header(model: EdgePolicy) -> dict[str, str]:
    return {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "model_type": "edge_policy",
        "model_config": json.dumps(asdict(model.config), sort_keys=True),
    }


def save_checkpoint(
    model: EdgePolicy,
    path: str | Path,
    *,
    metadata: dict[str, object] | None = None,
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    header = _checkpoint_header(model)
    header["metadata"] = json.dumps(metadata or {}, sort_keys=True, allow_nan=False)
    tensors = {key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()}
    save_file(tensors, str(output), metadata=header)


def _config_integer(config: dict[str, object], name: str) -> int:
    value = config.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"checkpoint model field {name!r} must be an integer")
    return value


def load_checkpoint(
    path: str | Path,
    *,
    device: torch.device | str = "cpu",
) -> tuple[EdgePolicy, dict[str, object]]:
    source = Path(path)
    with safe_open(str(source), framework="pt", device="cpu") as handle:
        header = handle.metadata()
        tensors = {key: handle.get_tensor(key) for key in handle.keys()}  # noqa: SIM118
    if header is None:
        raise ValueError("checkpoint metadata is missing")
    if header.get("checkpoint_schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("unsupported checkpoint schema version")
    if header.get("feature_schema_version") != FEATURE_SCHEMA_VERSION:
        raise ValueError("checkpoint feature schema is incompatible")
    if header.get("model_type") != "edge_policy":
        raise ValueError("checkpoint model type is unsupported")
    raw_config: object = json.loads(header["model_config"])
    if not isinstance(raw_config, dict):
        raise ValueError("checkpoint model configuration is invalid")
    config = cast(dict[str, object], raw_config)
    model = EdgePolicy(
        PolicyConfig(
            hidden_dim=_config_integer(config, "hidden_dim"),
            message_layers=_config_integer(config, "message_layers"),
            mlp_layers=_config_integer(config, "mlp_layers"),
        )
    )
    expected_keys = set(model.state_dict())
    if set(tensors) != expected_keys:
        raise ValueError("checkpoint tensor keys do not match the model schema")
    for key, tensor in tensors.items():
        if tensor.shape != model.state_dict()[key].shape:
            raise ValueError(f"checkpoint tensor {key!r} has an incompatible shape")
        if not torch.all(torch.isfinite(tensor)):
            raise ValueError(f"checkpoint tensor {key!r} contains non-finite values")
    model.load_state_dict(tensors, strict=True)
    model.to(device)
    raw_metadata: object = json.loads(header.get("metadata", "{}"))
    if not isinstance(raw_metadata, dict):
        raise ValueError("checkpoint metadata payload is invalid")
    return model, cast(dict[str, object], raw_metadata)


def model_state_fingerprint(model: EdgePolicy) -> str:
    """Hash model parameters without serializing executable Python objects."""

    digest = hashlib.sha256()
    for key, tensor in sorted(model.state_dict().items()):
        contiguous = tensor.detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tuple(contiguous.shape)).encode("utf-8"))
        digest.update(contiguous.numpy().tobytes(order="C"))
    return digest.hexdigest()


def validate_policy(model: EdgePolicy) -> None:
    """Fail closed on non-finite model parameters."""

    for name, parameter in model.named_parameters():
        if not torch.all(torch.isfinite(parameter)):
            raise ValueError(f"model parameter {name!r} contains non-finite values")
        if parameter.numel() == 0:
            raise ValueError(f"model parameter {name!r} is empty")
    if not math.isfinite(float(model.parameter_count)) or model.parameter_count <= 0:
        raise ValueError("model parameter count is invalid")
