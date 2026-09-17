"""First-order meta-learning for fast adaptation across TSP task distributions."""

from __future__ import annotations

import copy
from collections import defaultdict
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import Tensor, nn

from ttanco.dataset import TSPDataset, TSPRecord
from ttanco.domain import solution_from_tour
from ttanco.model import EdgePolicy
from ttanco.training import tour_edge_targets


@dataclass(frozen=True, slots=True)
class MetaLearningConfig:
    outer_epochs: int = 8
    inner_steps: int = 2
    support_size: int = 2
    query_size: int = 2
    inner_learning_rate: float = 5e-3
    outer_learning_rate: float = 1e-3
    gradient_clip: float = 1.0
    seed: int = 0

    def __post_init__(self) -> None:
        if self.outer_epochs <= 0 or self.inner_steps <= 0:
            raise ValueError("meta-learning epoch and inner-step counts must be positive")
        if self.support_size <= 0 or self.query_size <= 0:
            raise ValueError("support and query sizes must be positive")
        if self.inner_learning_rate <= 0.0 or self.outer_learning_rate <= 0.0:
            raise ValueError("meta-learning rates must be positive")
        if self.gradient_clip <= 0.0 or self.seed < 0:
            raise ValueError("gradient clip and seed are invalid")


@dataclass(frozen=True, slots=True)
class MetaTask:
    name: str
    regime: str
    node_count: int
    support_indices: tuple[int, ...]
    query_indices: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class FewShotReport:
    task: str
    support_size: int
    query_size: int
    query_loss_before: float
    query_loss_after: float
    mean_gap_before_pct: float
    mean_gap_after_pct: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class MetaEmbeddingAdapter(nn.Module):
    """Size-agnostic residual adapter that can be meta-trained across task distributions."""

    def __init__(self, hidden_dim: int, bottleneck_dim: int = 16) -> None:
        super().__init__()
        if hidden_dim <= 0 or bottleneck_dim <= 0:
            raise ValueError("adapter dimensions must be positive")
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim
        self.down = nn.Linear(hidden_dim, bottleneck_dim)
        self.up = nn.Linear(bottleneck_dim, hidden_dim)
        self.shift = nn.Parameter(torch.zeros(hidden_dim))
        self.log_temperature = nn.Parameter(torch.zeros(()))
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def adapted_logits(
        self,
        model: EdgePolicy,
        base_embeddings: Tensor,
        distances: Tensor,
    ) -> Tensor:
        if base_embeddings.ndim != 2 or base_embeddings.shape[1] != self.hidden_dim:
            raise ValueError("base embedding shape is incompatible with meta-adapter")
        residual = self.up(torch.nn.functional.silu(self.down(base_embeddings)))
        adapted = base_embeddings + residual + self.shift[None, :]
        logits = model.score_embeddings(adapted, distances)
        temperature = torch.exp(torch.clamp(self.log_temperature, min=-2.0, max=2.0))
        return logits / temperature


def build_meta_tasks(
    dataset: TSPDataset,
    *,
    support_size: int,
    query_size: int,
) -> tuple[MetaTask, ...]:
    if support_size <= 0 or query_size <= 0:
        raise ValueError("support_size and query_size must be positive")
    groups: dict[tuple[str, int], list[int]] = defaultdict(list)
    for index, record in enumerate(dataset.records):
        groups[(record.instance.regime, record.instance.node_count)].append(index)
    tasks: list[MetaTask] = []
    required = support_size + query_size
    for regime, node_count in sorted(groups):
        indices = groups[(regime, node_count)]
        if len(indices) < required:
            continue
        support = tuple(indices[:support_size])
        query = tuple(indices[support_size:required])
        tasks.append(
            MetaTask(
                name=f"{regime}-n{node_count}",
                regime=regime,
                node_count=node_count,
                support_indices=support,
                query_indices=query,
            )
        )
    if not tasks:
        raise ValueError("dataset does not contain enough records to form a meta-task")
    return tuple(tasks)


def _edge_loss(logits: Tensor, targets: Tensor) -> Tensor:
    n = logits.shape[0]
    if logits.shape != (n, n) or targets.shape != (n, n):
        raise ValueError("edge logits and targets must be aligned square matrices")
    mask = torch.triu(
        torch.ones((n, n), dtype=torch.bool, device=logits.device),
        diagonal=1,
    )
    selected_logits = logits[mask]
    selected_targets = targets[mask]
    positives = torch.sum(selected_targets)
    if float(positives.detach().cpu()) <= 0.0:
        raise RuntimeError("tour target contains no positive edges")
    negatives = float(selected_targets.numel()) - positives
    loss = nn.functional.binary_cross_entropy_with_logits(
        selected_logits,
        selected_targets,
        pos_weight=negatives / positives,
    )
    if not torch.isfinite(loss):
        raise RuntimeError("meta-learning edge loss is non-finite")
    return loss


def _record_loss(
    model: EdgePolicy,
    adapter: MetaEmbeddingAdapter,
    record: TSPRecord,
) -> Tensor:
    coordinates = torch.tensor(
        record.instance.coordinates,
        dtype=torch.float32,
        device=model.device,
    )
    targets = torch.tensor(
        tour_edge_targets(record.instance.node_count, record.optimum.tour),
        dtype=torch.float32,
        device=model.device,
    )
    base_embeddings, distances = model.encode(coordinates)
    logits = adapter.adapted_logits(
        model,
        base_embeddings.detach(),
        distances.detach(),
    )
    return _edge_loss(logits, targets)


def _mean_loss(
    model: EdgePolicy,
    adapter: MetaEmbeddingAdapter,
    dataset: TSPDataset,
    indices: tuple[int, ...],
) -> Tensor:
    losses = [
        _record_loss(model, adapter, dataset.records[index])
        for index in indices
    ]
    return torch.stack(losses).mean()


def adapt_to_task(
    model: EdgePolicy,
    initialization: MetaEmbeddingAdapter,
    dataset: TSPDataset,
    task: MetaTask,
    *,
    inner_steps: int,
    inner_learning_rate: float,
    gradient_clip: float = 1.0,
) -> MetaEmbeddingAdapter:
    if inner_steps <= 0 or inner_learning_rate <= 0.0 or gradient_clip <= 0.0:
        raise ValueError("inner adaptation configuration is invalid")
    adapted = copy.deepcopy(initialization).to(model.device)
    optimizer = torch.optim.SGD(adapted.parameters(), lr=inner_learning_rate)
    for _ in range(inner_steps):
        optimizer.zero_grad(set_to_none=True)
        loss = _mean_loss(model, adapted, dataset, task.support_indices)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            adapted.parameters(),
            gradient_clip,
        )
        if not torch.isfinite(gradient_norm):
            raise RuntimeError("meta-adaptation produced non-finite gradients")
        optimizer.step()
    return adapted


def meta_train_adapter(
    model: EdgePolicy,
    dataset: TSPDataset,
    *,
    adapter: MetaEmbeddingAdapter | None = None,
    config: MetaLearningConfig | None = None,
) -> tuple[MetaEmbeddingAdapter, tuple[float, ...]]:
    """First-order MAML-style training across (regime, size) task distributions."""

    cfg = config or MetaLearningConfig()
    tasks = build_meta_tasks(
        dataset,
        support_size=cfg.support_size,
        query_size=cfg.query_size,
    )
    result = adapter or MetaEmbeddingAdapter(model.config.hidden_dim)
    result = result.to(model.device)
    optimizer = torch.optim.AdamW(
        result.parameters(),
        lr=cfg.outer_learning_rate,
    )
    rng = np.random.default_rng(cfg.seed)
    history: list[float] = []
    result_parameters = list(result.parameters())

    for _ in range(cfg.outer_epochs):
        order = rng.permutation(len(tasks))
        gradient_sums = [
            torch.zeros_like(parameter)
            for parameter in result_parameters
        ]
        query_losses: list[float] = []
        for raw_index in order:
            task = tasks[int(raw_index)]
            adapted = adapt_to_task(
                model,
                result,
                dataset,
                task,
                inner_steps=cfg.inner_steps,
                inner_learning_rate=cfg.inner_learning_rate,
                gradient_clip=cfg.gradient_clip,
            )
            query_loss = _mean_loss(
                model,
                adapted,
                dataset,
                task.query_indices,
            )
            adapted_parameters = tuple(adapted.parameters())
            gradients = torch.autograd.grad(query_loss, adapted_parameters)
            for accumulator, gradient in zip(
                gradient_sums,
                gradients,
                strict=True,
            ):
                accumulator.add_(gradient.detach())
            query_losses.append(float(query_loss.detach().cpu()))

        optimizer.zero_grad(set_to_none=True)
        task_count = float(len(tasks))
        for parameter, gradient in zip(
            result_parameters,
            gradient_sums,
            strict=True,
        ):
            parameter.grad = gradient / task_count
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            result_parameters,
            cfg.gradient_clip,
        )
        if not torch.isfinite(gradient_norm):
            raise RuntimeError("meta-training produced non-finite outer gradients")
        optimizer.step()
        history.append(float(np.mean(query_losses)))
    return result, tuple(history)


def _greedy_gap_pct(
    model: EdgePolicy,
    adapter: MetaEmbeddingAdapter,
    record: TSPRecord,
) -> float:
    coordinates = torch.tensor(
        record.instance.coordinates,
        dtype=torch.float32,
        device=model.device,
    )
    with torch.no_grad():
        base_embeddings, distances = model.encode(coordinates)
        logits = adapter.adapted_logits(model, base_embeddings, distances)
        n = record.instance.node_count
        visited = torch.zeros(n, dtype=torch.bool, device=model.device)
        current = 0
        visited[current] = True
        tour = [current]
        for _ in range(n - 1):
            row = logits[current].masked_fill(visited, -1.0e9)
            nxt = int(torch.argmax(row).item())
            visited[nxt] = True
            tour.append(nxt)
            current = nxt
    solution = solution_from_tour(record.instance, tuple(tour))
    return 100.0 * (solution.length - record.optimum.length) / max(
        record.optimum.length,
        1e-12,
    )


def evaluate_few_shot(
    model: EdgePolicy,
    initialization: MetaEmbeddingAdapter,
    dataset: TSPDataset,
    task: MetaTask,
    *,
    inner_steps: int,
    inner_learning_rate: float,
) -> FewShotReport:
    before_loss = float(
        _mean_loss(
            model,
            initialization,
            dataset,
            task.query_indices,
        )
        .detach()
        .cpu()
    )
    before_gaps = [
        _greedy_gap_pct(model, initialization, dataset.records[index])
        for index in task.query_indices
    ]
    adapted = adapt_to_task(
        model,
        initialization,
        dataset,
        task,
        inner_steps=inner_steps,
        inner_learning_rate=inner_learning_rate,
    )
    after_loss = float(
        _mean_loss(model, adapted, dataset, task.query_indices)
        .detach()
        .cpu()
    )
    after_gaps = [
        _greedy_gap_pct(model, adapted, dataset.records[index])
        for index in task.query_indices
    ]
    return FewShotReport(
        task=task.name,
        support_size=len(task.support_indices),
        query_size=len(task.query_indices),
        query_loss_before=before_loss,
        query_loss_after=after_loss,
        mean_gap_before_pct=float(np.mean(before_gaps)),
        mean_gap_after_pct=float(np.mean(after_gaps)),
    )
