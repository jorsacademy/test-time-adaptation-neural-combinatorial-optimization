from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from ttanco.dataset import generate_instance
from ttanco.decoding import coordinates_tensor
from ttanco.model import (
    EdgePolicy,
    EmbeddingAdapter,
    PolicyConfig,
    load_checkpoint,
    model_state_fingerprint,
    save_checkpoint,
)


def small_model() -> EdgePolicy:
    torch.manual_seed(7)
    return EdgePolicy(PolicyConfig(hidden_dim=12, message_layers=1, mlp_layers=1))


def test_edge_logits_are_symmetric_finite_and_have_masked_diagonal() -> None:
    instance = generate_instance(7, seed=10)
    model = small_model()
    logits = model(coordinates_tensor(instance, device="cpu"))
    assert logits.shape == (7, 7)
    assert torch.all(torch.isfinite(logits))
    assert torch.allclose(logits, logits.T)
    assert torch.all(torch.diag(logits) < -1e8)


def test_policy_is_permutation_equivariant() -> None:
    instance = generate_instance(7, regime="clustered", seed=12)
    model = small_model().eval()
    coordinates = coordinates_tensor(instance, device="cpu")
    permutation = torch.tensor([3, 0, 6, 2, 1, 5, 4])
    with torch.no_grad():
        original = model(coordinates)
        permuted = model(coordinates[permutation])
    expected = original[permutation][:, permutation]
    assert torch.allclose(permuted, expected, atol=1e-5, rtol=1e-5)


def test_embedding_adapter_starts_as_source_policy() -> None:
    instance = generate_instance(6, seed=4)
    model = small_model().eval()
    coordinates = coordinates_tensor(instance, device="cpu")
    embeddings, distances = model.encode(coordinates)
    adapter = EmbeddingAdapter(6, model.config.hidden_dim)
    with torch.no_grad():
        source = model.score_embeddings(embeddings, distances)
        adapted = adapter.adapted_logits(model, embeddings, distances)
    assert torch.allclose(source, adapted)
    assert adapter.parameter_count == 6 * model.config.hidden_dim + 7


def test_safetensors_checkpoint_round_trip(tmp_path: Path) -> None:
    model = small_model()
    path = tmp_path / "model.safetensors"
    save_checkpoint(model, path, metadata={"purpose": "test"})
    loaded, metadata = load_checkpoint(path)
    assert metadata == {"purpose": "test"}
    assert loaded.config == model.config
    assert model_state_fingerprint(loaded) == model_state_fingerprint(model)


def test_coordinate_translation_and_scaling_leave_logits_approximately_unchanged() -> None:
    model = small_model().eval()
    coordinates = torch.tensor(
        [[0.0, 0.0], [1.0, 0.2], [0.7, 1.1], [-0.1, 0.8]], dtype=torch.float32
    )
    transformed = 37.0 * coordinates + torch.tensor([500.0, -700.0])
    with torch.no_grad():
        first = model(coordinates)
        second = model(transformed)
    assert np.allclose(first.numpy(), second.numpy(), atol=2e-5, rtol=2e-5)


def test_invalid_adapter_dimensions_are_rejected() -> None:
    with pytest.raises(ValueError):
        EmbeddingAdapter(3, 8)
