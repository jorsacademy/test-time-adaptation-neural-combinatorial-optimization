from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest
import torch

from ttanco.model import EdgePolicy, PolicyConfig
from ttanco.utils import (
    bootstrap_mean_interval,
    canonical_json,
    ensure_unique,
    finite_float,
    flatten_mapping,
    read_json,
    require_mapping,
    seed_everything,
    sha256_json,
    stable_seed,
    tensor_state_fingerprint,
    write_json,
)


def test_serialization_hashing_and_file_round_trip(tmp_path: Path) -> None:
    payload = {"b": [2, 1], "a": {"x": 3}}
    assert canonical_json(payload).startswith('{"a"')
    assert sha256_json(payload) == sha256_json({"a": {"x": 3}, "b": [2, 1]})
    path = tmp_path / "payload.json"
    write_json(payload, path)
    assert read_json(path) == payload
    assert flatten_mapping(payload) == {"a.x": 3, "b": "[2,1]"}


def test_seed_helpers_are_deterministic() -> None:
    seed_everything(123)
    first = (random.random(), float(np.random.random()), float(torch.rand(())))
    seed_everything(123)
    second = (random.random(), float(np.random.random()), float(torch.rand(())))
    assert first == second
    assert stable_seed("a", 1) == stable_seed("a", 1)
    assert stable_seed("a", 1) != stable_seed("a", 2)
    with pytest.raises(ValueError):
        seed_everything(-1)
    with pytest.raises(ValueError):
        stable_seed("x", modulus=1)


def test_validation_and_bootstrap_helpers() -> None:
    assert finite_float(3, name="x") == 3.0
    with pytest.raises(ValueError):
        finite_float(True, name="x")
    low, high = bootstrap_mean_interval([1.0, 2.0, 3.0], seed=2, draws=50)
    assert 1.0 <= low <= high <= 3.0
    with pytest.raises(ValueError):
        bootstrap_mean_interval([], seed=0, draws=10)
    with pytest.raises(ValueError):
        bootstrap_mean_interval([1.0], seed=0, draws=0)


def test_collection_and_mapping_guards() -> None:
    assert ensure_unique(["a", "b"], name="values") == ("a", "b")
    with pytest.raises(ValueError):
        ensure_unique([], name="values")
    with pytest.raises(ValueError):
        ensure_unique(["a", "a"], name="values")
    mapping = require_mapping({"a": 1}, name="payload")
    assert mapping["a"] == 1
    with pytest.raises(ValueError):
        require_mapping([1, 2], name="payload")


def test_tensor_fingerprint_changes_with_parameters() -> None:
    torch.manual_seed(1)
    model = EdgePolicy(PolicyConfig(hidden_dim=8, message_layers=0, mlp_layers=1))
    first = tensor_state_fingerprint(model)
    with torch.no_grad():
        next(model.parameters()).add_(1.0)
    assert tensor_state_fingerprint(model) != first
