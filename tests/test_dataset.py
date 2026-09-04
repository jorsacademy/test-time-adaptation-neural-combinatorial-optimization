from __future__ import annotations

import json
from pathlib import Path

import pytest

from ttanco.dataset import (
    SUPPORTED_REGIMES,
    generate_dataset,
    generate_instance,
    load_dataset_jsonl,
    save_dataset_jsonl,
)


def test_generators_are_deterministic_and_cover_all_regimes() -> None:
    for regime in SUPPORTED_REGIMES:
        first = generate_instance(8, regime=regime, seed=123)
        second = generate_instance(8, regime=regime, seed=123)
        assert first.coordinates == second.coordinates
        assert first.regime == regime
        assert len(set(first.coordinates)) == 8


def test_dataset_round_trip_recomputes_exact_labels(tmp_path: Path) -> None:
    dataset = generate_dataset(
        count=4,
        node_counts=(5, 6),
        regimes=("uniform", "clustered"),
        seed=100,
    )
    path = tmp_path / "corpus.jsonl"
    save_dataset_jsonl(dataset, path)
    loaded = load_dataset_jsonl(path)
    assert loaded == dataset
    assert loaded.to_metadata()["record_count"] == 4


def test_dataset_tampering_is_detected(tmp_path: Path) -> None:
    dataset = generate_dataset(count=2, node_counts=(5,), regimes=("uniform",), seed=1)
    path = tmp_path / "corpus.jsonl"
    save_dataset_jsonl(dataset, path)
    lines = path.read_text(encoding="utf-8").splitlines()
    record = json.loads(lines[1])
    record["instance"]["coordinates"][0][0] += 0.1
    lines[1] = json.dumps(record, sort_keys=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint"):
        load_dataset_jsonl(path)


def test_dataset_configuration_validation() -> None:
    with pytest.raises(ValueError):
        generate_instance(3)
    with pytest.raises(ValueError):
        generate_instance(5, regime="unknown")
    with pytest.raises(ValueError):
        generate_dataset(count=0, node_counts=(5,), regimes=("uniform",), seed=0)
    with pytest.raises(ValueError):
        generate_dataset(count=1, node_counts=(3,), regimes=("uniform",), seed=0)
