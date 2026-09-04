"""Controlled TSP generators and exact-labeled, fingerprinted JSONL corpora."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import numpy as np

from ttanco.domain import TourSolution, TSPInstance, audit_tour, solve_held_karp
from ttanco.utils import canonical_json, sha256_json

GeneratorRegime = Literal[
    "uniform",
    "clustered",
    "ring",
    "grid",
    "anisotropic",
    "outlier",
    "heavy_tail",
    "spiral",
    "coordinate_scale",
]

SUPPORTED_REGIMES: tuple[str, ...] = (
    "uniform",
    "clustered",
    "ring",
    "grid",
    "anisotropic",
    "outlier",
    "heavy_tail",
    "spiral",
    "coordinate_scale",
)
DATASET_SCHEMA_VERSION = "ttanco-tsp-corpus-v1"


@dataclass(frozen=True, slots=True)
class TSPRecord:
    instance: TSPInstance
    optimum: TourSolution

    def to_dict(self) -> dict[str, object]:
        return {"instance": self.instance.to_dict(), "optimum": self.optimum.to_dict()}

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> TSPRecord:
        raw_instance = payload.get("instance")
        raw_optimum = payload.get("optimum")
        if not isinstance(raw_instance, dict) or not isinstance(raw_optimum, dict):
            raise ValueError("dataset record must contain instance and optimum objects")
        return cls(
            TSPInstance.from_dict(cast(dict[str, object], raw_instance)),
            TourSolution.from_dict(cast(dict[str, object], raw_optimum)),
        )


@dataclass(frozen=True, slots=True)
class TSPDataset:
    records: tuple[TSPRecord, ...]
    fingerprint: str
    schema_version: str = DATASET_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.records:
            raise ValueError("dataset must contain at least one record")
        if self.schema_version != DATASET_SCHEMA_VERSION:
            raise ValueError("unsupported dataset schema version")
        expected = dataset_fingerprint(self.records)
        if expected != self.fingerprint:
            raise ValueError("dataset fingerprint does not match its records")
        identifiers = [record.instance.instance_id for record in self.records]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("dataset instance identifiers must be unique")

    @property
    def instances(self) -> tuple[TSPInstance, ...]:
        return tuple(record.instance for record in self.records)

    @property
    def optima(self) -> tuple[TourSolution, ...]:
        return tuple(record.optimum for record in self.records)

    @property
    def regimes(self) -> tuple[str, ...]:
        return tuple(sorted({record.instance.regime for record in self.records}))

    @property
    def node_counts(self) -> tuple[int, ...]:
        return tuple(sorted({record.instance.node_count for record in self.records}))

    def to_metadata(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "record_count": len(self.records),
            "fingerprint": self.fingerprint,
            "regimes": list(self.regimes),
            "node_counts": list(self.node_counts),
        }


def _rescale_unit_square(coordinates: np.ndarray) -> np.ndarray:
    minimum = np.min(coordinates, axis=0)
    maximum = np.max(coordinates, axis=0)
    span = maximum - minimum
    span = np.where(span < 1e-9, 1.0, span)
    return (coordinates - minimum) / span


def _ensure_distinct(coordinates: np.ndarray, *, seed: int) -> np.ndarray:
    result = np.asarray(coordinates, dtype=np.float64).copy()
    rng = np.random.default_rng(seed + 98_765)
    for _ in range(8):
        rounded = [tuple(np.round(row, decimals=12)) for row in result]
        if len(set(rounded)) == len(rounded):
            return result
        result += rng.normal(0.0, 1e-8, size=result.shape)
    raise RuntimeError("failed to construct pairwise distinct coordinates")


def generate_instance(
    node_count: int,
    *,
    regime: str = "uniform",
    seed: int = 0,
    instance_id: str | None = None,
) -> TSPInstance:
    """Generate a deterministic Euclidean TSP instance from a controlled regime."""

    if node_count < 4:
        raise ValueError("node_count must be at least four")
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    if regime not in SUPPORTED_REGIMES:
        raise ValueError(f"unsupported generator regime: {regime}")
    rng = np.random.default_rng(seed)

    if regime == "uniform":
        coordinates = rng.random((node_count, 2))
    elif regime == "clustered":
        cluster_count = min(3, max(2, node_count // 5))
        centers = rng.uniform(0.15, 0.85, size=(cluster_count, 2))
        assignments = rng.integers(0, cluster_count, size=node_count)
        coordinates = centers[assignments] + rng.normal(0.0, 0.075, size=(node_count, 2))
        coordinates = np.clip(coordinates, 0.0, 1.0)
    elif regime == "ring":
        angles = np.sort(rng.uniform(0.0, 2.0 * math.pi, size=node_count))
        radii = 0.38 + rng.normal(0.0, 0.025, size=node_count)
        coordinates = np.column_stack((np.cos(angles), np.sin(angles))) * radii[:, None] + 0.5
    elif regime == "grid":
        width = int(math.ceil(math.sqrt(node_count)))
        grid = np.asarray(
            [(column, row) for row in range(width) for column in range(width)][:node_count],
            dtype=np.float64,
        )
        coordinates = grid / max(1, width - 1)
        coordinates += rng.normal(0.0, 0.025, size=coordinates.shape)
        coordinates = np.clip(coordinates, 0.0, 1.0)
    elif regime == "anisotropic":
        base = rng.random((node_count, 2))
        base[:, 1] *= 0.12
        angle = math.radians(37.0)
        rotation = np.asarray(
            [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]],
            dtype=np.float64,
        )
        coordinates = _rescale_unit_square(base @ rotation.T)
    elif regime == "outlier":
        core_count = max(2, node_count - max(1, node_count // 6))
        core = rng.normal(loc=(0.35, 0.35), scale=(0.11, 0.11), size=(core_count, 2))
        outliers = rng.uniform(0.75, 1.0, size=(node_count - core_count, 2))
        coordinates = np.clip(np.vstack((core, outliers)), 0.0, 1.0)
        rng.shuffle(coordinates, axis=0)
    elif regime == "heavy_tail":
        coordinates = rng.standard_t(df=2.5, size=(node_count, 2))
        lower = np.quantile(coordinates, 0.05, axis=0)
        upper = np.quantile(coordinates, 0.95, axis=0)
        coordinates = np.clip(coordinates, lower, upper)
        coordinates = _rescale_unit_square(coordinates)
    elif regime == "spiral":
        angles = np.linspace(0.2, 3.8 * math.pi, node_count)
        radii = np.linspace(0.08, 0.48, node_count)
        coordinates = np.column_stack((radii * np.cos(angles), radii * np.sin(angles))) + 0.5
        coordinates += rng.normal(0.0, 0.015, size=coordinates.shape)
    elif regime == "coordinate_scale":
        coordinates = rng.random((node_count, 2)) * 1_000.0 + np.asarray([5_000.0, -7_000.0])
    else:  # pragma: no cover - exhaustive guard for static type checkers.
        raise AssertionError("unreachable regime")

    coordinates = _ensure_distinct(coordinates, seed=seed)
    identifier = instance_id or f"{regime}-n{node_count}-seed{seed}"
    return TSPInstance(
        tuple((float(row[0]), float(row[1])) for row in coordinates),
        instance_id=identifier,
        regime=regime,
        seed=seed,
    )


def generate_dataset(
    *,
    count: int,
    node_counts: Sequence[int],
    regimes: Sequence[str],
    seed: int,
    exact_limit: int = 18,
) -> TSPDataset:
    """Generate disjoint exact-labeled records by cycling through sizes and regimes."""

    if count <= 0:
        raise ValueError("count must be positive")
    if not node_counts or any(node_count < 4 for node_count in node_counts):
        raise ValueError("node_counts must contain values of at least four")
    if not regimes or any(regime not in SUPPORTED_REGIMES for regime in regimes):
        raise ValueError("regimes contains unsupported values")
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    records: list[TSPRecord] = []
    for index in range(count):
        node_count = int(node_counts[index % len(node_counts)])
        regime = regimes[index % len(regimes)]
        instance_seed = seed + index
        instance = generate_instance(
            node_count,
            regime=regime,
            seed=instance_seed,
            instance_id=f"corpus-{seed}-{index:05d}-{regime}-n{node_count}",
        )
        optimum = solve_held_karp(instance, maximum_nodes=exact_limit)
        records.append(TSPRecord(instance, optimum))
    frozen = tuple(records)
    return TSPDataset(frozen, dataset_fingerprint(frozen))


def dataset_fingerprint(records: Iterable[TSPRecord]) -> str:
    """Hash mathematical instances and exact solutions, excluding runtime metadata."""

    return sha256_json([record.to_dict() for record in records])


def save_dataset_jsonl(dataset: TSPDataset, path: str | Path) -> None:
    """Write a metadata header followed by exact-labeled JSONL records."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        canonical_json(
            {
                "record_type": "metadata",
                "schema_version": dataset.schema_version,
                "record_count": len(dataset.records),
                "fingerprint": dataset.fingerprint,
            }
        )
    ]
    lines.extend(
        canonical_json({"record_type": "record", **record.to_dict()}) for record in dataset.records
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_dataset_jsonl(
    path: str | Path,
    *,
    verify_exact: bool = True,
    exact_limit: int = 18,
    tolerance: float = 1e-9,
) -> TSPDataset:
    """Load a corpus, recomputing fingerprints and exact solutions when requested."""

    source = Path(path)
    lines = [line for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) < 2:
        raise ValueError("dataset file must contain metadata and at least one record")
    header = json.loads(lines[0])
    if not isinstance(header, dict) or header.get("record_type") != "metadata":
        raise ValueError("dataset metadata header is missing")
    if header.get("schema_version") != DATASET_SCHEMA_VERSION:
        raise ValueError("unsupported dataset schema version")
    records: list[TSPRecord] = []
    for line in lines[1:]:
        payload = json.loads(line)
        if not isinstance(payload, dict) or payload.get("record_type") != "record":
            raise ValueError("dataset contains a malformed record")
        records.append(TSPRecord.from_dict(cast(dict[str, object], payload)))
    declared_count = header.get("record_count")
    declared_fingerprint = header.get("fingerprint")
    if isinstance(declared_count, bool) or not isinstance(declared_count, int):
        raise ValueError("dataset record_count must be an integer")
    if declared_count != len(records):
        raise ValueError("dataset record count does not match the header")
    if not isinstance(declared_fingerprint, str):
        raise ValueError("dataset fingerprint must be a string")
    computed = dataset_fingerprint(records)
    if computed != declared_fingerprint:
        raise ValueError("dataset fingerprint mismatch")

    if verify_exact:
        for record in records:
            audit = audit_tour(
                record.instance,
                record.optimum.tour,
                reported_length=record.optimum.length,
                tolerance=tolerance,
            )
            if not (audit.permutation_valid and audit.reported_length_consistent):
                raise ValueError("stored optimum failed tour audit")
            recomputed = solve_held_karp(record.instance, maximum_nodes=exact_limit)
            if not math.isclose(
                recomputed.length,
                record.optimum.length,
                rel_tol=tolerance,
                abs_tol=tolerance,
            ):
                raise ValueError("stored optimum differs from exact recomputation")
    frozen = tuple(records)
    return TSPDataset(frozen, declared_fingerprint)
