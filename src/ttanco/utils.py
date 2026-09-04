"""Deterministic serialization, hashing, seeding, and numerical utilities."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch


def canonical_json(value: object) -> str:
    """Serialize JSON-compatible data with a stable byte representation."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_json(value: object) -> str:
    """Return SHA-256 of the stable JSON representation."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def write_json(value: object, path: str | Path) -> None:
    """Write indented UTF-8 JSON after creating parent directories."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def read_json(path: str | Path) -> object:
    """Read JSON from disk."""

    return json.loads(Path(path).read_text(encoding="utf-8"))


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for deterministic CPU experiments."""

    if seed < 0:
        raise ValueError("seed must be nonnegative")
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)


def stable_seed(*parts: object, modulus: int = 2**31 - 1) -> int:
    """Derive a deterministic positive integer seed from arbitrary JSON-compatible parts."""

    if modulus <= 1:
        raise ValueError("modulus must exceed one")
    digest = hashlib.sha256(canonical_json(list(parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % modulus


def finite_float(value: object, *, name: str) -> float:
    """Validate and convert a finite numeric value while excluding Booleans."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def percentile(values: Sequence[float], q: float) -> float:
    """Compute a deterministic linear percentile."""

    if not values:
        raise ValueError("cannot compute percentile of an empty sequence")
    if not 0.0 <= q <= 100.0:
        raise ValueError("percentile must lie in [0, 100]")
    return float(np.percentile(np.asarray(values, dtype=float), q, method="linear"))


def bootstrap_mean_interval(
    values: Sequence[float],
    *,
    seed: int,
    draws: int,
    lower_q: float = 2.5,
    upper_q: float = 97.5,
) -> tuple[float, float]:
    """Return a deterministic nonparametric bootstrap interval for the sample mean."""

    if not values:
        raise ValueError("bootstrap requires at least one value")
    if draws <= 0:
        raise ValueError("bootstrap draws must be positive")
    if not 0.0 <= lower_q < upper_q <= 100.0:
        raise ValueError("bootstrap quantiles are invalid")
    data = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(data)):
        raise ValueError("bootstrap values must be finite")
    rng = np.random.default_rng(seed)
    means = np.empty(draws, dtype=float)
    for draw in range(draws):
        indices = rng.integers(0, data.size, size=data.size)
        means[draw] = float(np.mean(data[indices]))
    return percentile(means.tolist(), lower_q), percentile(means.tolist(), upper_q)


def flatten_mapping(mapping: Mapping[str, object], *, prefix: str = "") -> dict[str, object]:
    """Flatten nested mappings for CSV output."""

    flattened: dict[str, object] = {}
    for key, value in mapping.items():
        joined = f"{prefix}.{key}" if prefix else key
        if isinstance(value, Mapping):
            flattened.update(flatten_mapping(value, prefix=joined))
        elif isinstance(value, (list, tuple, dict)):
            flattened[joined] = canonical_json(value)
        else:
            flattened[joined] = value
    return flattened


def ensure_unique(values: Iterable[str], *, name: str) -> tuple[str, ...]:
    """Validate a nonempty sequence of unique, nonempty strings."""

    materialized = tuple(values)
    if not materialized or any(not value for value in materialized):
        raise ValueError(f"{name} must contain nonempty values")
    if len(set(materialized)) != len(materialized):
        raise ValueError(f"{name} must not contain duplicates")
    return materialized


def tensor_state_fingerprint(module: torch.nn.Module) -> str:
    """Hash a module state dict without using pickle serialization."""

    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        contiguous = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(contiguous.shape)).encode("utf-8"))
        digest.update(str(contiguous.dtype).encode("utf-8"))
        digest.update(contiguous.numpy().tobytes(order="C"))
    return digest.hexdigest()


def require_mapping(value: object, *, name: str) -> dict[str, Any]:
    """Validate that a JSON value is a mapping with string keys."""

    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be a JSON object")
    return value
