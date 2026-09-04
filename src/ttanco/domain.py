"""Euclidean TSP domain, exact oracles, canonical tours, and local search."""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from functools import cached_property

import numpy as np


@dataclass(frozen=True)
class TSPInstance:
    """A finite two-dimensional Euclidean traveling-salesperson instance."""

    coordinates: tuple[tuple[float, float], ...]
    instance_id: str = "instance"
    regime: str = "unspecified"
    seed: int = 0

    def __post_init__(self) -> None:
        if len(self.coordinates) < 4:
            raise ValueError("a TSP instance requires at least four nodes")
        normalized: list[tuple[float, float]] = []
        for coordinate in self.coordinates:
            if len(coordinate) != 2:
                raise ValueError("each coordinate must have exactly two entries")
            x, y = float(coordinate[0]), float(coordinate[1])
            if not math.isfinite(x) or not math.isfinite(y):
                raise ValueError("coordinates must be finite")
            normalized.append((x, y))
        if len(set(normalized)) != len(normalized):
            raise ValueError("coordinates must be pairwise distinct")
        if not self.instance_id:
            raise ValueError("instance_id must be nonempty")
        if self.seed < 0:
            raise ValueError("seed must be nonnegative")
        object.__setattr__(self, "coordinates", tuple(normalized))

    @property
    def node_count(self) -> int:
        return len(self.coordinates)

    @cached_property
    def coordinate_array(self) -> np.ndarray:
        array = np.asarray(self.coordinates, dtype=np.float64)
        array.setflags(write=False)
        return array

    @cached_property
    def distance_matrix(self) -> np.ndarray:
        difference = self.coordinate_array[:, None, :] - self.coordinate_array[None, :, :]
        distances = np.linalg.norm(difference, axis=2)
        if not np.all(np.isfinite(distances)):
            raise RuntimeError("distance matrix contains non-finite entries")
        distances.setflags(write=False)
        return distances

    def to_dict(self) -> dict[str, object]:
        return {
            "coordinates": [[x, y] for x, y in self.coordinates],
            "instance_id": self.instance_id,
            "regime": self.regime,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> TSPInstance:
        raw_coordinates = payload.get("coordinates")
        if not isinstance(raw_coordinates, list):
            raise ValueError("instance coordinates must be a JSON array")
        coordinates: list[tuple[float, float]] = []
        for raw_coordinate in raw_coordinates:
            if not isinstance(raw_coordinate, list) or len(raw_coordinate) != 2:
                raise ValueError("coordinate entries must be two-element arrays")
            x, y = raw_coordinate
            if isinstance(x, bool) or isinstance(y, bool):
                raise ValueError("coordinates must be numeric, not Boolean")
            if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
                raise ValueError("coordinates must be numeric")
            coordinates.append((float(x), float(y)))
        instance_id = payload.get("instance_id", "instance")
        regime = payload.get("regime", "unspecified")
        seed = payload.get("seed", 0)
        if not isinstance(instance_id, str) or not isinstance(regime, str):
            raise ValueError("instance identifiers must be strings")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("instance seed must be an integer")
        return cls(tuple(coordinates), instance_id=instance_id, regime=regime, seed=seed)


@dataclass(frozen=True, slots=True)
class TourSolution:
    """A canonical Hamiltonian cycle and its Euclidean length."""

    tour: tuple[int, ...]
    length: float

    def to_dict(self) -> dict[str, object]:
        return {"tour": list(self.tour), "length": self.length}

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> TourSolution:
        raw_tour = payload.get("tour")
        length = payload.get("length")
        if not isinstance(raw_tour, list) or not all(
            isinstance(node, int) and not isinstance(node, bool) for node in raw_tour
        ):
            raise ValueError("tour must be an integer array")
        if isinstance(length, bool) or not isinstance(length, (int, float)):
            raise ValueError("tour length must be numeric")
        result = float(length)
        if not math.isfinite(result) or result < 0.0:
            raise ValueError("tour length must be finite and nonnegative")
        return cls(tuple(raw_tour), result)


@dataclass(frozen=True, slots=True)
class TourAudit:
    permutation_valid: bool
    canonical: bool
    recomputed_length: float | None
    reported_length_consistent: bool
    optimal: bool | None
    absolute_gap: float | None
    relative_gap_percent: float | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TwoOptResult:
    initial: TourSolution
    solution: TourSolution
    passes: int
    move_evaluations: int

    @property
    def improvement(self) -> float:
        return self.initial.length - self.solution.length

    def to_dict(self) -> dict[str, object]:
        return {
            "initial": self.initial.to_dict(),
            "solution": self.solution.to_dict(),
            "passes": self.passes,
            "move_evaluations": self.move_evaluations,
            "improvement": self.improvement,
        }


def canonicalize_tour(tour: Sequence[int], node_count: int | None = None) -> tuple[int, ...]:
    """Rotate a cycle to node zero and choose the lexicographically smaller orientation."""

    materialized = tuple(int(node) for node in tour)
    expected = len(materialized) if node_count is None else node_count
    if len(materialized) != expected or set(materialized) != set(range(expected)):
        raise ValueError("tour must be a permutation of all node indices")
    zero_index = materialized.index(0)
    forward = materialized[zero_index:] + materialized[:zero_index]
    reverse_raw = tuple(reversed(materialized))
    reverse_zero = reverse_raw.index(0)
    reverse = reverse_raw[reverse_zero:] + reverse_raw[:reverse_zero]
    return min(forward, reverse)


def tour_length(instance: TSPInstance, tour: Sequence[int]) -> float:
    """Recompute cycle length after validating the node permutation."""

    canonical = canonicalize_tour(tour, instance.node_count)
    distances = instance.distance_matrix
    total = 0.0
    for position, source in enumerate(canonical):
        target = canonical[(position + 1) % instance.node_count]
        total += float(distances[source, target])
    if not math.isfinite(total):
        raise RuntimeError("computed tour length is non-finite")
    return total


def solution_from_tour(instance: TSPInstance, tour: Sequence[int]) -> TourSolution:
    canonical = canonicalize_tour(tour, instance.node_count)
    return TourSolution(canonical, tour_length(instance, canonical))


def audit_tour(
    instance: TSPInstance,
    tour: Sequence[int],
    *,
    reported_length: float | None = None,
    optimum: TourSolution | None = None,
    tolerance: float = 1e-9,
) -> TourAudit:
    """Independently audit permutation feasibility, length, and optional optimality."""

    try:
        materialized = tuple(int(node) for node in tour)
        valid = len(materialized) == instance.node_count and set(materialized) == set(
            range(instance.node_count)
        )
    except (TypeError, ValueError):
        valid = False
        materialized = ()
    if not valid:
        return TourAudit(False, False, None, False, False if optimum else None, None, None)
    canonical = canonicalize_tour(materialized, instance.node_count)
    recomputed = tour_length(instance, canonical)
    consistent = reported_length is None or math.isclose(
        recomputed,
        float(reported_length),
        rel_tol=tolerance,
        abs_tol=tolerance,
    )
    optimal: bool | None = None
    absolute_gap: float | None = None
    relative_gap: float | None = None
    if optimum is not None:
        absolute_gap = recomputed - optimum.length
        if absolute_gap < -tolerance:
            raise RuntimeError("candidate appears better than the supplied exact optimum")
        absolute_gap = max(0.0, absolute_gap)
        relative_gap = 100.0 * absolute_gap / max(optimum.length, tolerance)
        optimal = absolute_gap <= tolerance
    return TourAudit(
        permutation_valid=True,
        canonical=canonical == materialized,
        recomputed_length=recomputed,
        reported_length_consistent=consistent,
        optimal=optimal,
        absolute_gap=absolute_gap,
        relative_gap_percent=relative_gap,
    )


def nearest_neighbor_tour(instance: TSPInstance, *, start: int = 0) -> TourSolution:
    """Construct a deterministic nearest-neighbor tour."""

    if not 0 <= start < instance.node_count:
        raise ValueError("start node is out of range")
    unvisited = set(range(instance.node_count))
    unvisited.remove(start)
    tour = [start]
    current = start
    while unvisited:
        next_node = min(unvisited, key=lambda node: (instance.distance_matrix[current, node], node))
        tour.append(next_node)
        unvisited.remove(next_node)
        current = next_node
    return solution_from_tour(instance, tour)


def two_opt(
    instance: TSPInstance,
    initial_tour: Sequence[int],
    *,
    maximum_passes: int = 100,
    tolerance: float = 1e-12,
) -> TwoOptResult:
    """Run deterministic best-improvement 2-opt while preserving tour feasibility."""

    if maximum_passes <= 0:
        raise ValueError("maximum_passes must be positive")
    initial = solution_from_tour(instance, initial_tour)
    tour = list(initial.tour)
    distances = instance.distance_matrix
    evaluations = 0
    passes = 0
    n = instance.node_count
    for _ in range(maximum_passes):
        passes += 1
        best_delta = 0.0
        best_move: tuple[int, int] | None = None
        for first in range(n - 1):
            a = tour[first]
            b = tour[(first + 1) % n]
            for second in range(first + 2, n):
                if first == 0 and second == n - 1:
                    continue
                c = tour[second]
                d = tour[(second + 1) % n]
                evaluations += 1
                delta = (
                    float(distances[a, c])
                    + float(distances[b, d])
                    - float(distances[a, b])
                    - float(distances[c, d])
                )
                if delta < best_delta - tolerance or (
                    math.isclose(delta, best_delta, abs_tol=tolerance)
                    and best_move is not None
                    and (first, second) < best_move
                ):
                    best_delta = delta
                    best_move = (first, second)
        if best_move is None:
            break
        first, second = best_move
        tour[first + 1 : second + 1] = reversed(tour[first + 1 : second + 1])
    solution = solution_from_tour(instance, tour)
    if solution.length > initial.length + 1e-9:
        raise RuntimeError("2-opt returned a worse tour")
    return TwoOptResult(initial, solution, passes, evaluations)


def _held_karp_reconstruct(
    parents: dict[tuple[int, int], int],
    mask: int,
    last: int,
) -> tuple[int, ...]:
    reversed_path = [last]
    current_mask = mask
    current = last
    while current_mask & (current_mask - 1):
        predecessor = parents[(current_mask, current)]
        reversed_path.append(predecessor)
        current_mask ^= 1 << (current - 1)
        current = predecessor
    return (0,) + tuple(reversed(reversed_path))


def solve_held_karp(instance: TSPInstance, *, maximum_nodes: int = 18) -> TourSolution:
    """Solve Euclidean TSP exactly using Held-Karp subset dynamic programming."""

    n = instance.node_count
    if n > maximum_nodes:
        raise ValueError(f"instance has {n} nodes; Held-Karp limit is {maximum_nodes}")
    distances = instance.distance_matrix
    costs: dict[tuple[int, int], float] = {}
    parents: dict[tuple[int, int], int] = {}
    for last in range(1, n):
        mask = 1 << (last - 1)
        costs[(mask, last)] = float(distances[0, last])
        parents[(mask, last)] = 0

    full_mask = (1 << (n - 1)) - 1
    for subset_size in range(2, n):
        for subset in itertools.combinations(range(1, n), subset_size):
            mask = sum(1 << (node - 1) for node in subset)
            for last in subset:
                previous_mask = mask ^ (1 << (last - 1))
                best_cost = math.inf
                best_parent = -1
                for predecessor in subset:
                    if predecessor == last:
                        continue
                    candidate = costs[(previous_mask, predecessor)] + float(
                        distances[predecessor, last]
                    )
                    if candidate < best_cost - 1e-12 or (
                        math.isclose(candidate, best_cost, rel_tol=1e-12, abs_tol=1e-12)
                        and predecessor < best_parent
                    ):
                        best_cost = candidate
                        best_parent = predecessor
                costs[(mask, last)] = best_cost
                parents[(mask, last)] = best_parent

    best_total = math.inf
    best_tour: tuple[int, ...] | None = None
    for last in range(1, n):
        total = costs[(full_mask, last)] + float(distances[last, 0])
        candidate_tour = canonicalize_tour(
            _held_karp_reconstruct(parents, full_mask, last),
            n,
        )
        if total < best_total - 1e-12 or (
            math.isclose(total, best_total, rel_tol=1e-12, abs_tol=1e-12)
            and (best_tour is None or candidate_tour < best_tour)
        ):
            best_total = total
            best_tour = candidate_tour
    if best_tour is None:
        raise RuntimeError("Held-Karp failed to produce a tour")
    solution = solution_from_tour(instance, best_tour)
    if not math.isclose(solution.length, best_total, rel_tol=1e-10, abs_tol=1e-10):
        raise RuntimeError("Held-Karp reconstruction is inconsistent with the DP objective")
    return solution


def solve_brute_force(instance: TSPInstance, *, maximum_nodes: int = 10) -> TourSolution:
    """Enumerate all tours for an independent tiny-instance exact oracle."""

    if instance.node_count > maximum_nodes:
        raise ValueError("instance exceeds the brute-force verification limit")
    best: TourSolution | None = None
    for permutation in itertools.permutations(range(1, instance.node_count)):
        if permutation[0] > permutation[-1]:
            continue
        candidate = solution_from_tour(instance, (0,) + permutation)
        if (
            best is None
            or candidate.length < best.length - 1e-12
            or (
                math.isclose(candidate.length, best.length, rel_tol=1e-12, abs_tol=1e-12)
                and candidate.tour < best.tour
            )
        ):
            best = candidate
    if best is None:
        raise RuntimeError("brute-force enumeration failed")
    return best
