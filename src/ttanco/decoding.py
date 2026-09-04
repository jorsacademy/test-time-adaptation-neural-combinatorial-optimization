"""Autoregressive masked tour construction and metamorphic test-time augmentation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

import numpy as np
import torch
from torch import Tensor

from ttanco.domain import TSPInstance, TourSolution, audit_tour, solution_from_tour
from ttanco.model import EdgePolicy, EmbeddingAdapter


@dataclass(slots=True)
class PolicyRollout:
    """One differentiable trajectory and its audited discrete tour."""

    solution: TourSolution
    raw_tour: tuple[int, ...]
    log_probability: Tensor
    mean_entropy: Tensor
    mean_kl_to_source: Tensor

    def detached_dict(self) -> dict[str, object]:
        return {
            "solution": self.solution.to_dict(),
            "raw_tour": list(self.raw_tour),
            "log_probability": float(self.log_probability.detach().cpu()),
            "mean_entropy": float(self.mean_entropy.detach().cpu()),
            "mean_kl_to_source": float(self.mean_kl_to_source.detach().cpu()),
        }


def coordinates_tensor(instance: TSPInstance, *, device: torch.device | str) -> Tensor:
    return torch.tensor(instance.coordinates, dtype=torch.float32, device=device)


def edge_logits(
    model: EdgePolicy,
    coordinates: Tensor,
    *,
    adapter: EmbeddingAdapter | None = None,
) -> Tensor:
    if adapter is None:
        return cast(Tensor, model(coordinates))
    embeddings, distances = model.encode(coordinates)
    return adapter.adapted_logits(model, embeddings, distances)


def rollout_tour(
    model: EdgePolicy,
    instance: TSPInstance,
    *,
    generator: torch.Generator,
    adapter: EmbeddingAdapter | None = None,
    greedy: bool = False,
    start: int | None = None,
    reference_model: EdgePolicy | None = None,
) -> PolicyRollout:
    """Sample or greedily decode one Hamiltonian cycle with hard visit masking."""

    coordinates = coordinates_tensor(instance, device=model.device)
    logits = edge_logits(model, coordinates, adapter=adapter)
    reference_logits = (
        cast(Tensor, reference_model(coordinates)).detach() if reference_model is not None else None
    )
    n = instance.node_count
    if start is None:
        start = int(torch.randint(n, (1,), generator=generator).item())
    if not 0 <= start < n:
        raise ValueError("start node is out of range")
    visited = torch.zeros(n, dtype=torch.bool, device=model.device)
    visited[start] = True
    tour = [start]
    current = start
    log_probabilities: list[Tensor] = []
    entropies: list[Tensor] = []
    divergences: list[Tensor] = []

    for _ in range(n - 1):
        mask_snapshot = visited.clone()
        row = logits[current].masked_fill(mask_snapshot, -1.0e9)
        log_probabilities_row = torch.log_softmax(row, dim=0)
        probabilities = torch.softmax(row, dim=0)
        if greedy:
            action = int(torch.argmax(row).item())
        else:
            action = int(
                torch.multinomial(
                    probabilities.detach().cpu(),
                    num_samples=1,
                    generator=generator,
                ).item()
            )
        if bool(visited[action]):
            raise RuntimeError("masked decoder selected an already visited node")
        log_probabilities.append(log_probabilities_row[action])
        finite_mask = ~mask_snapshot
        entropies.append(
            -torch.sum(probabilities[finite_mask] * log_probabilities_row[finite_mask])
        )
        if reference_logits is not None:
            reference_row = reference_logits[current].masked_fill(mask_snapshot, -1.0e9)
            reference_log_probabilities = torch.log_softmax(reference_row, dim=0)
            divergences.append(
                torch.sum(
                    probabilities[finite_mask]
                    * (
                        log_probabilities_row[finite_mask]
                        - reference_log_probabilities[finite_mask]
                    )
                )
            )
        visited[action] = True
        tour.append(action)
        current = action

    raw_tour = tuple(tour)
    solution = solution_from_tour(instance, raw_tour)
    audit = audit_tour(instance, solution.tour, reported_length=solution.length)
    if not (audit.permutation_valid and audit.reported_length_consistent):
        raise RuntimeError("decoded tour failed the independent audit")
    log_probability = torch.stack(log_probabilities).sum()
    mean_entropy = torch.stack(entropies).mean()
    mean_kl = (
        torch.stack(divergences).mean()
        if divergences
        else torch.zeros((), dtype=log_probability.dtype, device=log_probability.device)
    )
    if not torch.isfinite(log_probability) or not torch.isfinite(mean_entropy):
        raise RuntimeError("rollout produced non-finite policy statistics")
    if not torch.isfinite(mean_kl) or float(mean_kl.detach().cpu()) < -1e-7:
        raise RuntimeError("rollout produced an invalid KL divergence")
    return PolicyRollout(solution, raw_tour, log_probability, mean_entropy, mean_kl)


def dihedral_transform(coordinates: np.ndarray, transform_index: int) -> np.ndarray:
    """Apply one of eight distance-preserving square symmetries."""

    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("coordinates must have shape [nodes, 2]")
    index = transform_index % 8
    x = coordinates[:, 0]
    y = coordinates[:, 1]
    variants = (
        np.column_stack((x, y)),
        np.column_stack((-y, x)),
        np.column_stack((-x, -y)),
        np.column_stack((y, -x)),
        np.column_stack((-x, y)),
        np.column_stack((x, -y)),
        np.column_stack((y, x)),
        np.column_stack((-y, -x)),
    )
    return np.asarray(variants[index], dtype=np.float64)


def augmented_rollout(
    model: EdgePolicy,
    instance: TSPInstance,
    *,
    evaluation_index: int,
    seed: int,
    greedy: bool = False,
) -> PolicyRollout:
    """Decode an isometric, node-permuted view and map the tour back to original IDs."""

    if evaluation_index < 0 or seed < 0:
        raise ValueError("augmentation indices and seeds must be nonnegative")
    rng = np.random.default_rng(seed + 104_729 * evaluation_index)
    transformed = dihedral_transform(instance.coordinate_array, evaluation_index)
    permutation = rng.permutation(instance.node_count)
    permuted_coordinates = transformed[permutation]
    augmented = TSPInstance(
        tuple((float(row[0]), float(row[1])) for row in permuted_coordinates),
        instance_id=f"{instance.instance_id}-augmentation-{evaluation_index}",
        regime=instance.regime,
        seed=instance.seed,
    )
    torch_generator = torch.Generator(device="cpu")
    torch_generator.manual_seed(seed + evaluation_index)
    rollout = rollout_tour(
        model,
        augmented,
        generator=torch_generator,
        greedy=greedy,
    )
    mapped_raw = tuple(int(permutation[node]) for node in rollout.raw_tour)
    mapped_solution = solution_from_tour(instance, mapped_raw)
    audit = audit_tour(instance, mapped_solution.tour, reported_length=mapped_solution.length)
    if not (audit.permutation_valid and audit.reported_length_consistent):
        raise RuntimeError("augmented rollout failed mapping audit")
    # The transform is isometric, so the objective must be preserved up to floating-point noise.
    if not math.isclose(
        mapped_solution.length,
        rollout.solution.length,
        rel_tol=1e-6,
        abs_tol=1e-6,
    ):
        raise RuntimeError("augmentation changed the Euclidean tour objective")
    return PolicyRollout(
        mapped_solution,
        mapped_raw,
        rollout.log_probability,
        rollout.mean_entropy,
        rollout.mean_kl_to_source,
    )
