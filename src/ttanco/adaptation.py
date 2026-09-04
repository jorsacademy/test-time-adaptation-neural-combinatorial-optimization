"""Budget-matched frozen search, efficient adapters, full fine-tuning, and active search."""

from __future__ import annotations

import math
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
import torch
from torch import Tensor, nn

from ttanco.decoding import PolicyRollout, augmented_rollout, rollout_tour
from ttanco.domain import (
    TourSolution,
    TSPInstance,
    audit_tour,
    nearest_neighbor_tour,
    two_opt,
)
from ttanco.model import (
    EdgePolicy,
    EmbeddingAdapter,
    clone_policy,
    model_state_fingerprint,
    validate_policy,
)
from ttanco.utils import seed_everything

SearchMethod = Literal[
    "frozen_sampling",
    "augmentation_sampling",
    "adapter_tta",
    "full_tta",
    "scratch_active_search",
    "frozen_sampling_2opt",
    "nearest_neighbor_2opt",
]

AVAILABLE_METHODS: tuple[str, ...] = (
    "frozen_sampling",
    "augmentation_sampling",
    "adapter_tta",
    "full_tta",
    "scratch_active_search",
    "frozen_sampling_2opt",
    "nearest_neighbor_2opt",
)


@dataclass(frozen=True, slots=True)
class SearchConfig:
    budget: int = 64
    adaptation_steps: int = 8
    batch_size: int = 4
    learning_rate: float = 2e-2
    entropy_weight: float = 1e-3
    trust_region_weight: float = 5e-2
    adapter_l2_weight: float = 1e-4
    full_anchor_weight: float = 1e-5
    gradient_clip: float = 2.0
    two_opt_passes: int = 50
    seed: int = 0

    def __post_init__(self) -> None:
        if self.budget < 2:
            raise ValueError("objective-evaluation budget must be at least two")
        if self.adaptation_steps < 0 or self.batch_size < 2:
            raise ValueError("adaptation steps and batch size are invalid")
        if self.learning_rate <= 0.0 or self.gradient_clip <= 0.0:
            raise ValueError("learning rate and gradient clip must be positive")
        if self.entropy_weight < 0.0 or self.trust_region_weight < 0.0:
            raise ValueError("regularization weights must be nonnegative")
        if self.adapter_l2_weight < 0.0 or self.full_anchor_weight < 0.0:
            raise ValueError("regularization weights must be nonnegative")
        if self.two_opt_passes <= 0 or self.seed < 0:
            raise ValueError("two-opt passes or seed is invalid")


@dataclass(frozen=True, slots=True)
class BudgetPoint:
    objective_evaluations: int
    best_length: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SearchResult:
    method: str
    solution: TourSolution
    initial_solution: TourSolution
    objective_evaluations: int
    update_steps: int
    adapted_parameter_count: int
    best_found_evaluation: int
    runtime_seconds: float
    local_search_move_evaluations: int
    source_model_unchanged: bool
    adapter_norm: float | None
    mean_training_loss: float | None
    best_curve: tuple[BudgetPoint, ...]
    metadata: dict[str, object]

    @property
    def improvement(self) -> float:
        return self.initial_solution.length - self.solution.length

    def to_dict(self) -> dict[str, object]:
        return {
            "method": self.method,
            "solution": self.solution.to_dict(),
            "initial_solution": self.initial_solution.to_dict(),
            "improvement": self.improvement,
            "objective_evaluations": self.objective_evaluations,
            "update_steps": self.update_steps,
            "adapted_parameter_count": self.adapted_parameter_count,
            "best_found_evaluation": self.best_found_evaluation,
            "runtime_seconds": self.runtime_seconds,
            "local_search_move_evaluations": self.local_search_move_evaluations,
            "source_model_unchanged": self.source_model_unchanged,
            "adapter_norm": self.adapter_norm,
            "mean_training_loss": self.mean_training_loss,
            "best_curve": [point.to_dict() for point in self.best_curve],
            "metadata": self.metadata,
        }


class _BestTracker:
    def __init__(self, instance: TSPInstance) -> None:
        self.instance = instance
        self.best: TourSolution | None = None
        self.evaluations = 0
        self.best_found_evaluation = 0
        self.curve: list[BudgetPoint] = []

    def record(self, solution: TourSolution) -> None:
        audit = audit_tour(
            self.instance,
            solution.tour,
            reported_length=solution.length,
        )
        if not (audit.permutation_valid and audit.reported_length_consistent):
            raise RuntimeError("search produced an invalid tour")
        self.evaluations += 1
        if (
            self.best is None
            or solution.length < self.best.length - 1e-12
            or (
                math.isclose(solution.length, self.best.length, rel_tol=1e-12, abs_tol=1e-12)
                and solution.tour < self.best.tour
            )
        ):
            self.best = solution
            self.best_found_evaluation = self.evaluations
        if self.best is None:  # pragma: no cover - defensive assertion.
            raise RuntimeError("best tracker failed to retain a solution")
        self.curve.append(BudgetPoint(self.evaluations, self.best.length))


def _torch_generator(seed: int) -> torch.Generator:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return generator


def _initial_greedy(
    model: EdgePolicy,
    instance: TSPInstance,
    tracker: _BestTracker,
    *,
    generator: torch.Generator,
    adapter: EmbeddingAdapter | None = None,
) -> TourSolution:
    with torch.no_grad():
        rollout = rollout_tour(
            model,
            instance,
            generator=generator,
            adapter=adapter,
            greedy=True,
            start=0,
        )
    tracker.record(rollout.solution)
    return rollout.solution


def _frozen_sampling(
    model: EdgePolicy,
    instance: TSPInstance,
    config: SearchConfig,
) -> SearchResult:
    start_time = time.perf_counter()
    fingerprint_before = model_state_fingerprint(model)
    generator = _torch_generator(config.seed)
    tracker = _BestTracker(instance)
    initial = _initial_greedy(model, instance, tracker, generator=generator)
    model.eval()
    with torch.no_grad():
        while tracker.evaluations < config.budget:
            rollout = rollout_tour(model, instance, generator=generator)
            tracker.record(rollout.solution)
    fingerprint_after = model_state_fingerprint(model)
    return _finalize_result(
        method="frozen_sampling",
        tracker=tracker,
        initial=initial,
        start_time=start_time,
        update_steps=0,
        adapted_parameter_count=0,
        local_search_move_evaluations=0,
        source_unchanged=fingerprint_before == fingerprint_after,
        adapter_norm=None,
        losses=(),
        metadata={"test_time_gradients": False, "metamorphic_augmentations": False},
    )


def _augmentation_sampling(
    model: EdgePolicy,
    instance: TSPInstance,
    config: SearchConfig,
) -> SearchResult:
    start_time = time.perf_counter()
    fingerprint_before = model_state_fingerprint(model)
    tracker = _BestTracker(instance)
    first = augmented_rollout(
        model,
        instance,
        evaluation_index=0,
        seed=config.seed,
        greedy=True,
    )
    tracker.record(first.solution)
    initial = first.solution
    with torch.no_grad():
        while tracker.evaluations < config.budget:
            rollout = augmented_rollout(
                model,
                instance,
                evaluation_index=tracker.evaluations,
                seed=config.seed,
                greedy=False,
            )
            tracker.record(rollout.solution)
    fingerprint_after = model_state_fingerprint(model)
    return _finalize_result(
        method="augmentation_sampling",
        tracker=tracker,
        initial=initial,
        start_time=start_time,
        update_steps=0,
        adapted_parameter_count=0,
        local_search_move_evaluations=0,
        source_unchanged=fingerprint_before == fingerprint_after,
        adapter_norm=None,
        losses=(),
        metadata={
            "test_time_gradients": False,
            "metamorphic_augmentations": True,
            "augmentation_family": "dihedral-isometries-plus-node-permutations",
        },
    )


def _reinforce_loss(
    rollouts: Iterable[PolicyRollout],
    *,
    entropy_weight: float,
    trust_region_weight: float,
    regularizer: Tensor,
) -> Tensor:
    materialized = tuple(rollouts)
    if len(materialized) < 2:
        raise ValueError("REINFORCE updates require at least two rollouts")
    device = materialized[0].log_probability.device
    lengths = torch.tensor(
        [rollout.solution.length for rollout in materialized],
        dtype=torch.float32,
        device=device,
    )
    log_probabilities = torch.stack([rollout.log_probability for rollout in materialized])
    entropies = torch.stack([rollout.mean_entropy for rollout in materialized])
    divergences = torch.stack([rollout.mean_kl_to_source for rollout in materialized])
    advantages = lengths - torch.mean(lengths)
    loss = (
        torch.mean(advantages.detach() * log_probabilities)
        - entropy_weight * torch.mean(entropies)
        + trust_region_weight * torch.mean(divergences)
        + regularizer
    )
    if not torch.isfinite(loss):
        raise RuntimeError("test-time adaptation loss is non-finite")
    return loss


def _effective_steps(config: SearchConfig) -> int:
    return min(config.adaptation_steps, max(0, (config.budget - 1) // config.batch_size))


def _record_rollouts(tracker: _BestTracker, rollouts: Iterable[PolicyRollout]) -> None:
    for rollout in rollouts:
        tracker.record(rollout.solution)


def _remaining_sampling(
    model: EdgePolicy,
    instance: TSPInstance,
    tracker: _BestTracker,
    config: SearchConfig,
    *,
    generator: torch.Generator,
    adapter: EmbeddingAdapter | None = None,
) -> None:
    first = True
    model.eval()
    with torch.no_grad():
        while tracker.evaluations < config.budget:
            rollout = rollout_tour(
                model,
                instance,
                generator=generator,
                adapter=adapter,
                greedy=first,
            )
            tracker.record(rollout.solution)
            first = False


def _adapter_tta(
    model: EdgePolicy,
    instance: TSPInstance,
    config: SearchConfig,
) -> SearchResult:
    start_time = time.perf_counter()
    fingerprint_before = model_state_fingerprint(model)
    generator = _torch_generator(config.seed)
    tracker = _BestTracker(instance)
    model.eval()
    original_requires_grad = [parameter.requires_grad for parameter in model.parameters()]
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    adapter = EmbeddingAdapter(instance.node_count, model.config.hidden_dim).to(model.device)
    optimizer = torch.optim.Adam(adapter.parameters(), lr=config.learning_rate)
    losses: list[float] = []
    try:
        initial = _initial_greedy(
            model,
            instance,
            tracker,
            generator=generator,
            adapter=adapter,
        )
        steps = _effective_steps(config)
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            rollouts = [
                rollout_tour(
                    model,
                    instance,
                    generator=generator,
                    adapter=adapter,
                    reference_model=model,
                )
                for _ in range(config.batch_size)
            ]
            regularizer = (
                config.adapter_l2_weight
                * adapter.squared_norm
                / float(max(1, adapter.parameter_count))
            )
            loss = _reinforce_loss(
                rollouts,
                entropy_weight=config.entropy_weight,
                trust_region_weight=config.trust_region_weight,
                regularizer=regularizer,
            )
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                adapter.parameters(),
                max_norm=config.gradient_clip,
            )
            if not torch.isfinite(gradient_norm):
                raise RuntimeError("adapter TTA produced non-finite gradients")
            optimizer.step()
            _validate_parameters(adapter)
            _record_rollouts(tracker, rollouts)
            losses.append(float(loss.detach().cpu()))
        _remaining_sampling(
            model,
            instance,
            tracker,
            config,
            generator=generator,
            adapter=adapter,
        )
    finally:
        for parameter, requires_grad in zip(
            model.parameters(), original_requires_grad, strict=True
        ):
            parameter.requires_grad_(requires_grad)
    fingerprint_after = model_state_fingerprint(model)
    adapter_norm = math.sqrt(float(adapter.squared_norm.detach().cpu()))
    return _finalize_result(
        method="adapter_tta",
        tracker=tracker,
        initial=initial,
        start_time=start_time,
        update_steps=steps,
        adapted_parameter_count=adapter.parameter_count,
        local_search_move_evaluations=0,
        source_unchanged=fingerprint_before == fingerprint_after,
        adapter_norm=adapter_norm,
        losses=losses,
        metadata={
            "test_time_gradients": True,
            "adaptation_scope": "instance-embedding-node-bias-temperature",
            "trust_region_weight": config.trust_region_weight,
            "best_so_far_returned": True,
        },
    )


def _parameter_anchor(model: EdgePolicy, source: EdgePolicy) -> Tensor:
    terms = [
        torch.sum((parameter - reference.detach()) ** 2)
        for parameter, reference in zip(model.parameters(), source.parameters(), strict=True)
    ]
    return torch.stack(terms).sum() / float(max(1, model.parameter_count))


def _active_search_model(
    working_model: EdgePolicy,
    instance: TSPInstance,
    config: SearchConfig,
    *,
    method: str,
    source_model: EdgePolicy | None,
    anchor_weight: float,
) -> SearchResult:
    start_time = time.perf_counter()
    source_fingerprint_before = (
        model_state_fingerprint(source_model) if source_model is not None else None
    )
    generator = _torch_generator(config.seed)
    tracker = _BestTracker(instance)
    optimizer = torch.optim.Adam(working_model.parameters(), lr=config.learning_rate)
    losses: list[float] = []
    initial = _initial_greedy(working_model, instance, tracker, generator=generator)
    steps = _effective_steps(config)
    for _ in range(steps):
        working_model.train()
        optimizer.zero_grad(set_to_none=True)
        rollouts = [
            rollout_tour(
                working_model,
                instance,
                generator=generator,
                reference_model=source_model,
            )
            for _ in range(config.batch_size)
        ]
        regularizer = torch.zeros((), dtype=torch.float32, device=working_model.device)
        if source_model is not None and anchor_weight > 0.0:
            regularizer = anchor_weight * _parameter_anchor(working_model, source_model)
        loss = _reinforce_loss(
            rollouts,
            entropy_weight=config.entropy_weight,
            trust_region_weight=config.trust_region_weight if source_model is not None else 0.0,
            regularizer=regularizer,
        )
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            working_model.parameters(),
            max_norm=config.gradient_clip,
        )
        if not torch.isfinite(gradient_norm):
            raise RuntimeError(f"{method} produced non-finite gradients")
        optimizer.step()
        validate_policy(working_model)
        _record_rollouts(tracker, rollouts)
        losses.append(float(loss.detach().cpu()))
    _remaining_sampling(
        working_model,
        instance,
        tracker,
        config,
        generator=generator,
    )
    source_unchanged = True
    if source_model is not None and source_fingerprint_before is not None:
        source_unchanged = source_fingerprint_before == model_state_fingerprint(source_model)
    return _finalize_result(
        method=method,
        tracker=tracker,
        initial=initial,
        start_time=start_time,
        update_steps=steps,
        adapted_parameter_count=working_model.parameter_count,
        local_search_move_evaluations=0,
        source_unchanged=source_unchanged,
        adapter_norm=None,
        losses=losses,
        metadata={
            "test_time_gradients": True,
            "adaptation_scope": "all-policy-parameters",
            "source_warm_start": source_model is not None,
            "best_so_far_returned": True,
        },
    )


def _full_tta(model: EdgePolicy, instance: TSPInstance, config: SearchConfig) -> SearchResult:
    working = clone_policy(model)
    source = clone_policy(model)
    for parameter in source.parameters():
        parameter.requires_grad_(False)
    source.eval()
    return _active_search_model(
        working,
        instance,
        config,
        method="full_tta",
        source_model=source,
        anchor_weight=config.full_anchor_weight,
    )


def _scratch_active_search(
    model: EdgePolicy,
    instance: TSPInstance,
    config: SearchConfig,
) -> SearchResult:
    seed_everything(config.seed)
    scratch = EdgePolicy(model.config).to(model.device)
    return _active_search_model(
        scratch,
        instance,
        config,
        method="scratch_active_search",
        source_model=None,
        anchor_weight=0.0,
    )


def _nearest_neighbor_two_opt(
    model: EdgePolicy,
    instance: TSPInstance,
    config: SearchConfig,
) -> SearchResult:
    del model
    start_time = time.perf_counter()
    initial = nearest_neighbor_tour(instance)
    result = two_opt(instance, initial.tour, maximum_passes=config.two_opt_passes)
    tracker = _BestTracker(instance)
    tracker.record(initial)
    if result.solution != initial:
        tracker.best = result.solution
        tracker.best_found_evaluation = tracker.evaluations
        tracker.curve[-1] = BudgetPoint(tracker.evaluations, result.solution.length)
    return _finalize_result(
        method="nearest_neighbor_2opt",
        tracker=tracker,
        initial=initial,
        start_time=start_time,
        update_steps=0,
        adapted_parameter_count=0,
        local_search_move_evaluations=result.move_evaluations,
        source_unchanged=True,
        adapter_norm=None,
        losses=(),
        metadata={
            "test_time_gradients": False,
            "matched_tour_budget": False,
            "local_search": "deterministic-best-improvement-2opt",
        },
    )


def _frozen_sampling_two_opt(
    model: EdgePolicy,
    instance: TSPInstance,
    config: SearchConfig,
) -> SearchResult:
    base = _frozen_sampling(model, instance, config)
    start_time = time.perf_counter()
    improvement = two_opt(
        instance,
        base.solution.tour,
        maximum_passes=config.two_opt_passes,
    )
    return SearchResult(
        method="frozen_sampling_2opt",
        solution=improvement.solution,
        initial_solution=base.initial_solution,
        objective_evaluations=base.objective_evaluations,
        update_steps=0,
        adapted_parameter_count=0,
        best_found_evaluation=base.best_found_evaluation,
        runtime_seconds=base.runtime_seconds + (time.perf_counter() - start_time),
        local_search_move_evaluations=improvement.move_evaluations,
        source_model_unchanged=base.source_model_unchanged,
        adapter_norm=None,
        mean_training_loss=None,
        best_curve=base.best_curve,
        metadata={
            **base.metadata,
            "matched_tour_budget": False,
            "local_search": "deterministic-best-improvement-2opt",
        },
    )


def _validate_parameters(module: nn.Module) -> None:
    for name, parameter in module.named_parameters():
        if not torch.all(torch.isfinite(parameter)):
            raise RuntimeError(f"adapted parameter {name!r} contains non-finite values")


def _finalize_result(
    *,
    method: str,
    tracker: _BestTracker,
    initial: TourSolution,
    start_time: float,
    update_steps: int,
    adapted_parameter_count: int,
    local_search_move_evaluations: int,
    source_unchanged: bool,
    adapter_norm: float | None,
    losses: Iterable[float],
    metadata: dict[str, object],
) -> SearchResult:
    if tracker.best is None:
        raise RuntimeError("search returned no tour")
    materialized_losses = tuple(float(loss) for loss in losses)
    if any(not math.isfinite(loss) for loss in materialized_losses):
        raise RuntimeError("search loss history contains non-finite values")
    return SearchResult(
        method=method,
        solution=tracker.best,
        initial_solution=initial,
        objective_evaluations=tracker.evaluations,
        update_steps=update_steps,
        adapted_parameter_count=adapted_parameter_count,
        best_found_evaluation=tracker.best_found_evaluation,
        runtime_seconds=time.perf_counter() - start_time,
        local_search_move_evaluations=local_search_move_evaluations,
        source_model_unchanged=source_unchanged,
        adapter_norm=adapter_norm,
        mean_training_loss=(
            float(np.mean(np.asarray(materialized_losses, dtype=float)))
            if materialized_losses
            else None
        ),
        best_curve=tuple(tracker.curve),
        metadata=metadata,
    )


def run_method(
    method: str,
    model: EdgePolicy,
    instance: TSPInstance,
    config: SearchConfig,
) -> SearchResult:
    """Run one test-time method with explicit objective-evaluation accounting."""

    if method not in AVAILABLE_METHODS:
        raise ValueError(f"unsupported search method: {method}")
    dispatch = {
        "frozen_sampling": _frozen_sampling,
        "augmentation_sampling": _augmentation_sampling,
        "adapter_tta": _adapter_tta,
        "full_tta": _full_tta,
        "scratch_active_search": _scratch_active_search,
        "frozen_sampling_2opt": _frozen_sampling_two_opt,
        "nearest_neighbor_2opt": _nearest_neighbor_two_opt,
    }
    result = dispatch[method](model, instance, config)
    if method not in {"nearest_neighbor_2opt"} and result.objective_evaluations != config.budget:
        raise RuntimeError("method violated its declared objective-evaluation budget")
    if not result.source_model_unchanged:
        raise RuntimeError("test-time method mutated the shared source model")
    if result.solution.length > result.initial_solution.length + 1e-9:
        raise RuntimeError("best-so-far return policy regressed below the initial solution")
    return result
