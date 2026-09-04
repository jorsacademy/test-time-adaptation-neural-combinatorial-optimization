"""Solver-grounded, budget-matched evaluation of test-time adaptation methods."""

from __future__ import annotations

import csv
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from statistics import median

import numpy as np

from ttanco.adaptation import AVAILABLE_METHODS, SearchConfig, SearchResult, run_method
from ttanco.dataset import TSPDataset
from ttanco.domain import audit_tour
from ttanco.model import EdgePolicy, model_state_fingerprint
from ttanco.utils import bootstrap_mean_interval, percentile, stable_seed, write_json


@dataclass(frozen=True, slots=True)
class InstanceMethodResult:
    scenario: str
    instance_id: str
    regime: str
    node_count: int
    budget: int
    method: str
    tour_length: float
    optimum_length: float
    absolute_gap: float
    relative_gap_percent: float
    initial_length: float
    improvement_percent: float
    recovery_fraction: float
    objective_evaluations: int
    update_steps: int
    adapted_parameter_count: int
    best_found_evaluation: int
    best_found_fraction: float
    runtime_seconds: float
    local_search_move_evaluations: int
    source_model_unchanged: bool
    mean_training_loss: float | None
    gap_at_25_percent_budget: float
    gap_at_50_percent_budget: float
    gap_at_100_percent_budget: float
    mean_anytime_gap_percent: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class MethodSummary:
    scenario: str
    budget: int
    method: str
    instance_count: int
    mean_relative_gap_percent: float
    median_relative_gap_percent: float
    p90_relative_gap_percent: float
    mean_absolute_gap: float
    optimal_hit_rate: float
    mean_improvement_percent: float
    mean_recovery_fraction: float
    mean_objective_evaluations: float
    mean_update_steps: float
    mean_adapted_parameter_count: float
    mean_best_found_fraction: float
    mean_runtime_seconds: float
    mean_local_search_move_evaluations: float
    source_model_immutability_rate: float
    mean_gap_at_25_percent_budget: float
    mean_gap_at_50_percent_budget: float
    mean_gap_at_100_percent_budget: float
    mean_anytime_gap_percent: float
    mean_gap_difference_vs_frozen_sampling: float | None
    gap_difference_ci_low: float | None
    gap_difference_ci_high: float | None
    mean_length_improvement_vs_frozen_sampling_percent: float | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    scenario: str
    budgets: tuple[int, ...]
    methods: tuple[str, ...]
    instance_results: tuple[InstanceMethodResult, ...]
    summaries: tuple[MethodSummary, ...]
    metadata: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "scenario": self.scenario,
            "budgets": list(self.budgets),
            "methods": list(self.methods),
            "instance_results": [result.to_dict() for result in self.instance_results],
            "summaries": [summary.to_dict() for summary in self.summaries],
            "metadata": self.metadata,
        }


def _curve_gap(
    result: SearchResult,
    optimum_length: float,
    fraction: float,
) -> float:
    if not 0.0 < fraction <= 1.0:
        raise ValueError("curve fraction must lie in (0, 1]")
    target = max(1, int(math.ceil(result.objective_evaluations * fraction)))
    eligible = [point for point in result.best_curve if point.objective_evaluations <= target]
    if not eligible:
        raise RuntimeError("best-so-far curve contains no eligible point")
    length = eligible[-1].best_length
    return 100.0 * max(0.0, length - optimum_length) / max(optimum_length, 1e-12)


def _instance_result(
    *,
    scenario: str,
    budget: int,
    search: SearchResult,
    optimum_length: float,
    instance_id: str,
    regime: str,
    node_count: int,
) -> InstanceMethodResult:
    absolute_gap = search.solution.length - optimum_length
    if absolute_gap < -1e-8:
        raise RuntimeError("search result appears better than the exact optimum")
    absolute_gap = max(0.0, absolute_gap)
    relative_gap = 100.0 * absolute_gap / max(optimum_length, 1e-12)
    initial_gap = max(0.0, search.initial_solution.length - optimum_length)
    improvement = (
        100.0
        * max(
            0.0,
            search.initial_solution.length - search.solution.length,
        )
        / max(search.initial_solution.length, 1e-12)
    )
    if initial_gap <= 1e-12:
        recovery = 1.0
    else:
        recovery = min(
            1.0,
            max(
                0.0,
                (search.initial_solution.length - search.solution.length) / initial_gap,
            ),
        )
    curve_gaps = [
        100.0 * max(0.0, point.best_length - optimum_length) / max(optimum_length, 1e-12)
        for point in search.best_curve
    ]
    return InstanceMethodResult(
        scenario=scenario,
        instance_id=instance_id,
        regime=regime,
        node_count=node_count,
        budget=budget,
        method=search.method,
        tour_length=search.solution.length,
        optimum_length=optimum_length,
        absolute_gap=absolute_gap,
        relative_gap_percent=relative_gap,
        initial_length=search.initial_solution.length,
        improvement_percent=improvement,
        recovery_fraction=recovery,
        objective_evaluations=search.objective_evaluations,
        update_steps=search.update_steps,
        adapted_parameter_count=search.adapted_parameter_count,
        best_found_evaluation=search.best_found_evaluation,
        best_found_fraction=search.best_found_evaluation / max(1, search.objective_evaluations),
        runtime_seconds=search.runtime_seconds,
        local_search_move_evaluations=search.local_search_move_evaluations,
        source_model_unchanged=search.source_model_unchanged,
        mean_training_loss=search.mean_training_loss,
        gap_at_25_percent_budget=_curve_gap(search, optimum_length, 0.25),
        gap_at_50_percent_budget=_curve_gap(search, optimum_length, 0.50),
        gap_at_100_percent_budget=_curve_gap(search, optimum_length, 1.0),
        mean_anytime_gap_percent=float(np.mean(np.asarray(curve_gaps, dtype=float))),
    )


def _summary(
    *,
    scenario: str,
    budget: int,
    method: str,
    rows: list[InstanceMethodResult],
    frozen_by_instance: dict[str, InstanceMethodResult],
    bootstrap_seed: int,
    bootstrap_draws: int,
) -> MethodSummary:
    gaps = [row.relative_gap_percent for row in rows]
    gap_differences: list[float] = []
    length_improvements: list[float] = []
    for row in rows:
        frozen = frozen_by_instance.get(row.instance_id)
        if frozen is not None:
            gap_differences.append(row.relative_gap_percent - frozen.relative_gap_percent)
            length_improvements.append(
                100.0 * (frozen.tour_length - row.tour_length) / max(frozen.tour_length, 1e-12)
            )
    ci_low: float | None = None
    ci_high: float | None = None
    mean_difference: float | None = None
    mean_length_improvement: float | None = None
    if gap_differences:
        mean_difference = float(np.mean(np.asarray(gap_differences, dtype=float)))
        ci_low, ci_high = bootstrap_mean_interval(
            gap_differences,
            seed=bootstrap_seed,
            draws=bootstrap_draws,
        )
        mean_length_improvement = float(np.mean(np.asarray(length_improvements, dtype=float)))
    return MethodSummary(
        scenario=scenario,
        budget=budget,
        method=method,
        instance_count=len(rows),
        mean_relative_gap_percent=float(np.mean(np.asarray(gaps, dtype=float))),
        median_relative_gap_percent=float(median(gaps)),
        p90_relative_gap_percent=percentile(gaps, 90.0),
        mean_absolute_gap=float(np.mean(np.asarray([row.absolute_gap for row in rows]))),
        optimal_hit_rate=float(
            np.mean(np.asarray([row.relative_gap_percent <= 1e-8 for row in rows], dtype=float))
        ),
        mean_improvement_percent=float(
            np.mean(np.asarray([row.improvement_percent for row in rows], dtype=float))
        ),
        mean_recovery_fraction=float(
            np.mean(np.asarray([row.recovery_fraction for row in rows], dtype=float))
        ),
        mean_objective_evaluations=float(
            np.mean(np.asarray([row.objective_evaluations for row in rows], dtype=float))
        ),
        mean_update_steps=float(
            np.mean(np.asarray([row.update_steps for row in rows], dtype=float))
        ),
        mean_adapted_parameter_count=float(
            np.mean(np.asarray([row.adapted_parameter_count for row in rows], dtype=float))
        ),
        mean_best_found_fraction=float(
            np.mean(np.asarray([row.best_found_fraction for row in rows], dtype=float))
        ),
        mean_runtime_seconds=float(
            np.mean(np.asarray([row.runtime_seconds for row in rows], dtype=float))
        ),
        mean_local_search_move_evaluations=float(
            np.mean(np.asarray([row.local_search_move_evaluations for row in rows], dtype=float))
        ),
        source_model_immutability_rate=float(
            np.mean(np.asarray([row.source_model_unchanged for row in rows], dtype=float))
        ),
        mean_gap_at_25_percent_budget=float(
            np.mean(np.asarray([row.gap_at_25_percent_budget for row in rows], dtype=float))
        ),
        mean_gap_at_50_percent_budget=float(
            np.mean(np.asarray([row.gap_at_50_percent_budget for row in rows], dtype=float))
        ),
        mean_gap_at_100_percent_budget=float(
            np.mean(np.asarray([row.gap_at_100_percent_budget for row in rows], dtype=float))
        ),
        mean_anytime_gap_percent=float(
            np.mean(np.asarray([row.mean_anytime_gap_percent for row in rows], dtype=float))
        ),
        mean_gap_difference_vs_frozen_sampling=mean_difference,
        gap_difference_ci_low=ci_low,
        gap_difference_ci_high=ci_high,
        mean_length_improvement_vs_frozen_sampling_percent=mean_length_improvement,
    )


def evaluate_methods(
    model: EdgePolicy,
    dataset: TSPDataset,
    *,
    scenario: str,
    budgets: tuple[int, ...] = (8, 16, 32, 64),
    methods: tuple[str, ...] = (
        "frozen_sampling",
        "augmentation_sampling",
        "adapter_tta",
        "full_tta",
        "scratch_active_search",
        "frozen_sampling_2opt",
        "nearest_neighbor_2opt",
    ),
    search_template: SearchConfig | None = None,
    bootstrap_seed: int = 0,
    bootstrap_draws: int = 500,
) -> EvaluationReport:
    """Evaluate methods on exact-labeled instances under explicit test-time budgets."""

    if not scenario:
        raise ValueError("scenario must be nonempty")
    if not budgets or any(budget < 2 for budget in budgets):
        raise ValueError("budgets must be nonempty and at least two")
    if tuple(sorted(set(budgets))) != budgets:
        raise ValueError("budgets must be strictly increasing and unique")
    if not methods or len(set(methods)) != len(methods):
        raise ValueError("methods must be nonempty and unique")
    if any(method not in AVAILABLE_METHODS for method in methods):
        raise ValueError("methods contains an unsupported value")
    if bootstrap_draws <= 0:
        raise ValueError("bootstrap_draws must be positive")
    template = search_template or SearchConfig()
    fingerprint_before = model_state_fingerprint(model)
    instance_results: list[InstanceMethodResult] = []

    for budget in budgets:
        for record in dataset.records:
            for method in methods:
                method_seed = stable_seed(
                    template.seed,
                    dataset.fingerprint,
                    scenario,
                    budget,
                    record.instance.instance_id,
                    method,
                )
                config = replace(template, budget=budget, seed=method_seed)
                search = run_method(method, model, record.instance, config)
                audit = audit_tour(
                    record.instance,
                    search.solution.tour,
                    reported_length=search.solution.length,
                    optimum=record.optimum,
                )
                if not (
                    audit.permutation_valid
                    and audit.reported_length_consistent
                    and audit.relative_gap_percent is not None
                ):
                    raise RuntimeError("evaluated method returned an invalid tour")
                instance_results.append(
                    _instance_result(
                        scenario=scenario,
                        budget=budget,
                        search=search,
                        optimum_length=record.optimum.length,
                        instance_id=record.instance.instance_id,
                        regime=record.instance.regime,
                        node_count=record.instance.node_count,
                    )
                )

    fingerprint_after = model_state_fingerprint(model)
    if fingerprint_before != fingerprint_after:
        raise RuntimeError("evaluation mutated the shared source model")
    summaries: list[MethodSummary] = []
    for budget in budgets:
        budget_rows = [row for row in instance_results if row.budget == budget]
        frozen_by_instance = {
            row.instance_id: row for row in budget_rows if row.method == "frozen_sampling"
        }
        for offset, method in enumerate(methods):
            rows = [row for row in budget_rows if row.method == method]
            if len(rows) != len(dataset.records):
                raise RuntimeError("evaluation result matrix is incomplete")
            summaries.append(
                _summary(
                    scenario=scenario,
                    budget=budget,
                    method=method,
                    rows=rows,
                    frozen_by_instance=frozen_by_instance,
                    bootstrap_seed=bootstrap_seed + 1_000 * budget + offset,
                    bootstrap_draws=bootstrap_draws,
                )
            )
    return EvaluationReport(
        scenario=scenario,
        budgets=budgets,
        methods=methods,
        instance_results=tuple(instance_results),
        summaries=tuple(summaries),
        metadata={
            "dataset_fingerprint": dataset.fingerprint,
            "dataset_record_count": len(dataset.records),
            "dataset_regimes": list(dataset.regimes),
            "dataset_node_counts": list(dataset.node_counts),
            "source_model_fingerprint": fingerprint_before,
            "source_model_immutable": True,
            "bootstrap_seed": bootstrap_seed,
            "bootstrap_draws": bootstrap_draws,
            "objective_budget_definition": "number of complete tour-length evaluations",
            "exact_reference": "Held-Karp dynamic programming",
            "claims_boundary": (
                "Test-time methods return best observed feasible tours; only Held-Karp "
                "certifies optimality on configured small instances."
            ),
        },
    )


def save_report_json(report: EvaluationReport, path: str | Path) -> None:
    write_json(report.to_dict(), path)


def save_report_csv(report: EvaluationReport, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for summary in report.summaries:
        payload = summary.to_dict()
        payload["row_type"] = "summary"
        rows.append(payload)
    fieldnames = sorted({key for row in rows for key in row})
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
