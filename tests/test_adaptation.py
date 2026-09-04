from __future__ import annotations

from itertools import pairwise

import pytest
import torch

from ttanco.adaptation import SearchConfig, run_method
from ttanco.dataset import generate_instance
from ttanco.model import EdgePolicy, PolicyConfig, model_state_fingerprint


def source_model() -> EdgePolicy:
    torch.manual_seed(19)
    return EdgePolicy(PolicyConfig(hidden_dim=8, message_layers=1, mlp_layers=1))


def test_budget_matched_methods_preserve_source_and_best_so_far() -> None:
    instance = generate_instance(7, regime="clustered", seed=33)
    model = source_model()
    fingerprint = model_state_fingerprint(model)
    config = SearchConfig(
        budget=8,
        adaptation_steps=2,
        batch_size=2,
        learning_rate=0.01,
        seed=5,
    )
    for method in (
        "frozen_sampling",
        "augmentation_sampling",
        "adapter_tta",
        "full_tta",
        "scratch_active_search",
        "frozen_sampling_2opt",
    ):
        result = run_method(method, model, instance, config)
        assert result.objective_evaluations == 8
        assert result.solution.length <= result.initial_solution.length + 1e-9
        assert result.source_model_unchanged
        assert len(result.best_curve) == 8
        assert all(
            left.best_length >= right.best_length - 1e-12
            for left, right in pairwise(result.best_curve)
        )
        assert model_state_fingerprint(model) == fingerprint


def test_adapter_updates_fewer_parameters_than_full_finetuning() -> None:
    instance = generate_instance(6, seed=9)
    model = source_model()
    config = SearchConfig(budget=6, adaptation_steps=1, batch_size=2, seed=9)
    adapter = run_method("adapter_tta", model, instance, config)
    full = run_method("full_tta", model, instance, config)
    assert adapter.update_steps == 1
    assert adapter.adapted_parameter_count < full.adapted_parameter_count
    assert adapter.adapter_norm is not None


def test_classical_two_opt_reports_extra_move_evaluations() -> None:
    instance = generate_instance(8, seed=2)
    result = run_method(
        "nearest_neighbor_2opt",
        source_model(),
        instance,
        SearchConfig(budget=4, seed=2),
    )
    assert result.objective_evaluations == 1
    assert result.local_search_move_evaluations > 0


def test_invalid_search_configuration_and_method_are_rejected() -> None:
    with pytest.raises(ValueError):
        SearchConfig(budget=1)
    with pytest.raises(ValueError):
        SearchConfig(batch_size=1)
    with pytest.raises(ValueError):
        run_method(
            "unknown",
            source_model(),
            generate_instance(5, seed=1),
            SearchConfig(budget=4),
        )
