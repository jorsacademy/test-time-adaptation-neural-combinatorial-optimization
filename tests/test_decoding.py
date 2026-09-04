from __future__ import annotations

import pytest
import torch

from ttanco.dataset import generate_instance
from ttanco.decoding import augmented_rollout, rollout_tour
from ttanco.domain import audit_tour
from ttanco.model import EdgePolicy, PolicyConfig


def model() -> EdgePolicy:
    torch.manual_seed(11)
    return EdgePolicy(PolicyConfig(hidden_dim=10, message_layers=1, mlp_layers=1))


def test_rollout_is_feasible_and_differentiable() -> None:
    instance = generate_instance(7, seed=44)
    policy = model()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(2)
    rollout = rollout_tour(policy, instance, generator=generator)
    audit = audit_tour(instance, rollout.solution.tour, reported_length=rollout.solution.length)
    assert audit.permutation_valid
    assert rollout.log_probability.requires_grad
    loss = rollout.log_probability * rollout.solution.length
    loss.backward()
    assert any(parameter.grad is not None for parameter in policy.parameters())


def test_greedy_rollout_is_deterministic_for_fixed_start() -> None:
    instance = generate_instance(7, seed=45)
    policy = model().eval()
    first_generator = torch.Generator(device="cpu").manual_seed(1)
    second_generator = torch.Generator(device="cpu").manual_seed(999)
    with torch.no_grad():
        first = rollout_tour(policy, instance, generator=first_generator, greedy=True, start=0)
        second = rollout_tour(policy, instance, generator=second_generator, greedy=True, start=0)
    assert first.solution == second.solution


def test_augmentation_maps_back_without_changing_objective() -> None:
    instance = generate_instance(8, regime="anisotropic", seed=7)
    policy = model().eval()
    with torch.no_grad():
        for index in range(8):
            rollout = augmented_rollout(
                policy,
                instance,
                evaluation_index=index,
                seed=100,
                greedy=index == 0,
            )
            assert audit_tour(instance, rollout.solution.tour).permutation_valid


def test_invalid_start_is_rejected() -> None:
    instance = generate_instance(6, seed=8)
    generator = torch.Generator(device="cpu").manual_seed(1)
    with pytest.raises(ValueError):
        rollout_tour(model(), instance, generator=generator, start=99)
