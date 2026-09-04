from __future__ import annotations

import math

from ttanco.dataset import generate_dataset
from ttanco.model import PolicyConfig, validate_policy
from ttanco.training import TrainingConfig, tour_edge_targets, train_policy


def test_tour_edge_targets_have_degree_two() -> None:
    target = tour_edge_targets(5, (0, 1, 2, 3, 4))
    assert target.shape == (5, 5)
    assert (target.sum(axis=1) == 2).all()
    assert (target == target.T).all()


def test_small_training_run_is_finite_and_reproducible() -> None:
    training = generate_dataset(count=5, node_counts=(5, 6), regimes=("uniform",), seed=10)
    validation = generate_dataset(count=3, node_counts=(5,), regimes=("uniform",), seed=50)
    kwargs = {
        "model_config": PolicyConfig(hidden_dim=8, message_layers=1, mlp_layers=1),
        "training_config": TrainingConfig(epochs=2, patience=2, seed=3),
    }
    first_model, first = train_policy(training, validation, **kwargs)
    second_model, second = train_policy(training, validation, **kwargs)
    validate_policy(first_model)
    assert first.best_validation_loss == second.best_validation_loss
    assert first_model.state_dict().keys() == second_model.state_dict().keys()
    assert math.isfinite(first.best_validation_loss)
    assert len(first.epochs) == 2
