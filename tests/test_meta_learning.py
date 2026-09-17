from __future__ import annotations

import math

import torch

from ttanco.dataset import generate_dataset
from ttanco.meta_learning import (
    MetaEmbeddingAdapter,
    MetaLearningConfig,
    build_meta_tasks,
    evaluate_few_shot,
    meta_train_adapter,
)
from ttanco.model import EdgePolicy, PolicyConfig


def test_meta_tasks_and_adapter_are_size_agnostic() -> None:
    dataset = generate_dataset(
        count=8,
        node_counts=(5, 6),
        regimes=("uniform", "clustered"),
        seed=700,
    )
    tasks = build_meta_tasks(dataset, support_size=2, query_size=2)
    assert {task.name for task in tasks} == {"clustered-n6", "uniform-n5"}

    model = EdgePolicy(PolicyConfig(hidden_dim=8, message_layers=1, mlp_layers=1))
    adapter = MetaEmbeddingAdapter(hidden_dim=8, bottleneck_dim=4)
    counts = []
    for record in (dataset.records[0], dataset.records[1]):
        coordinates = torch.tensor(record.instance.coordinates, dtype=torch.float32)
        embeddings, distances = model.encode(coordinates)
        logits = adapter.adapted_logits(model, embeddings, distances)
        assert logits.shape == (record.instance.node_count, record.instance.node_count)
        counts.append(adapter.parameter_count)
    assert counts[0] == counts[1]


def test_first_order_meta_training_and_few_shot_report_are_finite() -> None:
    dataset = generate_dataset(
        count=8,
        node_counts=(5, 6),
        regimes=("uniform", "clustered"),
        seed=800,
    )
    model = EdgePolicy(PolicyConfig(hidden_dim=8, message_layers=1, mlp_layers=1))
    adapter = MetaEmbeddingAdapter(hidden_dim=8, bottleneck_dim=4)
    before = {key: value.detach().clone() for key, value in adapter.state_dict().items()}
    trained, history = meta_train_adapter(
        model,
        dataset,
        adapter=adapter,
        config=MetaLearningConfig(
            outer_epochs=1,
            inner_steps=1,
            support_size=2,
            query_size=2,
            inner_learning_rate=1e-2,
            outer_learning_rate=1e-3,
            seed=4,
        ),
    )
    assert len(history) == 1
    assert math.isfinite(history[0])
    assert any(
        not torch.equal(before[key], value)
        for key, value in trained.state_dict().items()
    )

    task = build_meta_tasks(dataset, support_size=2, query_size=2)[0]
    report = evaluate_few_shot(
        model,
        trained,
        dataset,
        task,
        inner_steps=1,
        inner_learning_rate=1e-2,
    )
    values = (
        report.query_loss_before,
        report.query_loss_after,
        report.mean_gap_before_pct,
        report.mean_gap_after_pct,
    )
    assert all(math.isfinite(value) for value in values)
    assert report.support_size == 2
    assert report.query_size == 2
