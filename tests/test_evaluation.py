from __future__ import annotations

from pathlib import Path

import torch

from ttanco.adaptation import SearchConfig
from ttanco.dataset import generate_dataset
from ttanco.evaluation import evaluate_methods, save_report_csv, save_report_json
from ttanco.model import EdgePolicy, PolicyConfig


def test_evaluation_builds_complete_budget_method_matrix(tmp_path: Path) -> None:
    torch.manual_seed(1)
    model = EdgePolicy(PolicyConfig(hidden_dim=8, message_layers=1, mlp_layers=1))
    dataset = generate_dataset(count=2, node_counts=(5,), regimes=("uniform",), seed=300)
    report = evaluate_methods(
        model,
        dataset,
        scenario="unit",
        budgets=(4, 6),
        methods=("frozen_sampling", "adapter_tta", "nearest_neighbor_2opt"),
        search_template=SearchConfig(
            budget=6,
            adaptation_steps=1,
            batch_size=2,
            seed=1,
        ),
        bootstrap_draws=20,
    )
    assert len(report.instance_results) == 2 * 2 * 3
    assert len(report.summaries) == 2 * 3
    assert report.metadata["source_model_immutable"] is True
    assert all(row.source_model_immutability_rate == 1.0 for row in report.summaries)
    json_path = tmp_path / "report.json"
    csv_path = tmp_path / "report.csv"
    save_report_json(report, json_path)
    save_report_csv(report, csv_path)
    assert json_path.exists()
    assert "mean_relative_gap_percent" in csv_path.read_text(encoding="utf-8")
