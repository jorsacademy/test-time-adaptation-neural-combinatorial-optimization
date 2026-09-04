from __future__ import annotations

from pathlib import Path

from ttanco.experiment import ResearchConfig, ScenarioConfig, run_research


def tiny_config() -> ResearchConfig:
    return ResearchConfig(
        training_count=4,
        training_node_counts=(5, 6),
        training_regimes=("uniform",),
        training_seed=1,
        validation_count=2,
        validation_node_counts=(5,),
        validation_regimes=("uniform",),
        validation_seed=20,
        hidden_dim=8,
        message_layers=1,
        mlp_layers=1,
        training_epochs=1,
        training_learning_rate=0.001,
        training_weight_decay=0.0,
        training_patience=1,
        training_seed_offset=30,
        budgets=(4,),
        methods=("frozen_sampling", "adapter_tta"),
        adaptation_steps=1,
        adaptation_batch_size=2,
        adaptation_learning_rate=0.01,
        entropy_weight=0.001,
        trust_region_weight=0.01,
        adapter_l2_weight=0.0001,
        full_anchor_weight=0.00001,
        gradient_clip=1.0,
        two_opt_passes=5,
        bootstrap_draws=10,
        bootstrap_seed=40,
        exact_limit=8,
        scenarios=(ScenarioConfig("unit", 1, 5, "clustered", 50),),
    )


def test_tiny_research_protocol_runs_and_saves_checkpoint(tmp_path: Path) -> None:
    config = tiny_config()
    report = run_research(config, checkpoint_directory=tmp_path)
    assert report.config_fingerprint
    assert len(report.evaluations) == 1
    assert Path(report.checkpoint).exists()
    restored = ResearchConfig.from_dict(config.to_dict())
    assert restored == config
