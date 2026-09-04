from __future__ import annotations

from pathlib import Path

from ttanco.cli import main


def test_cli_end_to_end_smoke(tmp_path: Path) -> None:
    training = tmp_path / "training.jsonl"
    validation = tmp_path / "validation.jsonl"
    checkpoint = tmp_path / "source.safetensors"
    benchmark = tmp_path / "benchmark.json"
    benchmark_csv = tmp_path / "benchmark.csv"
    oracle = tmp_path / "oracle.json"
    single = tmp_path / "single.json"

    assert main(["generate", "--nodes", "5", "--seed", "1", "--output", str(single)]) == 0
    assert single.exists()
    assert (
        main(
            [
                "collect",
                "--count",
                "4",
                "--node-counts",
                "5",
                "--seed",
                "10",
                "--output",
                str(training),
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "collect",
                "--count",
                "2",
                "--node-counts",
                "5",
                "--seed",
                "20",
                "--output",
                str(validation),
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "train",
                str(training),
                "--validation",
                str(validation),
                "--epochs",
                "1",
                "--patience",
                "1",
                "--hidden-dim",
                "8",
                "--message-layers",
                "1",
                "--mlp-layers",
                "1",
                "--checkpoint",
                str(checkpoint),
            ]
        )
        == 0
    )
    assert main(["oracle", str(validation), "--output", str(oracle)]) == 0
    assert (
        main(
            [
                "solve",
                str(single),
                "--checkpoint",
                str(checkpoint),
                "--method",
                "adapter_tta",
                "--budget",
                "4",
                "--adaptation-steps",
                "1",
                "--batch-size",
                "2",
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "benchmark",
                str(validation),
                "--checkpoint",
                str(checkpoint),
                "--budgets",
                "4",
                "--methods",
                "frozen_sampling",
                "adapter_tta",
                "--adaptation-steps",
                "1",
                "--batch-size",
                "2",
                "--bootstrap-draws",
                "10",
                "--output-json",
                str(benchmark),
                "--output-csv",
                str(benchmark_csv),
            ]
        )
        == 0
    )
    assert benchmark.exists() and benchmark_csv.exists() and oracle.exists()


def test_cli_reports_errors_without_traceback(tmp_path: Path) -> None:
    missing = tmp_path / "missing.jsonl"
    assert main(["oracle", str(missing)]) == 2
