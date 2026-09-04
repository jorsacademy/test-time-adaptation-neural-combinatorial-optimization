"""Command-line workflows for generation, training, adaptation, and evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import cast

from ttanco.adaptation import AVAILABLE_METHODS, SearchConfig, run_method
from ttanco.dataset import (
    SUPPORTED_REGIMES,
    TSPDataset,
    TSPRecord,
    generate_dataset,
    generate_instance,
    load_dataset_jsonl,
    save_dataset_jsonl,
)
from ttanco.domain import (
    TSPInstance,
    TourSolution,
    audit_tour,
    solve_brute_force,
    solve_held_karp,
)
from ttanco.evaluation import evaluate_methods, save_report_csv, save_report_json
from ttanco.experiment import load_research_config, run_research, save_research_report
from ttanco.model import PolicyConfig, load_checkpoint, save_checkpoint
from ttanco.training import TrainingConfig, train_policy
from ttanco.utils import read_json, write_json

SINGLE_INSTANCE_SCHEMA = "ttanco-single-instance-v1"


def _write_stdout(payload: object) -> None:
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False))


def _single_payload(instance: TSPInstance, optimum: TourSolution) -> dict[str, object]:
    return {
        "schema_version": SINGLE_INSTANCE_SCHEMA,
        "instance": instance.to_dict(),
        "optimum": optimum.to_dict(),
    }


def _load_single(path: str | Path) -> TSPRecord:
    payload = read_json(path)
    if not isinstance(payload, dict) or payload.get("schema_version") != SINGLE_INSTANCE_SCHEMA:
        raise ValueError("input is not a supported single-instance artifact")
    raw_instance = payload.get("instance")
    raw_optimum = payload.get("optimum")
    if not isinstance(raw_instance, dict) or not isinstance(raw_optimum, dict):
        raise ValueError("single-instance artifact is malformed")
    record = TSPRecord(
        TSPInstance.from_dict(cast(dict[str, object], raw_instance)),
        TourSolution.from_dict(cast(dict[str, object], raw_optimum)),
    )
    recomputed = solve_held_karp(record.instance)
    if abs(recomputed.length - record.optimum.length) > 1e-8:
        raise ValueError("stored single-instance optimum failed exact recomputation")
    return record


def _search_config(args: argparse.Namespace, *, budget: int | None = None) -> SearchConfig:
    return SearchConfig(
        budget=args.budget if budget is None else budget,
        adaptation_steps=args.adaptation_steps,
        batch_size=args.batch_size,
        learning_rate=args.adaptation_learning_rate,
        entropy_weight=args.entropy_weight,
        trust_region_weight=args.trust_region_weight,
        adapter_l2_weight=args.adapter_l2_weight,
        full_anchor_weight=args.full_anchor_weight,
        gradient_clip=args.gradient_clip,
        two_opt_passes=args.two_opt_passes,
        seed=args.seed,
    )


def _add_search_arguments(parser: argparse.ArgumentParser, *, include_budget: bool = True) -> None:
    if include_budget:
        parser.add_argument("--budget", type=int, default=32)
    parser.add_argument("--adaptation-steps", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--adaptation-learning-rate", type=float, default=0.02)
    parser.add_argument("--entropy-weight", type=float, default=0.001)
    parser.add_argument("--trust-region-weight", type=float, default=0.05)
    parser.add_argument("--adapter-l2-weight", type=float, default=0.0001)
    parser.add_argument("--full-anchor-weight", type=float, default=0.00001)
    parser.add_argument("--gradient-clip", type=float, default=2.0)
    parser.add_argument("--two-opt-passes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ttanco",
        description="Verification-first test-time adaptation for neural TSP policies.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="generate one exact-labeled TSP instance")
    generate.add_argument("--nodes", type=int, default=10)
    generate.add_argument("--regime", choices=SUPPORTED_REGIMES, default="uniform")
    generate.add_argument("--seed", type=int, default=42)
    generate.add_argument("--output", required=True)

    collect = subparsers.add_parser("collect", help="build an exact-labeled JSONL corpus")
    collect.add_argument("--count", type=int, default=64)
    collect.add_argument("--node-counts", type=int, nargs="+", default=[8, 10, 12])
    collect.add_argument("--regimes", nargs="+", choices=SUPPORTED_REGIMES, default=["uniform"])
    collect.add_argument("--seed", type=int, default=1000)
    collect.add_argument("--exact-limit", type=int, default=18)
    collect.add_argument("--output", required=True)

    oracle = subparsers.add_parser("oracle", help="audit Held-Karp against brute force")
    oracle.add_argument("input")
    oracle.add_argument("--sample-index", type=int, default=0)
    oracle.add_argument("--output")

    train = subparsers.add_parser("train", help="train a source edge policy")
    train.add_argument("training")
    train.add_argument("--validation", required=True)
    train.add_argument("--hidden-dim", type=int, default=64)
    train.add_argument("--message-layers", type=int, default=2)
    train.add_argument("--mlp-layers", type=int, default=2)
    train.add_argument("--epochs", type=int, default=30)
    train.add_argument("--learning-rate", type=float, default=0.001)
    train.add_argument("--weight-decay", type=float, default=0.00001)
    train.add_argument("--gradient-clip", type=float, default=1.0)
    train.add_argument("--patience", type=int, default=8)
    train.add_argument("--seed", type=int, default=2026)
    train.add_argument("--checkpoint", required=True)
    train.add_argument("--output-report")

    solve = subparsers.add_parser("solve", help="run one test-time method on one instance")
    solve.add_argument("input")
    solve.add_argument("--checkpoint", required=True)
    solve.add_argument("--method", choices=AVAILABLE_METHODS, default="adapter_tta")
    _add_search_arguments(solve)
    solve.add_argument("--output")

    benchmark = subparsers.add_parser("benchmark", help="compare methods at matched budgets")
    benchmark.add_argument("dataset")
    benchmark.add_argument("--checkpoint", required=True)
    benchmark.add_argument("--scenario", default="benchmark")
    benchmark.add_argument("--budgets", type=int, nargs="+", default=[8, 16, 32])
    benchmark.add_argument(
        "--methods",
        nargs="+",
        choices=AVAILABLE_METHODS,
        default=[
            "frozen_sampling",
            "augmentation_sampling",
            "adapter_tta",
            "full_tta",
            "scratch_active_search",
            "frozen_sampling_2opt",
            "nearest_neighbor_2opt",
        ],
    )
    _add_search_arguments(benchmark, include_budget=False)
    benchmark.add_argument("--bootstrap-draws", type=int, default=500)
    benchmark.add_argument("--bootstrap-seed", type=int, default=7000)
    benchmark.add_argument("--output-json", required=True)
    benchmark.add_argument("--output-csv")

    research = subparsers.add_parser("research", help="run the frozen research protocol")
    research.add_argument("--config", default="configs/research_v1.json")
    research.add_argument("--checkpoint-directory", default="artifacts/checkpoints")
    research.add_argument("--output-report", default="artifacts/research-report.json")

    return parser


def _run_command(args: argparse.Namespace) -> object:
    if args.command == "generate":
        instance = generate_instance(args.nodes, regime=args.regime, seed=args.seed)
        optimum = solve_held_karp(instance)
        payload = _single_payload(instance, optimum)
        write_json(payload, args.output)
        return payload

    if args.command == "collect":
        dataset = generate_dataset(
            count=args.count,
            node_counts=tuple(args.node_counts),
            regimes=tuple(args.regimes),
            seed=args.seed,
            exact_limit=args.exact_limit,
        )
        save_dataset_jsonl(dataset, args.output)
        return dataset.to_metadata()

    if args.command == "oracle":
        dataset = load_dataset_jsonl(args.input)
        if not 0 <= args.sample_index < len(dataset.records):
            raise ValueError("sample index is out of range")
        record = dataset.records[args.sample_index]
        held_karp = solve_held_karp(record.instance)
        brute_force = (
            solve_brute_force(record.instance)
            if record.instance.node_count <= 10
            else None
        )
        if brute_force is not None and abs(brute_force.length - held_karp.length) > 1e-8:
            raise RuntimeError("Held-Karp and brute-force oracles disagree")
        payload = {
            "instance_id": record.instance.instance_id,
            "held_karp": held_karp.to_dict(),
            "brute_force": brute_force.to_dict() if brute_force else None,
            "stored_optimum_agrees": abs(record.optimum.length - held_karp.length) <= 1e-8,
        }
        if args.output:
            write_json(payload, args.output)
        return payload

    if args.command == "train":
        training = load_dataset_jsonl(args.training)
        validation = load_dataset_jsonl(args.validation)
        model, report = train_policy(
            training,
            validation,
            model_config=PolicyConfig(
                hidden_dim=args.hidden_dim,
                message_layers=args.message_layers,
                mlp_layers=args.mlp_layers,
            ),
            training_config=TrainingConfig(
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                gradient_clip=args.gradient_clip,
                patience=args.patience,
                seed=args.seed,
                augment=True,
            ),
        )
        save_checkpoint(
            model,
            args.checkpoint,
            metadata={"training_report": report.to_dict()},
        )
        payload = report.to_dict()
        if args.output_report:
            write_json(payload, args.output_report)
        return payload

    if args.command == "solve":
        record = _load_single(args.input)
        model, metadata = load_checkpoint(args.checkpoint)
        result = run_method(args.method, model, record.instance, _search_config(args))
        audit = audit_tour(
            record.instance,
            result.solution.tour,
            reported_length=result.solution.length,
            optimum=record.optimum,
        )
        payload = {
            "search": result.to_dict(),
            "audit": audit.to_dict(),
            "checkpoint_metadata": metadata,
        }
        if args.output:
            write_json(payload, args.output)
        return payload

    if args.command == "benchmark":
        dataset = load_dataset_jsonl(args.dataset)
        model, metadata = load_checkpoint(args.checkpoint)
        template = _search_config(args, budget=max(args.budgets))
        report = evaluate_methods(
            model,
            dataset,
            scenario=args.scenario,
            budgets=tuple(args.budgets),
            methods=tuple(args.methods),
            search_template=template,
            bootstrap_seed=args.bootstrap_seed,
            bootstrap_draws=args.bootstrap_draws,
        )
        save_report_json(report, args.output_json)
        if args.output_csv:
            save_report_csv(report, args.output_csv)
        return {**report.to_dict(), "checkpoint_metadata": metadata}

    if args.command == "research":
        config = load_research_config(args.config)
        report = run_research(config, checkpoint_directory=args.checkpoint_directory)
        save_research_report(report, args.output_report)
        return report.to_dict()

    raise AssertionError("unreachable command")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        payload = _run_command(args)
        _write_stdout(payload)
        return 0
    except (ValueError, RuntimeError, OSError) as error:
        print(
            json.dumps(
                {"error": type(error).__name__, "message": str(error)},
                ensure_ascii=False,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
