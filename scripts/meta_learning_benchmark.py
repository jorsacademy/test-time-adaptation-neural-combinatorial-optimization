from __future__ import annotations

import argparse
import json
from pathlib import Path

from ttanco.dataset import load_dataset_jsonl
from ttanco.meta_learning import (
    MetaLearningConfig,
    build_meta_tasks,
    evaluate_few_shot,
    meta_train_adapter,
)
from ttanco.model import load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Meta-train a size-agnostic adapter and evaluate few-shot transfer."
    )
    parser.add_argument("source_dataset", type=Path)
    parser.add_argument("target_dataset", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--outer-epochs", type=int, default=4)
    parser.add_argument("--inner-steps", type=int, default=2)
    parser.add_argument("--support-size", type=int, default=2)
    parser.add_argument("--query-size", type=int, default=2)
    parser.add_argument("--inner-learning-rate", type=float, default=5e-3)
    parser.add_argument("--outer-learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = load_dataset_jsonl(args.source_dataset)
    target = load_dataset_jsonl(args.target_dataset)
    model, metadata = load_checkpoint(args.checkpoint)
    config = MetaLearningConfig(
        outer_epochs=args.outer_epochs,
        inner_steps=args.inner_steps,
        support_size=args.support_size,
        query_size=args.query_size,
        inner_learning_rate=args.inner_learning_rate,
        outer_learning_rate=args.outer_learning_rate,
        seed=args.seed,
    )
    adapter, history = meta_train_adapter(model, source, config=config)
    target_tasks = build_meta_tasks(
        target,
        support_size=config.support_size,
        query_size=config.query_size,
    )
    reports = [
        evaluate_few_shot(
            model,
            adapter,
            target,
            task,
            inner_steps=config.inner_steps,
            inner_learning_rate=config.inner_learning_rate,
        ).to_dict()
        for task in target_tasks
    ]
    payload = {
        "protocol": "supervised-few-shot-meta-adaptation-v1",
        "source_fingerprint": source.fingerprint,
        "target_fingerprint": target.fingerprint,
        "checkpoint_metadata": metadata,
        "config": {
            "outer_epochs": config.outer_epochs,
            "inner_steps": config.inner_steps,
            "support_size": config.support_size,
            "query_size": config.query_size,
            "inner_learning_rate": config.inner_learning_rate,
            "outer_learning_rate": config.outer_learning_rate,
            "seed": config.seed,
        },
        "meta_query_loss_history": list(history),
        "target_reports": reports,
    }
    serialized = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
    if args.output is None:
        print(serialized)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
