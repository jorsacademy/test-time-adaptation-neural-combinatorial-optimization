"""Exact-tour supervised pretraining for the source edge policy."""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass
from typing import cast

import numpy as np
import torch
from torch import Tensor, nn

from ttanco.dataset import TSPDataset
from ttanco.decoding import dihedral_transform
from ttanco.model import EdgePolicy, PolicyConfig, validate_policy
from ttanco.utils import seed_everything


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    epochs: int = 30
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    gradient_clip: float = 1.0
    patience: int = 8
    seed: int = 0
    augment: bool = True

    def __post_init__(self) -> None:
        if self.epochs <= 0 or self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("training epochs, learning rate, and weight decay are invalid")
        if self.gradient_clip <= 0.0 or self.patience <= 0 or self.seed < 0:
            raise ValueError("training gradient, patience, or seed configuration is invalid")


@dataclass(frozen=True, slots=True)
class EpochMetrics:
    epoch: int
    training_loss: float
    validation_loss: float
    validation_precision: float
    validation_recall: float
    validation_f1: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TrainingReport:
    config: dict[str, object]
    model_config: dict[str, object]
    training_fingerprint: str
    validation_fingerprint: str
    best_epoch: int
    best_validation_loss: float
    stopped_early: bool
    epochs: tuple[EpochMetrics, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "config": self.config,
            "model_config": self.model_config,
            "training_fingerprint": self.training_fingerprint,
            "validation_fingerprint": self.validation_fingerprint,
            "best_epoch": self.best_epoch,
            "best_validation_loss": self.best_validation_loss,
            "stopped_early": self.stopped_early,
            "epochs": [epoch.to_dict() for epoch in self.epochs],
        }


def tour_edge_targets(node_count: int, tour: tuple[int, ...]) -> np.ndarray:
    """Return a symmetric binary adjacency matrix for a Hamiltonian cycle."""

    if len(tour) != node_count or set(tour) != set(range(node_count)):
        raise ValueError("tour must be a node permutation")
    target = np.zeros((node_count, node_count), dtype=np.float32)
    for position, source in enumerate(tour):
        target_node = tour[(position + 1) % node_count]
        target[source, target_node] = 1.0
        target[target_node, source] = 1.0
    return target


def _instance_tensors(
    dataset: TSPDataset,
    index: int,
    *,
    rng: np.random.Generator | None,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    record = dataset.records[index]
    coordinates = record.instance.coordinate_array
    target = tour_edge_targets(record.instance.node_count, record.optimum.tour)
    if rng is not None:
        transform_index = int(rng.integers(0, 8))
        coordinates = dihedral_transform(coordinates, transform_index)
        permutation = rng.permutation(record.instance.node_count)
        coordinates = coordinates[permutation]
        target = target[np.ix_(permutation, permutation)]
    return (
        torch.tensor(coordinates, dtype=torch.float32, device=device),
        torch.tensor(target, dtype=torch.float32, device=device),
    )


def _edge_loss(logits: Tensor, targets: Tensor) -> Tensor:
    n = logits.shape[0]
    if logits.shape != (n, n) or targets.shape != (n, n):
        raise ValueError("edge logits and targets must be aligned square matrices")
    mask = torch.triu(torch.ones((n, n), dtype=torch.bool, device=logits.device), diagonal=1)
    selected_logits = logits[mask]
    selected_targets = targets[mask]
    positives = torch.sum(selected_targets)
    negatives = float(selected_targets.numel()) - positives
    if float(positives.detach().cpu()) <= 0.0:
        raise RuntimeError("tour target contains no positive edges")
    positive_weight = negatives / positives
    loss = nn.functional.binary_cross_entropy_with_logits(
        selected_logits,
        selected_targets,
        pos_weight=positive_weight,
    )
    if not torch.isfinite(loss):
        raise RuntimeError("edge-supervision loss is non-finite")
    return loss


def _validation_metrics(model: EdgePolicy, dataset: TSPDataset) -> tuple[float, float, float, float]:
    losses: list[float] = []
    true_positive = 0
    false_positive = 0
    false_negative = 0
    model.eval()
    with torch.no_grad():
        for index in range(len(dataset.records)):
            coordinates, targets = _instance_tensors(
                dataset,
                index,
                rng=None,
                device=model.device,
            )
            logits = cast(Tensor, model(coordinates))
            losses.append(float(_edge_loss(logits, targets).cpu()))
            n = logits.shape[0]
            mask = torch.triu(
                torch.ones((n, n), dtype=torch.bool, device=logits.device),
                diagonal=1,
            )
            predictions = logits[mask] >= 0.0
            truth = targets[mask] >= 0.5
            true_positive += int(torch.sum(predictions & truth).cpu())
            false_positive += int(torch.sum(predictions & ~truth).cpu())
            false_negative += int(torch.sum(~predictions & truth).cpu())
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
    return float(np.mean(losses)), precision, recall, f1


def train_policy(
    training: TSPDataset,
    validation: TSPDataset,
    *,
    model_config: PolicyConfig | None = None,
    training_config: TrainingConfig | None = None,
    device: torch.device | str = "cpu",
) -> tuple[EdgePolicy, TrainingReport]:
    """Train a source-generalization policy from exact optimal tour edges."""

    config = training_config or TrainingConfig()
    architecture = model_config or PolicyConfig()
    seed_everything(config.seed)
    model = EdgePolicy(architecture).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    rng = np.random.default_rng(config.seed)
    best_state = copy.deepcopy(model.state_dict())
    best_validation = math.inf
    best_epoch = 0
    remaining_patience = config.patience
    history: list[EpochMetrics] = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        order = rng.permutation(len(training.records))
        losses: list[float] = []
        for raw_index in order:
            optimizer.zero_grad(set_to_none=True)
            coordinates, targets = _instance_tensors(
                training,
                int(raw_index),
                rng=rng if config.augment else None,
                device=model.device,
            )
            logits = cast(Tensor, model(coordinates))
            loss = _edge_loss(logits, targets)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=config.gradient_clip,
            )
            if not torch.isfinite(gradient_norm):
                raise RuntimeError("source-policy training produced non-finite gradients")
            optimizer.step()
            validate_policy(model)
            losses.append(float(loss.detach().cpu()))

        validation_loss, precision, recall, f1 = _validation_metrics(model, validation)
        metrics = EpochMetrics(
            epoch=epoch,
            training_loss=float(np.mean(losses)),
            validation_loss=validation_loss,
            validation_precision=precision,
            validation_recall=recall,
            validation_f1=f1,
        )
        history.append(metrics)
        if validation_loss < best_validation - 1e-10:
            best_validation = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            remaining_patience = config.patience
        else:
            remaining_patience -= 1
            if remaining_patience == 0:
                break

    model.load_state_dict(best_state, strict=True)
    validate_policy(model)
    report = TrainingReport(
        config=asdict(config),
        model_config=asdict(architecture),
        training_fingerprint=training.fingerprint,
        validation_fingerprint=validation.fingerprint,
        best_epoch=best_epoch,
        best_validation_loss=best_validation,
        stopped_early=len(history) < config.epochs,
        epochs=tuple(history),
    )
    return model, report
