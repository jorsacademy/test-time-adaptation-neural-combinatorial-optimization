"""Frozen train-once, adapt-per-instance, evaluate-many research protocol."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

from ttanco.adaptation import AVAILABLE_METHODS, SearchConfig
from ttanco.dataset import SUPPORTED_REGIMES, TSPDataset, generate_dataset
from ttanco.evaluation import EvaluationReport, evaluate_methods
from ttanco.model import PolicyConfig, save_checkpoint
from ttanco.training import TrainingConfig, TrainingReport, train_policy
from ttanco.utils import read_json, sha256_json, write_json

RESEARCH_CONFIG_SCHEMA_VERSION = "ttanco-research-v1"


@dataclass(frozen=True, slots=True)
class ScenarioConfig:
    name: str
    count: int
    node_count: int
    regime: str
    seed: int

    def __post_init__(self) -> None:
        if not self.name or self.count <= 0 or self.node_count < 4:
            raise ValueError("scenario name, count, or node count is invalid")
        if self.regime not in SUPPORTED_REGIMES:
            raise ValueError(f"unsupported scenario regime: {self.regime}")
        if self.seed < 0:
            raise ValueError("scenario seed must be nonnegative")


@dataclass(frozen=True, slots=True)
class ResearchConfig:
    training_count: int
    training_node_counts: tuple[int, ...]
    training_regimes: tuple[str, ...]
    training_seed: int
    validation_count: int
    validation_node_counts: tuple[int, ...]
    validation_regimes: tuple[str, ...]
    validation_seed: int
    hidden_dim: int
    message_layers: int
    mlp_layers: int
    training_epochs: int
    training_learning_rate: float
    training_weight_decay: float
    training_patience: int
    training_seed_offset: int
    budgets: tuple[int, ...]
    methods: tuple[str, ...]
    adaptation_steps: int
    adaptation_batch_size: int
    adaptation_learning_rate: float
    entropy_weight: float
    trust_region_weight: float
    adapter_l2_weight: float
    full_anchor_weight: float
    gradient_clip: float
    two_opt_passes: int
    bootstrap_draws: int
    bootstrap_seed: int
    exact_limit: int
    scenarios: tuple[ScenarioConfig, ...]

    def __post_init__(self) -> None:
        if self.training_count <= 0 or self.validation_count <= 0:
            raise ValueError("training and validation counts must be positive")
        if not self.training_node_counts or not self.validation_node_counts:
            raise ValueError("training and validation node-count sets must be nonempty")
        if any(
            node_count < 4 for node_count in self.training_node_counts + self.validation_node_counts
        ):
            raise ValueError("training and validation node counts must be at least four")
        if not self.training_regimes or not self.validation_regimes:
            raise ValueError("training and validation regimes must be nonempty")
        if any(
            regime not in SUPPORTED_REGIMES
            for regime in self.training_regimes + self.validation_regimes
        ):
            raise ValueError("training or validation regime is unsupported")
        if self.hidden_dim <= 0 or self.message_layers < 0 or self.mlp_layers <= 0:
            raise ValueError("model configuration is invalid")
        if self.training_epochs <= 0 or self.training_learning_rate <= 0.0:
            raise ValueError("training configuration is invalid")
        if self.training_weight_decay < 0.0 or self.training_patience <= 0:
            raise ValueError("training regularization or patience is invalid")
        if not self.budgets or tuple(sorted(set(self.budgets))) != self.budgets:
            raise ValueError("budgets must be strictly increasing and unique")
        if any(budget < 2 for budget in self.budgets):
            raise ValueError("budgets must be at least two")
        if not self.methods or len(set(self.methods)) != len(self.methods):
            raise ValueError("methods must be nonempty and unique")
        if any(method not in AVAILABLE_METHODS for method in self.methods):
            raise ValueError("research methods contains an unsupported value")
        if self.bootstrap_draws <= 0 or self.exact_limit < max(
            scenario.node_count for scenario in self.scenarios
        ):
            raise ValueError("bootstrap draws or exact limit is invalid")
        if not self.scenarios:
            raise ValueError("research scenarios must be nonempty")
        # Reuse the deployment configuration validator for adaptation settings.
        SearchConfig(
            budget=self.budgets[-1],
            adaptation_steps=self.adaptation_steps,
            batch_size=self.adaptation_batch_size,
            learning_rate=self.adaptation_learning_rate,
            entropy_weight=self.entropy_weight,
            trust_region_weight=self.trust_region_weight,
            adapter_l2_weight=self.adapter_l2_weight,
            full_anchor_weight=self.full_anchor_weight,
            gradient_clip=self.gradient_clip,
            two_opt_passes=self.two_opt_passes,
            seed=0,
        )

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["schema_version"] = RESEARCH_CONFIG_SCHEMA_VERSION
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> ResearchConfig:
        if payload.get("schema_version") != RESEARCH_CONFIG_SCHEMA_VERSION:
            raise ValueError("unsupported research configuration schema")
        raw_scenarios = payload.get("scenarios")
        if not isinstance(raw_scenarios, (list, tuple)):
            raise ValueError("research scenarios must be a JSON array")
        scenarios: list[ScenarioConfig] = []
        for raw in raw_scenarios:
            if not isinstance(raw, dict):
                raise ValueError("research scenario entries must be JSON objects")
            scenarios.append(
                ScenarioConfig(
                    name=_string(raw, "name"),
                    count=_integer(raw, "count"),
                    node_count=_integer(raw, "node_count"),
                    regime=_string(raw, "regime"),
                    seed=_integer(raw, "seed"),
                )
            )
        return cls(
            training_count=_integer(payload, "training_count"),
            training_node_counts=tuple(_integer_list(payload, "training_node_counts")),
            training_regimes=tuple(_string_list(payload, "training_regimes")),
            training_seed=_integer(payload, "training_seed"),
            validation_count=_integer(payload, "validation_count"),
            validation_node_counts=tuple(_integer_list(payload, "validation_node_counts")),
            validation_regimes=tuple(_string_list(payload, "validation_regimes")),
            validation_seed=_integer(payload, "validation_seed"),
            hidden_dim=_integer(payload, "hidden_dim"),
            message_layers=_integer(payload, "message_layers"),
            mlp_layers=_integer(payload, "mlp_layers"),
            training_epochs=_integer(payload, "training_epochs"),
            training_learning_rate=_number(payload, "training_learning_rate"),
            training_weight_decay=_number(payload, "training_weight_decay"),
            training_patience=_integer(payload, "training_patience"),
            training_seed_offset=_integer(payload, "training_seed_offset"),
            budgets=tuple(_integer_list(payload, "budgets")),
            methods=tuple(_string_list(payload, "methods")),
            adaptation_steps=_integer(payload, "adaptation_steps"),
            adaptation_batch_size=_integer(payload, "adaptation_batch_size"),
            adaptation_learning_rate=_number(payload, "adaptation_learning_rate"),
            entropy_weight=_number(payload, "entropy_weight"),
            trust_region_weight=_number(payload, "trust_region_weight"),
            adapter_l2_weight=_number(payload, "adapter_l2_weight"),
            full_anchor_weight=_number(payload, "full_anchor_weight"),
            gradient_clip=_number(payload, "gradient_clip"),
            two_opt_passes=_integer(payload, "two_opt_passes"),
            bootstrap_draws=_integer(payload, "bootstrap_draws"),
            bootstrap_seed=_integer(payload, "bootstrap_seed"),
            exact_limit=_integer(payload, "exact_limit"),
            scenarios=tuple(scenarios),
        )


@dataclass(frozen=True, slots=True)
class ResearchReport:
    config: dict[str, object]
    config_fingerprint: str
    training_dataset_metadata: dict[str, object]
    validation_dataset_metadata: dict[str, object]
    training_report: TrainingReport
    checkpoint: str
    evaluations: tuple[EvaluationReport, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "config": self.config,
            "config_fingerprint": self.config_fingerprint,
            "training_dataset_metadata": self.training_dataset_metadata,
            "validation_dataset_metadata": self.validation_dataset_metadata,
            "training_report": self.training_report.to_dict(),
            "checkpoint": self.checkpoint,
            "evaluations": [evaluation.to_dict() for evaluation in self.evaluations],
        }


def _integer(payload: dict[str, object], name: str) -> int:
    value = payload.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"research field {name!r} must be an integer")
    return value


def _number(payload: dict[str, object], name: str) -> float:
    value = payload.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"research field {name!r} must be numeric")
    return float(value)


def _string(payload: dict[str, object], name: str) -> str:
    value = payload.get(name)
    if not isinstance(value, str):
        raise ValueError(f"research field {name!r} must be a string")
    return value


def _integer_list(payload: dict[str, object], name: str) -> list[int]:
    value = payload.get(name)
    if not isinstance(value, (list, tuple)) or not all(
        isinstance(entry, int) and not isinstance(entry, bool) for entry in value
    ):
        raise ValueError(f"research field {name!r} must be an integer array")
    return [int(entry) for entry in value]


def _string_list(payload: dict[str, object], name: str) -> list[str]:
    value = payload.get(name)
    if not isinstance(value, (list, tuple)) or not all(isinstance(entry, str) for entry in value):
        raise ValueError(f"research field {name!r} must be a string array")
    return [str(entry) for entry in value]


def load_research_config(path: str | Path) -> ResearchConfig:
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ValueError("research configuration must be a JSON object")
    return ResearchConfig.from_dict(cast(dict[str, object], payload))


def _training_datasets(config: ResearchConfig) -> tuple[TSPDataset, TSPDataset]:
    training = generate_dataset(
        count=config.training_count,
        node_counts=config.training_node_counts,
        regimes=config.training_regimes,
        seed=config.training_seed,
        exact_limit=config.exact_limit,
    )
    validation = generate_dataset(
        count=config.validation_count,
        node_counts=config.validation_node_counts,
        regimes=config.validation_regimes,
        seed=config.validation_seed,
        exact_limit=config.exact_limit,
    )
    return training, validation


def run_research(
    config: ResearchConfig,
    *,
    checkpoint_directory: str | Path,
) -> ResearchReport:
    """Train one source policy and reset adaptation independently for every test instance."""

    training, validation = _training_datasets(config)
    model, training_report = train_policy(
        training,
        validation,
        model_config=PolicyConfig(
            hidden_dim=config.hidden_dim,
            message_layers=config.message_layers,
            mlp_layers=config.mlp_layers,
        ),
        training_config=TrainingConfig(
            epochs=config.training_epochs,
            learning_rate=config.training_learning_rate,
            weight_decay=config.training_weight_decay,
            patience=config.training_patience,
            seed=config.training_seed_offset,
            augment=True,
        ),
    )
    checkpoint_path = Path(checkpoint_directory) / "source-edge-policy.safetensors"
    save_checkpoint(
        model,
        checkpoint_path,
        metadata={
            "training_report": training_report.to_dict(),
            "training_dataset_fingerprint": training.fingerprint,
            "validation_dataset_fingerprint": validation.fingerprint,
        },
    )
    search_template = SearchConfig(
        budget=config.budgets[-1],
        adaptation_steps=config.adaptation_steps,
        batch_size=config.adaptation_batch_size,
        learning_rate=config.adaptation_learning_rate,
        entropy_weight=config.entropy_weight,
        trust_region_weight=config.trust_region_weight,
        adapter_l2_weight=config.adapter_l2_weight,
        full_anchor_weight=config.full_anchor_weight,
        gradient_clip=config.gradient_clip,
        two_opt_passes=config.two_opt_passes,
        seed=config.training_seed_offset + 10_000,
    )
    evaluations: list[EvaluationReport] = []
    for offset, scenario in enumerate(config.scenarios):
        dataset = generate_dataset(
            count=scenario.count,
            node_counts=(scenario.node_count,),
            regimes=(scenario.regime,),
            seed=scenario.seed,
            exact_limit=config.exact_limit,
        )
        evaluations.append(
            evaluate_methods(
                model,
                dataset,
                scenario=scenario.name,
                budgets=config.budgets,
                methods=config.methods,
                search_template=search_template,
                bootstrap_seed=config.bootstrap_seed + 10_000 * offset,
                bootstrap_draws=config.bootstrap_draws,
            )
        )
    config_payload = config.to_dict()
    return ResearchReport(
        config=config_payload,
        config_fingerprint=sha256_json(config_payload),
        training_dataset_metadata=training.to_metadata(),
        validation_dataset_metadata=validation.to_metadata(),
        training_report=training_report,
        checkpoint=str(checkpoint_path),
        evaluations=tuple(evaluations),
    )


def save_research_report(report: ResearchReport, path: str | Path) -> None:
    write_json(report.to_dict(), path)
