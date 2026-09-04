"""Small deterministic adapter-TTA demonstration."""

from ttanco.adaptation import SearchConfig, run_method
from ttanco.dataset import generate_dataset, generate_instance
from ttanco.model import PolicyConfig
from ttanco.training import TrainingConfig, train_policy

training = generate_dataset(count=8, node_counts=(6, 7), regimes=("uniform",), seed=100)
validation = generate_dataset(count=3, node_counts=(6,), regimes=("uniform",), seed=200)
model, _ = train_policy(
    training,
    validation,
    model_config=PolicyConfig(hidden_dim=16, message_layers=1, mlp_layers=1),
    training_config=TrainingConfig(epochs=2, patience=2, seed=300),
)
instance = generate_instance(8, regime="clustered", seed=400)
result = run_method(
    "adapter_tta",
    model,
    instance,
    SearchConfig(budget=16, adaptation_steps=3, batch_size=4, seed=500),
)
print(result.to_dict())
