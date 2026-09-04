# Architecture

## System boundary

```text
exact-labeled source corpus
          │
          ▼
source edge-policy training
          │
          ▼
immutable Safetensors checkpoint
          │
          ├───────────────────────────────┐
          ▼                               ▼
unseen TSP instance                 exact Held–Karp oracle
          │                               │
          ▼                               │ withheld during adaptation
fresh per-instance method state           │
          │                               │
          ▼                               ▼
budgeted tour evaluations ─────────► offline exact-gap audit
```

The exact optimum is not passed to `run_method`. Adaptation can observe only coordinates, policy probabilities, and the length of tours it has actually constructed.

## Source policy

Coordinates are centered and divided by their root-mean-square radius. Shared node features include normalized coordinates, radial terms, quadratic orientation terms, mean pairwise distance, nearest-neighbor distance, and inverse node count.

A shared MLP produces node embeddings. Each message layer concatenates the local embedding with global mean and maximum pools, then applies a residual MLP and layer normalization. The edge scorer receives symmetric features `h_i + h_j`, `abs(h_i - h_j)`, and normalized distance, so the resulting edge-logit matrix is symmetric and permutation equivariant.

Tour construction is autoregressive. At each step, the current row of the static edge matrix is masked over already visited nodes. The policy samples or greedily selects the next node. The hard mask enforces permutation feasibility by construction.

## Adapter path

`EmbeddingAdapter` contains:

- one residual vector per test node;
- one scalar bias per node;
- one scalar log-temperature.

The source model is frozen. Adapted logits are generated from `base_embedding + delta`, then adjusted by symmetric node biases and temperature. Every instance starts from all-zero adapter parameters, exactly reproducing the source logits before the first update.

## Full and scratch paths

Full TTA clones the source model, updates the clone, and keeps a second frozen source copy for action-distribution KL and parameter anchoring. Scratch active search constructs a new random model with the same architecture and adapts it under the same complete-tour budget.

## Best-so-far archive

The archive is outside the neural policy. Each tour is independently audited and compared by objective. Parameter updates cannot delete a previously observed incumbent. The final returned tour is the archive minimum.
