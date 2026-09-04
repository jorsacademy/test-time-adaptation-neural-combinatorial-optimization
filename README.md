# Test-Time Adaptation for Neural Combinatorial Optimization

[![CI](https://github.com/jorsacademy/test-time-adaptation-neural-combinatorial-optimization/actions/workflows/ci.yml/badge.svg)](https://github.com/jorsacademy/test-time-adaptation-neural-combinatorial-optimization/actions/workflows/ci.yml)
[![Python 3.11–3.12](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)](https://www.python.org/)
[![License: PolyForm Noncommercial 1.0.0](https://img.shields.io/badge/license-PolyForm%20Noncommercial%201.0.0-orange)](LICENSE)

A verification-first research implementation of **budgeted test-time adaptation (TTA)** for neural combinatorial optimization on Euclidean Traveling Salesperson Problem instances.

The repository separates three mechanisms that are often reported under one “inference-time improvement” label:

1. **more samples from a frozen policy**;
2. **test-time augmentation without parameter updates**;
3. **objective-driven parameter adaptation on one unlabeled test instance**.

All learned methods construct valid Hamiltonian cycles with hard visit masks. Test-time adaptation uses only the test instance and its computable tour length; the exact optimum is withheld until evaluation. Held–Karp dynamic programming is the small-instance certification oracle, not part of adaptation.

## Research question

> Under the same number of complete tour-objective evaluations, does parameter-efficient instance adaptation recover more solution quality under size and geometric distribution shift than frozen sampling, metamorphic augmentation, naive full-model fine-tuning, or active search from random initialization?

The benchmark deliberately reports solution quality, anytime behavior, parameter-update scope, objective-evaluation count, local-search work, and wall-clock time separately. It does not hide these trade-offs in one weighted score.

## Claims boundary

This is a compact methodology benchmark. It does **not** claim:

- a state-of-the-art TSP solver;
- universal improvement from test-time adaptation;
- that an adapted neural policy certifies optimality;
- equivalence to EAS, Meta-SAGE, TTPL, TACO, or any other complete published system;
- industrial-scale exact evaluation;
- hardware-independent speedup;
- that a tour-evaluation budget equals a 2-opt move-evaluation budget;
- source-free pretraining—the source model is trained with exact small-instance tour labels;
- an OSI-approved open-source license.

Negative results are valid outcomes: adaptation may consume the budget, overfit sampled trajectories, or lose to frozen sampling and 2-opt on some regimes.

## Optimization problem

For coordinates \(u_i\in\mathbb R^2\), Euclidean edge costs are

\[
c_{ij}=\lVert u_i-u_j\rVert_2.
\]

A Hamiltonian cycle \(\pi\) has length

\[
L(\pi)=\sum_{t=1}^{n} c_{\pi_t,\pi_{t+1}},
\qquad \pi_{n+1}=\pi_1.
\]

The goal is

\[
\min_{\pi\in\mathcal T_n}L(\pi),
\]

where \(\mathcal T_n\) is the set of tours visiting every node exactly once.

Every reported tour is canonicalized by rotation and reversal, then audited by recomputing its permutation validity and objective from the original coordinates.

## Source-generalization policy

The source model is a permutation-equivariant static edge policy.

```text
coordinates
    │
    ▼
translation/RMS-scale normalization
    │
    ▼
shared node encoder
    │
    ▼
global mean/max message updates
    │
    ▼
symmetric pair scorer
    │
    ▼
edge-logit matrix
    │
    ▼
autoregressive masked tour construction
```

The edge score for \((i,j)\) is built from symmetric pair features:

- \(h_i+h_j\);
- \(|h_i-h_j|\);
- normalized Euclidean distance.

The matrix is therefore symmetric, while the construction policy remains sequential because the current node changes after each action. Already visited nodes are assigned a hard mask.

### Source training

Small source instances are solved exactly with Held–Karp. The optimal cycle is converted to a symmetric edge-adjacency target. Training uses weighted binary cross-entropy on the upper-triangular edge set, with deterministic geometric isometries and node permutations as source-time augmentation.

The source policy is trained once. Every test instance receives a fresh adaptation state; no adapted parameters carry over between instances.

## Test-time methods

| Method | Test-time gradients | Updated parameters | Complete-tour budget | Extra local-search work |
|---|---:|---|---:|---:|
| `frozen_sampling` | No | None | Matched | None |
| `augmentation_sampling` | No | None | Matched | None |
| `adapter_tta` | Yes | Per-instance node residuals, node biases, temperature | Matched | None |
| `full_tta` | Yes | All copied source-policy parameters | Matched | None |
| `scratch_active_search` | Yes | All parameters of a random policy | Matched | None |
| `frozen_sampling_2opt` | No | None | Matched before 2-opt | Separately counted |
| `nearest_neighbor_2opt` | No | None | Not matched | Separately counted |

### Frozen sampling

The source policy is never modified. One deterministic tour and the remaining stochastic tours compete under the same complete-tour evaluation budget.

### Metamorphic augmentation

The policy is evaluated on distance-preserving dihedral transformations and node permutations. Tours are mapped back to original node IDs before the objective is recomputed. This baseline allocates more inference diversity without gradients.

### Parameter-efficient adapter TTA

A fresh `EmbeddingAdapter` is created for each test instance:

\[
\widetilde h_i=h_i+\Delta_i,
\]

with an additional node bias and scalar temperature. The source network is frozen. The adapter has

\[
n d+n+1
\]

parameters for \(n\) nodes and hidden dimension \(d\), versus updating the entire source model.

The objective-driven update uses a batch REINFORCE estimator:

\[
\mathcal L_{\mathrm{AS}}
=
\frac{1}{B}
\sum_{b=1}^{B}
\bigl(L(\pi_b)-\bar L\bigr)
\log p(\pi_b)
-\eta\,\mathcal H
+\lambda_{\mathrm{KL}}D_{\mathrm{KL}}(p_{\mathrm{adapt}}\|p_{\mathrm{source}})
+\lambda_r\lVert\Delta\rVert_2^2.
\]

The implementation computes masked action-distribution KL values along sampled trajectories. Gradient clipping and finite-value checks fail closed.

### Full fine-tuning and scratch active search

`full_tta` copies the source model and updates all copied parameters, with source-policy KL and parameter anchoring. The shared source checkpoint remains immutable.

`scratch_active_search` initializes the same architecture randomly for each test instance and uses the same objective budget. It tests whether source inductive bias is actually useful rather than assuming that any instance-wise optimization should begin from a pretrained model.

### Best-so-far return rule

Every complete tour evaluated during adaptation is audited and placed in a best-so-far archive. The final output is the best observed tour, not merely the final policy’s greedy decode. Therefore parameter updates may be unstable, but the returned solution cannot be worse than the first audited tour observed by that method.

This is a monotone **observed-solution** guarantee, not an optimality guarantee and not a guarantee that the policy itself improves.

## Fair compute accounting

The matched budget is defined as:

> one evaluation = one complete Hamiltonian tour whose length is recomputed on the original instance.

For adaptation methods, the same budget covers:

- the initial greedy tour;
- all trajectories used for gradient updates;
- all post-update samples.

Local-search move evaluations are qualitatively different. They are therefore reported separately and are not silently treated as equal to one complete neural rollout. `frozen_sampling_2opt` and `nearest_neighbor_2opt` are useful controls, but their runtime and move counts must be read alongside the matched neural-budget results.

## Exactness and reliability contract

Exact components:

- Euclidean distance recomputation;
- tour permutation and length audit;
- Held–Karp exact dynamic programming up to a configured node limit;
- brute-force tour enumeration on tiny instances;
- exact optimality-gap calculation against Held–Karp;
- deterministic best-improvement 2-opt move evaluation.

Approximate components:

- source edge logits;
- sampled tours;
- test-time policy-gradient updates;
- adapter/full/scratch parameter states;
- finite-budget best-so-far results;
- bootstrap confidence intervals;
- wall-clock measurements.

A neural result is never labeled optimal without the independent exact oracle. See [`docs/exactness.md`](docs/exactness.md).

## Controlled distribution-shift protocol

The frozen protocol trains on node counts \(n\in\{8,10,12\}\) using uniform and mildly clustered source distributions. It evaluates disjoint seeds under:

1. `interpolation` — source-like uniform instances;
2. `size_14` — moderate size extrapolation;
3. `size_16` — harder size extrapolation;
4. `cluster_shift`;
5. `ring_shift`;
6. `grid_shift`;
7. `anisotropic_shift`;
8. `outlier_shift`;
9. `heavy_tail_shift`;
10. `spiral_shift`;
11. `coordinate_scale_shift`.

The model normalizes translation and positive coordinate scale. The coordinate-scale scenario is retained as a metamorphic regression check rather than being presented as a difficult semantic shift.

Budgets are evaluated separately—by default 8, 16, 32, and 64 complete tour evaluations. A method cannot hide poor low-budget behavior behind a large final budget.

## Metrics

### Final solution quality

- mean, median, and P90 optimality gap;
- exact-optimum hit rate;
- absolute gap;
- improvement from each method’s initial tour;
- recovered fraction of the initial exact gap.

### Anytime behavior

- gap at 25%, 50%, and 100% of the method’s own objective-evaluation budget;
- mean best-so-far gap across the trajectory;
- evaluation index and fraction at which the final incumbent was first found.

### Adaptation cost

- complete-tour objective evaluations;
- gradient-update count;
- adapted parameter count;
- local-search move evaluations;
- total runtime;
- source-model immutability rate.

### Paired comparisons

For each budget and scenario, the benchmark computes instance-paired gap differences against frozen sampling and deterministic bootstrap intervals. No unpaired aggregate is substituted for the matched-instance comparison.

## Installation

```bash
python -m pip install -e ".[dev]"
```

CPU-only PyTorch is sufficient for the supplied protocol.

## CLI

### Generate one exact-labeled instance

```bash
ttanco generate \
  --nodes 10 \
  --regime uniform \
  --seed 42 \
  --output artifacts/example.json
```

### Build source corpora

```bash
ttanco collect \
  --count 72 \
  --node-counts 8 10 12 \
  --regimes uniform uniform clustered \
  --seed 3200 \
  --output artifacts/train.jsonl

ttanco collect \
  --count 18 \
  --node-counts 8 10 12 \
  --regimes uniform clustered \
  --seed 4200 \
  --output artifacts/validation.jsonl
```

### Train the source policy

```bash
ttanco train artifacts/train.jsonl \
  --validation artifacts/validation.jsonl \
  --epochs 30 \
  --checkpoint artifacts/source-policy.safetensors \
  --output-report artifacts/training.json
```

### Adapt one test instance

```bash
ttanco solve artifacts/example.json \
  --checkpoint artifacts/source-policy.safetensors \
  --method adapter_tta \
  --budget 64 \
  --adaptation-steps 8 \
  --batch-size 4 \
  --output artifacts/adapted-solution.json
```

### Compare methods and budgets

```bash
ttanco benchmark artifacts/test.jsonl \
  --checkpoint artifacts/source-policy.safetensors \
  --scenario size_16 \
  --budgets 8 16 32 64 \
  --output-json artifacts/benchmark.json \
  --output-csv artifacts/benchmark.csv
```

### Run the frozen protocol

```bash
ttanco research \
  --config configs/research_v1.json \
  --checkpoint-directory artifacts/checkpoints \
  --output-report artifacts/research-report.json
```

## Repository layout

```text
src/ttanco/
├── domain.py       # TSP schema, audits, Held–Karp, brute force, nearest neighbor, 2-opt
├── dataset.py      # controlled shifts, exact-labeled JSONL, SHA-256 integrity
├── model.py        # equivariant edge policy, per-instance adapter, Safetensors
├── training.py     # exact-edge source supervision
├── decoding.py     # masked rollouts and metamorphic augmentation
├── adaptation.py   # frozen search, adapter/full/scratch TTA, budget ledger
├── evaluation.py   # exact gaps, anytime metrics, paired bootstrap reports
├── experiment.py   # frozen train-once/evaluate-many protocol
└── cli.py          # end-to-end commands
```

## Tests and CI

GitHub Actions runs on Python 3.11 and 3.12:

```text
package installation and dependency check
Ruff lint and formatting
strict mypy
branch-aware pytest coverage
collect → train → exact oracle → TTA benchmark smoke
```

The regression suite covers exact-oracle agreement, canonical tour audits, 2-opt monotonicity, every synthetic regime, corpus tamper detection, policy permutation equivariance, coordinate normalization, Safetensors round trips, differentiable masked rollouts, source-model immutability, objective-budget compliance, best-so-far monotonicity, adapter-vs-full parameter counts, report serialization, the frozen protocol, and CLI workflows.

## Methodological limitations

Held–Karp requires exponential time and memory, so exact gaps are limited to controlled small instances. Larger-instance studies would need certified lower bounds, trusted external solvers, or benchmark optima.

The source policy uses static edge scores. It is intentionally smaller than modern attention-based routing architectures. The adapter updates node embeddings and biases but does not implement every EAS variant, Meta-SAGE’s scale meta-learner, TTPL’s projection mechanism, or TACO’s strategic parameter relaxation.

REINFORCE estimates can have high variance. A finite budget can be too small for useful adaptation, while a large budget can make simple sampling or local search competitive. The repository reports these outcomes rather than asserting that gradient-based TTA must win.

The source model is supervised by exact optimal tours. The test-time stage is label-free in the standard optimization sense because tour cost is computable from the instance, but this is not an end-to-end unsupervised NCO training reproduction.

## Research context

The implementation is positioned relative to:

- Bello et al., [“Neural Combinatorial Optimization with Reinforcement Learning”](https://arxiv.org/abs/1611.09940), which introduced instance-level active search for learned constructive policies;
- Hottung, Kwon, and Tierney, [“Efficient Active Search for Combinatorial Optimization Problems”](https://openreview.net/forum?id=nO5caZwFwYu), ICLR 2022, which updates restricted parameter subsets instead of the full model;
- Choo et al., [“Simulation-guided Beam Search for Neural Combinatorial Optimization”](https://papers.neurips.cc/paper_files/paper/2022/hash/39b9b60f0d149eabd1fff2d7c7d5afc4-Abstract-Conference.html), NeurIPS 2022, which combines inference-time search with efficient active search;
- Son et al., [“Meta-SAGE”](https://proceedings.mlr.press/v202/son23a.html), ICML 2023, which adapts context parameters to scale-shifted routing instances;
- Wei et al., [“Extending Test-Time Augmentation with Metamorphic Relations for Combinatorial Problems”](https://proceedings.mlr.press/v235/wei24i.html), ICML 2024;
- Chen et al., [“Improving Generalization of Neural Combinatorial Optimization for Vehicle Routing Problems via Test-Time Projection Learning”](https://proceedings.neurips.cc/paper_files/paper/2025/hash/6edd46d69ef91f4555d67f7b321d6902-Abstract-Conference.html), NeurIPS 2025;
- Liao, Koushanfar, and Naghizadeh, [“Test-Time Adaptation for Unsupervised Combinatorial Optimization”](https://arxiv.org/abs/2601.21048), 2026 preprint, which proposes TACO as a bridge between generalization and instance-specific optimization.

This repository is not a reproduction of any single paper. Its narrower contribution is a transparent, exact-audited comparison of **where test-time compute is spent**: sampling, metamorphic transformations, small adapters, full fine-tuning, random-start active search, or local search.

## License

PolyForm Noncommercial 1.0.0. The repository is source-available for noncommercial use and is not offered under an OSI-approved open-source license.
