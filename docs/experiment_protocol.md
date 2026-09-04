# Frozen Experiment Protocol

## Hypotheses

The protocol is designed to test, not assume, the following hypotheses:

1. a source model provides a better active-search warm start than random initialization;
2. updating a small instance adapter is more compute-efficient than full fine-tuning;
3. adaptation gains exceed gains available from an equal number of frozen samples;
4. geometric augmentation recovers part of the distribution-shift loss without gradients;
5. larger test-time budgets improve best-so-far quality but may have diminishing returns;
6. neural adaptation can still lose to deterministic 2-opt once local-search work is counted.

## Data separation

Training, validation, and every evaluation scenario use disjoint deterministic seed ranges. Evaluation scenarios do not influence source training, early stopping, adaptation hyperparameters, or checkpoint selection.

## Test-time information

Permitted during adaptation:

- coordinates of the current instance;
- source-policy logits;
- sampled tours;
- exact length of each sampled tour;
- policy entropy and source-policy KL along visited states.

Prohibited during adaptation:

- Held–Karp optimum;
- optimal tour edges;
- exact optimality gap;
- evaluation-scenario aggregate statistics;
- adapted state from any previous test instance.

## Matched budgets

Budgets count complete tour evaluations. Adapter, full, and scratch methods reserve part of the budget for gradient-producing trajectories and spend the remainder on post-update sampling. Frozen and augmentation methods spend the full budget on search samples.

`nearest_neighbor_2opt` and `frozen_sampling_2opt` are deliberately marked as unmatched local-search controls. Their edge-swap evaluation counts and runtimes are reported separately.

## Evaluation matrix

Each combination of:

```text
scenario × budget × instance × method
```

produces one audited `InstanceMethodResult`. Summary rows are constructed only after the complete matrix is present.

## Statistical reporting

The primary paired comparison is method gap minus frozen-sampling gap on the same instance and budget. A deterministic nonparametric bootstrap gives a 95% interval for the mean paired difference. The protocol does not treat overlap or non-overlap of confidence intervals as a formal hypothesis test.

## Repeated runs

For publication-quality evidence, run multiple independent source-training seeds and report hierarchical variation across both source checkpoints and test instances. The included v1 protocol fixes one source-training seed to keep the repository reproducible and computationally accessible; it is not sufficient by itself for a broad empirical superiority claim.
