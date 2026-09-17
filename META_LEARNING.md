# Meta-Learning Extension for Fast NCO Adaptation

This repository now contains two deliberately distinct adaptation paradigms:

1. **test-time adaptation (TTA)** — one unlabeled test instance, objective-driven parameter updates, no target optimum used during adaptation;
2. **supervised few-shot meta-learning** — a distribution-level task provides a small labeled support set, and the goal is to adapt quickly to held-out query instances from that task.

They should not be reported as the same method.

## Research context

The extension is informed by two established NCO directions:

- **Towards Omni-generalizable Neural Methods for Vehicle Routing Problems** (ICML 2023) treats size/distribution combinations as tasks and meta-learns an initialization that can adapt quickly to new routing tasks.
- **Meta-SAGE** (ICML 2023) combines a scale meta-learner with scheduled test-time adaptation to improve scale-shift generalization.

The implementation here is independent and intentionally smaller. It does not reproduce either architecture or claim their benchmark performance.

References:

- Zhou et al., *Towards Omni-generalizable Neural Methods for Vehicle Routing Problems*, ICML 2023: https://proceedings.mlr.press/v202/zhou23o.html
- Son et al., *Meta-SAGE: Scale Meta-Learning Scheduled Adaptation with Guided Exploration for Mitigating Scale Shift on Combinatorial Optimization*, ICML 2023: https://proceedings.mlr.press/v202/son23a.html

## Task definition

A meta-task is identified by:

```text
(regime, node_count)
```

Examples:

```text
uniform-n8
uniform-n10
clustered-n8
clustered-n10
```

Each task is partitioned into:

```text
support set -> inner-loop adaptation
query set   -> meta-objective / evaluation
```

The current source corpora already contain exact Held-Karp solutions, so the meta-learning extension uses exact optimal-tour edge labels for the support/query protocol. This is therefore **supervised few-shot adaptation**, not label-free TTA.

## Size-agnostic meta-adapter

The existing `EmbeddingAdapter` is instance-specific and contains one residual vector per node, so its parameter count grows with graph size. That is useful for instance TTA but unsuitable as a shared meta-initialization across graph sizes.

The new `MetaEmbeddingAdapter` instead uses:

```text
base node embedding
      |
small shared bottleneck MLP
      |
residual embedding correction
      |
shared embedding shift
      |
temperature calibration
```

Its parameter count is independent of the number of TSP nodes. The same initialization can therefore be meta-trained across several sizes and regimes.

## Meta-training algorithm

`meta_train_adapter` implements a **first-order MAML-style** procedure:

1. copy the current meta-adapter initialization for one task;
2. take a small number of SGD steps on the task support set;
3. evaluate the adapted copy on the task query set;
4. copy the query gradients back to the shared initialization while ignoring second-order derivatives through the inner loop;
5. aggregate gradients across tasks and update the shared initialization.

This is explicitly a first-order approximation. It is not full second-order MAML.

## Decision-level evaluation

Few-shot evaluation reports both:

- query edge-supervision loss before and after adaptation;
- mean exact optimality gap of a deterministic greedy tour before and after adaptation.

This distinction matters: a lower surrogate/edge loss does not automatically imply a better combinatorial decision.

## Running the benchmark

First create a source model and exact-labeled corpora using the existing CLI. For example:

```bash
ttanco collect \
  --count 24 \
  --node-counts 8 10 \
  --regimes uniform clustered \
  --seed 1000 \
  --output artifacts/meta_source.jsonl

ttanco collect \
  --count 16 \
  --node-counts 12 14 \
  --regimes ring anisotropic \
  --seed 2000 \
  --output artifacts/meta_target.jsonl

ttanco train artifacts/meta_source.jsonl \
  --validation artifacts/meta_source.jsonl \
  --epochs 10 \
  --checkpoint artifacts/source-policy.safetensors
```

Then run:

```bash
python scripts/meta_learning_benchmark.py \
  artifacts/meta_source.jsonl \
  artifacts/meta_target.jsonl \
  artifacts/source-policy.safetensors \
  --outer-epochs 4 \
  --inner-steps 2 \
  --support-size 2 \
  --query-size 2 \
  --output artifacts/meta-report.json
```

## Recommended experimental comparisons

A research campaign should compare at least:

- source model with no adaptation;
- randomly initialized adapter + few-shot adaptation;
- ordinary jointly trained adapter initialization;
- meta-trained adapter initialization;
- existing objective-driven `adapter_tta` under an explicitly separate test-time-compute budget.

Do not compare supervised few-shot adaptation and label-free TTA as if they used the same information. The support labels are additional information and must be declared.

## Claims boundary

This extension does **not** claim:

- that meta-learning always improves query tour quality;
- that first-order MAML is equivalent to Omni-VRP or Meta-SAGE;
- zero-shot transfer to unseen combinatorial problem classes;
- state-of-the-art routing results;
- that exact small-instance supervision scales to industrial routing sizes.

A valid negative result is that the meta-initialization fails to improve exact query gaps after few-shot adaptation.
