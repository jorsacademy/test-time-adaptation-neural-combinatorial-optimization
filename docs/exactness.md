# Exactness and Reliability Contract

## Exact components

For every declared Euclidean TSP instance:

- coordinates are finite and pairwise distinct;
- pairwise distances are recomputed from the original coordinates;
- tours must be complete node permutations;
- tour length is recomputed after canonical rotation and reversal;
- Held–Karp dynamic programming certifies the exact optimum up to the configured node limit;
- brute-force permutation enumeration independently checks tiny instances;
- 2-opt evaluates exact Euclidean move deltas and never accepts a worsening move;
- optimality gaps are computed only after exact-oracle comparison.

## Approximate components

- source edge probabilities;
- stochastic tour samples;
- policy-gradient estimates;
- adapter, full-model, and scratch parameter updates;
- finite-budget incumbents;
- bootstrap intervals;
- wall-clock runtime.

## Non-binding adaptation

Neural parameters affect which feasible tour is sampled next. They do not change the TSP objective, node set, distance matrix, exact oracle, or audit rules. Hard visit masks prevent repeated nodes during construction. Every returned solution is audited again outside the model.

## Information boundary

`run_method` receives no exact tour and no exact optimum. It can evaluate a sampled tour because its cost follows directly from the test instance. Exact solutions are accessed only by the evaluation layer after adaptation has completed.

## Source immutability

Adapter TTA freezes the shared source model. Full TTA updates a clone. Scratch search uses a new random model. The shared source model is fingerprinted before and after every method; mutation is a hard failure.

## Best-so-far guarantee

The best-so-far archive guarantees only that the returned observed tour is not worse than the first observed tour for that method. It does not imply monotonic policy improvement, global convergence, or an approximation ratio.

## Budget boundary

The matched neural budget counts complete tour-length evaluations. Two-opt edge-swap evaluations are tracked separately. A comparison that ignores this distinction would overstate compute parity.

## Failure policy

The code raises rather than silently continuing on:

- malformed instances or tours;
- non-finite coordinates, logits, losses, gradients, or parameters;
- invalid checkpoint schemas or tensor shapes;
- corpus fingerprint mismatch;
- stored exact labels that disagree with recomputation;
- source-model mutation;
- objective-budget violations;
- a candidate that appears better than the exact optimum beyond tolerance;
- Held–Karp disagreement with brute force on tiny instances.
