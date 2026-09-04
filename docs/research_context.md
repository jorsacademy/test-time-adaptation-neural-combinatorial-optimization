# Research Context

Neural combinatorial optimization traditionally separates amortized generalization from instance-specific search. A generalization model pays training cost once and solves new instances quickly, but can degrade under scale or distribution shift. Instance-specific active search uses the objective of one test instance to improve a policy, but can be expensive and unstable.

Key reference points include:

- Bello et al. (2016), *Neural Combinatorial Optimization with Reinforcement Learning*: active search updates a policy on one test instance.
- Hottung, Kwon, and Tierney (ICLR 2022), *Efficient Active Search for Combinatorial Optimization Problems*: update restricted parameter subsets rather than every source-model weight.
- Choo et al. (NeurIPS 2022), *Simulation-guided Beam Search for Neural Combinatorial Optimization*: combine learned tree search with efficient active search.
- Son et al. (ICML 2023), *Meta-SAGE*: adapt context parameters to larger routing instances using scheduled exploration.
- Wei et al. (ICML 2024), *Extending Test-Time Augmentation with Metamorphic Relations for Combinatorial Problems*: formalize broader inference-time transformations for combinatorial tasks.
- Chen et al. (NeurIPS 2025), *Test-Time Projection Learning*: improve routing generalization at inference time through learned projection.
- Liao, Koushanfar, and Naghizadeh (2026 preprint), *Test-Time Adaptation for Unsupervised Combinatorial Optimization*: propose TACO to bridge generalizable and instance-specific unsupervised NCO.

This repository does not reproduce these systems. It constructs a small exact-audited laboratory around a narrower methodological question: after fixing a source policy and a complete-tour evaluation budget, what is gained by sampling, transformations, a small instance adapter, full fine-tuning, random-start active search, or local search?

## Why TSP?

Euclidean TSP provides:

- an objective available without labels at test time;
- strict discrete feasibility that can be enforced by masks;
- exact Held–Karp evaluation on small instances;
- classical nearest-neighbor and 2-opt controls;
- clear geometric distribution shifts;
- a direct bridge to the active-search literature.

The same methodology could be extended to CVRP, scheduling, independent set, or MILP subroutines, but exact audits and adaptation parameterization would need to change.
