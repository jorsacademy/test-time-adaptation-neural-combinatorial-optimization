# Verification

Local pre-merge verification exercises the complete exact and learned pipeline:

- 35 regression tests pass;
- branch-aware coverage is above the configured 84% threshold;
- Python compile-all succeeds for source and tests;
- Held–Karp agrees with brute-force tour enumeration on randomized tiny instances;
- all generated regimes are deterministic and pairwise-distinct;
- corpus fingerprints and stored optima are recomputed on load;
- edge logits are symmetric and permutation equivariant;
- coordinate translation and positive scaling leave normalized logits unchanged within tolerance;
- every adaptation method preserves the shared source model;
- matched methods consume exactly the declared complete-tour budget;
- best-so-far curves are monotone;
- adapter, full, scratch, augmentation, local-search, benchmark, research, checkpoint, and CLI paths complete locally.

GitHub Actions is the authoritative clean-environment check. It runs Ruff, formatting, strict mypy, branch-aware pytest, and an end-to-end collect–train–oracle–TTA benchmark on Python 3.11 and 3.12.
