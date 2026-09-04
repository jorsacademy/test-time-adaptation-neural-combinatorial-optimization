# Contributing

Changes should preserve the exact/approximate boundary and pass:

```bash
ruff check .
ruff format --check .
mypy src
pytest
```

Requirements for methodological changes:

- no exact optimum may enter a test-time adaptation function;
- every new learned method must report complete-tour objective evaluations;
- extra local-search or solver work must be counted separately;
- shared source checkpoints must remain immutable across test instances;
- every returned tour must pass the independent audit;
- new corpus fields must participate in schema and fingerprint validation;
- new claims require a frozen configuration and non-cherry-picked comparisons;
- failure cases must raise or be reported explicitly rather than silently repaired.

New benchmark methods should include a matched baseline and tests for budget accounting, determinism, and source-state isolation.
