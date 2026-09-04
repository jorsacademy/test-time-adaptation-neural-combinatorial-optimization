# Security Policy

This repository executes no model-generated source code and does not require network access at runtime.

## Untrusted artifacts

Treat datasets, JSON files, and checkpoints as untrusted input.

- Checkpoints use Safetensors rather than pickle.
- Checkpoint metadata, tensor keys, shapes, and finite values are validated.
- JSONL corpora use schema versions and SHA-256 fingerprints.
- Stored exact labels are recomputed when corpora are loaded with verification enabled.

Safetensors reduces arbitrary-code-deserialization risk but does not make an artifact semantically trustworthy.

## Resource exhaustion

Held–Karp and brute-force verification are exponential. Do not run untrusted node counts without explicit limits. Test-time budgets and adaptation steps should also be capped in services or shared compute environments.

## Reporting

Report suspected vulnerabilities privately through the repository owner’s GitHub security contact before public disclosure.
