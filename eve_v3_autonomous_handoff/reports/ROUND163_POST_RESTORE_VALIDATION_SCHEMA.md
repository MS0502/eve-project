# Round163 — Deterministic post-restore validation checklist/schema

Round163 adds a deterministic, artifact-free validation checklist for an operator restore. It does not include or create artifacts.

## Required checklist

1. Registered path presence: `vocab.txt`, `vectors.npy`, and `subset_manifest.json` exist at the manifest paths.
2. Manifest field match: expected manifest fields remain present and unchanged.
3. SHA256 match: restored file checksums match `seeds/MANIFEST.yaml` exactly.
4. Shape/dtype match: `vectors.npy` loads read-only with the registered shape and `float32` dtype.
5. Readiness gate green: seed/vector readiness reports `ready_for_explicit_operator_artifact_load`.
6. No-commit boundary clean: no `vectors.npy`, seed subsets, zip/part files, or operator artifacts enter the PR diff/staging area.
7. Policy flags unchanged: production persistence remains NO-GO, runtime mapping default remains false, enforcement remains disabled, and AGP is not bypassed.

## Expected focused validation commands after restore

- `python -m compileall -q adapters tests main.py`
- `python -m pytest --collect-only -q`
- `python -m pytest -q tests/test_v3_round162_164_restore_contract_preflight.py tests/test_v3_round159_seed_vector_artifact_gate.py`

Validation JSON: `eve_v3_autonomous_handoff/validation/ROUND163_POST_RESTORE_VALIDATION_SCHEMA_STATUS.json`.
