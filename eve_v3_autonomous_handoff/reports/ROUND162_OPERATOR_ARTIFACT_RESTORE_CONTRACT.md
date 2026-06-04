# Round162 — Operator artifact restore contract for registered vectors

Round162 defines the exact restore contract for the registered fastText subset `vectors.npy` artifacts. It is a documentation/data round only; no vector files, seed subsets, zips, part files, or operator-artifact bundles are included.

## Expected registered paths

- Medium 30k runtime-primary candidate: `seeds/subsets/cc.ko.300.subset.medium.30k/vectors.npy`.
- Small 5k retained production lexical seed: `seeds/subsets/cc.ko.300.subset.small.5k/vectors.npy`.
- Mini 1k fixture-level subset: `seeds/subsets/cc.ko.300.subset.mini.1k/vectors.npy`.

Each subset also requires its registered `vocab.txt` and `subset_manifest.json` at the same subset directory.

## Expected manifest fields

The restore contract requires the registered subset entry fields in `seeds/MANIFEST.yaml`, including `file_location`, `vocab_size`, `vector_dim`, `vocab_file`, `vectors_file`, `subset_manifest_file`, `vocab_checksum`, `vectors_checksum`, and `subset_manifest_checksum`.

## Verification contract

- SHA256 checksum of every restored file must match the manifest exactly.
- `vectors.npy` must load read-only with the manifest shape `[vocab_size, vector_dim]`.
- `vectors.npy` dtype must be `float32`.
- The readiness gate must be green before load-dependent repair starts.

## No-commit safety list

- `seeds/subsets/**/vectors.npy`
- `seeds/subsets/**`
- `*.zip`
- `*.part`
- `*_operator_artifacts/**`
- `_operator_artifacts/**`

Validation JSON: `eve_v3_autonomous_handoff/validation/ROUND162_OPERATOR_ARTIFACT_RESTORE_CONTRACT_STATUS.json`.
