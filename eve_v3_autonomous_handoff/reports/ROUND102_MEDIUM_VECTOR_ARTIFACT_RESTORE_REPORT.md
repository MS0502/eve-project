# Round102 Medium Vector Artifact Restore / Audit Report

## Goal

Round102 consumes the operator-supplied GitHub Release artifact for the medium 30k vector subset without committing any binary artifact to the PR diff.

Release inputs supplied by the operator:

- tag: `eve-medium-30k-20260603`
- assets:
  - `subset_medium_30k-20260603T024008Z-3-001.zip.part01.upload.zip`
  - `subset_medium_30k-20260603T024008Z-3-001.zip.part02.upload.zip`
  - `subset_medium_30k_split_manifest.json`
- reconstructed zip SHA-256: `ecfeabea37cf947a09c3d2d11f83f5ac9b7bc29cb026d32b18c4be616970b98f`
- internal vector path: `subset_medium_30k/vectors.npy`
- internal vector SHA-256: `f228cbca9816d539ce9532e63fbb1ea95e4c66a7c3df286c788f817e2055bd05`
- expected vector shape/dtype: `(30000, 300)` / `float32`

## Implemented restore surface

Added `adapters/medium_vector_release_restore.py`.

The helper can:

1. Download the Release assets into an explicit temporary work directory.
2. Unwrap the two wrapper zip files to raw `.part01` / `.part02` files.
3. Concatenate raw parts into the reconstructed zip.
4. Verify reconstructed zip SHA-256.
5. Run `zipfile.testzip()` integrity checking.
6. Extract only `subset_medium_30k/vectors.npy` into the temporary work directory.
7. Reuse the Round100 vector audit to verify SHA-256, shape, and dtype.
8. Optionally copy the verified vector file into the ignored seed path with `--install-to-repo`.

The helper never stages files, never writes wrapper/raw/restored zip assets into the repository, and refuses to install `vectors.npy` unless all audit gates pass.

Safe local/manual command if Release downloads are already available outside the repo:

```bash
python -m adapters.medium_vector_release_restore \
  --work-dir /tmp/eve_round102_medium_restore \
  --asset-dir /path/to/downloaded/release-assets \
  --no-download \
  --install-to-repo \
  --output eve_v3_autonomous_handoff/validation/ROUND102_MEDIUM_VECTOR_ARTIFACT_RESTORE_STATUS.json
```

Do not `git add` wrapper zip files, raw part files, the restored zip, or `seeds/subsets/cc.ko.300.subset.medium.30k/vectors.npy`.

## Current execution result

The current execution environment could not download GitHub Release assets. Each attempted Release asset request failed with:

```text
<urlopen error Tunnel connection failed: 403 Forbidden>
```

Because the assets could not be downloaded here:

- no wrapper zip was written into the repo;
- no raw part was created in the repo;
- no restored zip was created in the repo;
- no `vectors.npy` was installed;
- medium vector artifact availability remains blocked in this checkout;
- Round97/98 validation remains blocked for the same missing known-context vector prerequisite.

This is an environment download blocker, not a checksum mismatch, vector audit failure, or AGP/runtime-mapping failure.

## Validation

Passed:

- `python -m compileall -q adapters tests main.py`
- `pytest -q tests/test_v3_round102_medium_vector_release_restore.py tests/test_v3_round100_medium_vector_restoration.py` — 7 passed.

Blocked/partial:

- `python -m adapters.medium_vector_release_restore --work-dir /tmp/eve_round102_medium_restore --repo-root . --install-to-repo --output eve_v3_autonomous_handoff/validation/ROUND102_MEDIUM_VECTOR_ARTIFACT_RESTORE_STATUS.json` — blocked by HTTPS CONNECT 403 for Release downloads.
- `pytest -q tests/test_v3_round97_98_runtime_mapping_enable_smoke.py` — 3 prerequisite failures remain because `민석` cannot be committed without known fastText context vectors.
- Round92~Round98 adjacent focused validation — 14 prerequisite failures remain for the same reason.
- `pytest --collect-only -q` — blocked/partial after 1227 collected tests because legacy root tests still import missing `spreading_activation`.

## Hard-stop state

Hard stop is not lifted in this execution environment because the required conditions were not observed locally:

- reconstructed zip SHA-256 match: not reached;
- `vectors.npy` SHA-256 match: not reached;
- shape/dtype match: not reached;
- `vectors.npy` absent from PR diff: preserved.

The hard stop can be lifted in a network-enabled or manually prepared environment by running the helper with downloaded Release assets and observing `hard_stop_released=true` in the status JSON.

## Next recommendation

Use the manual/local restore command above after downloading the three Release assets outside the repo. Once `hard_stop_released=true` and the ignored seed-path `vectors.npy` is installed locally, rerun:

```bash
pytest -q tests/test_v3_round97_98_runtime_mapping_enable_smoke.py
pytest -q tests/test_v3_round92_runtime_mapping_gate_dry_run.py tests/test_v3_round93_runtime_mapping_proposal_report.py tests/test_v3_round94_runtime_mapping_enforcement_dry_run.py tests/test_v3_round95_runtime_mapping_operator_acceptance_fixture.py tests/test_v3_round96_runtime_mapping_enable_smoke_precheck.py tests/test_v3_round97_98_runtime_mapping_enable_smoke.py
```

Only after those focused validations pass should the next autonomous rounds proceed to runtime mapping persistence approval gate and AGP proof object expansion.
