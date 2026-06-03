# Round101 Medium Vector Artifact Workflow

## Goal

Round101 finalizes the operating workflow for supplying `cc.ko.300.subset.medium.30k/vectors.npy` without committing the binary artifact to the repository.

Round101 does **not** restore the artifact itself. It chooses and documents the safe supply path so the next validation run can be unblocked without fake vectors, dummy `.npy` files, or PR-diff binary churn.

## Selected workflow

Preferred path:

```text
GitHub Release asset or operator-supplied artifact outside the repo
```

Repository contents remain limited to:

- manifest constants and checksums
- deterministic audit helpers
- operator instructions
- validation status/report files
- tests for fail-closed behavior

The repository must not contain:

- `vectors.npy`
- generated dummy `.npy` or `.npz` artifacts
- fake checksums
- split binary chunks committed to `packages/`

## Release asset convention

Recommended release metadata:

```text
release tag: eve-v3-medium-vector-artifacts
asset name: cc.ko.300.subset.medium.30k.vectors.npy
```

The release asset must be the original operator-extracted medium 30k vector file and must verify as:

```text
target path: seeds/subsets/cc.ko.300.subset.medium.30k/vectors.npy
sha256: SHA256:f228cbca9816d539ce9532e63fbb1ea95e4c66a7c3df286c788f817e2055bd05
shape: (30000, 300)
dtype: float32
```

## Operator workflow

1. Upload the exact `vectors.npy` artifact as a GitHub Release asset, or keep it as a local operator-supplied file outside the repository.
2. Audit the candidate locally:

   ```bash
   python -m adapters.medium_vector_restoration --candidate /path/to/vectors.npy
   ```

3. Continue only if the output reports:

   ```json
   "acceptable_for_manual_install": true
   ```

4. Manually install the verified artifact at:

   ```text
   seeds/subsets/cc.ko.300.subset.medium.30k/vectors.npy
   ```

5. Run medium artifact validation:

   ```bash
   pytest -q tests/test_v3_round50_subset_medium_30k.py tests/test_v3_round51_wrapper_primary_medium_swap.py
   ```

6. Run focused runtime mapping validation:

   ```bash
   pytest -q tests/test_v3_round97_98_runtime_mapping_enable_smoke.py
   pytest -q tests/test_v3_round92_runtime_mapping_gate_dry_run.py tests/test_v3_round93_runtime_mapping_proposal_report.py tests/test_v3_round94_runtime_mapping_enforcement_dry_run.py tests/test_v3_round95_runtime_mapping_operator_acceptance_fixture.py tests/test_v3_round96_runtime_mapping_enable_smoke_precheck.py tests/test_v3_round97_98_runtime_mapping_enable_smoke.py
   ```

## Fail-closed behavior

If the artifact is absent, the workflow status remains:

```text
blocked_waiting_for_operator_supplied_artifact
```

If a candidate exists but checksum, shape, or dtype does not match, it is rejected. The helper does not download, copy, create, or install any vector artifact.

## Runtime and persistence boundary

Round101 does not change runtime mapping or persistence:

- Runtime mapping remains disabled by default.
- Runtime mapping persistence is not applied.
- AGP is not bypassed.
- Vectors remain lexical evidence, not AGP anchors.
- Round97/98 and Round92~98 validation can be rerun only after the verified artifact is installed.

## Validation result

Passed:

- `python -m compileall -q adapters tests main.py`
- `pytest -q tests/test_v3_round100_medium_vector_restoration.py` — 7 passed.

Still blocked:

- Medium/full validation, because `seeds/subsets/cc.ko.300.subset.medium.30k/vectors.npy` is absent.
- Round97/98 focused validation rerun, because known fastText context remains unavailable until the artifact is installed.
- Round92~98 focused chain rerun, because it shares the same EveSpecific commit prerequisite.
