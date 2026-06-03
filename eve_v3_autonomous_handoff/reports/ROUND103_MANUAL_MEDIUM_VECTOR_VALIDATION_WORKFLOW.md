# Round103 Manual Medium Vector Validation Workflow

## Goal

Round103 accepts the Round102 conclusion: this Codex environment cannot download GitHub Release assets because HTTPS CONNECT returns 403. Instead of retrying direct Release download, Round103 adds the operator-facing single command to run after the medium 30k vector artifact has been manually restored in a network-enabled local/Termux/Codespaces environment.

The workflow does not create, fake, stage, or commit `vectors.npy`.

## Implemented surface

Added `adapters/medium_vector_manual_validation.py`.

The module provides:

- `build_round103_manual_validation_plan(...)`: read-only readiness plan over the medium subset manifest scan and git artifact-boundary audit.
- `artifact_boundary_audit(...)`: confirms the target `vectors.npy` path is not tracked by git.
- `run_round103_manual_validation(...)`: fail-closed validation runner. It executes focused validation commands only when the medium artifact is present, checksum/shape/dtype-valid through the Round100 scan, and not tracked by git.
- `write_round103_manual_validation_status(...)`: JSON status export only.

Single command after manual artifact installation:

```bash
python -m adapters.medium_vector_manual_validation \
  --output eve_v3_autonomous_handoff/validation/ROUND103_MANUAL_MEDIUM_VECTOR_VALIDATION_STATUS.json
```

Manual restore command retained from Round102:

```bash
python -m adapters.medium_vector_release_restore \
  --work-dir /tmp/eve_round102_medium_restore \
  --asset-dir /path/to/downloaded/release-assets \
  --no-download \
  --install-to-repo \
  --output eve_v3_autonomous_handoff/validation/ROUND102_MEDIUM_VECTOR_ARTIFACT_RESTORE_STATUS.json
```

## Validation behavior

When the verified local medium artifact exists at:

```text
seeds/subsets/cc.ko.300.subset.medium.30k/vectors.npy
```

and remains untracked by git, the Round103 command runs:

```bash
python -m compileall -q adapters tests main.py
pytest -q tests/test_v3_round100_medium_vector_restoration.py
pytest -q tests/test_v3_round97_98_runtime_mapping_enable_smoke.py
pytest -q tests/test_v3_round92_runtime_mapping_gate_dry_run.py tests/test_v3_round93_runtime_mapping_proposal_report.py tests/test_v3_round94_runtime_mapping_enforcement_dry_run.py tests/test_v3_round95_runtime_mapping_operator_acceptance_fixture.py tests/test_v3_round96_runtime_mapping_enable_smoke_precheck.py tests/test_v3_round97_98_runtime_mapping_enable_smoke.py
```

If the artifact is absent or invalid, it records `blocked_manual_artifact_not_ready` and does not claim downstream validation.

## Current execution result

In this checkout, `vectors.npy` is still absent. Round103 therefore correctly reports:

```text
status = blocked_manual_artifact_not_ready
hard_stop_released = false
commands_executed = false
```

This is the expected state for the Codex environment. The next validation transition requires operator/manual artifact installation outside the PR diff.

## Tests and checks

Passed:

- `python -m compileall -q adapters tests main.py`
- `pytest -q tests/test_v3_round103_manual_medium_vector_validation.py tests/test_v3_round102_medium_vector_release_restore.py tests/test_v3_round100_medium_vector_restoration.py` — 10 passed.
- `python -m json.tool eve_v3_autonomous_handoff/validation/ROUND103_MANUAL_MEDIUM_VECTOR_VALIDATION_STATUS.json >/dev/null`

Blocked/expected:

- `python -m adapters.medium_vector_manual_validation --output eve_v3_autonomous_handoff/validation/ROUND103_MANUAL_MEDIUM_VECTOR_VALIDATION_STATUS.json` — exits 2 because the medium vector artifact is not installed in this checkout.

## Next recommendation

Merge the Round102/103 code-only helper work. Then run the manual restore command in an environment that can access the Release assets. Once `hard_stop_released=true`, run the Round103 single validation command above and use its JSON output to decide whether to proceed to runtime mapping persistence approval gate and AGP proof object expansion.
