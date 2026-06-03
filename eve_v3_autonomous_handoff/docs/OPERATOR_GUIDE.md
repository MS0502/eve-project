# OPERATOR_GUIDE

## Round97/98 operator summary

Round97 opened runtime lexical→concept mapping only inside a controlled smoke path for `민석`, then rolled back. Round98 audited the result and did not persist runtime mapping.

Current flags after Round98:

- `runtime_mapping_enabled=False`
- `enforcement_enabled=False`

## What passed

- Round96 package manifest SHA/size validation and zip integrity.
- Round97/98 focused tests.
- Round92~Round98 focused/adjacent tests.
- Focused compileall for `adapters`, `tests`, and `main.py`.

## What is partial / blocked

- Medium fastText validation is blocked because `seeds/subsets/cc.ko.300.subset.medium.30k/vectors.npy` is absent.
- Full collect-only is partial due legacy root tests importing missing `spreading_activation`.
- Full repository compileall is partial due pre-existing syntax errors in legacy root files.

## Operator decision needed before persistence

Do not persist runtime mapping unless you approve one of these paths:

1. Restore the medium vector artifact and require full validation.
2. Explicitly approve a partial-validation persistence experiment.

In both paths, vectors remain evidence only and must not become AGP anchors.

## Round99 operator note — post-merge validation

Post-merge validation did not produce a full pass. The code surfaces compile, but runtime-mapping validation fixtures cannot complete without a restored subset vector artifact.

What passed:

- `python -m compileall -q adapters tests main.py`

What is blocked/partial:

- `pytest -q tests/test_v3_round97_98_runtime_mapping_enable_smoke.py` — blocked because the `민석` prerequisite vector cannot be committed without known fastText context.
- Round92~Round98 adjacent focused tests — blocked for the same missing-vector reason.
- `pytest --collect-only -q` — partial due legacy root missing `spreading_activation`.
- `python -m compileall -q .` — partial due legacy root SyntaxError blockers.

Operator decision needed:

1. Restore the medium 30k subset `vectors.npy` artifact according to the existing manifest/checksum policy, then re-run Round99 validation.
2. Or explicitly approve a partial-validation path, understanding that runtime mapping persistence must remain disabled unless separately approved.

Recommended next task:

- Round100 should be a medium vector restoration / validation plan, not AGP proof expansion yet.

## Round100 operator guide — restoring medium vectors

Round100 provides a verification path but does not include a binary vector artifact.

To restore medium validation:

```bash
python -m adapters.medium_vector_restoration --candidate /path/to/vectors.npy
```

Proceed only when the JSON output reports:

```json
"acceptable_for_manual_install": true
```

Then place the verified file at:

```text
seeds/subsets/cc.ko.300.subset.medium.30k/vectors.npy
```

After restoration, rerun:

```bash
pytest -q tests/test_v3_round50_subset_medium_30k.py tests/test_v3_round51_wrapper_primary_medium_swap.py
pytest -q tests/test_v3_round97_98_runtime_mapping_enable_smoke.py
pytest -q tests/test_v3_round92_runtime_mapping_gate_dry_run.py tests/test_v3_round93_runtime_mapping_proposal_report.py tests/test_v3_round94_runtime_mapping_enforcement_dry_run.py tests/test_v3_round95_runtime_mapping_operator_acceptance_fixture.py tests/test_v3_round96_runtime_mapping_enable_smoke_precheck.py tests/test_v3_round97_98_runtime_mapping_enable_smoke.py
```

Do not treat small/mini fallback results as medium/full validation. Do not persist runtime mapping until validation is honestly passed or a partial-validation path is explicitly approved.

## Round101 operator guide — autonomous hard stop

The Issue #5 autonomous multi-round policy is now the operating model for future tasks: internal round records first, one final integrated PR at the end.

This run stops after Round101 because the next required step is external/operator-controlled:

- restore the original medium 30k `vectors.npy` artifact outside the PR diff; or
- explicitly approve a partial-validation path.

Do not ask the next autonomous task to continue runtime mapping persistence, AGP proof expansion, or legacy blocker isolation until one of those actions is complete. Without the vector artifact, the Round92~Round98 focused validation substrate cannot create the prerequisite Eve-specific vector from known fastText context.

After artifact restoration, rerun the Round100 guide commands before approving persistence or further runtime changes.

## Round102 operator guide — GitHub Release artifact restore

Release assets supplied:

- `subset_medium_30k-20260603T024008Z-3-001.zip.part01.upload.zip`
- `subset_medium_30k-20260603T024008Z-3-001.zip.part02.upload.zip`
- `subset_medium_30k_split_manifest.json`

The automated download path was attempted, but this environment returned HTTPS CONNECT 403. To continue safely, download the three Release assets outside the repo and run:

```bash
python -m adapters.medium_vector_release_restore \
  --work-dir /tmp/eve_round102_medium_restore \
  --asset-dir /path/to/downloaded/release-assets \
  --no-download \
  --install-to-repo \
  --output eve_v3_autonomous_handoff/validation/ROUND102_MEDIUM_VECTOR_ARTIFACT_RESTORE_STATUS.json
```

The helper verifies the reconstructed zip SHA-256, zip integrity, internal `vectors.npy` SHA-256, shape, and dtype before installing the ignored local seed vector.

Never `git add`:

- wrapper zip files;
- raw `.part01` / `.part02` files;
- restored zip;
- `seeds/subsets/cc.ko.300.subset.medium.30k/vectors.npy`.

Only proceed to Round97/98 validation when `hard_stop_released=true`.

## Round107 operator guide — activation dry-run only

Round107 is the final harness before a possible future runtime mapping persistence activation. It does not enable persistence.

Artifacts to review:

- `eve_v3_autonomous_handoff/reports/ROUND107_RUNTIME_MAPPING_PERSISTENCE_ACTIVATION_DRYRUN.md`
- `eve_v3_autonomous_handoff/validation/ROUND107_RUNTIME_MAPPING_PERSISTENCE_ACTIVATION_DRYRUN_STATUS.json`

What Round107 defines:

- Future checkpoint JSON format.
- Future rollback JSON format.
- Future append-only audit JSONL schema.
- State-debug export key: `lex_concept_mapping.runtime_mapping_persistence_activation`.
- Exact files and state a future activation patch would touch.

Current defaults after Round107:

- `runtime_mapping_enabled=False`
- `enforcement_enabled=False`

Do not treat Round107 as activation approval. A later patch must explicitly write checkpoint/rollback/audit artifacts and pass validation before enabling persistent runtime mapping.

## Round108 operator guide — guarded activation candidate

Round108 is not default persistence enablement. It provides a guarded candidate runner for operator-reviewed activation drills.

Operator-approved candidate runs must provide:

- Round106 decision packet ready state.
- Round107 activation dry-run no-mutation proof.
- `operator_approved=True`.
- Approval token: `ROUND108_OPERATOR_APPROVED_RUNTIME_MAPPING_PERSISTENCE_CANDIDATE`.
- `apply_candidate=True`.

The candidate writes checkpoint, rollback, audit JSONL, and before/after state-debug exports, then rolls runtime mapping back to disabled flags. Enforcement remains disabled throughout.
