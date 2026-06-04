# NEXT_ACTIONS

## Current position

Latest completed rounds:

- Round97 controlled runtime mapping enable smoke
- Round98 runtime mapping persistence gate audit

Status:

- `runtime_mapping_enabled=False` after rollback
- `enforcement_enabled=False`
- Ephemerally smoke-mapped token: `민석`
- Persistence gate: `ready_for_operator_persistence_decision`
- Persistence applied now: `false`

## Validation boundary

Passed focused/adjacent validation is recorded in `validation/ROUND97_VALIDATION_STATUS.json` and `validation/ROUND98_VALIDATION_STATUS.json`.

Blocked/partial validation is intentionally separated:

- `seeds/subsets/cc.ko.300.subset.medium.30k/vectors.npy` is absent from the code-only package.
- Full medium fastText validation is blocked until that artifact is restored.
- Repository-wide collect-only/compileall still hits legacy root issues unrelated to Round97/98.

## Highest-value next round

Round99 should design an operator persistence decision path, but must not persist runtime mapping unless one of these is true:

1. Medium vectors are restored and split/full validation passes.
2. The operator explicitly accepts a partial-validation persistence experiment.

Required Round99 outputs:

- explicit persistence preconditions
- operator approval schema
- rollback checklist
- validation plan distinguishing focused pass from medium/full blocked
- no AGP bypass
- no vector-as-anchor shortcut

## Round99 update — validation-first gate

Current position after post-merge validation:

- Focused compile check passed.
- Round97/98 focused tests are blocked/partial due absent subset vector files.
- Round92~Round98 adjacent tests are blocked/partial for the same reason.
- Collect-only and repository-wide compile probes still have separated pre-existing legacy root blockers.

Highest-value next round is now:

```text
Round100: medium vector restoration / validation plan
```

Required Round100 outputs:

- Decide how the medium 30k vector artifact is restored or validated outside the code-only package.
- Preserve manifest provenance/checksum rules; do not create fake checksums or seed files.
- Re-run Round97/98 focused and Round92~Round98 adjacent validation after restoration.
- Keep runtime mapping persistence disabled until validation is honestly passed or the operator explicitly approves partial validation.

Deferred until validation is unblocked:

- AGP proof object expansion.
- Runtime mapping persistence approval gate implementation.
- Any persistence mutation or enforcement enablement.

## Round100 update — artifact restoration gate

Current position:

- Round100 restoration/audit helper is implemented and tested.
- Medium, small, and mini vector artifacts are absent in this checkout.
- Runtime mapping focused validation remains blocked until known fastText context vectors are available.

Next required operator action:

1. Obtain the original medium 30k `vectors.npy` outside the PR diff.
2. Run `python -m adapters.medium_vector_restoration --candidate /path/to/vectors.npy`.
3. Install the artifact only if the audit reports `acceptable_for_manual_install=true`.
4. Rerun medium and focused runtime mapping validation.

Still deferred:

- AGP proof object expansion.
- Runtime mapping persistence approval gate.
- Any persistence/enforcement mutation.

## Round101 update — final integrated PR hard stop

Current position:

- Issue #5 autonomous multi-round policy is acknowledged for this task.
- No intermediate PR was created during the round loop.
- Round100 restoration helper exists, but the required medium 30k `vectors.npy` is still absent.
- Round101 confirms hard stop because operator artifact action is required.

Immediate next action:

```text
Create one final integrated PR for Round100~Round101, then wait for operator action.
```

Operator action required before more autonomous implementation:

1. Restore and audit the medium 30k `vectors.npy` artifact outside the PR diff.
2. Or explicitly approve a partial-validation path.

Still deferred:

- Runtime mapping persistence approval gate.
- AGP proof object expansion.
- Legacy root blocker isolation.

## Round102 update — Release artifact restore blocked by environment download

Current position:

- The operator supplied the medium 30k vector artifact through GitHub Release tag `eve-medium-30k-20260603`.
- Round102 added a deterministic restore/audit helper for those assets.
- This execution environment could not download the Release assets due HTTPS CONNECT 403.
- No binary artifact was committed or installed.

Immediate next action:

```bash
python -m adapters.medium_vector_release_restore \
  --work-dir /tmp/eve_round102_medium_restore \
  --asset-dir /path/to/downloaded/release-assets \
  --no-download \
  --install-to-repo \
  --output eve_v3_autonomous_handoff/validation/ROUND102_MEDIUM_VECTOR_ARTIFACT_RESTORE_STATUS.json
```

Proceed only when the status JSON reports `hard_stop_released=true`.

Then rerun Round97/98 and Round92~98 focused validation. Runtime mapping persistence approval gate and AGP proof object expansion remain deferred until those validations are unblocked.

## Round103~106 clean replacement PR update

Current position:

- Round103~106 code and documentation have been reapplied on latest main as a clean replacement for conflicted PR #7.
- No `vectors.npy`, wrapper zips, raw parts, restored zips, `_operator_artifacts`, or upload artifacts are included.
- Runtime mapping persistence is still not applied.

Immediate next actions:

1. Restore the medium 30k `vectors.npy` outside the PR diff.
2. Run Round102 restore and Round103 manual validation until the local validation status passes.
3. Re-run the runtime mapping approval/proof/decision tests.
4. If persistence should actually be applied, do so in a later explicit mutation patch with rollback and full validation.

## Round107 update — activation dry-run harness complete

Current position:

- Round107 defines checkpoint, rollback, audit-log, state-debug, and future touch-plan surfaces for runtime mapping persistence activation.
- The dry-run execution proves disabled defaults remain unchanged.
- Persistence is still not enabled.

Required next step if activation is requested:

1. Create a separate explicit activation patch.
2. Write real checkpoint and rollback JSON artifacts before changing flags.
3. Append future activation audit events using the Round107 schema.
4. Apply only the reviewed runtime mapping persistence state changes.
5. Rerun focused, adjacent, collect-only, compileall, and any broader validation required by the operator.
6. Confirm rollback restores `runtime_mapping_enabled=False` and `enforcement_enabled=False`.

Still forbidden until that explicit patch:

- Runtime mapping default changes.
- Enforcement default changes.
- AGP bypass or vector-as-anchor shortcuts.
- `vectors.npy` commits.
- Semantic memory/quarantine mutation.

## Round108 update — guarded activation candidate complete

- Round108 adds a guarded runtime mapping persistence activation candidate.
- The candidate requires Round106 decision readiness, Round107 dry-run no-mutation proof, explicit operator approval, checkpoint creation, audit log emission, before/after state-debug export, and rollback verification.
- Defaults remain disabled: `runtime_mapping_enabled=False`, `enforcement_enabled=False`.
- Next round may review whether an operator-approved persistence activation should become a production startup path, but only as a separate explicit patch with full validation.

## After Round109

- Review the Round109 approval fixture and rollback drill artifacts before considering any real persistence enablement.
- Keep persistence disabled by default and enforcement disabled unless a future explicit round introduces a guarded, checkpointed, audited mutation path.
- Do not include `vectors.npy` or seed subset artifacts in runtime mapping persistence PR diffs.

## After Round112

1. Review `validation/ROUND110_RUNTIME_MAPPING_LIMITED_PERSISTENCE_SANDBOX_STATUS.json` for sandbox checkpoint/audit/rollback proof.
2. Review `validation/ROUND111_SANDBOX_ROLLBACK_CLEANUP_STATUS.json` for sandbox state cleanup proof.
3. Review `validation/ROUND112_POST_SANDBOX_AUDIT_REPLAY_STATUS.json` for read-only replay proof.
4. Next highest-value round: Round113 state-debug/audit replay viewer.
5. Keep `runtime_mapping_enabled=False` and `enforcement_enabled=False` by default.
6. Do not enable production persistence until a later explicit operator approval round.
7. Continue forbidding `vectors.npy`, `_operator_artifacts`, zip/part/upload artifacts, and seed subset binary diffs.

## Broader validation follow-up

- Before any production persistence discussion, resolve or explicitly quarantine the current broader validation blockers:
  - root-level legacy import failure for `spreading_activation` under `pytest -q` collection;
  - missing local seed `vectors.npy` fixture artifacts required by historical seed tests;
  - older baseline expectation failures in `pytest -q tests`.
- Do not report full-suite success until those checks pass in the target environment.

## After Round117

1. Review `validation/ROUND113_STATE_DEBUG_AUDIT_REPLAY_VIEWER_STATUS.json` for the state-debug/audit replay viewer timeline.
2. Review `validation/ROUND114_LEGACY_ROOT_BLOCKER_ISOLATION_STATUS.json` before interpreting root-level collect-only failures.
3. Review `validation/ROUND115_BROADER_VALIDATION_TRIAGE_STATUS.json` for focused-vs-broader validation separation.
4. Review `validation/ROUND116_SANDBOX_REPLAY_REGRESSION_GUARD_STATUS.json` for replay regression guard evidence.
5. Review `validation/ROUND117_OPERATOR_GO_NO_GO_PACKAGE_STATUS.json` for the no-go package.
6. Next recommended work: resolve or explicitly quarantine the legacy root `spreading_activation` collection blocker in a separate validation hygiene round before considering real production persistence.

## Round118-121 update — final pre-activation package is NO-GO

Current position:

- Round118 production persistence readiness audit: `NO-GO`.
- Round119 minimal enablement risk matrix/checklist: required items unsatisfied.
- Round120 final pre-activation gate package: `NO-GO`.
- Round121 required blocker isolation completed.

Blocked before real persistence enablement:

1. Broader validation must pass, or the operator must explicitly accept a partial-validation activation risk.
2. Explicit operator approval for production persistence must be supplied for a separate activation patch.
3. `runtime_mapping_enabled` default and `enforcement_enabled` default must remain false until that separate patch.
4. AGP boundary must remain intact; lexical persistence must not become an anchor bypass.
5. No `vectors.npy`, seed subset, zip/part, or `_operator_artifacts` files may enter the PR.

Next safe action:

```text
Review the Round120 gate and Round121 blockers. Do not activate production persistence until explicit operator approval and validation disposition are provided.
```

## After Round126

1. Review `validation/ROUND122_LEGACY_ROOT_IMPORT_BLOCKER_DIAGNOSIS_STATUS.json` for the pre-shim root import diagnosis.
2. Review `validation/ROUND123_SPREADING_ACTIVATION_COMPAT_SHIM_STATUS.json` for the shim decision and legacy re-export proof.
3. Review `validation/ROUND124_COLLECT_ONLY_RECOVERY_VERIFICATION_STATUS.json` for collect-only recovery evidence and the remaining `working_memory` blocker.
4. Treat broader validation as blocked/partial until the remaining legacy root imports are resolved or explicitly quarantined by a separate validation-hygiene round.
5. Keep production persistence `NO-GO`; do not enable `runtime_mapping_enabled` by default and do not enable enforcement.

## After Round131

Current position:

- `spreading_activation` root import blocker remains recovered by a minimal retained-implementation shim.
- `working_memory` root import blocker is now recovered by a minimal retained-implementation shim.
- `pytest --collect-only -q` has advanced to the next legacy blocker family: module-level validation side effects in `test_natural_lang_v2.py` call `sys.exit(1)` during collection.
- Production persistence remains NO-GO.
- `runtime_mapping_enabled` default remains false.
- `enforcement_enabled` remains false.

Recommended next actions:

1. Start the next round by diagnosing the `test_natural_lang_v2.py` collection-time `SystemExit` without deleting or weakening the test.
2. Prefer an import-safe isolation plan or test-entrypoint guard only if it preserves the legacy validation behavior when run intentionally.
3. Re-run `pytest --collect-only -q` after any isolation patch.
4. Keep broader validation marked blocked/partial until collect-only is green.
5. Keep production persistence NO-GO.

## Round136 next action update — post-SystemExit isolation

Current status after Round132-136:

1. `test_natural_lang_v2.py` no longer aborts pytest collection with `SystemExit`.
2. The NaturalLanguage v2 legacy validation is still preserved and now fails at test runtime (`pytest -q test_natural_lang_v2.py`) with 8/28 checks passing. Do not mark it skipped/xfail or weaken its assertions.
3. `pytest --collect-only -q` now reaches the next blockers: missing root `dmn` compatibility imports through `test_eve_main_ab.py` and `test_eve_main_abc.py`.
4. Production persistence remains **NO-GO**. Do not enable production persistence, runtime mapping defaults, or enforcement.

Recommended next loop:

- Round137: diagnose root `dmn` import blocker provenance and retained legacy source, if any.
- Round138: isolate or add a minimal compatibility shim only if it re-exports retained behavior and does not fake runtime behavior.
- Round139: rerun collect-only and record the next blocker honestly.
- Continue to keep NaturalLanguage v2 behavior failures visible as test failures after collection is recovered.

## Next after Round141

1. Round142: diagnose legacy root `digital_somatic` import blockers in `test_eve_main_ab.py` and `test_eve_main_abc.py`.
2. Round143: add a minimal compatibility shim only if a retained implementation exists, otherwise hard-stop with an isolation plan.
3. Round144: rerun `python -m pytest --collect-only -q` and record whether collection advances.
4. Keep production persistence **NO-GO** until collect-only is green and broader validation is no longer blocked/partial.

Guardrails remain unchanged: do not enable production persistence, do not change `runtime_mapping_enabled` default to true, do not enable enforcement, do not bypass AGP, do not weaken legacy tests, and do not commit vector/seed/operator artifacts.

## Next after Round146

1. Treat the root `digital_somatic` missing-import blocker as recovered by a retained-implementation re-export shim.
2. Do **not** enable production persistence: production persistence remains **NO-GO**.
3. Do not change `runtime_mapping_enabled` default to true, do not enable enforcement, and do not bypass AGP.
4. Next safe loop should diagnose and isolate the two remaining legacy root collection side effects without weakening, deleting, skipping, or xfail-marking the tests:
   - `test_eve_main_ab.py` import-time `learn_beliefs(path='/home/claude/eve/beliefs.json')` file dependency.
   - `test_eve_main_abc.py` import-time `learn_beliefs(beliefs_dict={...})` dict/object mismatch.
5. Keep broader validation marked blocked/partial until `python -m pytest --collect-only -q` is green.
6. Keep the `test_natural_lang_v2.py` behavior failure visible until it is fixed honestly.

## Next after Round151

1. Treat the `test_eve_main_ab.py` and `test_eve_main_abc.py` collection-time script side-effect blockers as recovered for pytest collection.
2. Keep production persistence **NO-GO**; do not attempt production persistence enablement in the next patch unless the operator explicitly requests a separate activation round.
3. Do not change `runtime_mapping_enabled` default to true, do not enable enforcement, and do not bypass AGP.
4. Broader validation is still red. Next safe work should address real runtime validation failures honestly, without skips/xfails, dummy vectors, semantic case hardcoding, or translation of Korean behavior fixtures.
5. Candidate next priorities:
   - Korean NaturalLanguage behavior failures (`test_natural_lang_v2.py`, `tests/test_round2_nl_sd.py`).
   - Missing vector artifact policy/disposition without committing `vectors.npy` or seed subset artifacts.
   - EVE-specific vector commit prerequisite cascade failures.
6. Continue preserving Korean-first behavior cases, including tokens such as `민석`.

## Round152~156 update — next recommended cluster after NaturalLanguage fix

Current position:

- Round152~156 completed in one integrated loop.
- Korean NaturalLanguage v2 focused failures are fixed and verified.
- `python -m pytest --collect-only -q` remains green (`1294 tests collected`).
- Full pytest remains red: `210 failed, 1084 passed` after improving by two tests.

Next recommended cluster:

```text
Seed/vector artifact dependency readiness or operator real-artifact restoration
```

Constraints for the next round:

- Do not fabricate or commit `vectors.npy`, seed subsets, zips, part files, or `_operator_artifacts`.
- Do not enable production persistence.
- Do not change `runtime_mapping_enabled` default to true.
- Do not enable enforcement.
- Do not bypass AGP.
- If real artifacts are not restored by the operator, keep next work diagnostic/read-only and improve failure/readiness reporting honestly.

## Next after Round161

1. Operator should restore real registered fastText subset `vectors.npy` artifacts outside the PR boundary.
2. Rerun `python -m pytest -q tests/test_v3_round159_seed_vector_artifact_gate.py` to confirm readiness status changes from blocked to ready.
3. Only after real artifact readiness is green, re-run focused fastText wrapper/load tests.
4. Keep production persistence NO-GO, `runtime_mapping_enabled` default false, and enforcement disabled.
5. If artifacts remain unavailable, continue with code-only diagnostic/readiness work; do not create dummy vectors or skip tests.

## Next after Round166

1. Restore real registered operator-owned `vectors.npy` artifacts outside the PR boundary:
   - `seeds/subsets/cc.ko.300.subset.medium.30k/vectors.npy`
   - `seeds/subsets/cc.ko.300.subset.small.5k/vectors.npy`
   - `seeds/subsets/cc.ko.300.subset.mini.1k/vectors.npy`
2. Do not commit `vectors.npy`, `seeds/subsets/**`, zip files, part files, upload bundles, or `_operator_artifacts/**`.
3. Verify SHA256 checksums against `seeds/MANIFEST.yaml`, then verify `vectors.npy` shape `[vocab_size, vector_dim]` and dtype `float32`.
4. Rerun the Round164 preflight; start load-dependent repair only if it reports green.
5. Keep production persistence NO-GO, `runtime_mapping_enabled` default false, enforcement disabled, and AGP unbypassed.
6. If real artifacts remain unavailable, continue with a non-artifact code-only cluster instead of dummy vectors, fake checksums, skips, or xfails.
