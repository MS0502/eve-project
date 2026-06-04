# TECHNICAL_MAP

## Runtime mapping surfaces through Round98

- `adapters/lex_concept_mapping_adapter.py`
  - `runtime_mapping_enable_smoke_precheck(...)`: Round96 read-only readiness gate.
  - `controlled_runtime_mapping_enable_smoke(...)`: Round97 ephemeral enable smoke with rollback.
  - `runtime_mapping_persistence_gate_audit(...)`: Round98 read-only persistence gate audit.
  - `stats()`: exposes Round97/Round98 versions and availability.

- `adapters/runtime_smoke_runner.py`
  - `run_round97_controlled_runtime_mapping_enable_smoke(...)`
  - `write_round97_controlled_runtime_mapping_enable_smoke(...)`
  - `run_round98_runtime_mapping_persistence_gate_audit(...)`
  - `write_round98_runtime_mapping_persistence_gate_audit(...)`

- `adapters/state_debug_adapter.py`
  - exposes controlled enable smoke and persistence gate audit versions/availability.

- `main.py`
  - code-only handoff fallback: if the medium 30k vectors file is absent, focused runtime tests may load the preserved small 5k subset; medium/full validation remains blocked/partial and must be reported as such.

## Round97 invariants

- Runtime mapping may be true only during the smoke method.
- Runtime mapping must be false after rollback.
- Enforcement must remain false.
- Ephemeral mapping table must be cleared.
- No AGP verify call is made by the runtime mapping smoke.
- No embedding lookup or EveSpecific vector commit is made by the runtime mapping smoke.

## Round98 invariants

- Audit is read-only.
- Persistence is not applied.
- Operator approval and full validation remain required before persistence.

## Round99 post-merge validation map

Validation surfaces checked:

- `main.py`: `build_full_engine()` still attempts medium 30k fastText load first and records a code-only blocked state if subset vectors are absent.
- `adapters/eve_self_learning_adapter.py`: `commit_eve_specific_vectors(...)` still requires observed Eve-specific candidates plus known fastText context.
- `adapters/eve_vector_store.py`: vector creation still derives deterministic 300d vectors from loaded fastText context vectors only.
- `tests/test_v3_round92_runtime_mapping_gate_dry_run.py` through `tests/test_v3_round97_98_runtime_mapping_enable_smoke.py`: fixtures require `민석` EveSpecific vector creation before runtime mapping dry-run/smoke checks.

Observed technical blocker:

- No subset `vectors.npy` is present under `seeds/subsets/cc.ko.300.subset.medium.30k/`, `small.5k/`, or `mini.1k/`.
- FastText adapter remains unloaded, so context words `오늘` and `군대` are not known context.
- The Eve-specific commit gate rejects `민석` with `insufficient_known_context`; this blocks Round92~Round98 focused runtime-mapping validations.

Boundary preserved:

- No vectors were treated as AGP anchors.
- No runtime mapping persistence was applied.
- No enforcement was enabled.
- No AGP bypass or threshold relaxation was introduced.

## Round100 medium vector restoration surfaces

- `adapters/medium_vector_restoration.py`
  - `scan_subset_artifact_paths(...)`: read-only medium/small/mini artifact path, checksum, shape, and dtype scan.
  - `audit_operator_supplied_medium_vectors(...)`: read-only fail-closed audit for an externally supplied medium `vectors.npy`.
  - `build_round100_restoration_plan(...)`: deterministic plan that separates medium/full, small/focused, and blocked validation states.
  - `write_round100_restoration_status(...)`: writes JSON status only; never writes vectors.

Validation boundary:

- Medium/full validation requires `seeds/subsets/cc.ko.300.subset.medium.30k/vectors.npy` with checksum `SHA256:f228cbca9816d539ce9532e63fbb1ea95e4c66a7c3df286c788f817e2055bd05`, shape `(30000, 300)`, and dtype `float32`.
- Small/focused fallback requires the exact small 5k artifact; it cannot be reported as medium/full validation.
- Mini 1k remains fixture boundary only.
- The helper does not copy, create, or mutate vector artifacts.

## Round101 autonomous hard-stop map

- `eve_v3_autonomous_handoff/reports/ROUND101_AUTONOMOUS_MULTI_ROUND_HARD_STOP.md`
  - Records the Issue #5 autonomous multi-round operating policy and the reason this run stops after Round100.
  - Does not add runtime behavior.

- `eve_v3_autonomous_handoff/validation/ROUND101_AUTONOMOUS_HARD_STOP_STATUS.json`
  - Structured status for the final integrated PR.
  - Classifies the blocker as `external_artifact_operator_action_required`.
  - Records that no dummy vectors, binary vector commits, AGP bypass, nondeterminism, weakened tests, or runtime mapping persistence were introduced.

Boundary:

- Round101 is documentation/reporting only.
- The next technical implementation round remains blocked until the medium 30k vector artifact is restored or the operator explicitly approves a partial-validation path.

## Round102 Release artifact restore surfaces

- `adapters/medium_vector_release_restore.py`
  - `restore_medium_vectors_from_release(...)`: downloads or consumes local Release assets in a temporary work directory, unwraps wrapper zips, concatenates raw parts, verifies reconstructed zip SHA-256, verifies zip integrity, extracts the internal `vectors.npy`, and reuses the Round100 vector audit.
  - `write_round102_release_restore_status(...)`: writes JSON status only.
  - `--install-to-repo`: copies `vectors.npy` into the ignored seed path only after all gates pass; it does not stage or commit binary artifacts.

- `tests/test_v3_round102_medium_vector_release_restore.py`
  - Proves the helper fails closed without assets.
  - Proves status export writes JSON only and does not create binary seed artifacts.

Boundary:

- Release assets are operator-supplied external artifacts.
- Wrapper zips, raw parts, restored zip, and `vectors.npy` remain forbidden in PR diffs.
- Runtime mapping validation remains blocked until `hard_stop_released=true` is observed and the ignored local seed vector is installed.

## Round103 manual medium vector validation surfaces

- `adapters/medium_vector_manual_validation.py`
  - `validate_round103_manual_medium_vectors(...)`: read-only fail-closed validation over release-restore status and a candidate medium `vectors.npy` audit.
  - `write_round103_manual_medium_vector_validation(...)`: writes JSON only.

Boundary:

- Does not copy, install, create, or commit vector artifacts.
- Does not relax manifest checksum, shape, or dtype gates.

## Round104 runtime mapping persistence approval surfaces

- `adapters/runtime_mapping_persistence_approval.py`
  - `build_runtime_mapping_persistence_approval(...)`: builds an operator approval packet from Round98 and Round103 evidence.
  - `write_round104_runtime_mapping_persistence_approval(...)`: writes JSON only.

Boundary:

- Does not enable or persist runtime mapping.
- Does not mutate concept memory, categories, AGP, or vector stores.

## Round105 AGP proof object expansion surfaces

- `adapters/agp_proof_object_expansion.py`
  - `build_agp_proof_object_expansion(...)`: expands proof rows for approved runtime mapping candidates.
  - `write_round105_agp_proof_object_expansion(...)`: writes JSON only.

Boundary:

- Anchor source remains explicit category plus SA activation only.
- Lexical, EveSpecific, and seed vectors are evidence only, never AGP anchors.

## Round106 runtime mapping persistence decision surfaces

- `adapters/runtime_mapping_persistence_decision.py`
  - `build_runtime_mapping_persistence_decision(...)`: records a decision packet and defers any state-changing persistence application.
  - `write_round106_runtime_mapping_persistence_decision(...)`: writes JSON only.

Boundary:

- Runtime mapping and enforcement remain disabled.
- Persistent application requires a later explicit patch.

## Round107 runtime mapping persistence activation dry-run surfaces

- `adapters/runtime_mapping_persistence_activation_dryrun.py`
  - `checkpoint_format()`: defines the future pre-activation checkpoint JSON schema without writing or applying it.
  - `rollback_format()`: defines the future rollback JSON schema and required disabled-flag restoration path.
  - `audit_log_schema()`: defines the future append-only JSONL activation audit schema.
  - `future_activation_touch_plan()`: enumerates exact files and state a later activation patch would touch.
  - `state_debug_export_surface()`: documents the state-debug export key and fields.
  - `build_runtime_mapping_persistence_activation_dryrun(...)`: executes a read-only before/after proof that flags and AGP/vector/category/memory state remain unchanged.
  - `write_round107_runtime_mapping_persistence_activation_dryrun(...)`: writes the dry-run report JSON only.
- `adapters/lex_concept_mapping_adapter.py`
  - `stats()` now advertises the Round107 dry-run state-debug surface while keeping `runtime_mapping_enabled=False` and `enforcement_enabled=False`.
- `adapters/state_debug_adapter.py`
  - snapshots `lex_concept_mapping.runtime_mapping_persistence_activation` as read-only debug data.

Boundary:

- No persistence activation is applied in Round107.
- Runtime mapping and enforcement defaults remain disabled.
- AGP, vectors, categories, concept memory, semantic memory, and quarantine remain unmutated.

Next:

- A future activation patch must create real checkpoint/rollback/audit artifacts and explicitly apply state changes under tests; Round107 is only the harness and proof surface.

## Round108 runtime mapping persistence activation candidate

- `adapters/runtime_mapping_persistence_activation_candidate.py`
  - `build_operator_approval_guard(...)`: fail-closed prerequisite/approval check over Round106 decision, Round107 dry-run, and the explicit Round108 operator token.
  - `build_runtime_mapping_activation_checkpoint(...)`: JSON checkpoint payload created before any candidate mutation.
  - `run_runtime_mapping_persistence_activation_candidate(...)`: default blocked/read-only; with explicit approval and `apply_candidate=True`, writes checkpoint/audit/state-debug artifacts, applies an ephemeral runtime mapping candidate flag, rolls back, and verifies protected state.
  - `verify_runtime_mapping_rollback(...)`: verifies disabled flags plus unchanged category/audit/vector/SA/concept-memory/AGP surfaces.
- `LexConceptMappingAdapter.stats()` and `StateDebugAdapter.snapshot_state()` now advertise the Round108 candidate surface while keeping `runtime_mapping_enabled=False` and `enforcement_enabled=False` by default.
- Round108 does not add default startup wiring and does not enable enforcement.

## Round109 runtime mapping persistence approval fixture

- `adapters/runtime_mapping_persistence_approval_fixture.py`
  - `build_round109_operator_approval_fixture(...)`: builds the limited-scope operator approval fixture for `runtime_mapping_persistence_only` and `["민석"]`.
  - `run_round109_runtime_mapping_persistence_approval_fixture_drill(...)`: runs the Round108 candidate in test/dry-run drill mode, verifies checkpoint-before-mutation, audit ordering, rollback, protected surfaces, and state-debug exports.
  - `write_round109_runtime_mapping_persistence_approval_fixture(...)`: writes JSON only and does not enable runtime mapping or enforcement.

## Round110-112 runtime mapping sandbox surfaces

- `adapters/runtime_mapping_limited_persistence_sandbox.py`
  - `run_round110_runtime_mapping_limited_persistence_sandbox(...)`: guarded JSON-only sandbox persistence drill. It writes checkpoint/audit/state-debug/sandbox-state/rollback artifacts, briefly enables runtime mapping inside the sandbox, disables it before return, and keeps enforcement and production persistence disabled.
  - `run_round111_sandbox_rollback_cleanup_verification(...)`: verifies Round110 checkpoint/audit/rollback evidence, removes the transient sandbox state JSON, and writes cleanup audit/receipt JSON.
  - `run_round112_post_sandbox_focused_validation_audit_replay(...)`: read-only replay of Round110 and Round111 audit evidence.
  - `write_round110_runtime_mapping_limited_persistence_sandbox(...)`, `write_round111_sandbox_rollback_cleanup_verification(...)`, and `write_round112_post_sandbox_focused_validation_audit_replay(...)`: JSON status writers only.
- `adapters/lex_concept_mapping_adapter.py`
  - `stats()` now advertises Round110 sandbox, Round111 cleanup, and Round112 replay surfaces while retaining disabled defaults.
- `adapters/state_debug_adapter.py`
  - State-debug now surfaces the Round110-112 sandbox metadata under `lex_concept_mapping`.
- `tests/test_v3_round110_112_runtime_mapping_sandbox.py`
  - Focused tests cover guarded sandbox execution, blocked guard behavior, cleanup, replay, state-debug exposure, and JSON export helpers.

Boundary:

- No production persistence is enabled.
- `runtime_mapping_enabled` and `enforcement_enabled` default to `False`.
- No `vectors.npy`, seed subset, zip/part/upload, or `_operator_artifacts` file is part of the patch.

## Round113-117 runtime mapping validation loop surfaces

- `adapters/runtime_mapping_limited_persistence_sandbox.py`
  - `build_round113_state_debug_audit_replay_viewer(...)`: read-only viewer over Round110-112 checkpoint, audit, rollback, cleanup, replay, and state-debug artifacts.
  - `build_round114_legacy_root_blocker_isolation(...)`: static isolation of root-level legacy collection blockers, currently missing `spreading_activation` imports.
  - `build_round115_broader_validation_triage_report(...)`: focused-vs-broader validation triage data.
  - `run_round116_runtime_mapping_sandbox_replay_regression_guard(...)`: JSON-only regression guard that reruns Round110-112 and verifies cleanup.
  - `build_round117_operator_go_no_go_package(...)`: operator package recommending no-go for production persistence in this PR.
- `adapters/lex_concept_mapping_adapter.py`
  - `stats()` now advertises Round113-117 surfaces while keeping `runtime_mapping_enabled=False` and `enforcement_enabled=False` by default.
- `adapters/state_debug_adapter.py`
  - State-debug now surfaces Round113-117 availability and metadata under `lex_concept_mapping`.
- `tests/test_v3_round113_117_runtime_mapping_validation_loop.py`
  - Focused tests cover viewer read-only behavior, blocker isolation, validation triage, regression guard cleanup, no-go package, state-debug exposure, and JSON export helpers.

Boundary:

- Production persistence remains disabled.
- `runtime_mapping_enabled` and `enforcement_enabled` default to `False`.
- No AGP bypass is added.
- No `vectors.npy`, seed subset, zip/part/upload, or `_operator_artifacts` file is part of the patch.

## Round122-126 import blocker recovery surfaces

- `spreading_activation.py`: root compatibility shim that re-exports `legacy.eve_modules.spreading_activation.SpreadingActivation`.
- `adapters/runtime_mapping_import_blocker_recovery.py`: read-only report builders for Round122 diagnosis, Round123 shim decision, Round124 collect-only verification, Round125 validation taxonomy, and Round126 go/no-go refresh.
- `tests/test_v3_round122_124_import_blocker_recovery.py`: focused tests for the diagnosis, shim identity, collect-only recovery recording, and JSON export safety.

## Round132-136 NaturalLanguage v2 SystemExit isolation surfaces

- `test_natural_lang_v2.py`
  - `run_natural_language_v2_validation(verbose=True)`: deterministic wrapper around the legacy NaturalLanguage v2 validation body.
  - `test_natural_language_v2_validation_behavior()`: pytest-visible behavior assertion preserving the original validation intent at test runtime instead of collection time.
  - `if __name__ == "__main__"`: preserves script-mode non-zero exit when the legacy validation fails.
- `adapters/runtime_mapping_import_blocker_recovery.py`
  - `build_round132_natural_lang_v2_system_exit_diagnosis(...)`: records the observed collection-time `SystemExit` blocker.
  - `build_round133_collection_side_effect_isolation_decision(...)`: records the main-guard/wrapper isolation decision and no-skip/no-xfail preservation.
  - `build_round134_collect_only_after_system_exit_isolation(...)`: records collect-only status after SystemExit isolation.
  - `build_round135_broader_validation_taxonomy_refresh(...)`: records compile/focused/collect-only/legacy behavior/broader validation taxonomy.
  - `build_round136_go_no_go_refresh_after_system_exit(...)`: keeps the production-persistence recommendation NO-GO unless collect-only and broader validation are green.
- `tests/test_v3_round132_136_system_exit_isolation.py`: focused tests for diagnosis, import safety, test-intent preservation, collect-only status recording, taxonomy, go/no-go, and JSON export safety.
