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
