# Round641-660 AppraisalClassifier / AGP Input Stabilization Audit

## Feature track

`read_only_appraisal_classifier_agp_input_stabilization_audit`

Round641-660 starts the first feature-safe track after the Round621-640 baseline lock.  This packet is intentionally audit/design/test-surface work only.  It does not change runtime behavior, classifier rules, semantic guards, AGP verification, fallback behavior, persistence, runtime mapping defaults, enforcement, runtime loading, or vector state.

## Operator audit command

```bash
python scripts/operator_audit_round641_660_appraisal_agp_input.py
```

The command emits compact JSON to stdout.  It is read-only, does not write an operator artifact, does not open vector/vocab/subset artifact contents, does not load runtime vectors, and includes a `git status --short` artifact-safety proof.

## Boundary inventory

### AppraisalClassifier / semantic guard boundary

Relevant modules:

- `adapters/appraisal_classifier.py`
- `adapters/orchestrator_adapter.py`
- `adapters/state_debug_adapter.py`

Current path:

1. `orchestrator_adapter._analyze_appraisal_statement(text)` delegates to the shared classifier.
2. `AppraisalClassifier.analyze(text)` keeps the frozen round73 patch8-12 compatibility guard surface.
3. `AppraisalResult.as_meaning_dict()` exposes a stable appraisal meaning summary.
4. Orchestrator traces may include the appraisal payload and `small_talk` route override for target appraisal.

Compatibility finding: the current semantic guard marker sets remain frozen.  This audit adds no noun lists, no thresholds, no route changes, and no classifier behavior changes.

### AGP input / verification boundary

Relevant modules:

- `adapters/agp_adapter.py`
- `adapters/speech_hub.py`
- `adapters/compositor_adapter.py`

Current path:

1. `AGPAdapter.accept_input_meaning(meaning)` normalizes AppraisalClassifier-compatible meaning dictionaries without veto.
2. `speech_hub` and `compositor` capture explicit generation-time `meaning_categories` as the meaning bridge.
3. `AGPAdapter.verify(candidate_response, meaning, activated_categories, hormone_state)` verifies candidate categories against active categories and hormone compatibility.
4. AGP must verify EVE-internal category activation; raw lexical vectors, seed vectors, and EveSpecific vectors are not AGP anchors.

Compatibility finding: this audit does not change AGP thresholds, mode, category extraction, hormone checks, or verification behavior.

### Fallback boundary

Relevant modules:

- `adapters/agp_adapter.py`
- `adapters/speech_hub.py`
- `adapters/compositor_adapter.py`
- `main.py`

Current path:

1. `AGPAdapter.verify(...)` returns `AGPResult` fallback data when category anchoring or hormone compatibility fails.
2. `AGPAdapter.fallback_to_surface(...)` maps only AGPResult reason/fallback data to the minimal fallback surface pool.
3. `speech_hub` and `compositor` may apply fallback surfaces only under explicit veto-mode conditions.
4. The embedding wrapper fallback remains the PMI+SVD backup path and is not modified by this audit.

Compatibility finding: this audit introduces no AGP bypass and no fallback bypass.

## Korean fixture coverage

Relevant fixture:

- `tests/fixtures/korean_conversation_fixtures.py`

The audit reports 20 deterministic Korean smoke fixtures and preserves the three `minsok` category rows exactly:

- `군대 생활 어때`
- `코딩 좋아해`
- `EVE 프로젝트`

No new runtime hardcode for `민석` is added.

## Focused regression tests

Added test file:

```bash
python -m pytest -q tests/test_v3_round641_660_appraisal_agp_input_audit.py
```

The tests prove that the audit surface is read-only and that it does not change runtime behavior, classifier defaults, AGP verification, fallback behavior, Korean fixture order, `minsok` fixture rows, vector/artifact safety, persistence, runtime loading, runtime mapping defaults, or enforcement defaults.

## Recommended next implementation step

Exactly one next step is recommended:

`add_read_only_AppraisalClassifier_boundary_contract_object_without_changing_routes_or_AGP_verification`

This should add a read-only boundary contract object for current AppraisalClassifier outputs so tests can pin `route_override` and meaning fields before any behavior-changing classifier work.

## Still forbidden

- Changing AppraisalClassifier behavior or thresholds.
- Changing SemanticGuard behavior or adding guard keywords.
- Changing AGP verification behavior or thresholds.
- Bypassing AGP.
- Bypassing fallback.
- Enabling production persistence.
- Changing `runtime_mapping_enabled` default to `true`.
- Enabling enforcement by default.
- Loading runtime vectors or reading vector/vocab/subset contents.
- Creating, staging, or committing operator artifacts, seed subsets, vectors, vocab files, subset manifests, zip files, or part files.
