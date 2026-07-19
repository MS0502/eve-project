# M0-D Neural, Vector, Adaptive, and Life-Loop Inventory

Baseline: `main` at `fe10cd954bdf445400ea6aa9708dd214ed761114`

Status: audit and recommendation only. No production module is imported, no runtime or loop is started, no vector/model is loaded, and no runtime state is changed.

## Regeneration

```bash
python scripts/audit/m0_d_component_inventory.py --pretty
```

The scanner is stdlib-only, scans tracked Python source with `ast.parse`, cross-references the merged M0-A/B/C inventory scripts, and emits canonical JSON to stdout. Generated JSON must remain ephemeral.

## Evidence schema

Every component and life-loop entry records repository-relative path, exact line range, symbol, detection method, mechanical evidence, classification, confidence, unresolved state, and manual-only state. Life-loop entries additionally record trigger evidence, mutation references, clock/concurrency/output calls, and mapping to the v4 taxonomy: Vital, Cognitive, Goal, Activity, Learning, Memory, Social, Expression, or `no-v4-equivalent`.

## Validation snapshot

Pending first branch validation. Counts in this section will be replaced with exact script output before review.

## A/B/C retrospective verification

A disposable exact-main run re-executed the merged M0-A/B/C scripts and selected three entries from each map by a deterministic SHA ordering, then stored their source excerpts. Current-main headline results:

- M0-A test classification: `KEEP 225 / REWRITE 0 / RETIRE 0`; unresolved entries `1,865`; parse errors `2`.
- M0-B: broad exception handlers `614`; silent handlers `607`; silent broad handlers `532`; bypass/override candidates `37`.
- M0-C: persistence-intended state occurrences `3,845`; persistence I/O `856`; hormone/affect sites `1,777`; drive/need sites `386`; bridge candidates `54`.

The count increases relative to the original M0-A/B documents are caused by subsequently merged audit files entering the tracked Python scan. The underlying historical snapshots remain valid for their exact merge heads; the current-main rerun is the input used by M0-D.

### Deterministic source samples

All nine generated excerpts were independently reopened at exact `main` and matched byte-for-line:

- M0-A: `scripts/operator_report_round1001_1020_visual_observation_schema.py:12`, `legacy/eve_modules/metacognition.py:297`, `adapters/lex_concept_mapping_adapter.py:2463`.
- M0-B: `scripts/audit/m0_b_controlflow_concurrency_inventory.py:328-329`, `legacy/eve_modules/emotion_regulation.py:280-281`, `learning/code_synthesis.py:54-61`.
- M0-C: `scripts/operator_run_local_validation_suite.py:503`, `active_inference.py:150`, `legacy/eve_main_abc.py:510`.

These samples confirm that the detectors point to actual constructor/mutation, exception, vector-state, and hormone-state syntax. They do not by themselves validate every occurrence; the full rerunnable JSON remains the primary evidence.

## M0-C constitutional gap

EVE v4 requires M0 to propose migration from the current hormone architecture toward core drives, appraisal, and derived emotion. The merged M0-C document inventories `1,777` hormone/affect sites, `386` drive/need sites, and `54` bridge candidates, but it does not provide a concrete migration plan, compatibility projection, persistence/event mapping, rollback design, or acceptance criteria. M0-D therefore records `M0_C_REQUIRED_MIGRATION_PLAN_ABSENT` as a high-confidence unresolved governance blocker. This document does not repair M0-C or invent a migration plan outside the allowed M0-D scope.

## Scope boundary

M0-D does not modify production code, existing tests, data, models, vectors, configuration, persistence, defaults, or frozen PRs. It does not close any PR and does not implement a disposition recommendation.
