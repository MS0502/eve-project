# M0-D Neural, Vector, Adaptive, and Life-Loop Inventory

Baseline: `main` at `fe10cd954bdf445400ea6aa9708dd214ed761114`

Status: audit and recommendation only. No production module is imported, no runtime or loop is started, no vector/model is loaded, and no runtime state is changed.

## Regeneration

```bash
python scripts/audit/m0_d_component_inventory.py --pretty
```

The scanner is stdlib-only, scans tracked Python source with `ast.parse`, cross-references the merged M0-A/B/C inventory scripts, and emits canonical JSON to stdout. Generated JSON must remain ephemeral. Two consecutive runs are byte-identical.

## Evidence schema

Every component and life-loop entry records repository-relative path, exact line range, symbol, detection method, mechanical evidence, classification, confidence, unresolved state, and manual-only state. Life-loop entries additionally record trigger evidence, mutation references, clock/concurrency/output calls, and mapping to the v4 taxonomy: Vital, Cognitive, Goal, Activity, Learning, Memory, Social, Expression, or `no-v4-equivalent`.

## Validated snapshot

```text
tracked Python files: 584
runtime modules classified: 288
component evidence entries: 1225
life-loop entries: 75
parse errors: 2
unresolved module dispositions: 278
standalone unresolved items: 3
```

Component evidence:

```text
adaptive or learning method candidates: 106
adaptive state-transition candidates: 282
numeric or learned state candidates: 329
NumPy dependency sites: 59
numeric representation operations: 373
vector or numeric method candidates: 74
vector/vocabulary artifact I/O candidates: 2
```

Life-loop taxonomy occurrences:

```text
Vital: 5
Cognitive: 16
Goal: 3
Activity: 9
Learning: 5
Memory: 1
Social: 0
Expression: 27
no-v4-equivalent: 31
```

Counts are occurrence-level mechanical evidence. One callable may map to more than one v4 loop category.

## High-impact life-loop assessment

| Evidence | Symbol | Trigger evidence | M0-A mutation refs | v4 taxonomy |
|---|---|---|---:|---|
| `adapters/allostatic_adapter.py:61` | `AllostaticAdapter.tick` | explicit_callable_invocation | 0 | Vital |
| `adapters/continual_adapter.py:61` | `ContinualAdapter.tick` | explicit_callable_invocation | 0 | Learning |
| `adapters/dmn_adapter.py:64` | `DMNAdapter.tick` | explicit_callable_invocation | 0 | Cognitive, Expression |
| `adapters/dmn_adapter.py:68` | `DMNAdapter.force_spontaneous` | internal_proactive_or_dmn_condition | 0 | Cognitive, Expression |
| `adapters/dmn_adapter.py:74` | `DMNAdapter.try_spontaneous_meaning` | internal_proactive_or_dmn_condition | 0 | Cognitive, Expression |
| `adapters/goal_adapter.py:89` | `GoalAdapter.tick` | explicit_callable_invocation | 0 | Goal |
| `adapters/hormone_adapter.py:104` | `HormoneAdapter.tick` | explicit_callable_invocation | 1 | Vital |
| `adapters/live_loop.py:113` | `LiveLoop._run` | clock_or_delay, explicit_callable_invocation, thread_queue_or_task_lifecycle | 13 | Vital, Activity |
| `adapters/live_loop.py:237` | `LiveLoop.wait_for_tick` | clock_or_delay, thread_queue_or_task_lifecycle | 1 | Vital, Activity |
| `core/autonomous.py:68` | `AutonomousLoop.step` | explicit_callable_invocation, thread_queue_or_task_lifecycle | 21 | Activity |
| `core/autonomous.py:178` | `AutonomousLoop._internal_tick` | call_graph_unresolved | 0 | Activity |
| `core/autonomous.py:225` | `AutonomousLoop._emit_via_proactive` | internal_proactive_or_dmn_condition | 0 | Activity, Expression |
| `language/planner.py:452` | `ResponsePlanner._build_proactive` | clock_or_delay, internal_proactive_or_dmn_condition | 22 | Expression |
| `language/streaming.py:439` | `StreamingEngine.proactive_stream` | internal_proactive_or_dmn_condition | 2 | Expression |
| `main.py:369` | `proactive` | internal_proactive_or_dmn_condition | 0 | Expression |

The complete life-loop list, including legacy and versioned surfaces, is emitted by canonical JSON. `no-v4-equivalent` does not mean useless or removable; it means the callable has not yet been mapped to a constitutional loop owner.

## Neural, vector, and adaptive interpretation

- NumPy imports and calls prove numeric representation dependency, not learning quality or runtime activation.
- `train`, `observe`, `learn`, `update`, `consolidate`, and related method names are candidates only.
- Vector/vocabulary path and call evidence is cross-referenced with M0-C; it does not authorize artifact loading.
- Reachable learned/numeric modules default to `EXPERIMENTAL` or `WRAP` unless a stronger manual disposition is evidence-backed.
- Legacy hormone coupling does not automatically justify `REWRITE`. The conservative automatic result is `WRAP/unresolved`; only six manually evidenced architecture conflicts remain `REWRITE`.
- No module is automatically classified `REMOVE`.

## A/B/C retrospective verification

Disposable PR #124 re-executed the merged M0-A/B/C scripts at exact main and selected three entries from each map by deterministic SHA ordering.

```text
M0-A tests: KEEP 225 / REWRITE 0 / RETIRE 0
M0-A unresolved: 1,865
M0-A parse errors: 2
M0-B broad exception handlers: 614
M0-B silent exception handlers: 607
M0-B silent broad handlers: 532
M0-B bypass candidates: 37
M0-C persistence-intended state: 3,845
M0-C persistence I/O: 856
M0-C hormone/affect sites: 1,777
M0-C drive/need sites: 386
M0-C hormone-to-drive bridge candidates: 54
```

All nine generated excerpts were independently reopened at exact main and matched:

- M0-A: `scripts/operator_report_round1001_1020_visual_observation_schema.py:12`, `legacy/eve_modules/metacognition.py:297`, `adapters/lex_concept_mapping_adapter.py:2463`.
- M0-B: `scripts/audit/m0_b_controlflow_concurrency_inventory.py:328-329`, `legacy/eve_modules/emotion_regulation.py:280-281`, `learning/code_synthesis.py:54-61`.
- M0-C: `scripts/operator_run_local_validation_suite.py:503`, `active_inference.py:150`, `legacy/eve_main_abc.py:510`.

Evidence:

```text
workflow run: 29678859485
artifact: m0-d-abc-retrospective
artifact SHA-256: 4b7db261834f2a81342d4ae627366cc2d164d0a0f1422a1e98b3b8df4c78d39f
```

## M0-C constitutional gap

EVE v4 requires M0 to propose migration from the current hormone architecture toward core drives, appraisal, and derived emotion. The merged M0-C document inventories `1,777` hormone/affect sites, `386` drive/need sites, and `54` bridge candidates, but it does not provide migration phases, state mapping, compatibility projection, persistence/event migration, rollback, or acceptance criteria.

M0-D records `M0_C_REQUIRED_MIGRATION_PLAN_ABSENT` as a high-confidence unresolved governance blocker. M0-D does not repair M0-C or invent constitutional migration policy outside its allowed scope.

## Validation

```text
compileall: PASS
deterministic double-run: byte-identical
focused M0-D tests: 12 passed in 25.12s
collection: 2,585 tests collected in 3.01s
full suite: 2,585 passed in 50.14s
temporary six-file scope: PASS
```

Validation run `29680825833`, artifact `eve-m0-d-validation`, SHA-256 `c9fbec7a0635173b487ab15cc712d73f5509bd7fb63dfbee85adfa2db1ee21f3`.

## Scope boundary

M0-D does not modify production code, existing tests outside its new audit test, data, models, vectors, configuration, persistence, defaults, or frozen PRs. It does not close any frozen PR and does not implement a disposition recommendation.
