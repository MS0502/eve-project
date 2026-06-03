# Round77 — lexical → concept mapping planning

## Purpose

Round77 begins the next axis after `self_learning_v1` freeze. It does **not** enable runtime lexical→concept mapping. It creates a read-only planning boundary so EveSpecific lexical vectors cannot silently become concept categories or AGP anchors.

## Added

```text
+ adapters/lex_concept_mapping_adapter.py
+ tests/test_v3_round77_lexical_to_concept_mapping_planning.py
+ LEXICAL_CONCEPT_MAPPING_PLAN_R77.json
+ ROUND77_SPLIT_CHUNKS.json
+ ROUND77_SPLIT_SUITE_STATUS.json
+ ROUND77_SPLIT_SUITE_BY_CHUNK_RESULTS.json
+ ROUND_V3_R77_REPORT.md
```

## Modified

```text
- main.py
- adapters/runtime_smoke_runner.py
- adapters/state_debug_adapter.py
- CURRENT_STATUS.md
- AGENTS.md
```

## Main contract

```text
lexical vector = evidence only
lexical vector != concept
EveSpecific vector != AGP anchor
seed vector != AGP anchor
AGP anchor = explicit category + SA activation
```

## New adapter

`LexConceptMappingAdapter` is planning-only in Round77.

It exposes:

```text
mapping_contract_snapshot()
candidate_record_schema()
plan_for_tokens(tokens)
write_mapping_contract_snapshot(path)
stats()
```

It does not:

```text
observe text
commit EveSpecific vectors
call embedding lookup
create categories
mutate ConceptMemory
mutate Frame/Hypergraph
call AGP verify
change thresholds
change context-diversity gate
change generation behavior
```

## Runtime wiring

`build_full_engine()` now attaches:

```python
engine.lex_concept_mapping = LexConceptMappingAdapter(engine)
```

This is a read-only planning boundary. Runtime mapping remains disabled:

```text
runtime_mapping_enabled = False
enforcement_enabled = False
```

## Round77 artifact

`LEXICAL_CONCEPT_MAPPING_PLAN_R77.json` records:

```text
planning_version = v3_round77_lexical_to_concept_mapping_planning
source_baseline = self_learning_v1_round76
status = planning_only_not_enforced
```

It includes:

```text
- self_learning_v1 active policy
- wrapper route policy
- component roles
- lexical source boundaries
- concept anchor source requirements
- future dry-run/proposal/enforcement path
- candidate record schema plan
- token plan rows for 민석 / EVE with mapping_status=not_mapped_round77_planning_only
```

## Policy unchanged

```text
auto_observe_enabled = True
auto_promotion_enabled = False
commit_gate_enabled = True
min_observations_for_commit = 2
context_diversity_gate_enabled = True
automatic_rollback_enabled = False
AGP anchor = explicit categories + SA activation only
```

## Validation

```text
Round77 focused: 5 passed
Round75~77 focused: 13 passed
Round60~77 focused sweep: 71 passed
Round50~77 adjacent sweep: 179 passed
collect-only: 1177 tests collected
split suite: 13/13 chunks passed
passed tests by chunk sum: 1177
failures: 0
timeouts: 0
compileall: passed
```

Note: the split-suite runner process window was interrupted mid-run, so chunks 3 and 8~13 were resumed with equivalent explicit pytest chunk commands and the final JSON records that resume note.

## Next recommendation

Round78: lexical-concept candidate schema dry-run.

Scope:

```text
- produce candidate rows from lexical evidence
- still no category creation
- still no AGP anchor change
- still no runtime enforcement
```
