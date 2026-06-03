# EVE v3 Round71 — Self-learning Policy Consolidation

Round71 is a stabilization round. It does not add a new learning shortcut, does not change the active commit gate, and does not add automatic rollback or automatic promotion.

## Active self-learning pipeline

```text
chat_stream input
  -> EveSelfLearningAdapter.observe_text()
  -> EveVocabTracker.observe_word()
  -> EveVocabTracker.is_eve_specific()
  -> explicit commit_eve_specific_vectors()
  -> audit_commit_gate()
  -> EveSpecificVectorStore.add_or_update_vector()
  -> EmbeddingWrapper lookup can use EveSpecific vectors after fastText miss
```

## Component roles

| Component | Role | Mutation it owns | What it must not do |
|---|---|---|---|
| `EveVocabTracker` | Lexical observation only | observed word counts + context log | vector creation, wrapper routing, automatic promotion |
| `EveSpecificVectorStore` | Deterministic 300d vector storage | explicit add/update from known fastText context vectors | runtime auto-observe, memory/quarantine mutation, AGP anchoring |
| `EmbeddingWrapper` | Lookup routing | telemetry counters only | learning policy decisions, AGP anchor decisions |
| `EveSelfLearningAdapter` | Coordinator/gate/audit | event log, audit log, explicit vector commit when gate passes | automatic promotion, automatic rollback, memory/quarantine mutation |
| `AGPAdapter` | Anchored generation verification | none in observation mode | seed-vector or EveSpecific-vector anchor inference |

## Active commit gate

A candidate can be committed only through `commit_eve_specific_vectors()` and only when all conditions pass:

```text
1. observed_count >= 2
2. fastText-OOV / EVE-specific candidate
3. known fastText context word count >= 1
4. context_diverse = True
```

Current policy:

```text
auto_observe_enabled = True
auto_promotion_enabled = False
commit_gate_enabled = True
min_observations_for_commit = 2
min_known_context_words_for_commit = 1
context_diversity_gate_enabled = True
automatic_rollback_enabled = False
```

## Round55-57 connection check

```text
Round55: EveSpecificVectorStore implemented only as deterministic vector store.
Round56: EmbeddingWrapper receives eve_specific=engine.eve_specific_vector_store.
Round57: telemetry and drift baseline expose eve_specific hit/rate observations.
```

Runtime wrapper order:

```text
get_vector(): fastText primary -> EveSpecificVectorStore -> PMI+SVD fallback
get_embedding(): PMI+SVD local learned embedding if present -> fastText primary -> EveSpecificVectorStore -> PMI+SVD fallback
```

The `get_embedding()` exception preserves old local PMI+SVD composition spaces and is not an AGP anchoring path.

## AGP boundary

AGP must pass only when candidate categories are explicit and active in EVE's internal SA/activated category input. EveSpecific vectors do not count as AGP anchors.

Round71 locks this with a regression test:

```text
- store EveSpecific vector for "민석"
- call AGP verify with meaning_categories=["민석"] and activated_categories=[]
- expected: fail unknown_category
- wrapper telemetry unchanged during AGP verify
```

## Full-suite validation status

Round71 started a file-by-file full-suite runner because a single `pytest tests` run exceeds the sandbox execution window. The runner writes:

```text
ROUND71_FULL_SUITE_STATUS.json
ROUND71_FULL_SUITE_BY_FILE_RESULTS.json
ROUND71_FULL_SUITE_BY_FILE_SUMMARY.json  # only when complete
ROUND71_FULL_SUITE_RUNNER.log
```

A package must not claim full 1150 pass unless `ROUND71_FULL_SUITE_BY_FILE_SUMMARY.json` exists and reports zero failures/timeouts for all 119 test files.
