# EVE v3 Agent Instructions

## Project identity

EVE v3 is a deterministic Korean digital personality project.
EVE is not a generic chatbot and must not be optimized as one.
The current v3 line starts from the v41/round73 codebase as retained assets, not as a rewrite.

## Absolute principles

1. Reproducible determinism: no random sampling. Same timestep and same state must produce the same output.
2. EVE Core is built by Kim Minseok and project agents; core algorithms remain owned and inspectable.
3. Case hardcoding is permanently forbidden. Do not add new keyword lists to patch around individual examples.
4. Transformers, LLMs, RWKV, Mamba, and SSM bodies are forbidden inside EVE Core.
5. N-gram language models are forbidden as generation engines.
6. Hormones are somatic markers and must remain part of decision context.
7. AGP, the Anchored Generation Principle, is the generation gate: output must be anchored in active categories and hormone-compatible.
8. Lexical representation and conceptual representation are distinct layers.

## External seed policy

External generation bodies are forbidden. External lexical or conceptual seed data may be evaluated only as initial maps, not as a speaking subject.
Allowed seed candidates require provenance, license, version, deterministic loading, and later EVE-specific drift tracking.
AGP anchors must be based on EVE internal category activation, not on external seed vectors alone.
Round21 adds the seed manifest gate: every future external seed must be declared in `seeds/MANIFEST.yaml` and pass manifest validation before import.
Round22 adds read-only seed registration dry-run: dry-run may simulate a future manifest entry but must not load seed files, modify `seeds/MANIFEST.yaml`, or rewrite embeddings.

Round23 adds the acquisition workflow and explicit seed state ladder. If the real seed file is absent, the only valid outcome is deferred acquisition: do not create fake checksums, do not modify `seeds/MANIFEST.yaml`, and do not rewrite embeddings.

Round24 registers concrete `cc.ko.300.bin` provenance in `seeds/MANIFEST.yaml`. This advances External Seed Policy state to `registered` only. Do not treat registration as runtime load or usage. Do not import fastText/gensim, do not rewrite `self_embedding_adapter.py`, and do not use the seed until a later explicit round.

Round25 adds only the fastText runtime loader scaffold. The optional dependency is declared in `requirements-optional.txt`, but the loader must not import fastText at module import time, must not load `cc.ko.300.bin`, must not mark the seed as loaded, and must not rewrite `self_embedding_adapter.py`. Actual seed loading is reserved for a later explicit round after checksum verification against the manifest.

Round26 implements explicit fastText checksum verification and load workflow. Do not call the loader during normal startup. Loading requires an explicit file path, checksum verification, and optional fastText runtime availability. The default seed state remains registered. Do not rewrite `self_embedding_adapter.py` or mark the seed as used in round26.

## Current transition rule

The round73 patch8-12 semantic guards are frozen temporary safety rails.
Do not add new guard keywords for cases such as classes, companies, news, exams, or other nouns.
Future work must consolidate those guards into an AppraisalClassifier under AGP input stabilization.

## Engineering rules

- Keep changes targeted.
- Do not weaken tests.
- Do not modify semantic memory or quarantine files unless explicitly requested.
- Do not add external API calls.
- Do not add non-deterministic behavior.
- Run the full test suite and compile check before reporting success.


Round27 adds only a seed verification runner. The runner may read a path from `EVE_FASTTEXT_SEED_PATH` or an explicit argument, compute SHA256, and compare it to `seeds/MANIFEST.yaml`. It must not load fastText, must not mark the seed as loaded by default, must not rewrite `self_embedding_adapter.py`, and must return structured data only. Missing path or mismatch must fail closed.

Round28 records external verification of `cc.ko.300.bin` and adds only a subset extraction plan. The verified checksum matches the manifest, but the runtime state remains registered. Do not extract a subset, load fastText, rewrite `self_embedding_adapter.py`, or mark the seed as used in round28. Round29 may perform operator-side subset extraction only after preserving deterministic selection and provenance rules.

Round29 registers the operator-extracted `cc.ko.300.subset.mini.1k` artifact under `seeds/subsets/`. A subset entry must reference a registered parent seed and must include deterministic selection metadata plus checksums for `vocab.txt`, `vectors.npy`, and `subset_manifest.json`. The mini 1k subset is fixture-level: it validates extraction/provenance/checksum workflow only. Do not use it as the production lexical map, do not rewrite `self_embedding_adapter.py`, do not load it at runtime by default, and do not mark it as used in round29.


## EVE v3 round30 subset audit rule

- The mini 1k subset is a fixture-level artifact, not production lexical memory.
- `cc.ko.300.subset.mini.1k` may be used to validate manifest/checksum/shape/load boundaries.
- Do not rewrite `self_embedding_adapter.py` from the mini 1k subset unless the task explicitly says scaffold-only.
- Prefer extracting a small 5k subset before meaningful self_embedding usage.
- Readiness assessments are data only and must not auto-apply recommendations.
- Subset audit helpers must be read-only: no AGP runtime changes, no seed state transitions, no fallback/semantic guard changes.

## v3 round31: small 5k subset status

- `cc.ko.300.subset.small.5k` is registered as an extracted `production_lexical_seed` subset.
- The subset may be validated with `external_seed_manifest` helpers, but it must not be used by `self_embedding_adapter` until a later explicit rewrite round.
- Mini 1k remains a fixture-level extraction/checksum artifact.
- Small 5k is the preferred candidate for future self_embedding 300d scaffold work.
- Do not remove mini 1k. Do not auto-load small 5k. Do not mark any subset as used without an explicit future round and tests.

## v3 round32: self_embedding rewrite scaffold rule

- `FasttextEmbeddingAdapter` is a separate scaffold for the small 5k production lexical seed.
- Do not replace `engine.self_embedding` in round32.
- Do not modify `adapters/self_embedding_adapter.py` in round32.
- Do not load `seeds/subsets/cc.ko.300.subset.small.5k/vectors.npy` at runtime in round32.
- Interface and affected-module audits are read-only data. They must not mark a subset as loaded or used.
- Future migration must proceed module-by-module with explicit tests, not by a broad global swap.

## v3 round33: FasttextEmbeddingAdapter actual load rule

- `FasttextEmbeddingAdapter` may explicitly load the committed small 5k subset as an isolated instance.
- Loading must verify subset files/checksums and must not import the external fastText runtime package.
- Do not replace `engine.self_embedding` in round33.
- Do not modify `adapters/self_embedding_adapter.py` in round33.
- Do not migrate affected modules in round33.
- Unknown word handling remains explicit: word lookup returns `None`, word similarity returns `0.0` when a word is missing, and text embedding returns a deterministic zero vector when no known tokens exist.
- Future migration must still proceed module by module with explicit tests and no broad global swap.

## v3 round34: state debug fastText exposure

- `engine.fasttext_embedding` may exist as an unloaded `FasttextEmbeddingAdapter` for debug visibility only.
- `state_debug_adapter` may report fastText subset status, but it must not call `load()`.
- `self_embedding_adapter.py` remains the active generation embedding path.
- `fasttext_embedding.in_use_by_generation` must remain `False` until a later explicit migration round.
- Do not migrate attention/compositor/concept memory/situation/streaming/main in round34.

## v3 round35: attention analyzer fastText observation

- `attention_analyzer` may run `FasttextEmbeddingAdapter` only in parallel observation mode when `engine.fasttext_embedding` has been explicitly loaded.
- Attention decisions and generation remain driven by the existing `engine.self_embedding` PMI+SVD path.
- `attention_analyzer` must not call `fasttext_embedding.load()` automatically.
- FastText attention traces are data-only diagnostics and must not alter returned `AttentionResult` values.
- `state_debug_adapter` may report attention parallel-observation status, but it must not load fastText or change migration state.
- Do not migrate compositor/concept memory/situation/streaming/main in round35.

## v3 round36: compositor fastText observation

- `compositor_adapter` may run `FasttextEmbeddingAdapter` only in parallel observation mode when `engine.fasttext_embedding` has been explicitly loaded.
- Compositor output and composition decisions remain driven by the existing `engine.self_embedding` PMI+SVD path.
- AGP observation/veto and fastText observation must remain independent systems with separate traces.
- `compositor_adapter` must not call `fasttext_embedding.load()` automatically.
- FastText compositor traces are data-only diagnostics and must not alter output, fallback, AGP mode, or route selection.
- `state_debug_adapter` may report compositor parallel-observation status, but it must not load fastText or change migration state.
- Do not migrate concept memory/situation/streaming/main in round36.

## v3 round37: concept memory fastText observation

- `concept_memory_adapter` may run `FasttextEmbeddingAdapter` only in parallel observation mode when `engine.fasttext_embedding` has been explicitly loaded.
- Concept-memory query and write decisions remain driven by the existing `engine.self_embedding` PMI+SVD path.
- Concept memory must not call `fasttext_embedding.load()` automatically.
- FastText concept-memory traces are data-only diagnostics and must not alter stored concepts, query results, quarantine policy, or route selection.
- `state_debug_adapter` may report concept-memory parallel-observation status and operation counts, but it must not load fastText or change migration state.
- Do not migrate situation_responder/streaming/main in round37.

## v3 round38 migration checkpoint

- `situation_responder` is migrated to fastText parallel observation only.
- User-visible response selection and output must remain unchanged.
- `engine.self_embedding` remains the active generation embedding source until the explicit round40 swap.
- Do not auto-load fastText from situation/streaming/main paths.


## v3 round39 migration checkpoint

- `language/streaming` is migrated to fastText parallel observation only.
- Streaming chunk text, chunk order, and timing shape must remain unchanged.
- `engine.self_embedding` remains the active generation embedding source until the explicit final swap round.
- `language/streaming` must not call `fasttext_embedding.load()` automatically.
- FastText streaming traces are data-only diagnostics and must not alter generation, partial output, sleep/timing, history, memory, or route selection.
- `state_debug_adapter` may report streaming parallel-observation status and operation counts, but it must not load fastText or change migration state.
- Next step is a pre-swap audit round before the global `engine.self_embedding` swap.

## v3 round40 pre-swap audit checkpoint

- Round40 is an audit-only safety gate between 6/7 fastText observation migration and the final global swap.
- `engine.self_embedding` must remain `SelfEmbeddingAdapter(PMI+SVD, 50d)` through round40.
- `main.py` must not be rewired in round40.
- `FasttextEmbeddingAdapter` must remain unloaded by default and must never be auto-loaded by migrated modules.
- The migrated trace modules are: `attention_analyzer`, `compositor_adapter`, `concept_memory_adapter`, `situation_responder`, and `language/streaming`.
- `state_debug_adapter` is the sixth migration surface and remains read-only debug exposure.
- FastText observation must be fail-open: load failure, lookup failure, observation exceptions, corrupted adapter behavior, or runtime unload must not break the active self_embedding path.
- AGP traces, fastText traces, and self_embedding decisions must remain separate systems.
- `assess_main_py_swap_readiness(engine)` is read-only data. It may report readiness, swap strategy, rollback strategy, and concerns; it must not apply the recommendation.
- Next step is round41 final `main.py` / `engine.self_embedding` swap.

## v3 round41 final swap checkpoint

- `engine.self_embedding` is now an `EmbeddingWrapper`.
- Wrapper primary is the loaded `FasttextEmbeddingAdapter` using `cc.ko.300.subset.small.5k`.
- Wrapper fallback is the preserved `engine.self_embedding_backup`, the original `SelfEmbeddingAdapter(PMI+SVD, 50d)`.
- `self_embedding_adapter.py` must not be deleted; it is the rollback and local-learning fallback.
- Six migrated surfaces must report `in_use_by_generation = "wrapper"` after round41.
- FastText primary failures must fail open to PMI+SVD fallback.
- Rollback strategy must stay explicit: restore `engine.self_embedding = engine.self_embedding_backup` and unload fastText.
- Do not start drift/promotion logic inside round41. That begins in round42+.

## v3 round42 telemetry + drift baseline checkpoint

- `EmbeddingWrapper.telemetry()` is read-only and may report call counts, primary hit rate, fallback rate, error rate, and bounded OOV samples.
- OOV telemetry must be bounded (`cap = 1000`) and must not alter response decisions.
- Seed drift measurement begins at round42 through `measure_seed_drift_baseline(engine)`.
- AGP/wrapper correlation is data-only through `correlate_agp_and_wrapper_telemetry(engine)`.
- Drift measurement must not auto-adjust AGP thresholds, wrapper thresholds, routing, or fallback policy.
- PMI+SVD fallback remains mandatory and must not be removed.
- Do not extract new subsets or promote larger fastText subsets during telemetry baseline rounds.

## v3 round43 runtime smoke checkpoint

- Round43 establishes the first post-swap runtime conversation baseline.
- `adapters/runtime_smoke_runner.py` may run deterministic Korean fixtures through the real streaming path and collect wrapper telemetry, OOV samples, and AGP trace summaries.
- Smoke sampling is data-only. It must not tune wrapper thresholds, AGP thresholds, routing, fallback policy, or subset size.
- `tests/fixtures/korean_conversation_fixtures.py` is the stable fixture source for the round43 baseline.
- `PMI+SVD` fallback remains mandatory. Do not remove it based on one telemetry run.
- Runtime smoke must not invoke `rollback()`.
- Round43 findings are a post-swap baseline for later analysis rounds, not an automatic recommendation.

## v3 round44 first-pass smoke data analysis checkpoint

- Round44 is analysis-only over the round43 post-swap runtime smoke baseline.
- `adapters/smoke_data_analyzer.py` may classify wrapper telemetry, bounded OOV samples, fixture categories, and AGP unknown-category patterns.
- Analysis output must be a data dict, not an automatic recommendation executor.
- Do not tune wrapper thresholds, AGP thresholds, routing, fallback policy, or subset size from round44 analysis.
- Do not promote or extract a medium 30k subset in round44.
- PMI+SVD fallback remains mandatory.
- `primary_hit_rate < 0.50` is a manual review flag only.
- `AGP unknown_category` patterns must be root-caused before any threshold or category-anchor change.


## v3 round45 coverage + AGP root cause checkpoint

- Round45 is analysis-only over round44 smoke-data findings.
- Lexical coverage below 0.50 is not a blocker while PMI+SVD fallback remains active.
- AGP `unknown_category` is the blocker and must be root-caused before medium subset work.
- `analyze_category_coverage`, `analyze_agp_unknown_category_root_cause`, and `confirm_problem_separation` must remain read-only data functions.
- Do not auto-tune AGP thresholds, wrapper thresholds, routing, fallback policy, or subset size from this analysis.
- Do not extract or promote medium 30k until AGP blocker is understood/resolved.
- Next round should make one manual, narrow AGP probe rather than broad refactoring.

## v3 round46 AGP extraction trace checkpoint

- Round46 is a read-only trace probe for AGP `unknown_category` root cause.
- `verify_with_trace`, `run_conversation_smoke_with_agp_trace`, and `analyze_extraction_traces` must remain data-only surfaces.
- Round46 evidence shows H1 as the likely root cause: candidate category extraction returns no categories for all observed AGP traces.
- SA activation is not empty, so do not treat the current blocker as an SA-empty problem without new evidence.
- Threshold relaxation is not justified by round46 because weak-overlap evidence is zero.
- Do not extract/promote medium 30k until the AGP blocker is resolved.
- Next fix should target a narrow AGP meaning bridge before any AGP threshold change.

## v3 round47 AGP meaning bridge checkpoint

- Generation modules are the source of truth for candidate meaning categories.
- `speech_hub` and `compositor` may pass explicit `meaning_categories` captured at generation time to AGP.
- AGP must verify explicit generation-time categories against active categories.
- AGP must not invent categories from raw response text.
- If no explicit bridge data exists, legacy extraction may fail closed; do not infer broad categories to make output pass.
- Explicit veto mode must keep candidates when AGP returns anchored/pass.
- Do not relax AGP thresholds as a substitute for missing bridge data.
- Do not change SA activation mechanics without a separate evidence round.
- Do not extract or promote medium 30k until post-bridge AGP stability is confirmed.

## v3 round48 post-bridge smoke checkpoint

- Round48 establishes the post-bridge AGP baseline.
- Post-bridge smoke comparison must remain read-only and must not tune AGP thresholds, wrapper thresholds, routing, fallback policy, or subset size.
- `compare_pre_post_bridge` and `identify_residual_issues` are data-only analyzers.
- Round48 data shows the meaning bridge resolved the observed AGP `unknown_category` blocker for the smoke fixtures: candidates, overlaps, and anchored results are present for all observed traces.
- Do not infer that lexical coverage is solved. Wrapper primary/fallback telemetry remains a separate problem from AGP anchoring.
- Do not extract or promote medium 30k automatically. Lexical coverage work requires a separate planning round.
- AGP must still not invent categories; explicit generation-time `meaning_categories` remain the valid bridge source.

## v3 round49 lexical coverage strategy

Round49 confirms lexical coverage strategy C: hybrid.

```text
strategy = C_hybrid
sequence = A first, B continuous
A = medium 30k extraction for quick general Korean coverage
B = continuous self-learning for EVE/Minsok-specific drift
```

Rules:
- Round49 is planning-only. Do not extract or promote medium 30k in round49.
- Round50 may perform medium 30k extraction/registration using the deterministic round29/31 subset pattern.
- Round51+ may begin continuous self-learning.
- Do not remove PMI+SVD fallback. It remains the EVE-specific learning substrate.
- Do not treat external fastText seed expansion as a replacement for EVE-specific drift.
- Lexical coverage strategy C is now the project policy unless a later explicit design update changes it.

## v3 round50 medium 30k subset policy

- `cc.ko.300.subset.medium.30k` is registered as `production_lexical_seed_expanded`.
- It supersedes small 5k for lexical coverage planning, but round50 does not swap runtime primary.
- Mini 1k, small 5k, and medium 30k must coexist. Do not delete earlier subsets.
- Round50 confirms round49 projection: general Korean OOV sample `어때/그래/뭐야/좋아해/군대/코딩` is resolved 6/6 by medium 30k.
- EVE-specific OOV terms such as `EVE` and `민석` remain self-learning targets.
- Do not remove PMI+SVD fallback. It remains the EVE-specific learning substrate.
- Round51 may perform a narrow wrapper primary swap to medium 30k.
- Round52 must measure primary hit rate by smoke rerun before any further lexical decision.

## v3 round51 wrapper primary medium swap policy

- `FasttextEmbeddingAdapter` default runtime subset is now `cc.ko.300.subset.medium.30k`.
- Wrapper primary should load the medium 30k subset by default.
- Mini 1k and small 5k remain extracted artifacts and must not be deleted.
- Round51 is a narrow swap only: do not rerun smoke as a decision input in the same round.
- Round52 must measure post-swap primary hit rate before any further lexical decision.
- Round51 does not implement continuous self-learning; that remains round53+ work.
- EVE-specific OOV terms such as `EVE` and `민석` are not solved by medium 30k and remain self-learning targets.
- PMI+SVD fallback remains mandatory and must not be removed.

## v3 round52 post-medium smoke measurement policy

- Round52 measured the medium 30k runtime primary after round51.
- `primary_hit_rate` improved to 0.7708 and `fallback_rate` dropped to 0.2292 on the smoke fixture sample.
- Round49 projection was conservative; medium 30k exceeded the projected 0.50–0.65 range.
- AGP remains stable with pass rate 1.0; do not mix lexical coverage work with AGP threshold changes.
- Round52 is measurement-only. It does not implement self-learning, change wrapper logic, extract another subset, or remove the PMI+SVD fallback.
- Round53+ should begin the B continuous self-learning design for EVE/Minsok-specific terms while keeping medium 30k as the external lexical primary.

---

## v3 round53 self-learning scaffold rule

B continuous is now started as a scaffold. The project strategy is:

```text
A first: medium 30k for general Korean lexical coverage
B continuous: deterministic EVE-specific self-learning for EVE/민석/project terms
```

Round53 establishes these modules only:

```text
adapters/eve_vocab_tracker.py
adapters/eve_vector_store.py
```

Rules:

- Do not implement runtime observation before the designated tracker round.
- Do not implement vector creation before the designated vector-store round.
- Do not integrate `EveSpecificVectorStore` into `EmbeddingWrapper` before the wrapper-integration round.
- Do not use randomness for EVE-specific vocabulary learning.
- Do not auto-promote vocabulary into memory or semantic memory.
- Keep medium 30k as fastText primary and PMI+SVD as fallback until explicit integration.
- Preserve Appendix D: external seed is an initial map; EVE-specific drift must be deterministic and auditable.

## v3 round54 observation rule

EveVocabTracker manual observation is implemented.

Rules:
- Observation must remain deterministic.
- Do not use random or sampling.
- Do not auto-observe from runtime before the wrapper integration round.
- Do not generate EVE-specific vectors in round54.
- Do not integrate EveSpecificVectorStore into the wrapper before the dedicated integration round.
- `EVE` and `민석` are examples of EVE-specific observed/OOV terms, but they must not be hardcoded into runtime behavior.

---

## v3 round55 policy — EVE-specific vector store

Round55 implements `EveSpecificVectorStore` only.

Rules:
- Use deterministic context averaging from loaded fastText medium 30k vectors.
- Store EVE-specific vectors as 300d float32.
- Do not use random initialization.
- Do not auto-observe runtime text in round55.
- Do not integrate the store into `EmbeddingWrapper` until round56.
- Do not remove PMI+SVD fallback.
- Do not modify AGP, memory, or quarantine files for this round.

Round56 may route lookup priority as:

```text
fastText medium 30k → EveSpecificVectorStore → PMI+SVD fallback
```

---

## v3 round56 policy — wrapper fallback priority integration

Round56 introduces the Eve-specific vector store into the live lookup path by
inserting it between the fastText primary and the PMI+SVD fallback. The
wrapper must consult the eve-specific store after querying fastText and
before deferring to the PMI+SVD fallback. This preserves compatibility with
previous rounds while enabling deterministic self-learning vectors.

Rules:
- The `EmbeddingWrapper` must accept an optional `eve_specific` parameter and
  query it for vectors prior to calling the PMI+SVD fallback.
- Telemetry must expose `eve_specific_hits` and `eve_specific_rate` alongside
  existing primary and fallback metrics.
- `stats()` must include `eve_specific_class`, `eve_specific_dimension`, and
  `eve_specific_count` fields.
- `state_debug_adapter` must surface eve-specific dimensions, counts, class,
  and telemetry rates.
- The wrapper’s `method` string remains
  `fasttext_primary_pmi_svd_fallback` for backward compatibility; do not
  invent a new method name in state debug.
- Automatic observation and promotion remain disabled; vectors must be added
  manually via `EveVocabTracker` and `EveSpecificVectorStore` APIs.
- Do not modify AGP, memory, or quarantine files in this round.

Next:
- Round57 will re-run smoke sampling and establish a post-eve-specific drift
  baseline.

---

## v3 round58 policy — controlled continuous EVE-specific observation

Round58 enables runtime observation of lexical tokens for EVE-specific
self-learning, but it does **not** enable automatic vector promotion.

Rules:
- Runtime observation may call `EveSelfLearningAdapter.observe_text(...)`.
- Vector creation/update must go through explicit
  `commit_eve_specific_vectors(...)`.
- `auto_promotion_enabled` must remain `False` unless a later round explicitly
  adds a deterministic gate and tests.
- Do not promote observed words into semantic memory or quarantine bypass paths.
- Do not change AGP thresholds based on drift metrics.
- Do not mutate the fastText seed vectors.
- Wrapper order remains `fastText → EveSpecificVectorStore → PMI+SVD`.
- State/debug and drift reports must expose observation and commit counters.

Next intended guard:
- Round59 should require observation-count and known-context-count checks before
  explicit vector commit is considered safe.

---

## v3 round59 policy — explicit vector commit gate

Round59 adds a deterministic gate before EVE-specific vector mutation.

Rules:
- Runtime observation may continue, but it is still observation only.
- `commit_eve_specific_vectors(...)` must call the commit-gate audit before
  mutating `EveSpecificVectorStore`.
- A candidate may be committed only when it is already observed, classified as
  EVE-specific by the tracker, and has enough known fastText context words to
  derive a deterministic vector.
- Failed candidates must be rejected with explicit reasons; silent mutation is
  forbidden.
- `audit_commit_gate(...)` is read-only and must not mutate the vector store.
- `auto_promotion_enabled` remains `False`.
- Do not mutate fastText seed vectors.
- Do not promote observed words into semantic memory or bypass quarantine.
- Do not change AGP thresholds based on commit-gate or drift metrics.

Next:
- Round60 may add persistence/export for audit records, but should not enable
  automatic promotion.

## v3 round60 policy — commit audit persistence/export

Round60 may persist and export commit-gate audit records, but export is not a
learning or promotion path.

Rules:
- Commit-gate audit records may be stored in memory for observability.
- `export_commit_audit_snapshot()` must be read-only.
- `write_commit_audit_export(path)` may write JSON only when explicitly called.
- Audit export must not mutate EveSpecificVectorStore, fastText seed vectors,
  semantic memory, quarantine, AGP thresholds, or drift thresholds.
- Rejection reasons must remain explicit and deterministic.
- `auto_promotion_enabled` remains `False`.

Next:
- Round61 may add dashboard summarization or stricter commit thresholds, but not
  automatic promotion.

## v3 round61 policy — commit audit dashboard snapshot

Round61 may summarize commit-gate audit records for observability only.

Rules:
- `commit_audit_dashboard_snapshot()` must be read-only.
- Dashboard aggregation must not mutate EveSpecificVectorStore, fastText seed
  vectors, semantic memory, quarantine, AGP thresholds, or drift thresholds.
- Dashboard counters may include rejection reasons, event-type counts, recent
  trend entries, and most-attempted words.
- The dashboard is an operator/debug surface, not a commit decision override.
- `auto_promotion_enabled` remains `False`.
- Commit mutation must still go through the Round59 gate.

Next:
- Round62 may tighten commit thresholds or extend dashboard export, but must not
  enable automatic promotion.

## v3 round62 policy — multi-observation commit threshold dry-run

Round62 may inspect stricter future commit thresholds without changing the active
commit policy.

Rules:
- `dry_run_commit_thresholds()` must be read-only and must not append audit
  records.
- The active `min_observations_for_commit` remains `1` unless a later explicit
  round changes it.
- Dry-run output may show what would pass or fail under observation thresholds
  such as `2` or `3`.
- Threshold readiness in the dashboard is an observability surface only.
- No vector-store mutation, fastText seed mutation, memory/quarantine mutation,
  AGP bypass, drift-based runtime adjustment, or automatic promotion is allowed.
- Commit mutation must still go through the Round59 gate.

Next:
- Round63 may convert the dry-run evidence into a stricter threshold proposal,
  but should still avoid automatic promotion.


## v3 round63 policy — threshold proposal report

Round63 may summarize Round62 dry-run evidence into an operator-facing proposal.
It must not change the active commit policy.

Rules:
- `threshold_proposal_report()` is read-only.
- The active `min_observations_for_commit` remains `1` unless a later explicit
  round changes it.
- Proposal output may recommend `defer_threshold_change` or
  `operator_may_consider_threshold_change`, but must not apply the change.
- No vector-store mutation, audit record append, fastText seed mutation,
  memory/quarantine mutation, AGP bypass, drift-based runtime adjustment, or
  automatic promotion is allowed.
- Commit mutation must still go through the Round59 gate.

Next:
- Round64 may explicitly apply a stricter threshold, but only as a separate
  policy-changing patch with tests.

## v3 round64 policy — commit threshold enforcement

Round64 explicitly applies the stricter commit threshold proposed in Round63.
This is a general safety rule, not a keyword/case hardcode.

Rules:
- The active `min_observations_for_commit` is now `2`.
- A single observation is not enough to create/update an EVE-specific vector.
- Explicit vector mutation must still pass the Round59 commit gate.
- Known fastText context evidence is still required.
- `auto_promotion_enabled` remains `False`.
- No vector-store mutation may happen during dry-run, dashboard, proposal,
  state-debug, or drift-measurement calls.
- No fastText seed mutation, semantic-memory/quarantine mutation, AGP bypass,
  drift-based runtime adjustment, or automatic threshold adjustment is allowed.

Next:
- Round65 may add a rollback/config snapshot for the threshold, but must not
  enable automatic promotion.

## v3 round65 policy — threshold config / manual rollback snapshot

Round65 may expose the active commit threshold configuration and a manual
rollback reference for operator review.

Rules:
- `threshold_policy_snapshot()` must be read-only.
- The active `min_observations_for_commit` remains `2`.
- The snapshot may show policy history and a previous manual rollback target,
  but must not apply rollback automatically.
- `write_threshold_policy_snapshot(path)` may export JSON only as an operator
  artifact.
- No vector-store mutation, audit record append, fastText seed mutation,
  memory/quarantine mutation, AGP bypass, drift-based runtime adjustment,
  automatic threshold adjustment, automatic rollback, or automatic promotion is
  allowed.
- Commit mutation must still go through the Round59 gate and Round64 threshold.

Next:
- Round66 may add observation evidence quality/context-diversity summaries, but
  should keep them read-only before changing any policy.

## v3 round66 policy — observation evidence quality summary

Round66 may expose context-diversity and repeated-evidence summaries for
observed EVE-specific candidates. This is observability only, not a new commit
rule.

Rules:
- `observation_evidence_quality_summary()` must be read-only.
- The active `min_observations_for_commit` remains `2`.
- The summary may report observation count, unique context count, unique source
  count, peer-token diversity, duplicate-context count, and bounded context
  samples.
- The summary must not change commit thresholds, append audit records, create or
  update vectors, mutate the fastText seed, promote memory/quarantine entries,
  bypass AGP, adjust runtime behavior from drift, or enable automatic promotion.
- Commit mutation must still go through the Round59 gate and Round64 threshold.
- Context-diversity must remain a report signal only until a later explicit
  policy-changing round.

Next:
- Round67 may add context-diversity gate dry-run/reporting, but should not
  enforce it yet.

## v3 round67 policy — context-diversity gate dry-run

Round67 may simulate a future context-diversity commit requirement, but it must
not enforce that requirement.

Rules:
- `dry_run_context_diversity_gate()` must be read-only.
- The active `min_observations_for_commit` remains `2`.
- The active commit gate still does not require context diversity.
- The dry-run may report `insufficient_context_diversity` and
  `newly_blocked_by_context_diversity` for operator review only.
- No threshold change, context-diversity policy change, audit record append,
  vector-store mutation, fastText seed mutation, memory/quarantine mutation,
  AGP bypass, drift-based runtime adjustment, automatic rollback, or automatic
  promotion is allowed.
- Commit mutation must still go through the Round59 gate and Round64 threshold.

Next:
- Round68 may turn the dry-run output into a proposal report, but should not
  enforce context diversity unless a later explicit policy-changing patch does
  so with tests.

## v3 round68 policy — context-diversity proposal report

Round68 may convert the Round67 context-diversity dry-run into an operator-facing
proposal report. It must not enforce context diversity.

Rules:
- `context_diversity_proposal_report()` must be read-only.
- The active `min_observations_for_commit` remains `2`.
- The active commit gate still does not require context diversity.
- The proposal may report `newly_blocked_by_context_diversity`,
  `eligible_under_proposal`, and an operator recommendation.
- The recommendation is advisory only; it must not change thresholds, gate
  policy, vectors, audit records, memory/quarantine state, fastText seed data,
  AGP behavior, or runtime behavior.
- Automatic rollback and automatic promotion remain disabled.
- Commit mutation must still go through the Round59 gate and Round64 threshold.

Next:
- Round69 may either enforce context diversity with explicit tests or keep
  accumulating read-only evidence. No automatic promotion.

## v3 round69 policy — context-diversity gate enforcement

Round69 makes context-diversity an active explicit vector-commit requirement.
This is a safety hardening patch, not a new learning shortcut.

Rules:
- `context_diversity_gate_enabled` is `True` by default.
- A candidate must satisfy the Round64 observation threshold, known fastText
  context requirement, EVE-specific/OOV candidate check, and `context_diverse=True`.
- Repeated same-context observations must be rejected with
  `insufficient_context_diversity` before vector-store mutation.
- Diverse-context candidates may commit only through the existing explicit
  `commit_eve_specific_vectors()` path.
- No automatic promotion, automatic rollback, automatic threshold adjustment,
  fastText seed mutation, memory/quarantine mutation, AGP bypass, or drift-based
  runtime policy change is allowed.
- Round67 dry-run and Round68 proposal surfaces remain read-only observability
  surfaces, but their reports must now reflect that the active gate is enforced.

Next:
- Round70 may add a read-only blocked-candidate/rollback-drill report. It must
  not disable the context-diversity gate automatically.

## v3 round70 policy — context-diversity rollback drill / blocked-candidate report

Round70 may expose candidates blocked by the active context-diversity gate and
may run a manual rollback drill. It must not disable the gate automatically.

Rules:
- `context_diversity_gate_enabled` remains `True`.
- `context_diversity_blocked_candidate_report()` must read existing audit records
  and aggregate `insufficient_context_diversity` blocks only.
- `context_diversity_rollback_drill()` may compute what would pass if a later
  operator patch manually disabled the context-diversity gate.
- The rollback drill is advisory only. It must not change runtime config,
  thresholds, vector-store contents, audit records, memory/quarantine state,
  fastText seed data, AGP behavior, or drift-based runtime behavior.
- Automatic rollback and automatic promotion remain disabled.
- Commit mutation must still go through the Round59 gate, Round64 observation
  threshold, and Round69 context-diversity enforcement.

Next:
- Round71 may add explicit export for the blocked-candidate report. It must stay
  read-only and must not introduce automatic rollback or promotion.

## v3 round71 policy — self-learning consolidation gate

Round71 is a stabilization round. It must not add new automatic learning behavior.

Rules:
- `EveVocabTracker` is lexical observation only. It must not generate vectors,
  route wrapper lookups, commit vectors, mutate memory/quarantine, or make AGP
  anchor decisions.
- `EveSelfLearningAdapter` is the continuous-observation coordinator and the
  only Round58+ explicit vector commit surface.
- `EveSpecificVectorStore` stores deterministic vectors only after an explicit
  gate-approved commit.
- `EmbeddingWrapper` may read EveSpecific vectors after fastText misses, but it
  must not decide AGP anchors or learning policy.
- `AGPAdapter` must verify anchors from explicit meaning categories and EVE
  internal SA/activated categories only. EveSpecific vectors are not anchors.
- Active commit policy remains: `observed_count >= 2`, fastText-OOV/EVE-specific
  candidate, known fastText context count >= 1, and `context_diverse=True`.
- Automatic promotion, automatic rollback, fastText seed mutation,
  memory/quarantine mutation, AGP bypass, and drift-based runtime policy changes
  remain forbidden.
- Before Round72, full-suite status must be checked from the Round71 split-run
  artifacts or any incomplete/failed files must be triaged.

## v3 round72 policy — smoke/drift baseline remeasurement

Round72 remeasures the EveSpecific routing surface after Round71 consolidation.
It may use isolated direct probe vectors only inside smoke/baseline measurement.
It must not call the self-learning commit path, change thresholds, alter context
diversity policy, promote memory, mutate fastText seeds, or bypass AGP.

## v3 round73 policy — explicit EveSpecific commit smoke

Round73 may run a first real explicit commit smoke through
`EveSelfLearningAdapter`:

```text
observe_text -> commit_eve_specific_vectors -> EveSpecificVectorStore -> EmbeddingWrapper lookup
```

Rules:
- The Round72 direct probe path must not be used for this smoke.
- Commit must pass the active gate: `observed_count >= 2`, fastText-OOV /
  EVE-specific candidate, known fastText context count >= 1, and
  `context_diverse=True`.
- `auto_promotion_enabled` remains `False`.
- `min_observations_for_commit` remains `2`.
- `context_diversity_gate_enabled` remains `True`.
- No automatic rollback, automatic threshold adjustment, memory/quarantine
  mutation, fastText seed mutation, AGP bypass, or drift-based runtime policy
  change is allowed.
- The smoke may create a vector only in the current explicit test/runtime engine
  through the gate-approved commit path.

Next:
- Round74 may compare pre/post explicit-commit drift and telemetry. It must not
  change the active self-learning policy.


## v3 round74 policy — explicit commit drift/telemetry delta

Round74 may compare baseline/telemetry before and after the existing Round73 explicit EveSpecific commit smoke.

Rules:
- It must use the real explicit self-learning commit path, not the Round72 direct probe path.
- It may create a vector only through `EveSelfLearningAdapter.commit_eve_specific_vectors()` after active gate approval.
- It must report pre-commit and post-commit lookup route deltas separately.
- It must not change thresholds, context-diversity policy, automatic rollback policy, automatic promotion policy, memory/quarantine state, fastText seed data, AGP behavior, or drift-based runtime policy.
- Any delta report is observability only. It must not feed back into runtime decisions automatically.

Next:
- Round75 may consolidate commit audit replay/export. It must remain read-only except for explicit gate-approved commits performed inside test/smoke engines.

## v3 round75 policy — commit audit replay/export consolidation

Round75 may consolidate Round73 explicit commit smoke and Round74 drift/telemetry delta evidence into a replay/export artifact.

Rules:
- The replay/export layer must be read-only.
- It may consume existing Round74 delta reports and existing `EveSelfLearningAdapter.commit_audit_records()` snapshots.
- It must not call a new commit path during replay construction.
- It must not call lookup during replay construction.
- It must not append audit records during replay construction.
- It must not mutate EveSpecific vectors, fastText seed data, memory/quarantine state, AGP behavior, thresholds, context-diversity policy, automatic rollback, or automatic promotion.
- Any file write is an explicit operator export only; it must not recompute the smoke or create vectors.

Next:
- Round76 should freeze the Round57~75 self-learning safety pipeline as a v1 baseline before starting a new architectural axis.

## v3 round76 policy — self-learning v1 freeze baseline

Round76 freezes the Round57~75 self-learning safety pipeline as `self_learning_v1`.

Rules:
- No new learning shortcut is introduced in Round76.
- The active commit gate remains: `observed_count >= 2`, fastText-OOV / EVE-specific candidate, known fastText context count >= 1, and `context_diverse=True`.
- `auto_promotion_enabled` remains `False`.
- `automatic_rollback_enabled` remains `False`.
- AGP anchors remain explicit meaning categories plus SA/activated categories only. EveSpecific vectors are not anchors.
- Round76 freeze/export functions are read-only. They must not call commit, lookup, append audit records, mutate vectors, change thresholds, alter context-diversity policy, mutate memory/quarantine, mutate fastText seed data, or bypass AGP.
- Any future policy change must use the existing pattern: dry-run → proposal report → explicit enforcement patch.

Next:
- Round77 may begin a new axis only from the frozen `self_learning_v1` baseline.

## v3 round77 policy — lexical → concept mapping planning

Round77 starts the post-`self_learning_v1` lexical→concept axis as planning only.

Rules:
- Lexical vectors are evidence, not concepts.
- EveSpecific vectors are not AGP anchors.
- External seed vectors are not AGP anchors.
- AGP anchors remain explicit meaning categories plus SA/activated categories only.
- `LexConceptMappingAdapter` is a read-only planning contract owner in Round77.
- Runtime lexical→concept mapping remains disabled.
- No category creation, ConceptMemory mutation, Frame/Hypergraph mutation, AGP verify shortcut, memory/quarantine mutation, fastText seed mutation, automatic promotion, automatic rollback, threshold change, or context-diversity policy change is allowed.
- Future mapping enforcement must follow the existing safety pattern: dry-run → proposal report → explicit enforcement patch.

Next:
- Round78 may add a lexical-concept candidate schema dry-run. It must remain read-only and must not create categories or change AGP anchors.

## v3 round78-79 policy — lexical→concept candidate dry-run and evidence report

Round78 and Round79 are read-only planning/reporting layers after the Round77 boundary.

Allowed:
- Build lexical→concept candidate rows from existing lexical evidence.
- Summarize candidate evidence quality.
- Export JSON reports for operator review.

Forbidden:
- Runtime lexical→concept mapping enforcement.
- Automatic category creation.
- Treating fastText/EveSpecific/PMI+SVD vectors as AGP anchors.
- Concept-memory mutation.
- Frame/hypergraph mutation.
- Wrapper lookup side effects from candidate reporting.
- Vector commit from candidate reporting.
- AGP verify or fallback changes from candidate reporting.

Current required boundary:
- lexical vector = evidence only
- concept/category = explicit future proposal/enforcement path only
- AGP anchor = explicit category + SA activation only


## v3 round80 policy — concept proposal report

Round80 is an operator proposal/reporting layer only.

Allowed:
- Convert lexical-evidence-ready candidate rows into concept proposal records for operator review.
- Export proposal reports as explicit artifacts.

Forbidden:
- Runtime lexical→concept mapping enforcement.
- Automatic category creation.
- Concept-memory mutation.
- Frame/hypergraph mutation.
- SA activation creation.
- Calling AGP verify or changing AGP fallback behavior.
- Treating fastText/EveSpecific/PMI+SVD vectors as AGP anchors.
- Wrapper lookup side effects from proposal reporting.
- Vector commit from proposal reporting.

Current boundary remains:
- lexical vector = evidence only
- proposed concept category = operator-review label only
- actual AGP anchor = explicit category + SA activation only

Next:
- Round81 may add a concept mapping gate dry-run. It must remain read-only and must not create categories or AGP anchors.

## v3 round81 policy — concept mapping gate dry-run

Round81 is a read-only dry-run layer.

Allowed:
- Evaluate Round80 concept proposal records against future mapping-gate requirements.
- Report which candidates would pass or block.
- Export dry-run JSON artifacts.

Forbidden:
- Runtime lexical→concept mapping enforcement.
- Automatic or implicit category creation.
- Concept-memory mutation.
- Frame/hypergraph mutation.
- SA activation creation.
- AGP anchor creation.
- Calling AGP verify or changing AGP fallback behavior.
- Treating fastText/EveSpecific/PMI+SVD vectors as AGP anchors.
- Wrapper lookup side effects from dry-run reporting.
- Vector commit from dry-run reporting.

Current boundary remains:
- lexical vector = evidence only
- Round80 concept proposal = operator-review label only
- future mapping gate requires operator acceptance, explicit category creation, concept evidence, SA activation path, and AGP bridge smoke
- actual AGP anchor = explicit category + SA activation only

Next:
- Round82 may add a concept mapping gate proposal report. It must remain read-only unless explicitly promoted to an enforcement patch after review.

## v3 round82 policy — concept mapping gate proposal report

Round82 is a read-only operator proposal/report layer over the Round81 gate dry-run.

Allowed:
- Consolidate blocked mapping-gate reasons into operator action items.
- Export the operator proposal report for review.

Forbidden:
- Runtime lexical→concept mapping enforcement.
- Automatic or implicit category creation.
- Concept-memory mutation.
- Frame/hypergraph mutation.
- SA activation creation.
- AGP anchor creation.
- Calling AGP verify or changing AGP fallback behavior.
- Treating fastText/EveSpecific/PMI+SVD vectors as AGP anchors.
- Wrapper lookup side effects from proposal reporting.
- Vector commit from proposal reporting.

Current boundary remains:
- lexical vector = evidence only
- Round80 concept proposal = operator-review label only
- Round82 operator action item = review checklist only
- actual AGP anchor = explicit category + SA activation only

Next:
- Round83 may add an operator acceptance fixture or explicit category creation dry-run. It must still avoid runtime mapping enforcement until concept evidence, SA activation path, and AGP bridge smoke tests exist.

## v3 round83 policy — operator acceptance fixture / category creation dry-run

Round83 is a read-only fixture/dry-run layer.

Allowed:
- Model explicit operator acceptance as a test fixture.
- Produce a dry-run explicit category creation plan.
- Export dry-run JSON artifacts.

Forbidden:
- Persisting operator acceptance as runtime state.
- Runtime lexical→concept mapping enforcement.
- Actual category creation.
- Concept-memory mutation.
- Frame/hypergraph mutation.
- SA activation creation.
- AGP anchor creation.
- Calling AGP verify or changing AGP fallback behavior.
- Treating fastText/EveSpecific/PMI+SVD vectors as AGP anchors.
- Wrapper lookup side effects from dry-run reporting.
- Vector commit from dry-run reporting.

Current boundary remains:
- lexical vector = evidence only
- operator acceptance fixture = non-persisted test fixture only
- category creation dry-run = would-create plan only
- actual AGP anchor = explicit category + SA activation only

Next:
- Round84 may add concept memory or frame evidence dry-run. It must remain read-only and must not create SA activation or AGP anchors.

## v3 round84~88 policy — lexical→concept read-only batch

Rounds84~88 are a read-only batch over lexical→concept mapping plans.

Allowed:
- Round84: attach concept-memory/frame evidence as dry-run plans only.
- Round85: produce SA activation path dry-run plans only.
- Round86: produce AGP bridge smoke dry-run plans only.
- Round87: aggregate readiness in a dashboard only.
- Round88: freeze concept mapping v0 proposal for operator review only.

Forbidden:
- Runtime lexical→concept mapping enforcement.
- Actual category creation.
- Concept-memory mutation.
- Frame/hypergraph mutation.
- SA activation creation.
- AGP anchor creation.
- Calling AGP verify from the dry-run/report path.
- Treating fastText/EveSpecific/PMI+SVD vectors as AGP anchors.
- Wrapper lookup side effects from dry-run/report/freeze.
- Vector commit from dry-run/report/freeze.
- Persisting operator acceptance fixture as runtime state.

Current boundary remains:
- lexical vector = evidence only
- EveSpecific vector != concept
- EveSpecific vector != AGP anchor
- seed vector != AGP anchor
- AGP anchor = explicit category + SA activation only

Next:
- Round89 may attempt explicit concept commit smoke. Because it is a mutation/enforcement round, it must be isolated and must run split full suite.

## v3 round89 policy — explicit concept commit smoke

Round89 introduced the first explicit concept-layer mutation smoke.

Allowed only in the explicit smoke path:
- Create an explicit category record for an operator-ready token.
- Persist a minimal concept-memory record.
- Persist minimal frame/hypergraph evidence.
- Create SA activation for the explicit category.
- Call AGP verify with `meaning_categories=[explicit_category]` and active SA category.

Forbidden:
- Enabling runtime lexical→concept mapping.
- Enabling broad enforcement.
- Auto-creating categories from lexical vectors.
- Treating fastText, EveSpecific, or PMI+SVD vectors as AGP anchors.
- Mutating fastText seed vectors.
- Committing new EveSpecific vectors from the concept commit path.
- Changing AGP fallback/veto behavior.

Boundary remains:
- lexical vector = evidence only
- EveSpecific vector != concept
- EveSpecific vector != AGP anchor
- seed vector != AGP anchor
- AGP anchor = explicit category + SA activation only

Any future runtime mapping enforcement must be a separate round and must run split full suite.

## v3 round90 policy — concept commit delta/replay report

Round90 is read-only. It may inspect and replay the existing Round89 explicit concept commit, but it must not create another concept category or enable runtime mapping.

Allowed:
- Read existing explicit concept category records.
- Read concept-memory presence.
- Read SA activation presence.
- Call AGP verify as a replay check only.
- Export replay/delta artifacts.

Forbidden:
- Creating additional categories.
- Writing concept memory.
- Writing frame/hypergraph edges.
- Creating SA activation.
- Enabling runtime lexical→concept mapping.
- Enabling broad enforcement.
- Treating fastText/EveSpecific/PMI+SVD vectors as AGP anchors.
- Mutating fastText seed vectors or EveSpecific vector store.

Boundary remains:
- lexical vector = evidence only
- EveSpecific vector != concept
- EveSpecific vector != AGP anchor
- seed vector != AGP anchor
- AGP anchor = explicit category + SA activation only

## v3 round91 policy — concept commit replay export / v0 checkpoint

Round91 is read-only. It may consolidate Round77~90 lexical→concept artifacts and export a checkpoint summary, but it must not create new categories or enable runtime mapping.

Allowed:
- Read existing explicit concept category records.
- Read concept-memory/SA/AGP replay evidence.
- Export checkpoint artifacts for operator review.
- Recommend the next dry-run round.

Forbidden:
- Creating additional categories.
- Writing concept memory.
- Writing frame/hypergraph edges.
- Creating SA activation.
- Enabling runtime lexical→concept mapping.
- Enabling broad enforcement.
- Treating fastText/EveSpecific/PMI+SVD vectors as AGP anchors.
- Mutating fastText seed vectors or EveSpecific vector store.

Boundary remains:
- lexical vector = evidence only
- EveSpecific vector != concept
- EveSpecific vector != AGP anchor
- seed vector != AGP anchor
- AGP anchor = explicit category + SA activation only

Round92 may be a runtime lexical→concept mapping gate dry-run, but must remain read-only unless explicitly approved as a separate mutation/enforcement round.


## Round92 runtime mapping gate dry-run

- Added read-only runtime lexical→concept mapping gate dry-run.
- Existing committed concept category `민석` would map if runtime mapping is enabled later.
- `EVE` remains blocked due to missing committed category/concept/SA/AGP evidence.
- Runtime mapping and enforcement remain disabled.
- No category/concept memory/frame/SA/AGP/vector mutation.
- Validation: Round92 focused 2 passed; Round77~92 focused 76 passed; Round50~92 adjacent split total 220 passed; collect-only 1208; compileall passed.

## v3 round93 policy — runtime mapping proposal report

Round93 is read-only. It may consolidate the Round92 runtime mapping gate dry-run into an operator proposal report, but it must not enable runtime lexical→concept mapping or enforcement.

Allowed:
- Read Round92 dry-run rows.
- Propose eligible committed concept categories for future operator review.
- Export proposal/report artifacts.

Forbidden:
- Enabling runtime lexical→concept mapping.
- Enabling enforcement.
- Creating categories.
- Writing concept memory.
- Writing frame/hypergraph edges.
- Creating SA activation.
- Calling AGP verify as part of runtime mapping.
- Performing embedding lookup as part of the proposal.
- Treating fastText/EveSpecific/PMI+SVD vectors as AGP anchors.

Boundary remains:
- lexical vector = evidence only
- EveSpecific vector != concept
- EveSpecific vector != AGP anchor
- seed vector != AGP anchor
- AGP anchor = explicit category + SA activation only

Round94 may be a runtime mapping enforcement dry-run, but actual runtime mapping enablement must remain a separate mutation/enforcement round and requires split full suite.

## v3 round94 policy — runtime mapping enforcement dry-run

Round94 is read-only. It may simulate the future runtime lexical→concept mapping API/result surface, but it must not enable runtime mapping or enforcement.

Allowed:
- Read Round93 runtime mapping proposal rows.
- Produce simulated mapping results for eligible committed concept categories.
- Export dry-run/report artifacts for operator review.

Forbidden:
- Enabling runtime lexical→concept mapping.
- Enabling enforcement.
- Creating categories.
- Writing concept memory.
- Writing frame/hypergraph edges.
- Creating SA activation.
- Calling AGP verify as part of runtime mapping.
- Performing embedding lookup as part of enforcement dry-run.
- Treating fastText/EveSpecific/PMI+SVD vectors as AGP anchors.

Boundary remains:
- lexical vector = evidence only
- EveSpecific vector != concept
- EveSpecific vector != AGP anchor
- seed vector != AGP anchor
- AGP anchor = explicit category + SA activation only

Round95 may add an operator acceptance fixture for runtime mapping, but actual runtime mapping enablement must remain a separate mutation/enforcement round and requires split full suite.

## v3 round95/round96 policy — runtime mapping fixture and enable-smoke precheck

- Round95 operator acceptance fixture is read-only.
- Round96 enable-smoke precheck is read-only.
- Do not enable `runtime_mapping_enabled=True` in these rounds.
- Do not enable `enforcement_enabled=True` in these rounds.
- Do not create categories, concept memory, frame/hypergraph edges, SA activation, or AGP anchors in these rounds.
- Lexical vectors, EveSpecific vectors, and seed vectors remain evidence only; they are not AGP anchors.
- A later controlled enable-smoke round requires checkpoint, rollback plan, focused tests, adjacent tests, collect-only, compileall, and broader validation.


## v3 round236-260 policy — runtime mapping acceptance handoff

Rounds236-260 are reporting-only over the green Round231-235 runtime-mapping
acceptance stage matrix.

Allowed:
- Build stage-matrix quality gates over existing accepted/blocked rows.
- Build operator handoff/replay rows for future review.
- Record validation manifests and handoff docs every five rounds.
- Summarize remaining taxonomy and no-go items.

Forbidden:
- Enabling production persistence.
- Changing `runtime_mapping_enabled` default to true.
- Enabling enforcement.
- Bypassing AGP.
- Creating or fabricating vectors.
- Mutating fastText seed vectors, EveSpecific vectors, semantic memory, or quarantine.
- Committing operator artifacts, seed subset artifacts, zip files, part files, `vectors.npy`, `vocab.txt`, or `subset_manifest.json`.

Boundary remains:
- `민석` may be reported as accepted for future operator review only.
- `EVE` remains a blocked control in this measurement line.
- The default runtime remains no-load unless explicitly operator-authorized.
- Any future runtime-mapping rehearsal must be a separate explicit round with no production persistence and split full-suite validation.
