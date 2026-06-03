# ROUND_V3_R53_REPORT — B continuous self-learning design + scaffold

## Result

```text
v3 round53
scope: self-learning design + scaffold only
status: passed
```

Round53 starts **B continuous** after the successful A-first medium 30k expansion.
It does not implement runtime learning yet. It only adds the interfaces and
read-only architecture data needed for deterministic EVE-specific vocabulary
learning in later rounds.

## Why

Round52 showed that medium 30k improved runtime lexical coverage strongly:

```text
primary_hit_rate: 0.7708
fallback_rate:    0.2292
AGP_pass_rate:    1.0
```

Remaining lexical gaps are EVE-specific proper nouns and identity terms:

```text
EVE
민석
```

These should not be solved by relying further on external seed expansion alone.
They are the first target for Appendix D drift: external seed as initial map,
EVE-specific deterministic learning as the long-term distribution.

## Files added

```text
adapters/eve_vocab_tracker.py
adapters/eve_vector_store.py
tests/test_v3_round53_self_learning_design.py
```

## Files changed

```text
adapters/external_seed_manifest.py
AGENTS.md
CURRENT_STATUS.md
```

## New scaffold modules

### EveVocabTracker

Purpose:

```text
Observe EVE-specific lexical use and deterministic counts.
```

Round53 status:

```text
observe_word(...) = NotImplementedError
is_eve_specific(...) = NotImplementedError
get_observation_count(...) = NotImplementedError
tracked_vocab(...) = NotImplementedError
stats() = read-only scaffold data
```

Implementation target:

```text
round54
```

### EveSpecificVectorStore

Purpose:

```text
Store deterministic vectors for EVE-specific vocabulary.
```

Round53 status:

```text
dimension = 50  # PMI+SVD compatibility
add_vector(...) = NotImplementedError
get_vector(...) = NotImplementedError
is_eve_specific(...) = NotImplementedError
stats() = read-only scaffold data
```

Implementation target:

```text
round55
```

## Architecture plan

```text
round53: design + scaffold
round54: EveVocabTracker observe implementation
round55: EveSpecificVectorStore deterministic vector update
round56: Wrapper fallback priority integration
round57: smoke rerun + drift baseline measurement
round58+: continuous observation + drift accumulation
```

## Future wrapper priority

Current round53 runtime remains unchanged:

```text
primary:  fasttext medium 30k
fallback: PMI+SVD legacy backup
```

Future design after round56:

```text
primary:             fasttext medium 30k (immutable external seed)
fallback_priority_1: EveSpecificVectorStore (EVE-specific deterministic learning)
fallback_priority_2: PMI+SVD legacy fallback
```

## Appendix D alignment

```text
alignment: high
reason: EVE-specific vocabulary is learned deterministically from EVE's own use.
```

Draft drift tracking remains:

```text
round 200+ avg drift > 0.3
```

Round53 refines the metric inputs:

```text
EveSpecificVectorStore.stats.stored_count
EveVocabTracker.stats.observed_count
fallback_path_distribution: fastText vs EveSpecificVectorStore vs PMI+SVD
EVE-specific OOV resolution: EVE, 민석
```

## Preserved invariants

```text
no actual observation yet
no actual vector update yet
no wrapper integration yet
no smoke rerun
no drift auto measurement activation
fasttext primary remains medium 30k
PMI+SVD fallback remains unchanged
AGP unchanged
memory/quarantine files unchanged
random.random() = 0 policy preserved
```

## Next recommended round

```text
v3 round54: EveVocabTracker observe implementation
```

Round54 should implement deterministic observation only. It should still avoid
vector creation and wrapper integration.
