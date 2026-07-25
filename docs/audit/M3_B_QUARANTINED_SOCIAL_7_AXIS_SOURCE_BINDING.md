# M3-B Quarantined Social-Relationship Seven-Axis Source Binding

## Baseline

`d82652d74b9051d227ec415b648ec209b9de8b3d` — PR #181 squash merge.

## Purpose

This artifact binds the seven `social_relationship` registry source contracts to an explicit detached raw-record schema. It advances source-binding coverage after the merged four operational, two appraised survival, and six quarantined risk-defense axes without installing any production capture path.

Bound axes:

1. `social_pain`
2. `social_trust`
3. `attachment`
4. `care_drive`
5. `loneliness_pressure`
6. `belonging_need`
7. `rejection_sensitivity`

Combined registry source-binding progress becomes `19/37`; `18/37` appraised axes remain unbound.

## Social quarantine and appraisal boundary

The canonical source family is exactly:

```text
quarantined_social_appraisal_trace
```

Each record is caller-supplied and immutable. The module performs no social-input acquisition. Evidence is admissible only after an explicit versioned social-input quarantine trace and a separate versioned bounded social appraisal trace.

The integrity chain is fail-closed:

```text
raw source evidence
  -> quarantine input digest
  -> verified social-quarantine integrity digest
  -> exact social-appraisal input digest
  -> verified social-appraisal integrity digest
  -> canonical raw observation digest
  -> detached registry evidence
```

`appraisal_input_digest` must equal the exact `quarantine_integrity_digest`. Raw social feedback cannot be passed directly into any of the seven axes. Direct hardware input, synthetic values, proposal-only records, registry-owner circular sourcing, and runtime-polled records are rejected.

## Canonical raw fields

The implementation consumes the exact five fields frozen by the merged 37-axis source manifest:

| Axis | Exact raw fields |
|---|---|
| `social_pain` | `appraisal_version`, `injury_evidence_score`, `intent_confidence`, `recurrence_count`, `source_trust` |
| `social_trust` | `contradiction_count`, `fulfilled_commitment_count`, `observation_span_ticks`, `repair_count`, `source_trust` |
| `attachment` | `appraisal_version`, `interaction_continuity`, `mutual_reliability`, `relationship_span_ticks`, `separation_tolerance` |
| `care_drive` | `appraisal_version`, `capability_to_help`, `consent_status`, `cost_boundary`, `welfare_need_score` |
| `loneliness_pressure` | `appraisal_version`, `available_relationship_context`, `chosen_solitude_flag`, `meaningful_contact_gap_ticks`, `unmet_connection_signal_count` |
| `belonging_need` | `appraisal_version`, `context_span_ticks`, `group_continuity`, `reciprocal_inclusion_count`, `role_clarity` |
| `rejection_sensitivity` | `ambiguous_signal_count`, `false_positive_count`, `observation_span_ticks`, `source_trust`, `verified_rejection_count` |

The default social minimum is three records over eight logical ticks. Two manifest overrides remain exact:

```text
attachment: 3 records / 12 ticks
care_drive: 2 records / 2 ticks
```

Records must remain sorted, unique in logical time and evidence identity, and tied to one source contract.

## Deterministic derivation

All numeric inputs are bounded or non-negative integer evidence. Count evidence uses the bounded transform `count / (count + 1)`. Long-horizon span evidence uses `span / (span + 4)`.

`care_drive` uses an explicit finite consent map:

```text
withheld -> 0.0
limited  -> 0.5
granted  -> 1.0
```

`loneliness_pressure` preserves the distinction between chosen solitude and unmet connection: chosen solitude contributes no loneliness-pressure unit by itself.

`rejection_sensitivity` uses verified rejection, ambiguous signal, false-positive, source-trust, and observation-span evidence. Ambiguous signals are weighted below verified rejection, and false positives remain part of the calibration denominator.

Each axis score is deterministic, bounded to the registry range, and averaged across the accepted record set. Confidence is deterministic, strictly positive, and derived from accepted-record score variance within `(0,1]`.

## Recalculation and identity

Each raw observation digest binds:

- axis and logical tick;
- observation identity;
- source instance, snapshot, schema, and integrity digest;
- social-quarantine trace identity, input digest, integrity digest, schema, method, outcome, and verification state;
- social-appraisal trace identity, exact input digest, integrity digest, schema, method, outcome, and verification state;
- exact ordered raw values;
- canonical acquisition and verification provenance;
- raw-record and derivation rule versions.

Final registry evidence carries a deterministic raw-bundle digest and derived source-integrity digest so value and confidence remain independently recalculable from the retained record set.

## Authority boundary

This PR remains detached and `shadow_only`.

It does **not**:

- install production capture;
- ingest raw social feedback;
- poll hardware;
- install a runtime observer or scheduler;
- materialize the registry owner;
- access persistence;
- append events;
- mutate live affect, drives, named state, goals, memory, self, or expression;
- start or satisfy the observation window;
- complete M3-B;
- open M3-C;
- authorize cutover;
- open M3-E authority.

Legacy runtime and legacy persistence remain authoritative.

## Validation and cross-chat reuse

PR #181 is a merged prerequisite and must be reused by exact pin. Moving this work to another chat, editing PR metadata, changing comments/review state, or changing Draft/Ready state is not a validation invalidator.

The validation order remains:

```text
focused / M0 / M2-B discovery
  -> full suite blocked
  -> exact append-only M2-B decisions
  -> exact forward fingerprints
  -> final registered head
  -> full suite exactly once
```

A prior pin may be invalidated only by a relevant code-head change, artifact loss/corruption or digest mismatch, validation-scope/dependency change, or loss of required merge ancestry.

## Current authority status

```text
operational source bindings:          4/37
appraised survival bindings:          2/37
quarantined risk-defense bindings:    6/37
quarantined social bindings:          7/37
combined source bindings:            19/37
remaining appraised bindings:        18/37
production capture:                   absent
positive-confidence real observation: absent
observation window started:           false
M3-B complete:                         false
M3-C open:                             false
M3-E authority open:                   false
cutover authorized:                    false
```

Active blockers remain:

```text
REGISTRY_APPRAISED_18_AXIS_SOURCE_BINDINGS_INCOMPLETE
REGISTRY_POSITIVE_CONFIDENCE_COVERAGE_INCOMPLETE
```

## Next required artifact

Bind the remaining 18 appraised axes in bounded source-family stages. Retained real production capture and any observation-window start decision remain separate later artifacts.
