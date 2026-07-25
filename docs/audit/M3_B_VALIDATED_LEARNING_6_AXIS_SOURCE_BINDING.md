# M3-B Validated Learning-Exploration Six-Axis Source Binding

## Baseline

`6c57f41114fbe0a203e559a27b187f6801ad7640` — PR #182 squash merge.

## Purpose

This artifact binds the six `learning_exploration` registry source contracts to detached caller-supplied immutable validated records:

1. `curiosity_drive`
2. `novelty_seeking`
3. `learning_pressure`
4. `memory_consolidation_pressure`
5. `prediction_error_pressure`
6. `competence_drive`

Combined registry source-binding progress becomes `25/37`; `12/37` appraised axes remain unbound.

## Validation and appraisal boundary

The canonical source family is:

```text
validated_learning_and_prediction_trace
```

Each record must carry a versioned learning/prediction validation trace followed by a separate bounded appraisal trace. `appraisal_input_digest` must equal the exact `validation_integrity_digest`.

The module does not perform learning or prediction updates. It does not consolidate memory. It consumes already validated caller-supplied records and only derives detached registry evidence.

Raw social feedback is rejected rather than admitted as a learning signal. Direct hardware input, synthetic values, proposal-only records, registry-owner circular sourcing, runtime-polled records, records claiming learning mutation, and records claiming memory writes are rejected.

## Canonical raw fields

| Axis | Exact raw fields |
|---|---|
| `curiosity_drive` | `exploration_cost`, `information_gain_estimate`, `relevance_score`, `sampling_window_ticks`, `unknown_count` |
| `novelty_seeking` | `appraisal_version`, `expected_information_gain`, `novelty_score`, `reversibility`, `safety_score` |
| `learning_pressure` | `available_training_signal`, `competence_gap`, `error_recurrence`, `task_relevance`, `validation_status` |
| `memory_consolidation_pressure` | `causal_relevance`, `emotional_relevance`, `provenance_completeness`, `recurrence_count`, `salience_score` |
| `prediction_error_pressure` | `model_version`, `normalized_error`, `observed_value_digest`, `predicted_value_digest`, `verification_status` |
| `competence_drive` | `calibrated_error_rate`, `evaluation_version`, `learning_progress`, `skill_gap`, `success_rate` |

Default minimum coverage is two records over two logical ticks. Manifest overrides remain exact:

```text
prediction_error_pressure: 2 records / 1 tick
competence_drive:          3 records / 4 ticks
```

## Deterministic derivation

All continuous evidence is range checked to `[0,1]`; counts and spans must be non-negative integers. Count pressure uses `count / (count + 1)` and span support uses `span / (span + 4)`.

`prediction_error_pressure` accepts only verified prediction-error evidence with non-placeholder observed and predicted digests. `learning_pressure` accepts only explicit `verified` or `operator_validated` validation status.

Values and confidence are deterministic. Confidence is positive and variance-derived. Every result carries a recalculable raw-bundle digest and source-integrity digest.

## Authority boundary

This artifact remains detached and `shadow_only`.

It does not:

- acquire training or prediction input;
- mutate a model, skill, category, vector, or memory;
- install production capture;
- poll hardware;
- install a runtime observer or scheduler;
- access persistence;
- append events;
- materialize the registry owner;
- start or satisfy the observation window;
- complete M3-B;
- open M3-C;
- authorize cutover;
- open M3-E authority.

Legacy runtime and legacy persistence remain authoritative.

## Cross-chat validation reuse

PR #182 is a merged prerequisite. Its exact validation pin must be reused across chat/session changes. PR metadata, comments, reviews, and Draft/Ready transitions do not invalidate the pin.

The next PR must continue the standard sequence:

```text
focused / M0 / M2-B discovery
  -> full suite blocked
  -> exact append-only M2-B decisions
  -> exact forward fingerprints
  -> final registered head
  -> full suite exactly once
```

Only a relevant code-head change, artifact integrity failure, validation-scope/dependency change, or required ancestry loss invalidates a prior pin.

## Current authority status

```text
operational source bindings:          4/37
appraised survival bindings:          2/37
quarantined risk-defense bindings:    6/37
quarantined social bindings:          7/37
validated learning bindings:          6/37
combined source bindings:            25/37
remaining appraised bindings:        12/37
production capture:                   absent
positive-confidence real observation: absent
observation window started:           false
M3-B complete:                         false
M3-C open:                             false
M3-E authority open:                  false
cutover authorized:                    false
```

Active blockers:

```text
REGISTRY_APPRAISED_12_AXIS_SOURCE_BINDINGS_INCOMPLETE
REGISTRY_POSITIVE_CONFIDENCE_COVERAGE_INCOMPLETE
```

## Next required artifact

Bind the remaining 12 axes (`self_identity` and `expression_action`) in separate bounded stages. Retained real production capture and any observation-window start remain separate later artifacts.
