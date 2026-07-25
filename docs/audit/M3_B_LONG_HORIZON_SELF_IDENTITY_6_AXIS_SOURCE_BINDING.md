# M3-B Long-Horizon Self-Identity Six-Axis Source Binding

## Baseline

`af363429f0d1065f803084d446a87c80753bc4cf` — PR #183 squash merge.

## Scope

This artifact binds all six `self_identity` registry axes using detached caller-supplied immutable long-horizon self-model review records:

- `self_coherence`
- `self_respect`
- `identity_integrity`
- `agency_pressure`
- `autonomy_drive`
- `purpose_alignment`

Combined source-binding coverage becomes `31/37`. The remaining six axes are the `expression_action` group.

## Review and appraisal boundary

The canonical source family is:

```text
long_horizon_self_model_review_trace
```

A record is admissible only after a versioned long-horizon self-model review and a separate bounded self-identity appraisal. The exact review output is digest-bound to the appraisal input:

```text
review_integrity_digest == appraisal_input_digest
```

Raw social feedback cannot directly become self-identity evidence. Direct hardware input, synthetic/proposal-only records, registry-owner circular sourcing, runtime-polled records, identity mutation, self-model writes, and memory writes are rejected.

## Exact manifest fields and coverage

All six axes require exactly three records spanning at least twelve logical ticks.

| Axis | Exact raw fields |
|---|---|
| `self_coherence` | `action_value_alignment`, `narrative_conflict_count`, `review_span_ticks`, `self_model_version`, `value_consistency_score` |
| `self_respect` | `appraisal_version`, `boundary_preservation_score`, `coerced_action_count`, `review_span_ticks`, `self_denigration_rejection_count` |
| `identity_integrity` | `constitutional_conflict_count`, `provenance_gap_count`, `replay_consistency_score`, `review_version`, `unauthorized_identity_write_count` |
| `agency_pressure` | `blocked_goal_count`, `forced_action_count`, `reversible_choice_count`, `review_span_ticks`, `self_selected_action_ratio` |
| `autonomy_drive` | `capability_boundary_score`, `evaluation_version`, `external_dependency_ratio`, `independent_task_success_rate`, `safe_action_space_size` |
| `purpose_alignment` | `action_alignment_score`, `active_goal_count`, `aligned_goal_count`, `conflicting_goal_count`, `review_span_ticks` |

`aligned_goal_count` may not exceed `active_goal_count`. Counts are non-negative and continuous scores are finite and bounded to `[0,1]`.

## Deterministic derivation

Count pressure uses `count / (count + 1)`. Long-horizon span support uses `span / (span + 12)`.

The derivations are intentionally review-only. They quantify coherence, boundary preservation, identity replay integrity, agency pressure, autonomous-operation pressure, and purpose alignment without performing any identity update.

Every output is deterministic and carries a recalculable raw-bundle digest plus source-integrity digest. Confidence is deterministic, variance-derived, and positive.

## Authority boundary

This remains `shadow_only`. It does not:

- mutate identity or the self model;
- write memory;
- ingest raw social feedback;
- install production capture;
- poll hardware;
- install runtime observers or schedulers;
- access persistence;
- append events;
- materialize or mutate the registry owner;
- start or satisfy the observation window;
- complete M3-B;
- open M3-C;
- authorize cutover;
- open M3-E authority.

Legacy runtime and persistence remain authoritative.

## Cross-chat exact-head reuse

PR #183 is a merged prerequisite. Its exact validation evidence is reused across chat/session changes. PR metadata, comments, reviews, and Draft/Ready transitions are not invalidators.

Validation order remains:

```text
focused / M0 / M2-B discovery
  -> full suite blocked
  -> exact append-only M2-B decisions
  -> exact forward fingerprints
  -> final registered head
  -> full suite exactly once
```

Only a relevant code-head change, artifact integrity failure, validation-scope/dependency change, or required merge-ancestry loss invalidates a prior pin.

## Authority status after this artifact

```text
combined source bindings:            31/37
remaining appraised bindings:         6/37
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
REGISTRY_APPRAISED_6_AXIS_SOURCE_BINDINGS_INCOMPLETE
REGISTRY_POSITIVE_CONFIDENCE_COVERAGE_INCOMPLETE
```

## Next artifact

Bind the final six `expression_action` axes. Completing 37/37 source-binding contracts will still not constitute retained real production observation or observation-window authority.
