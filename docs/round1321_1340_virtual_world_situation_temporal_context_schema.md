# Round1321-1340: read-only virtual world situation temporal context schema

This round adds a deterministic, read-only schema for temporal context candidates inside non-visual virtual situations. A temporal context can describe an ongoing situation candidate, logical before/after order, simultaneity candidate, abstract interval, waiting/traveling duration candidate, logical sequence position, or symbolic/DMN/simulation/dream time.

## Logical candidates, not external time facts

The schema only represents internal temporal candidates. It never confirms current time, wall-clock time, external timestamps, verified event history, completed external events, scheduler state, timers, alarms, reminders, deadlines, memory facts, executable plans, or world-state transitions.

## Supported temporal types

- `situation_ongoing_candidate`
- `situation_started_candidate`
- `situation_ended_candidate`
- `situation_before_candidate`
- `situation_after_candidate`
- `situation_simultaneous_candidate`
- `situation_sequence_candidate`
- `situation_interval_candidate`
- `situation_recurring_pattern_candidate`
- `situation_future_window_candidate`
- `waiting_duration_candidate`
- `traveling_duration_candidate`
- `learning_session_candidate`
- `symbolic_time_candidate`
- `dmn_time_candidate`
- `simulation_time_candidate`
- `dream_time_candidate`
- `mixed_unknown_temporal_candidate`

## Temporal anchor structure

Each valid candidate requires a temporal anchor with `anchor_id`, `anchor_kind`, and `label`. Supported anchor kinds are `logical_step`, `relative_order`, `bounded_logical_interval`, `symbolic_marker`, and `unknown_marker`. Optional `logical_start`, `logical_end`, and `sequence_index` are non-negative integers; `logical_end` must be greater than or equal to `logical_start`. These values are abstract logical values, not seconds or wall-clock timestamps.

## Boundary classes and confidence states

Boundary classes are `internal_logical_time`, `symbolic_virtual_time`, `dmn_virtual_time`, `simulated_virtual_time`, `dream_virtual_time`, and `mixed_virtual_external_time_boundary`. If `temporal_boundary_classification` is absent, the builder uses the deterministic temporal-type default. If it is present, it must be a non-empty supported string; falsey or container values fail closed with `malformed_temporal_boundary_class`, and unsupported strings fail with `unknown_temporal_boundary_class`. Mixed virtual/external boundaries add review flags and warnings while remaining candidate-only when otherwise valid.

Confidence states remain non-factual: `temporal_unverified`, `temporal_low_confidence`, `temporal_medium_confidence`, `temporal_high_confidence_but_not_fact`, `temporal_conflict_detected`, and `temporal_origin_unknown`. If `temporal_confidence_state` is absent, the builder uses `temporal_unverified`. If it is present, it must be a non-empty supported string; falsey or container values fail closed with `malformed_temporal_confidence_state`, and unsupported strings fail with `unknown_temporal_confidence_state`. Conflict and unknown-origin states add uncertainty or integrity flags, not runtime changes.

## Fail-closed behavior

The builder and validator fail closed for missing or unknown temporal types, missing or malformed situation IDs, malformed optional reference situation IDs, missing required reference situations, missing or malformed anchors, malformed or unknown boundary/confidence values, identical before/after situation IDs, invalid logical intervals, negative sequence indexes, forbidden external-time requests, scheduler/timer/alarm/calendar/reminder/deadline requests, non-JSON-serializable or non-JSON-native semantic input, and incoherent pass/reject statuses. A shared recursive validator inspects metadata and temporal anchors, including preserved unknown anchor fields, and rejects any forbidden request field set to `true` using that field name as the reason. `false` remains non-requesting, while any non-boolean forbidden-request value fails with `malformed_forbidden_request_field`.

## Deterministic IDs

`temporal_context_id` is computed from normalized semantic input using canonical JSON with sorted keys and SHA-256. Reordered dictionary keys do not change the ID, semantic changes do change it, and validator recomputation detects tampering. Strict JSON-native validation runs before metadata, anchor, enum, or string inspection in both the builder and validator. A single canonical serialization helper enforces strict JSON-native semantic input, uses `allow_nan=False`, and catches `TypeError`, `ValueError`, `OverflowError`, and `RecursionError` so non-JSON-native semantic input fails closed instead of raising. The validator also rebuilds the expected candidate payload from normalized semantic input and rejects tampering of protected derived fields such as origin/fact summaries, sequence constraints, duration candidates, uncertainty and boundary flags, temporal-integrity flags, candidate-only fields, and warnings. The schema does not use UUIDs, random sampling, process-local identity, filesystem state, or clock state.

## No clock, scheduler, timer, or activation access

The implementation does not read wall-clock/current time and does not import or call system clock APIs. It does not create timers, alarms, schedules, calendar actions, reminders, deadlines, device activations, model loads, vector loads, tool calls, plans, persistence writes, memory writes, relationship updates, affect transitions, hormone transitions, world mutations, AGP bypasses, or fallback bypasses.

## Candidate-only downstream plans

Downstream plan builders for situation, snapshot, transition preflight, memory candidate, appraisal, and AGP input produce read-only candidate plans. They report readiness from validation only and keep all side-effect flags false.

## Recommended next implementation step

read_only_virtual_world_situation_causal_context_schema
