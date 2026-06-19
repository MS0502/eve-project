# Round1341-1360: read-only virtual world situation causal context schema

## Track

`read_only_virtual_world_situation_causal_context_schema`

## Purpose

This round adds a deterministic, read-only schema for **internal causal-context candidates** inside non-visual virtual situations. These records may represent candidate direct causes, indirect causes, consequences, common causes, correlations, unknown direction, multi-link causal chains, simulated cause/effect relations, counterfactuals, and symbolic/DMN/dream causality.

They are explicitly not verified real-world causes, confirmed external consequences, factual event history, medical/legal/financial/physical/scientific causal claims, memory facts, completed external events, guaranteed predictions, executable plans, interventions, world-state transitions, tool actions, behavior commands, relationship updates, self-model updates, affect/hormone transitions, persistence writes, or vector/model loads.

## Supported causal types

- `direct_cause_candidate`
- `indirect_cause_candidate`
- `contributing_factor_candidate`
- `enabling_condition_candidate`
- `preventing_condition_candidate`
- `candidate_consequence`
- `common_cause_candidate`
- `causal_chain_candidate`
- `causal_direction_unknown_candidate`
- `correlation_only_candidate`
- `counterfactual_cause_candidate`
- `simulation_cause_candidate`
- `symbolic_cause_candidate`
- `dmn_cause_candidate`
- `dream_cause_candidate`
- `mixed_unknown_causal_candidate`

## Causal-link structure and link kinds

`causal_links` is a non-empty list. Each link requires `link_id`, `source_situation_id`, `target_situation_id`, and `link_kind`. Optional fields are `sequence_index`, `weight_candidate`, and `label`. Unknown link fields are preserved only if they remain strict JSON-native and do not contain forbidden request, mutation, assertion, activation, scheduling, intervention, or execution fields.

Supported link kinds are `cause_candidate`, `consequence_candidate`, `contributing_candidate`, `enabling_candidate`, `preventing_candidate`, `common_cause_candidate`, `correlation_candidate`, `counterfactual_candidate`, and `unknown_direction_candidate`.

Link IDs must be unique, source and target situation IDs must be distinct, duplicate semantic links fail closed, boolean values are not accepted as integers/floats, sequence indexes are non-negative integers, and weights are finite numbers in `[0.0, 1.0]`. Duplicate semantic edge identity is based on source, target, and link kind; link ID, label, weight, and sequence index cannot make the same causal edge semantically distinct. Correlation and unknown-direction duplicates use unordered endpoint pairs.

## Boundary classes and confidence states

Boundary classes are `internal_logical_causality`, `symbolic_virtual_causality`, `dmn_virtual_causality`, `simulated_virtual_causality`, `dream_virtual_causality`, and `mixed_virtual_external_causal_boundary`. Missing boundary metadata is deterministically defaulted from causal type. Present malformed values fail with `malformed_causal_boundary_class`; unsupported non-empty strings fail with `unknown_causal_boundary_class`.

Confidence states are `causal_unverified`, `causal_low_confidence`, `causal_medium_confidence`, `causal_high_confidence_but_not_fact`, `causal_conflict_detected`, `causal_direction_unknown`, and `causal_origin_unknown`. Missing confidence metadata defaults to `causal_unverified`. High confidence remains non-factual. Conflict and unknown origin add review flags rather than changing runtime state.

## Causal type/link compatibility and endpoint coherence

Every causal type has a deterministic compatible link-kind set. Examples include direct and indirect cause candidates requiring `cause_candidate`, contributing-factor candidates requiring `contributing_candidate`, correlation-only candidates requiring `correlation_candidate`, unknown-direction and mixed-unknown candidates requiring `unknown_direction_candidate`, counterfactual candidates requiring `counterfactual_candidate`, and common-cause candidates requiring `common_cause_candidate`. Incompatible link kinds fail closed with `incompatible_causal_type_link_kind`; the builder never rewrites link kinds.

Single-direction contexts require exactly one link whose source equals `subject_situation_id` and target equals `object_situation_id`. Correlation-only, unknown-direction, and mixed-unknown contexts require exactly one compatible link whose unordered endpoint set equals the declared subject/object set, while `direction_known` remains false. Endpoint mismatches fail with `causal_link_context_endpoint_mismatch`.

## Causal direction, paths, chains, common cause, and counterfactuals

Directional candidate types require distinct subject/object situations. Correlation-only and unknown-direction candidates keep `direction_known=false`; correlation-only explicitly records that correlation is not causation.

Indirect cause candidates require at least two `cause_candidate` links before any sequence validation occurs. After cardinality passes, they require unique contiguous sequence indexes from zero, first source equal to the subject, last target equal to the object, and a connected non-cyclic path where each target equals the next source. Disconnected, cyclic, orphaned, branched, or endpoint-mismatched paths fail closed with explicit indirect-cause reasons.

Causal chains validate internal sequence only. They require at least two compatible links before any sequence validation occurs, then require unique contiguous `sequence_index` values beginning at zero, connected path semantics, subject/object endpoint coherence, and no cycles. Chain order is normalized by sequence index for deterministic IDs, and `causal_chain_summary.sequence_validated` is true only after sequence, connectivity, endpoint, and cycle checks pass.

Common-cause candidates require at least two `common_cause_candidate` links sharing one candidate common-cause source. One target must equal the subject and one target must equal the object; unrelated targets, mismatched common-cause sources, duplicate semantic edges, or a common-cause source equal to either declared outcome fail closed. Common-cause candidates do not verify a shared external origin.

Counterfactual candidates never perform interventions or counterfactual execution. Simulation candidates remain simulated. Symbolic, DMN, and dream causes remain internal symbolic candidates.

## Strict JSON-native and recursive forbidden-request contracts

Semantic input is validated before metadata, enum, string, causal-link, or nested-field inspection. Allowed values are JSON-native dictionaries with string keys, lists, strings, integers, finite floats, booleans, and null. Tuples, sets, frozensets, bytes, bytearrays, NaN, infinities, custom objects, non-string dictionary keys, circular structures, and recursion-hostile structures fail closed with exactly `non_json_serializable_semantic_input`. Rejected payloads remain JSON-serializable.

One recursive typed validator is shared for metadata and causal-link fields. Forbidden request fields such as memory writes, fact promotion, causal verification, intervention, execution, scheduling aliases (`schedule_requested`, `timer_requested`, `cron_expression`, and related scheduler/alarm/calendar/reminder/deadline fields), persistence, vector/model/device/network activation, AGP bypass, and fallback bypass fail closed when `true`. A forbidden field set to `false` is allowed. Any non-boolean forbidden field value fails with `malformed_forbidden_request_field`.

## Deterministic ID and exact payload integrity

`causal_context_id` is built from schema version, causal type, boundary, confidence, subject situation ID, object situation ID, normalized causal links, and normalized metadata. The implementation uses canonical JSON with string-only keys, sorted dictionary keys, compact separators, `ensure_ascii=False`, `allow_nan=False`, and SHA-256. It does not use UUIDs, random sampling, current time, Python `hash()`, object identity, filesystem state, process state, models, vectors, or devices.

The public validator accepts only passed payloads. It strict-JSON validates first, rebuilds the complete expected valid payload from normalized semantic input, requires exact top-level key-set equality, protects the canonical algorithm field, recomputes the deterministic ID, and compares the full payload for equality. Missing fields, extra fields, altered summaries, altered normalized links, immutable flag tampering, request-field injection, and blocked status incoherence are rejected.

## Status coherence and downstream plans

A valid payload has `situation_causal_context_passed=true`, status `VALIDATED`, no blocked reasons, a valid deterministic ID, and the complete stable schema. A rejected payload has `situation_causal_context_passed=false`, status `REJECTED`, at least one blocked reason, and safe immutable fields.

Downstream plan builders for situation, snapshot, transition preflight, memory candidate, appraisal, and AGP input always call the validator. Valid sources may produce `ready=true`; rejected, malformed, non-dict, non-JSON, or tampered sources produce `ready=false`. Plans are candidate-only and read-only and never execute or mutate anything.

## Prohibited side effects

This schema adds no runtime wiring change, no runtime mutation, no memory write, no memory fact promotion, no self-model or relationship update, no affect or hormone transition, no world-state mutation, no transition application, no intervention execution, no persistence, no vector/vocabulary/subset/model loading, no device/network/browser/tool activation, no planning execution, and no AGP or fallback bypass.

## Recommended next implementation step

`read_only_virtual_world_situation_uncertainty_context_schema`
