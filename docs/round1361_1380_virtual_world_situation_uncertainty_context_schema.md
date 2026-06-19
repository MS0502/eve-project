# Round1361-1380: read-only virtual world situation uncertainty context schema

## Track

`read_only_virtual_world_situation_uncertainty_context_schema`

## Purpose

This round adds a deterministic, read-only schema for internal uncertainty candidates attached to non-visual virtual situations. The records distinguish internal uncertainty candidates from verified external facts: they do not assert real-world uncertainty, resolve truth, forecast probabilities, mutate world state, update identity or relationships, write memory, schedule work, execute tools, persist artifacts, load vectors/models/devices, or bypass AGP/fallback paths.

## Supported uncertainty types

The schema supports `missing_information_candidate`, `conflicting_evidence_candidate`, `source_reliability_uncertainty_candidate`, `temporal_uncertainty_candidate`, `causal_uncertainty_candidate`, `state_uncertainty_candidate`, `identity_uncertainty_candidate`, `boundary_uncertainty_candidate`, `stochastic_uncertainty_candidate`, `ambiguity_candidate`, `simulation_uncertainty_candidate`, `symbolic_uncertainty_candidate`, `dmn_uncertainty_candidate`, `dream_uncertainty_candidate`, `unknown_origin_uncertainty_candidate`, and `mixed_unknown_uncertainty_candidate`.

## Factor schema, kinds, polarity, and compatibility

`uncertainty_factors` is a non-empty list of dictionaries. Each factor requires `factor_id`, `situation_id`, `factor_kind`, and `polarity`. Optional fields are `related_context_id`, `weight_candidate`, and `label`; unknown fields are preserved only when strict JSON-native and free of forbidden request/mutation/resolution fields. Factor kinds and polarities are validated literally. `supporting_evidence` must use `support_candidate`, `challenging_evidence` must use `challenge_candidate`, and every other kind must use `neutral_unknown`. Type/factor compatibility follows the round matrix and incompatible pairs fail with `incompatible_uncertainty_type_factor_kind`; incompatible kind/polarity pairs fail with `incompatible_factor_kind_polarity`.

## Situation coherence and duplicate identity

Every factor `situation_id` must equal the top-level `situation_id`, otherwise `uncertainty_factor_situation_mismatch` is returned. Factor IDs must be unique. Semantic duplicate identity is exactly `situation_id`, `factor_kind`, `polarity`, and `related_context_id` or null; semantic duplicates fail even when factor ID, label, weight, unknown fields, or list position differ.

## Type-specific cardinality and composition

Missing-information records require missing-information factors. Conflicting evidence requires at least two factors with at least one support and one challenge. Ambiguity and mixed-unknown records require at least two factors with at least two distinct compatible kinds. Other specialized types require their corresponding uncertainty factor kind. Cardinality checks precede detailed composition checks when applicable.

## Boundary classes and confidence states

Boundary classes include internal epistemic, internal stochastic, source provenance, temporal/causal/state/identity/symbolic/DMN/simulated/dream virtual boundaries, and the mixed virtual/external uncertainty boundary. Defaults are deterministic by uncertainty type, with remaining types defaulting to `internal_epistemic_uncertainty`. Confidence states include unverified, low/medium confidence, high-confidence-but-not-fact, conflict detected, and origin unknown. Present malformed boundary or confidence values are rejected; unsupported strings are rejected separately. High confidence remains non-factual, conflict confidence adds review flags, and unknown-origin confidence adds origin flags.

## Evidence balance and unresolved-resolution contract

Evidence balance counts normalized factor polarities and sets `conflict_present` only when support and challenge counts are both positive. External evidence is never verified. Resolution fields are immutable: records remain candidate-only, unresolved, not externally resolved, and not automatically resolvable.

## Strict JSON-native and recursive forbidden-request contracts

Strict JSON-native validation runs first and allows only dictionaries with string keys, lists, strings, integers, finite floats, booleans, and null. Tuples, sets, bytes, NaN/infinity, custom objects, circular structures, and recursion-hostile inputs fail closed with `non_json_serializable_semantic_input`; rejected payloads remain JSON-serializable. Metadata and factors share one recursive forbidden-field validator. Forbidden fields set to `true` fail with the exact field name; forbidden fields set to `false` are allowed; non-boolean forbidden values fail with `malformed_forbidden_request_field`.

## Validation precedence and exact payload integrity

Validation order is strict: JSON, type, situation ID, metadata, forbidden metadata, factor list, factor fields, forbidden factor fields, factor ID uniqueness, semantic duplicates, situation coherence, kind/polarity compatibility, type/factor compatibility, type cardinality and composition, boundary, confidence, summaries, and ID generation. The validator rebuilds the complete expected payload, requires exact top-level keys, protects nested summaries and immutable flags, and rejects added, removed, altered, or tampered fields. A valid object has status `VALIDATED`, an ID, a canonical algorithm, and empty blocked reasons. A rejected object has status `REJECTED`, no ID, and non-empty blocked reasons.

## Deterministic ID generation

The `uncertainty_context_id` is SHA-256 over canonical JSON with sorted string keys, compact separators, `ensure_ascii=False`, and `allow_nan=False`. The basis contains schema version, uncertainty type, boundary classification, confidence state, situation ID, normalized uncertainty factors, and normalized metadata. Metadata key order, factor key order, and factor list order do not affect the ID; semantic changes do.

## Read-only downstream plans and prohibited side effects

Situation, snapshot, transition-preflight, memory-candidate, appraisal, and AGP-input plans always call the validator. Valid contexts may produce `ready=true`; malformed, rejected, non-dict, non-JSON, or tampered contexts produce `ready=false`. Plans are read-only, candidate-only, non-executing, non-mutating, non-sampling, non-resolving, non-persistent, and do not perform scheduling, vector/model/device/network actions, AGP bypass, or fallback bypass.

## Fail-closed behavior

Malformed or unsafe input returns a stable rejected payload with safe immutable fields. The builder and validator never raise for hostile semantic input.

## Recommended next implementation step

`read_only_virtual_world_situation_constraint_context_schema`
