# Round1421-1440: Read-only virtual-world situation hypothesis context schema

## Scope

This round adds a deterministic, read-only schema for **hypothesis candidates** attached to an internal non-visual virtual situation. Hypotheses are candidate-only internal review records: explanatory, causal, temporal, state, identity, boundary, simulation, symbolic, DMN, dream, predictive, counterfactual, competing, unknown-origin, and mixed/unresolved candidates.

They are not verified facts, accepted beliefs, proven explanations, selected conclusions, resolved truth, guaranteed predictions, factual memories, automatic decisions, action permissions, action blocks, world-state mutations, transition applications, executable plans, schedules, persistence writes, vector/model/device/network operations, AGP bypasses, or fallback bypasses.

## Public API

- `build_virtual_world_situation_hypothesis_context_schema_summary()`
- `build_virtual_world_situation_hypothesis_context(hypothesis_type=None, situation_id=None, hypothesis_items=None, metadata=None)`
- `validate_virtual_world_situation_hypothesis_context(hypothesis_context)`
- `build_hypothesis_context_to_situation_plan(hypothesis_context)`
- `build_hypothesis_context_to_snapshot_plan(hypothesis_context)`
- `build_hypothesis_context_to_transition_preflight_plan(hypothesis_context)`
- `build_hypothesis_context_to_memory_candidate_plan(hypothesis_context)`
- `build_hypothesis_context_to_appraisal_plan(hypothesis_context)`
- `build_hypothesis_context_to_agp_input_plan(hypothesis_context)`
- `virtual_world_situation_hypothesis_context_schema_summary()`

All public functions fail closed and never raise for malformed, hostile, non-JSON, rejected, or tampered inputs.

## Enums and matrices

Supported hypothesis types, hypothesis kinds, candidate roles, boundary classes, confidence states, type↔kind compatibility, and kind↔role compatibility are exposed literally by the schema summary. Incompatible type/kind pairs fail with `incompatible_hypothesis_type_kind`; incompatible kind/role pairs fail with `incompatible_hypothesis_kind_candidate_role`. Hypothesis kinds are never rewritten.

## Item schema

`hypothesis_items` must be a non-empty list of dictionaries. Required fields are `hypothesis_item_id`, `situation_id`, `hypothesis_kind`, `candidate_role`, `subject_ref_id`, `claim_ref_id`, and `basis_evidence_ref_ids`. Optional recognized fields are `object_ref_id`, `related_context_id`, `confidence_candidate`, `plausibility_candidate`, and `label`. Unknown fields are preserved only when they are strict JSON-native and recursively free of forbidden request fields; preserved unknown fields participate in deterministic IDs.

Basis-evidence references must be one or more unique non-empty strings and are normalized lexicographically. Item key order, item-list order, and basis-reference order are non-semantic for normalized payload construction and deterministic IDs.

## Relational and unary reference shape

Relational kinds require a non-empty `object_ref_id` distinct from `subject_ref_id`: causal, temporal, boundary, predictive, counterfactual, and alternative proposition candidates. Missing, malformed, or identical relational references fail closed. Unary kinds must not include `object_ref_id`; the field is never silently discarded.

## Semantic duplicate identity and situation coherence

Semantic duplicate identity is the canonical JSON key over situation ID, kind, role, subject, object-or-null, claim, sorted basis references, and related context-or-null. It ignores item ID, confidence, plausibility, label, unknown fields, and caller list position. Duplicate semantic hypotheses fail with `duplicate_semantic_hypothesis_item`. Every item situation ID must equal the top-level situation ID or fail with `hypothesis_item_situation_mismatch`.

## Type composition

Standard non-competing and non-mixed types must contain their compatible corresponding kind. `competing_hypothesis_candidate` requires multiple items, a focal candidate, and an alternative or counter candidate without selecting a winner. `mixed_unknown_hypothesis_candidate` requires multiple items and at least two distinct kinds or roles; it remains unresolved and defaults to the mixed virtual/external boundary, with review warnings but no blocked reasons when otherwise valid.

## Boundaries and confidence

Absent `hypothesis_boundary_classification` uses deterministic type defaults. Malformed actual values fail with `malformed_hypothesis_boundary_class`; unsupported non-empty strings fail with `unknown_hypothesis_boundary_class`. Absent `hypothesis_confidence_state` defaults to `hypothesis_unverified`. Malformed confidence values fail with `malformed_hypothesis_confidence_state`; unsupported non-empty strings fail with `unknown_hypothesis_confidence_state`. High confidence remains non-factual and unaccepted. Competing confidence adds review flags but fabricates no competing items. Unknown-origin confidence adds origin-review flags.

## Nested summaries and status coherence

Valid and rejected payloads share one stable schema with origin, fact-status, scope, competition, support, and resolution summaries. Immutable false fields remain false, including acceptance, rejection, verification, conclusion assertion, truth resolution, prediction guarantee, counterfactual application, mutation, execution, scheduling, persistence, vector/model/device/network, AGP-bypass, and fallback-bypass fields. Immutable true fields keep candidate-only, read-only, review, quarantine, appraisal, and AGP-input requirements.

## Strict JSON and recursive forbidden fields

Strict JSON validation runs before semantic inspection. Only exact built-in dictionaries with string keys, lists, strings, integers, finite floats, booleans, and null are accepted. Subclasses, non-string keys, tuples, sets, bytes, byte arrays, NaN, infinities, custom objects, circular structures, hostile containers, and recursion-hostile depths fail with `non_json_serializable_semantic_input`.

Metadata and item payloads are recursively inspected in deterministic order for assertion, verification, conclusion, resolution, mutation, execution, scheduling, persistence, vector/model/device/network, AGP-bypass, and fallback-bypass request fields. A forbidden field set to true fails with the exact field name; false is allowed; non-boolean values fail with `malformed_forbidden_request_field`.

## Validation precedence and deterministic invalid ordering

Validation follows the fixed order required by the round: strict JSON, type, situation, metadata, forbidden metadata, item list, canonical raw item ordering, item fields, forbidden item fields, unique IDs, basis references, semantic duplicates, situation coherence, reference shape, role compatibility, type compatibility, composition, boundary, confidence, summaries, and canonical ID. Invalid-list ordering is deterministic because raw items and recursive list entries use canonical JSON ordering before semantic checks.

## Canonical ID and recursive type-exact integrity

`hypothesis_context_id` is SHA-256 over canonical JSON with sorted string keys, compact separators, `ensure_ascii=False`, and `allow_nan=False`. The ID basis includes schema version, hypothesis type, boundary classification, confidence state, situation ID, normalized hypothesis items, and normalized metadata. Validator strict-JSON validates, requires exact top-level keys, rebuilds the complete expected payload, and recursively compares exact Python types, rejecting ID tampering, algorithm tampering, immutable flag tampering, bool/int confusion, int/float confusion, added or removed fields, and normalized item tampering.

## Downstream plans and prohibited side effects

All six downstream plan builders validate their source. Valid sources may be ready; invalid, rejected, malformed, hostile, deep, or tampered sources are not ready. Plans remain candidate-only and read-only and keep required false fields false. No plan performs acceptance, rejection, verification, winner selection, truth resolution, prediction guarantee, counterfactual application, memory write, world mutation, transition, execution, scheduling, persistence, vector read/load, model/device/network action, AGP bypass, or fallback bypass.

## Permanent EVE Validation Contract

The permanent contract covers exact input types, required and optional fields, literal enums, compatibility matrices, situation/reference coherence, relational/unary shape, cardinality, composition, ordering, duplicate identity, validation precedence, strict JSON, recursive forbidden fields, deterministic IDs, exact top-level schema, nested payload integrity, immutable safety fields, status coherence, downstream plan safety, and fail-closed behavior.

Recommended next implementation step: `read_only_virtual_world_situation_inference_context_schema`.
