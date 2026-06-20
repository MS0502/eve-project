# Round1401-1420: Read-only virtual-world situation evidence context schema

## Scope

This round adds a deterministic, read-only schema for **evidence candidates** attached to an internal non-visual virtual situation. The schema represents supporting, challenging, conflicting, neutral, provenance, temporal, causal, state, identity, boundary, simulation, symbolic, DMN, dream, tool-state, unknown-origin, and mixed/unresolved evidence candidates.

Evidence contexts are candidate-only review payloads. They are not verified external facts, proven evidence, resolved truth, selected conclusions, predictions, memories, decisions, permissions, blocks, mutations, transition applications, executable plans, schedules, persistence writes, vector/model/device/network operations, AGP bypasses, or fallback bypasses.

## Public API

- `build_virtual_world_situation_evidence_context_schema_summary()`
- `build_virtual_world_situation_evidence_context(evidence_type=None, situation_id=None, evidence_items=None, metadata=None)`
- `validate_virtual_world_situation_evidence_context(evidence_context)`
- `build_evidence_context_to_situation_plan(evidence_context)`
- `build_evidence_context_to_snapshot_plan(evidence_context)`
- `build_evidence_context_to_transition_preflight_plan(evidence_context)`
- `build_evidence_context_to_memory_candidate_plan(evidence_context)`
- `build_evidence_context_to_appraisal_plan(evidence_context)`
- `build_evidence_context_to_agp_input_plan(evidence_context)`
- `virtual_world_situation_evidence_context_schema_summary()`

All public functions fail closed and never raise for malformed, hostile, non-JSON, rejected, or tampered inputs.

## Supported enums and compatibility matrices

The schema supports the exact evidence types, evidence item kinds, stances, source classes, boundary classes, and confidence states exposed by `virtual_world_situation_evidence_context_schema_summary()`.

Validation applies three compatibility matrices in a fixed order:

1. evidence item kind to stance, failing with `incompatible_evidence_item_kind_stance`;
2. evidence item kind to source class, failing with `incompatible_evidence_item_kind_source_class`;
3. evidence type to item kind, failing with `incompatible_evidence_type_item_kind`.

Item kinds are never rewritten to satisfy a type.

## Evidence item schema

`evidence_items` must be a non-empty list of dictionaries. Each item requires:

- `evidence_id`
- `situation_id`
- `evidence_kind`
- `stance`
- `source_class`
- `source_ref_id`
- `claim_ref_id`

Recognized optional fields are:

- `related_context_id`
- `confidence_candidate`
- `weight_candidate`
- `label`

Unknown fields are preserved only after strict JSON validation and recursive forbidden-request inspection. Preserved unknown fields participate in deterministic context IDs.

## Strict JSON and hostile input contract

Strict JSON validation runs before semantic validation. Accepted values are exact built-in dictionaries with string keys, lists, strings, integers, finite floats, booleans, and null. Rejected values include built-in subclasses, non-string dictionary keys, tuples, sets, bytes, byte arrays, NaN, infinities, custom objects, circular structures, hostile containers, raising string/repr objects, and recursion-hostile deep structures.

All strict JSON failures return `non_json_serializable_semantic_input` in a stable rejected payload that is itself JSON-serializable with `allow_nan=False`.

## Recursive forbidden fields

Metadata and item payloads are recursively inspected in deterministic order for assertion, verification, conclusion, resolution, mutation, execution, scheduling, persistence, vector/model/device/network, AGP-bypass, and fallback-bypass request fields. A forbidden field set to `True` fails with that exact field name. A forbidden field set to `False` is allowed. Any non-boolean forbidden-field value fails with `malformed_forbidden_request_field`.

## Validation precedence

Validation order is fixed:

1. strict JSON-native validation;
2. evidence type;
3. top-level situation ID;
4. metadata dictionary validation;
5. recursive forbidden metadata validation;
6. evidence item list presence and shape;
7. canonical raw item ordering;
8. per-item required fields and field types;
9. per-item recursive forbidden validation;
10. unique evidence IDs;
11. semantic duplicate identity;
12. situation coherence;
13. item-kind/stance compatibility;
14. item-kind/source-class compatibility;
15. evidence-type/item-kind compatibility;
16. type-specific cardinality;
17. type-specific composition;
18. boundary validation;
19. confidence validation;
20. derived summaries;
21. deterministic ID generation.

Later errors do not mask earlier required reasons.

## Deterministic IDs and ordering

`evidence_context_id` is SHA-256 over canonical JSON with sorted string keys, compact separators, `ensure_ascii=False`, and `allow_nan=False`. The ID basis includes schema version, evidence type, boundary classification, confidence state, situation ID, normalized evidence items, and normalized metadata.

Metadata key order, item key order, and item-list order do not change the ID. Semantic payload changes do change the ID. Evidence IDs, candidate confidence/weight, labels, and preserved unknown fields remain part of the full ID payload.

## Duplicate and coherence rules

Semantic duplicate identity is the canonical JSON key over:

- `situation_id`
- `evidence_kind`
- `stance`
- `source_class`
- `source_ref_id`
- `claim_ref_id`
- `related_context_id` or null

Duplicate semantic items fail with `duplicate_semantic_evidence_item`. Item situation IDs must match the top-level situation ID or fail with `evidence_item_situation_mismatch`.

## Summaries and immutable safety fields

Valid and rejected payloads use one complete top-level schema. Summaries include origin, fact status, evidence scope, stance, provenance, and resolution. Resolution remains candidate-only and unresolved. Winner selection and automatic resolution are always false.

Immutable false fields keep assertion, verification, resolution, prediction, random sampling, probability draw, identity, memory, relationship, self-model, affect, hormone, world-state, transition, planning, tool execution, runtime mutation, persistence, vector, model, device, network, artifact staging, scheduling, action permission/blocking, AGP-bypass, and fallback-bypass paths disabled.

Immutable true fields keep candidate-only, read-only, review-required, quarantine-required, appraisal-required, and AGP-input-required markers enabled.

## Recursive type-exact integrity

The validator accepts only complete valid passed payloads. It strict-JSON validates the source, requires exact top-level keys, rebuilds the complete expected payload from the normalized semantic inputs, and recursively compares exact Python types, dictionary key sets, list lengths/order, and scalar values. A matching ID is never sufficient.

This rejects missing fields, extra fields, nested-summary tampering, normalized-item tampering, ID or algorithm tampering, immutable flag tampering, bool/int confusion, and int/float confusion.

## Downstream plans

All six downstream plan builders call the validator. Valid sources may produce `ready == True`; invalid, rejected, malformed, hostile, or tampered sources produce `ready == False`. Plans remain candidate-only, read-only, non-verifying, non-resolving, non-executing, non-mutating, non-persistent, non-scheduling, non-vector, non-model, non-device, non-network, non-AGP-bypass, and non-fallback-bypass.

## Prohibited side effects

The implementation performs no external evidence assertion, verification, fact promotion, conclusion selection, conflict resolution, memory write, self-model update, identity update, relationship update, affect or hormone transition, world-state mutation, transition application, plan/tool execution, scheduling, persistence, vector read/load, model load, device activation, network action, AGP bypass, or fallback bypass.

## Operator report

The operator report prints exactly one compact JSON object, writes no files, and exits nonzero if any check fails.

## Permanent EVE Validation Contract

Validate:

- input types
- required and optional fields
- enums
- compatibility matrices
- situation and reference coherence
- cardinality
- composition
- ordering
- duplicates
- validation precedence
- strict JSON
- recursive forbidden fields
- deterministic IDs
- exact top-level schema
- nested payload integrity
- immutable safety fields
- downstream plan safety

Never silently default a present malformed value.

Never invent required semantic values.

The validator must rebuild and type-exactly compare the complete expected payload.

Tests must use literal contracts and mutation-based adversarial cases rather than merely repeating implementation constants.

Every plan builder validates its source and remains candidate-only, read-only, non-verifying, non-resolving, non-executing, non-mutating, non-persistent, and non-scheduling.

## Recommended next implementation step

read_only_virtual_world_situation_hypothesis_context_schema
