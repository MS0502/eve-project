# Round1441-1460 virtual-world situation inference context schema

This round adds `read_only_virtual_world_situation_inference_context_schema`, a deterministic schema for candidate inference products attached to an internal non-visual virtual situation. Inference candidates are candidate reasoning products only: they are not verified facts, accepted beliefs, accepted or rejected hypotheses, selected conclusions, resolved truth, guaranteed predictions, applied counterfactuals, factual memories, automatic decisions, action permissions or blocks, world-state mutations, transition applications, executable plans, schedules, persistence writes, vector/model/device/network operations, AGP bypasses, or fallback bypasses.

## Public API

- `build_virtual_world_situation_inference_context_schema_summary()`
- `build_virtual_world_situation_inference_context(inference_type=None, situation_id=None, inference_items=None, metadata=None)`
- `validate_virtual_world_situation_inference_context(inference_context)`
- `build_inference_context_to_situation_plan(inference_context)`
- `build_inference_context_to_snapshot_plan(inference_context)`
- `build_inference_context_to_transition_preflight_plan(inference_context)`
- `build_inference_context_to_memory_candidate_plan(inference_context)`
- `build_inference_context_to_appraisal_plan(inference_context)`
- `build_inference_context_to_agp_input_plan(inference_context)`
- `virtual_world_situation_inference_context_schema_summary()`

## Permanent EVE Validation Contract

The schema is read-only, candidate-only, strict JSON-native, type-exact, deterministic, and fail-closed. It preserves EVE v3 invariants: no randomness, timestamps, UUIDs, Python `hash()`, filesystem-dependent identity, model/vector/device/network loading, runtime wiring, fact promotion, belief promotion, conclusion selection, truth resolution, prediction guarantee, counterfactual application, memory write, identity update, relationship update, affect or hormone update, world mutation, transition application, execution, scheduling, persistence, AGP bypass, or fallback bypass.

## Enums and compatibility

Supported inference types are exactly the 17 public values from deductive, abductive, inductive, causal, temporal, state, identity, boundary, simulation, symbolic, DMN, dream, predictive, counterfactual, competing, unknown-origin, and mixed-unknown candidate classes. Supported inference kinds are exactly the 16 step-candidate values. Supported roles are focal conclusion, alternative conclusion, counter conclusion, and neutral unknown. Supported derivation-source classes are internal hypothesis/evidence/constraint/uncertainty/temporal/causal contexts, simulation, symbolic, DMN, dream, external-unverified candidate, and unknown-origin candidate.

The implementation exposes the exact type-kind, kind-role, and kind-source compatibility matrices through the schema summary. Incompatible pairs fail with `incompatible_inference_type_kind`, `incompatible_inference_kind_candidate_role`, or `incompatible_inference_kind_derivation_source`; kinds are never rewritten.

## Item and reference-list schema

`inference_items` must be a non-empty exact built-in list of exact built-in dictionaries. Required fields are `inference_item_id`, `situation_id`, `inference_kind`, `candidate_role`, `derivation_source_class`, `subject_ref_id`, `conclusion_ref_id`, non-empty unique `premise_ref_ids`, and non-empty unique `hypothesis_context_ref_ids`. Optional recognized fields are `object_ref_id`, unique `evidence_ref_ids` which may be empty, `related_context_id`, finite numeric `confidence_candidate`, finite numeric `coherence_candidate`, and `label`. Booleans are not numeric candidates, and huge integers are checked by direct range comparison.

Reference-list ordering is normalized lexicographically. Item key order and item-list order are non-semantic. Unknown preserved fields participate in the deterministic context ID.

## Relational/unary shape and semantic duplicate identity

Causal, temporal, boundary, predictive, counterfactual, and alternative inference kinds are relational. They require a non-empty `object_ref_id` that differs from `subject_ref_id`; failures are `missing_object_ref_id_for_relational_inference`, `malformed_object_ref_id`, or `identical_inference_refs`. Unary kinds fail on any present `object_ref_id` with `unexpected_object_ref_id_for_unary_inference`.

Semantic duplicate identity uses situation ID, kind, role, derivation-source class, subject, optional object, conclusion, sorted premise references, sorted hypothesis-context references, sorted evidence references, and optional related context. It ignores item ID, confidence, coherence, label, unknown fields, and caller list position. Duplicates fail with `duplicate_semantic_inference_item`.

## Situation coherence, composition, boundary, and confidence

Every item situation ID must equal the top-level situation ID. Standard inference types require compatible items. Competing inference requires at least two items, focal plus alternative or counter candidates, and distinct semantic candidates; it never selects a winner. Mixed unknown inference requires at least two items and at least two distinct kinds, roles, or derivation-source classes; it remains unresolved and warns only for mixed-boundary review.

Boundary classification defaults by inference type and may be supplied in metadata via `inference_boundary_classification`. Malformed values fail with `malformed_inference_boundary_class`; unsupported non-empty strings fail with `unknown_inference_boundary_class`. Confidence defaults to `inference_unverified`; malformed values fail with `malformed_inference_confidence_state`; unsupported non-empty strings fail with `unknown_inference_confidence_state`. High confidence remains non-factual, competing confidence does not fabricate competition, and unknown-origin confidence does not verify origin.

## Stable payload and nested summaries

Valid and rejected payloads share a complete stable top-level schema. Summaries are protected by recursive type-exact validation:

- `origin_summary`: internal virtual inference candidate, external origin unverified.
- `fact_status_summary`: candidate-only, no external fact verification, no fact assertion, no acceptance/rejection, no conclusion assertion, no truth resolution.
- `inference_scope_summary`: sorted unique kinds, roles, source classes, conclusion references, and item count.
- `inference_competition_summary`: focal/alternative/counter/neutral counts, true competition only when focal plus alternative/counter exists, and no winner.
- `inference_support_summary`: sorted premise, hypothesis-context, and evidence references with counts; no external basis verification or sufficiency assertion.
- `resolution_summary`: candidate-only and fully unresolved.

Immutable false and immutable true flags are always present. Status coherence requires passed/`VALIDATED`/empty blocked reasons for valid payloads and failed/`REJECTED`/non-empty blocked reasons for rejected payloads. The public validator accepts only complete valid passed payloads.

## Strict JSON, forbidden fields, precedence, and deterministic IDs

Inputs are accepted recursively only when they are exact built-in dicts with string keys, lists, strings, integers, finite floats, booleans, or null. Subclasses, non-string keys, tuples, sets, frozensets, bytes, bytearrays, NaN, infinities, custom objects, circular containers, excessive depth, and hostile containers fail closed with `non_json_serializable_semantic_input`. Builder, validator, and plans never raise for hostile inputs.

Metadata and all item fields are recursively inspected for forbidden requests including inference assertion, verification, fact promotion, acceptance/rejection, conclusion selection, truth resolution, prediction guarantee, counterfactual application, memory writes, identity/relationship/affect/hormone/world/transition mutation, execution, scheduling, persistence, vector/model/device/network activation, AGP bypass, and fallback bypass. `True` fails with the exact field name; `False` is allowed; any other value fails with `malformed_forbidden_request_field`.

Validation precedence is strict: JSON-native validation; inference type; situation ID; metadata; forbidden metadata; item-list shape; canonical raw-item ordering; required fields/basic scalars; forbidden items; item ID uniqueness; premise, hypothesis, and evidence references; semantic duplicates; situation coherence; relational/unary shape; role compatibility; source compatibility; type compatibility; composition; boundary; confidence; derived summaries; and canonical ID generation. Invalid item ordering is deterministic.

`inference_context_id` is SHA-256 over canonical JSON containing schema version, inference type, boundary, confidence, situation ID, normalized items, and normalized metadata, using string keys, sorted keys, compact separators, `ensure_ascii=False`, and `allow_nan=False`. Reordering metadata, item keys, item list, premise references, hypothesis-context references, or evidence references does not change the ID; semantic changes do.

## Recursive integrity and downstream plans

The validator rebuilds the complete expected valid payload and recursively compares exact Python types. It rejects added/removed fields, ID or algorithm tampering, summary tampering, normalized reference tampering, immutable flag tampering, bool/int confusion, and int/float confusion; a matching ID is never sufficient.

All six downstream plans validate their source and remain candidate-only, read-only, and inference-candidate-only. Valid sources may be ready. Invalid, rejected, malformed, hostile, deep, or tampered sources produce `ready is False` while preserving every required false side-effect field.

## Prohibited side effects and fail-closed behavior

The adapter writes no files, creates no artifacts, stages no data, schedules nothing, executes no tools, performs no network/device/model/vector activity, and mutates no virtual world, memory, identity, relationship, affect, hormone, AGP, fallback, transition, or persistence state. Rejected payloads serialize under strict JSON with `allow_nan=False`.

## Recommended next implementation step

read_only_virtual_world_situation_conclusion_candidate_schema
