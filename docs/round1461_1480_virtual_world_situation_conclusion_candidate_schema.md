# Round1461-1480 virtual-world situation conclusion-candidate schema

This round adds `read_only_virtual_world_situation_conclusion_candidate_schema` as a strict-JSON, deterministic, read-only layer derived only from validated Round1441-1460 virtual-world situation inference contexts.

Conclusion candidates are candidate-only, unresolved, unaccepted, unrejected, non-factual, non-executable, and non-persistent. The adapter never selects a winner, asserts truth, promotes a fact, applies a counterfactual, guarantees a prediction, writes memory, executes a plan, schedules work, activates vectors/models/devices/network access, or mutates memory, identity, relationships, affect, hormones, world state, transitions, AGP, or fallback behavior.

## Source contract

`build_virtual_world_situation_conclusion_candidate(...)` first requires its `inference_context` input to pass `validate_virtual_world_situation_inference_context(...)`. Invalid, hostile, tampered, rejected, or non-JSON sources fail closed as rejected conclusion-candidate payloads. By default, conclusion items are deterministically derived from source inference items while preserving source inference item IDs, source conclusion references, premise references, hypothesis-context references, evidence references, subject/object references, role, confidence candidate, and coherence candidate.

## Strict JSON and hostile inputs

The adapter accepts only exact built-in JSON-native values: dicts with string keys, lists, strings, integers, finite floats, booleans, and null. It rejects subclasses, tuples, sets, bytes, bytearrays, objects, non-string keys, circular structures, excessive depth, NaN, and infinities with `non_json_serializable_semantic_input`. Builder, validator, and downstream plans fail closed rather than raising on hostile input.

## Validation precedence

Validation precedence is deterministic: JSON-native validation; validated source inference-context validation; conclusion type; metadata shape; recursive forbidden metadata fields; conclusion item list shape; canonical item ordering; required item fields and basic scalar checks; recursive forbidden item fields; item ID uniqueness; source-reference membership; situation coherence; reference-list shape; numeric candidate shape/range; semantic duplicate detection; kind-role compatibility; conclusion type-kind compatibility; composition checks for competing and mixed candidates; boundary classification; confidence state; summary derivation; canonical ID generation; recursive type-exact integrity validation.

## Deterministic canonical ID

`conclusion_candidate_id` is SHA-256 over canonical JSON containing the schema version, conclusion type, boundary, confidence, source inference context ID, situation ID, normalized conclusion items, and normalized metadata. Canonical JSON uses sorted keys, compact separators, `ensure_ascii=False`, and `allow_nan=False`. Non-semantic ordering of metadata keys, item keys, input item list order, and reference-list order does not change the ID; semantic payload changes do.

## Forbidden fields and side-effect boundaries

Metadata and conclusion items are recursively inspected for forbidden request fields covering fact promotion, acceptance/rejection, conclusion selection, truth resolution, winner selection, prediction guarantee, counterfactual application, memory writes, identity/relationship/affect/hormone/world/transition mutation, execution, scheduling, persistence, vector/model/device/network activation, AGP bypass, and fallback bypass. A forbidden field set to `True` fails with the exact field name; malformed non-`False` values fail with `malformed_forbidden_request_field`; explicit `False` remains inert.

## Downstream plans

The six downstream plan builders validate their source conclusion candidate before reporting readiness. Valid source candidates can produce read-only review plans; invalid, rejected, hostile, tampered, or malformed sources produce `ready: false` while preserving candidate-only/read-only flags and all side-effect fields as `False`.

## Operator report

`scripts/operator_report_round1461_1480_virtual_world_situation_conclusion_candidate_schema.py` emits exactly one compact JSON object. Every boolean is computed from actual adapter checks, including valid/rejected cases, literal schema contract, deterministic IDs, exact validation precedence, recursive forbidden-field matrix, semantic duplicate detection, tamper detection, downstream plan validation, hostile-object failures, no-side-effect flags, exact four-file scope, policy grep, and artifact inspection.

## Recommended next implementation step

read_only_virtual_world_situation_decision_candidate_schema
