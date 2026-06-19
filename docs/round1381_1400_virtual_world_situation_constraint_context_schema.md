# Round1381-1400: read-only virtual world situation constraint context schema

## Track

`read_only_virtual_world_situation_constraint_context_schema`

## Purpose

This round adds a deterministic, read-only schema for **internal constraint candidates** attached to non-visual virtual situations. Constraint contexts describe possible hard/soft constraints, preconditions, postconditions, invariants, resource limits, temporal ordering, causal guards, role/relation guards, uncertainty guards, boundary limits, exclusions, dependencies, symbolic guards, conflicts, and unresolved mixed cases.

They are candidate-only descriptions. They do not enforce a constraint, permit or prevent an action, evaluate satisfaction or violation, reserve or consume resources, mutate virtual world state, apply transitions, execute plans or tools, schedule anything, promote facts, write memory, update identity/self-model/relationships/affect/hormones, persist state, read vectors, load models/devices, perform network actions, bypass AGP, or bypass fallback gates.

## Supported constraint types and clause kinds

The schema supports exactly these constraint types: `hard_constraint_candidate`, `soft_constraint_candidate`, `precondition_constraint_candidate`, `postcondition_constraint_candidate`, `invariant_constraint_candidate`, `resource_constraint_candidate`, `temporal_constraint_candidate`, `causal_constraint_candidate`, `role_relation_constraint_candidate`, `uncertainty_guard_constraint_candidate`, `boundary_constraint_candidate`, `exclusion_constraint_candidate`, `dependency_constraint_candidate`, `ordering_constraint_candidate`, `symbolic_constraint_candidate`, `conflicting_constraint_candidate`, and `mixed_unknown_constraint_candidate`.

The schema supports exactly these clause kinds: `must_hold_candidate`, `should_hold_candidate`, `must_not_hold_candidate`, `requires_candidate`, `excludes_candidate`, `before_candidate`, `after_candidate`, `depends_on_candidate`, `resource_limit_candidate`, `boundary_limit_candidate`, `uncertainty_guard_candidate`, `causal_guard_candidate`, `role_relation_guard_candidate`, `symbolic_guard_candidate`, and `unknown_constraint_candidate`.

## Type/clause compatibility

Each constraint type has a literal compatible clause-kind set. Incompatible pairs fail closed with `incompatible_constraint_type_clause_kind`; the builder never rewrites clause kinds. Hard constraints accept must-hold, must-not-hold, requires, and excludes candidates. Soft constraints accept should-hold candidates. Precondition, postcondition, invariant, resource, temporal, causal, role/relation, uncertainty, boundary, exclusion, dependency, ordering, and symbolic types accept only their corresponding clause kinds. Conflicting constraints accept the exact contradiction-capable kinds. Mixed-unknown constraints accept all supported clause kinds.

## Clause schema and reference shape

`constraint_clauses` must be a non-empty list. Each clause requires `clause_id`, `situation_id`, `clause_kind`, and `subject_ref_id`. Optional recognized fields are `object_ref_id`, `strength_candidate`, `priority_candidate`, and `label`. Unknown fields are preserved only when strict JSON-native and free of forbidden assertion, enforcement, evaluation, mutation, execution, scheduling, activation, persistence, vector/model/device/network, AGP-bypass, or fallback-bypass request fields.

Relational kinds are `requires_candidate`, `excludes_candidate`, `before_candidate`, `after_candidate`, and `depends_on_candidate`. They require a non-empty `object_ref_id`, and subject/object references must differ. All other clause kinds are unary and must not include `object_ref_id`.

## Situation coherence and duplicate identity

Every clause `situation_id` must equal the top-level `situation_id`; otherwise the context fails with `constraint_clause_situation_mismatch`. Clause IDs must be unique. Semantic duplicate identity is exactly `situation_id`, `clause_kind`, `subject_ref_id`, and `object_ref_id` or null. Duplicate semantic clauses fail even if `clause_id`, `label`, `strength_candidate`, `priority_candidate`, preserved unknown fields, or input position differ.

## Type-specific cardinality and composition

Every type requires at least one compatible clause. Specialized types require their matching clause family. Conflicting contexts require at least two clauses and at least one detected contradiction pair. Mixed-unknown contexts require at least two clauses and at least two distinct compatible clause kinds. Cardinality checks precede detailed conflict composition where required.

## Exact conflict detection and type coherence

The schema detects exactly three contradiction forms: `must_hold_candidate` versus `must_not_hold_candidate` for the same subject; `requires_candidate` versus `excludes_candidate` for the same subject/object; and `before_candidate` versus `after_candidate` for the same subject/object. It infers no other contradiction. Conflict pair clause IDs and the final pair list are sorted deterministically. A detected conflict is valid only for `conflicting_constraint_candidate` and `mixed_unknown_constraint_candidate`; other types fail with `constraint_conflict_requires_conflicting_type`. Conflicting contexts select no winner and resolve nothing.

## Boundary classes and confidence states

Supported boundary classes are `internal_logical_constraint`, `internal_soft_constraint`, `internal_precondition_constraint`, `internal_postcondition_constraint`, `internal_invariant_constraint`, `internal_resource_constraint`, `temporal_virtual_constraint`, `causal_virtual_constraint`, `role_relation_virtual_constraint`, `uncertainty_virtual_constraint`, `boundary_virtual_constraint`, `symbolic_virtual_constraint`, and `mixed_virtual_external_constraint_boundary`. Missing boundary metadata uses deterministic type defaults; present malformed values fail as malformed, while unsupported non-empty strings fail as unknown.

Supported confidence states are `constraint_unverified`, `constraint_low_confidence`, `constraint_medium_confidence`, `constraint_high_confidence_but_not_enforced`, `constraint_conflict_detected`, and `constraint_origin_unknown`. Missing confidence defaults to `constraint_unverified`. High confidence remains non-factual and unenforced. Conflict confidence adds review flags but does not invent a conflict pair. Unknown-origin confidence adds origin review flags.

## Summaries and immutable safety

The output contains origin, fact-status, scope, conflict, and enforcement summaries. Scope records sorted unique clause kinds plus unary and relational counts. Conflict summary records detected pairs, keeps `winner_selected`, `conflict_resolved`, and `automatic_resolution_allowed` false, and sets `conflict_present` from actual pairs only. Enforcement summary always remains candidate-only, unenforced, not automatically enforceable, not action-blocking, and not action-permitting. Immutable safety flags keep every enforcement/evaluation/mutation/execution/scheduling/persistence/vector/model/device/network/AGP/fallback side-effect flag false and every review/candidate/read-only requirement true.

## Strict JSON-native validation and deterministic invalid ordering

Strict JSON-native validation runs first. Allowed values are exact built-in dictionaries with string keys, exact built-in lists, strings, integers, finite floats, booleans, and null. Container or scalar subclasses, hostile containers, non-string dictionary keys, tuples, sets, frozensets, bytes, bytearrays, NaN, infinities, custom objects, circular structures, recursion-hostile structures, and serialization exceptions fail closed with `non_json_serializable_semantic_input`. `strength_candidate` validates booleans as malformed, integers by direct range check without `math.isfinite()`, floats by finite check then range check, and huge integers outside `[0.0, 1.0]` as `strength_candidate_out_of_range`. After strict JSON validation succeeds, raw clause-list entries are canonically sorted before per-clause validation so reordered invalid inputs fail deterministically.

## Recursive forbidden requests and validation precedence

Metadata, clauses, unknown clause fields, and dictionaries inside lists are inspected recursively. Forbidden request fields set to `true` fail with the exact field name; `false` is allowed; non-boolean values fail with `malformed_forbidden_request_field`. Validation precedence is strict: JSON, type, situation ID, metadata, forbidden metadata, clause list, canonical raw clause ordering, clause fields, forbidden clause requests, clause ID uniqueness, semantic duplicates, situation coherence, reference shape, compatibility, cardinality, conflict detection, composition/type coherence, boundary, confidence, summaries, and ID generation.

## Canonical ID and type-exact integrity

`constraint_context_id` is SHA-256 over canonical JSON with sorted keys, compact separators, `ensure_ascii=False`, and `allow_nan=False`. The basis contains schema version, constraint type, boundary classification, confidence state, situation ID, normalized constraint clauses, and normalized metadata. Metadata key order, clause key order, and clause-list order do not affect the ID; semantic payload changes do. The validator strict-JSON validates first, rebuilds the complete expected valid payload, requires exact top-level keys, and compares recursively with type-exact equality so boolean/integer and integer/float confusion is rejected.

## Status coherence, downstream plans, and fail-closed behavior

A valid payload has `situation_constraint_context_passed=true`, status `VALIDATED`, an ID, the canonical algorithm, and empty `blocked_reasons`. A rejected payload has status `REJECTED`, passed false, a non-empty blocked reason list, a complete safe schema, no ID, and no side-effect flag true. Downstream situation, snapshot, transition-preflight, memory-candidate, appraisal, and AGP-input plans always call the validator. Valid sources may produce `ready=true`; invalid, rejected, malformed, hostile, or tampered sources produce `ready=false`. Plans remain read-only, candidate-only, unenforced, unevaluated, non-executing, non-mutating, non-persistent, non-scheduling, and do not activate vectors, models, devices, networks, AGP bypass, or fallback bypass.

## Recommended next implementation step

read_only_virtual_world_situation_evidence_context_schema
