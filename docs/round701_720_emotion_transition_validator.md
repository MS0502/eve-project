# Round701-720 emotion transition validator

## Scope

Round701-720 adds a pure, read-only validator for proposed emotion transition
payloads. The validator is a design/test surface only: it returns deterministic
pass/fail results and reasons, but it does not apply emotion transitions.

## Validator module

- Module: `adapters/emotion_transition_validator.py`
- Operator command: `python scripts/operator_validate_round701_720_emotion_transition_payloads.py`
- Contract basis: `adapters/emotion_state_transition_contract.py` from Round681-700

## Required result fields

The validator returns structured data containing:

- `passed`
- `status`
- `reasons`
- `blocked_reasons`
- `warnings`
- `required_quarantine`
- `required_appraisal`
- `core_identity_protected`
- `runtime_mutation_allowed = false`
- `persistence_allowed = false`
- `vector_read_performed = false`
- `state_mutation_performed = false`

The result also reports read-only proof fields for memory, live emotion state,
live hormone state, AGP route, fallback route, classifier route, runtime mapping,
and enforcement defaults.

## Validation rules

- Unknown `event_category` fails closed.
- Known Round681-700 social feedback categories are accepted only for read-only
  validation.
- All social feedback requires quarantine.
- Long-term memory or self-model update requests require appraisal before memory
  or self-model update.
- `malicious_comment`, `social_threat`, and `identity_attack` require
  quarantine and block direct core identity update.
- High-risk social feedback also blocks direct self-model and direct long-term
  memory update in this validator round.
- `empathy_mode` must be
  `other_state_inference_plus_relationship_aware_action_selection`.
- `proposed_effects` may describe future behavior tendency but must not request
  application of runtime state.
- `runtime_mutation_requested` fails.
- `persistence_write_requested` fails.
- Vector read/load requests fail.

## Read-only guarantees

Round701-720 does not:

- mutate live emotion state;
- mutate hormone state;
- write memory;
- enable production persistence;
- change `runtime_mapping_enabled` defaults;
- enable enforcement by default;
- read vector, vocab, or subset manifest contents;
- load vectors;
- alter AGP, fallback, or classifier routes;
- create operator artifacts.

## Next implementation recommendation

Exactly one next step is recommended: integrate the read-only validator as an
explicit operator or test gate before any future emotion transition apply round.
