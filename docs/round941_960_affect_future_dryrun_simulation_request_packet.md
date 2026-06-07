# Round941-960 affect future dry-run simulation request packet

## Track

`read_only_future_dryrun_simulation_request_packet_from_operator_decision`

## Scope

Round941-960 adds an intent-only request packet derived from a valid operator decision packet. The packet can record that a future read-only dry-run simulation may be requested later, but it does not execute the simulation, does not authorize dry-run apply, and does not authorize live apply.

## Request packet surface

Module:

- `adapters/affect_future_dryrun_simulation_request_packet.py`

Pure functions:

- `build_affect_future_dryrun_simulation_request_packet(event_category, proposed_axis_deltas, operator_decision, metadata=None)`
- `build_affect_future_dryrun_simulation_request_from_decision_packet(decision_packet, metadata=None)`
- `validate_affect_future_dryrun_simulation_request_packet(packet)`
- `affect_future_dryrun_simulation_request_packet_summary()`

The request packet is built only through the operator decision packet surface. The only passing source decision is `approve_for_future_dryrun_simulation`; `reject`, `hold_for_review`, and `request_revision` remain non-request outcomes.

## Schema summary

The packet includes:

- `packet_version`
- `event_category`
- `source_operator_decision`
- `source_decision_packet_summary`
- `dryrun_trace_summary`
- `requested_future_action = "future_dryrun_simulation_request_only"`
- `requested_now = false`
- `execution_allowed_now = false`
- `dryrun_apply_allowed = false`
- `live_apply_allowed = false`
- `approval_scope`
- `explicit_non_permissions`
- `required_followup_before_any_simulation`
- `required_followup_before_any_apply`
- `deterministic_sequence_id_policy`
- `integrity_hash_policy`
- `no_write_guarantee`
- `no_apply_guarantee`
- `notes`

## Safety rules

The request packet fails closed when:

- the event category is unknown,
- the operator decision is invalid,
- the source decision packet fails,
- the source trace fails,
- the source decision is anything other than `approve_for_future_dryrun_simulation`,
- event-specific safety requirements are not preserved,
- one event attempts an all-axis request packet,
- global synchrony is not blocked.

## Explicit non-permissions

A passing request packet still grants no permission to:

- execute a dry-run simulation now,
- execute dry-run apply,
- execute live apply,
- create checkpoint files,
- create rollback files,
- write audit files,
- mutate emotion or hormone state,
- mutate runtime state,
- write memory or self-model state,
- enable persistence,
- enable runtime mapping by default,
- enable enforcement by default,
- read or load vectors,
- create or stage artifacts,
- bypass AGP,
- bypass fallback,
- schedule autonomous runtime work.

## Followup requirements

Before any future simulation, a separate explicit operator simulation request round must revalidate the source decision packet, source trace, read-only simulation scope, and no-write paths.

Before any apply, a separate explicit operator apply authorization round must review checkpoint/rollback design, validator/gate status, dry-run apply plan, focused/full tests, and artifact safety.

## Operator report

Command:

```bash
python scripts/operator_request_round941_960_affect_future_dryrun_simulation.py
```

The command emits compact JSON for the request summary, schema summary, passing and non-request decision outcomes, fail-closed proofs, compatibility proofs, non-permission proofs, and exactly one next implementation recommendation. It is read-only and writes no artifacts.

## Recommended next implementation step

Add a separate read-only future simulation runner that consumes this request packet without apply or artifact writes.
