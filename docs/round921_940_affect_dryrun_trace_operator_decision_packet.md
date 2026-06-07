# Round921-940 affect dry-run trace operator decision packet

Round921-940 adds `adapters/affect_dryrun_trace_operator_decision_packet.py` as a pure read-only operator decision packet surface for existing Round901-920 affect checkpoint/rollback execution dry-run traces.

## Scope

The packet is review-decision data only. It does not create checkpoints, create rollback records, write audits, stage operator artifacts, apply emotion transitions, apply hormone transitions, mutate memory, enable persistence, enable runtime mapping, enable enforcement, read/load vectors, bypass AGP, or bypass fallback.

## Supported operator decisions

- `approve_for_future_dryrun_simulation`
- `reject`
- `hold_for_review`
- `request_revision`

`approve_for_future_dryrun_simulation` only records that an operator may later request another read-only dry-run simulation. It is not dry-run apply permission and is not live apply permission.

## Packet schema summary

Each packet includes:

- `packet_version`
- `event_category`
- `operator_decision`
- `decision_reason`
- `dryrun_trace_summary`
- `preflight_validation_sequence_summary`
- `checkpoint_capture_sequence_summary`
- `bounded_apply_candidate_sequence_summary`
- `rollback_restore_sequence_summary`
- `audit_event_sequence_summary`
- `failure_handling_sequence_summary`
- `guard_preservation_sequence_summary`
- `approval_scope`
- `explicit_non_permissions`
- `required_followup_before_any_apply`
- `deterministic_sequence_id_policy`
- `integrity_hash_policy`
- `no_write_guarantee`
- `no_apply_guarantee`
- `notes`

## Decision safety rules

The packet fails closed for unknown event categories and invalid operator decisions. Trace failure blocks `approve_for_future_dryrun_simulation`. All decisions keep `dryrun_apply_allowed`, `live_apply_allowed`, `apply_permission_granted`, checkpoint creation, rollback creation, audit writing, runtime mutation, memory writes, persistence writes, vector reads/loads, artifact staging, AGP bypass, and fallback bypass false.

Social feedback packets preserve quarantine, appraisal, and gate requirements. Useful-criticism packets preserve appraisal-before-memory/self-model requirements. Speech pressure packets preserve AGP/fallback gates. Listening uncertainty cannot relabel neutral input as hostile by itself. Imagination negative spiral packets preserve scenario budget, cooldown, and reality-check boundaries. Memory/self update candidates preserve appraisal/quarantine. Hardware normal remains zero-delta; low-power-and-below hardware packets remain non-panic operational-only; hardware prediction error remains diagnostic/operational only; hardware polling tick cannot create a recursive concern loop.

## Compatibility surfaces

The operator report command summarizes compatibility with:

- Round901-920 execution dry-run trace
- Round881-900 checkpoint/rollback plan
- Round861-880 dry-run bridge
- Round841-860 handoff
- Round821-840 payload builder
- Round801-820 proposal validator
- Round701-720 emotion transition validator
- Round721-740 emotion transition gate
- Round741-760 dry-run apply plan

## Operator command

```bash
python scripts/operator_decision_round921_940_affect_dryrun_trace_packet.py
```

The command emits compact JSON only and writes no files.

## Recommended next step

Exactly one recommended next implementation step is included in the operator report: add an operator-reviewed dry-run simulation request intake schema in a later round while keeping all apply/write/mutation/vector permissions disabled until separate explicit authorization.
