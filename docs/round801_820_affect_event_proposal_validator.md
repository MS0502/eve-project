# Round801-820 affect event proposal validator

Track: `read_only_event_proposal_validator_against_transition_payloads_before_any_apply_round`

## Scope

Round801-820 adds a pure read-only validator for event-derived affect/hormone
axis proposals before any future emotion or hormone apply round. The validator
checks event category, proposed axis deltas, proposal-map constraints,
interaction-matrix constraints, registry axis definitions, transition payload
safety fields, hardware non-panic rules, AGP/fallback non-bypass rules, and
quarantine/appraisal/gate/operator-authorization requirements.

## Files

- `adapters/affect_event_proposal_validator.py`
- `scripts/operator_validate_round801_820_affect_event_proposals.py`
- `tests/test_v3_round801_820_affect_event_proposal_validator.py`
- `docs/round801_820_affect_event_proposal_validator.md`

## Validator rules

The validator fails closed for unknown event categories. Proposed deltas must
reference Round761-780 registry axes, remain a subset of the Round781-800 event
proposal map, avoid forbidden direct effects, stay within each event's
`max_delta_per_axis`, and avoid all-axis/global-synchrony patterns.

Social feedback and memory/self candidates preserve quarantine, appraisal, gate,
and operator-authorization requirements. Hostile social events cannot directly
modify core identity, self-model, self-worth, or long-term memory. Useful
criticism requires appraisal before memory/self-model updates. Praise cannot
overboost attachment or risk tolerance.

Hardware events remain operational governor proposals only. `hardware_normal`
passes only with zero affect deltas. Low-power and lower hardware proposals stay
on non-panic operational axes. Hardware prediction errors remain diagnostic and
operational, not existential or identity threats. Hardware polling ticks cannot
create recursive concern loops.

Speech/listening proposals cannot bypass AGP/fallback or directly emit speech.
Listening uncertainty cannot relabel neutral input as hostile by itself.
Imagination negative spiral proposals require scenario budget, cooldown, and a
reality-check boundary.

## Read-only guarantees

Validation success does **not** imply live mutation permission, memory write
permission, persistence permission, runtime-mapping permission, enforcement
permission, speech emission permission, AGP bypass, or fallback bypass. The
module performs no file writes, no vector reads/loads, no runtime route
registration, no persistence enablement, no memory writes, and no live emotion or
hormone state mutation.

## Operator command

```bash
python scripts/operator_validate_round801_820_affect_event_proposals.py
```

The command emits compact JSON with validator summary, valid and blocked sample
results, social/cognitive/speech/memory/hardware safety summaries,
interaction-matrix proof, hardware non-panic proof, anti-global-synchrony proof,
AGP/fallback non-bypass proof, no mutation/persistence/memory/vector/artifact
proofs, validation command list, and exactly one next implementation
recommendation.

## Recommended next step

`add_an_operator_review_payload_handoff_that_consumes_validator_passed_reports_without_applying_emotion_or_hormone_transitions`
