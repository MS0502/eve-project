# Round781-800 Affect Event Proposal Map

Track: `read_only_event_to_axis_affect_proposal_map_and_hormone_interaction_matrix`

This round adds a design/registry/test surface only. It defines future event-to-axis affect proposals and hormone/affect interactions without applying transitions or mutating live runtime state.

## Event-to-axis proposal map

`adapters/affect_event_to_axis_proposal_map.py` defines the required social feedback, cognitive/neural rhythm, speech/listening, memory/self, and hardware governor event categories. Each row includes allowed bounded axis deltas, forbidden direct effects, required quarantine/appraisal/gates, operator authorization requirements for any future apply, and hard safety flags forbidding core identity writes, self-model writes, long-term memory writes, AGP/fallback bypass, persistence, vector reads, and global synchrony.

Key safety decisions:

- Hostile social events propose only bounded defense/recovery/appraisal deltas and cannot directly alter core identity, self-worth, self-model, or long-term memory.
- Hardware events are operational governor proposals only; `hardware_normal` and `hardware_polling_tick` have zero affect deltas.
- Low-power hardware states are bounded to survival/stability axes and remain non-panic.
- Speech pressure can never emit speech directly and requires AGP/fallback gates for any later output path.
- Negative imagination spirals require scenario budget, cooldown, and a reality-check boundary.
- Memory/self update candidates require appraisal and quarantine before any later long-term or self-model update path.
- No single event can trigger global synchrony or propose deltas across all axes.

## Hormone/affect interaction matrix

`adapters/affect_hormone_interaction_matrix.py` covers every Round761-780 registry axis. Each axis row defines bounded amplification/dampening relationships, opponent axes, saturation/decay/refractory requirements, global synchrony blocking, direct identity write blocking, and notes.

Required examples are encoded as read-only matrix facts:

- `social_pain` may increase `recovery_need` and `expression_inhibition`, but cannot lower `self_respect` directly.
- `threat_pressure` may increase `self_protection` and `boundary_defense`, but cannot relabel neutral input as hostile by itself.
- `curiosity_drive` may increase `novelty_seeking` and `learning_pressure`, but is bounded by `fatigue_pressure`, `overload_risk`, and `self_protection`.
- Praise-related competence/trust increases are bounded and cannot overboost `risk_tolerance` or `attachment`.
- `recovery_need` dampens `action_readiness` and `expression_pressure` rather than causing panic.
- Hardware governor interactions remain operational and non-panic.

## Operator report

Run:

```bash
python scripts/operator_report_round781_800_affect_event_proposal_map.py
```

The command emits compact JSON with event-map summaries, group safety summaries, matrix summary, anti-global-synchrony proof, hardware non-panic proof, AGP/fallback non-bypass proof, no-runtime-mutation proof, no-persistence proof, no-memory-write proof, no-vector-read/load proof, no forbidden artifact staging proof, and exactly one next implementation recommendation.

## Next recommendation

`add_a_read_only_event_proposal_validator_against_transition_payloads_before_any_apply_round`
