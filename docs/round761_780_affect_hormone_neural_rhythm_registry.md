# Round761-780 Affect/Hormone/Neural-Rhythm Registry

Track:
`full_read_only_affect_hormone_neural_rhythm_registry_with_hardware_governor_non_panic_policy`

Round761-780 adds a read-only constitution/design and registry surface for EVE's
always-on multi-rhythm internal activation architecture.  The implementation is
pure data and pure functions only.

## Constitution/design update summary

The EVE v3 constitution now includes **Always-On Multi-Rhythm Neural Activation
and Affect Governance**.  The section establishes that EVE has always-on
neural/concept activation rhythms, while explicitly rejecting globally
synchronized panic.  Thought, imagination, listening, speaking, memory, affect,
self-model, and hardware governor rhythms are separate systems with their own
cycle class, decay, refractory, evidence threshold, and safety boundary.

## Full affect/hormone axis registry summary

The registry lives in `adapters/affect_hormone_neural_rhythm_registry.py` and
contains 37 axes across six groups:

- survival/stability: `energy_budget`, `fatigue_pressure`, `recovery_need`,
  `stress_load`, `stability_need`, `overload_risk`
- risk/defense: `threat_pressure`, `uncertainty_pressure`, `self_protection`,
  `boundary_defense`, `trust_risk`, `exposure_risk`
- social/relationship: `social_pain`, `social_trust`, `attachment`,
  `care_drive`, `loneliness_pressure`, `belonging_need`,
  `rejection_sensitivity`
- learning/exploration: `curiosity_drive`, `novelty_seeking`,
  `learning_pressure`, `memory_consolidation_pressure`,
  `prediction_error_pressure`, `competence_drive`
- self/identity: `self_coherence`, `self_respect`, `identity_integrity`,
  `agency_pressure`, `autonomy_drive`, `purpose_alignment`
- expression/action: `expression_pressure`, `expression_inhibition`,
  `action_readiness`, `risk_tolerance`, `patience_level`, `conflict_avoidance`

Every axis declares its baseline/default/min/max, decay rate, spike limit,
saturation policy, refractory ticks, evidence requirement, rhythm class, cycle
class, modulation targets, activation-pattern inputs, quarantine/appraisal
requirements, hardware direct-input policy, and hard false permissions for panic,
AGP bypass, fallback bypass, persistence writes, vector reads, raw feedback
self-model rewrite, raw feedback memory write, and core identity rewrite.

## Multi-timescale rhythm schema summary

The schema defines ten rhythms without scheduling them:

- `activation_tick`: concept activation, inhibition, recency, co-activation
- `attention_tick`: focus selection, salience arbitration
- `listening_tick`: syllable/token/intent/social-signal integration
- `inner_speech_tick`: verbal thought and reflection fragments
- `imagination_tick`: scenario simulation, counterfactuals, future rehearsal,
  DMN-like processing
- `affect_tick`: bounded emotion/hormone proposal readiness
- `recovery_tick`: decay, baseline return, cooldown, refractory release
- `memory_tick`: consolidation, forgetting, importance weighting
- `self_model_tick`: identity coherence, values, long-term narrative stability
- `hardware_tick`: battery/thermal/memory/storage operational governor only

## Connection map summary

The connection map links thought, imagination, speech, listening, memory, and
action to activation-pattern inputs.  Speaking requires AGP and fallback.
Listening may request extra appraisal under uncertainty or threat, but cannot
relabel neutral input as hostile by threat pressure alone.  Imagination has a
scenario budget, cooldown, and reality-check boundary.  Social memory update
requires quarantine and appraisal.

## Hardware governor non-panic policy summary

Battery bands are operational governor bands only:

- `battery >= 50`: normal, no affect effect, no warning
- `35 <= battery < 50`: light conserve, no affect effect, heavy background tasks
  may defer
- `20 <= battery < 35`: conserve, tiny fatigue pressure only, reduce background
  cadence
- `10 <= battery < 20`: low power, mild recovery need only, no panic, checkpoint
  recommended
- `5 <= battery < 10`: critical prepare, operational caution only, graceful pause
  preparation
- `battery < 5`: shutdown imminent, no panic/no death framing, graceful pause only

Battery drop alone cannot trigger death fear, identity threat, social pain,
abandonment fear, self-worth change, or panic.  Hardware prediction error creates
diagnostic flags, not existential threat.  Hardware polling cannot create
recursive concern loops and must require hysteresis, debounce, trend windows, and
rate limiting before any future live governor work.

## Anti-global-synchrony safety summary

Always-on activation does not synchronize all axes.  One event cannot spike all
axes.  Every axis requires saturation, decay, refractory cooldown, evidence
thresholds, and baseline return.  Activation patterns can modulate proposal
readiness but cannot directly mutate live state, bypass AGP, bypass fallback, or
write persistence/memory/vector state.

## Operator command/report path

Run:

```bash
python scripts/operator_report_round761_780_affect_hormone_neural_rhythm_registry.py
```

The command emits compact JSON only.  It does not write files, create artifacts,
read vector contents, load vectors, poll hardware, mutate runtime state, or stage
operator artifacts.

## Exactly one next implementation recommendation

Add a read-only affect proposal validator against this registry before any
transition apply round.
