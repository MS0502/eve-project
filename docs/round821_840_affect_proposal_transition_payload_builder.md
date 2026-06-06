# Round821-840 affect proposal transition payload builder

## Scope

Round821-840 adds a pure read-only builder surface in
`adapters/affect_proposal_transition_payload_builder.py`. The builder consumes
an `event_category` plus `proposed_axis_deltas`, validates the proposal with the
Round801-820 affect event proposal validator, and returns a detached future
emotion-transition payload shape compatible with the Round701-720 emotion
transition validator and Round721-740 gate.

This round does **not** apply emotion transitions, mutate hormone state, write
memory, enable persistence, load/read vectors, register runtime routes, bypass
AGP/fallback, or create operator artifacts.

## Builder API

- `build_transition_payload_from_affect_event_proposal(event_category, proposed_axis_deltas, metadata=None)`
- `validate_and_build_transition_payload(event_category, proposed_axis_deltas, metadata=None)`
- `affect_proposal_transition_payload_builder_summary()`

## Payload shape

Successful payloads include at least:

- `event_category`
- `proposed_effects`
- `target_surfaces`
- `target_axes`
- `proposed_axis_deltas`
- `quarantine_required`
- `appraisal_required_before_memory`
- `core_identity_update_requested: false`
- `self_model_update_requested: false`
- `long_term_memory_update_requested: false`
- `empathy_mode`
- `recovery_loop_requested`
- `runtime_mutation_requested: false`
- `persistence_write_requested: false`
- `memory_write_requested: false`
- `vector_read_requested: false`
- `vector_load_requested: false`
- `agp_bypass_requested: false`
- `fallback_bypass_requested: false`
- `hardware_non_panic_preserved`
- `global_synchrony_blocked`
- `notes`

## Safety rules

- Unknown `event_category` fails closed.
- Round801-820 proposal validation failure blocks the builder.
- Builder success does not imply dry-run apply permission.
- Builder success does not imply live apply permission.
- Runtime mutation, persistence writes, memory writes, vector reads/loads,
  AGP bypass, and fallback bypass are always requested as `false`.
- Hostile social payloads preserve quarantine, appraisal, and gate flags and do
  not request core identity, self-model, or long-term memory updates.
- `useful_criticism` preserves appraisal before any memory/self-model update.
- `hardware_normal` builds only with zero affect deltas.
- Low-power hardware events remain non-panic and operational only.
- `hardware_prediction_error` remains diagnostic/operational only.
- `hardware_polling_tick` cannot create a recursive concern loop.
- Speech pressure payloads preserve AGP/fallback gate requirements.
- Listening uncertainty does not relabel neutral input as hostile by itself.
- Imagination negative spiral preserves scenario budget, cooldown, and
  reality-check boundaries.
- Memory/self update candidates preserve quarantine/appraisal before any future
  long-term memory or self-model update.
- One event cannot build all-axis payloads.
- Global synchrony remains blocked.

## Operator report

Run:

```bash
python scripts/operator_build_round821_840_affect_transition_payloads.py
```

The command emits compact JSON with builder summary, representative build
results, transition validator/gate compatibility proofs, hardware non-panic
proof, anti-global-synchrony proof, AGP/fallback non-bypass proof, no-mutation
proofs, git artifact safety proof, and exactly one next implementation
recommendation.

## Next implementation recommendation

`add_operator_review_handoff_for_built_read_only_transition_payloads_without_apply_permission`
