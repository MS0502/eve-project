# Round861-880 affect reviewed-payload dry-run bridge

## Scope

Round861-880 adds a pure read-only bridge from Round841-860 operator-review handoff packets to an explicit dry-run preflight request shape. The bridge is a preflight/test surface only: it does not apply emotion transitions, does not mutate hormone state, does not write memory, does not enable persistence, does not read or load vectors, and does not bypass AGP or fallback.

## Bridge module

The bridge lives in `adapters/affect_reviewed_payload_dryrun_bridge.py` and exposes:

- `build_dryrun_preflight_from_operator_handoff(event_category, proposed_axis_deltas, metadata=None)`
- `build_dryrun_preflight_from_review_packet(review_packet, metadata=None)`
- `affect_reviewed_payload_dryrun_bridge_summary()`

A successful bridge result means only `dryrun_preflight_eligible=True`. It explicitly keeps:

- `dryrun_apply_allowed=False`
- `live_apply_allowed=False`
- `apply_permission_granted=False`
- `runtime_mutation_performed=False`
- `state_mutation_performed=False`
- `memory_write_performed=False`
- `persistence_write_performed=False`
- `vector_read_performed=False`
- `vector_load_performed=False`
- `artifact_created_or_staged=False`
- `agp_bypass_allowed=False`
- `fallback_bypass_allowed=False`

## Preflight request shape

The bridge emits a detached `preflight_request` with:

- `request_version`
- `event_category`
- `transition_payload`
- `review_packet_summary`
- `operator_review_ready=True`
- `operator_review_required=True`
- `operator_review_recorded`
- `dryrun_preflight_only=True`
- `dryrun_apply_requested=False`
- `live_apply_requested=False`
- `runtime_mutation_requested=False`
- `persistence_write_requested=False`
- `memory_write_requested=False`
- `vector_read_requested=False`
- `vector_load_requested=False`
- `agp_bypass_requested=False`
- `fallback_bypass_requested=False`
- `hardware_non_panic_preserved`
- `global_synchrony_blocked`
- `notes`

## Safety rules

The bridge fails closed when:

- handoff failed or the review packet is missing
- `operator_review_ready` is false
- `operator_review_required` is not true
- `apply_permission_granted` is true
- `dryrun_apply_allowed` is true
- `live_apply_allowed` is true
- runtime mutation, persistence, memory write, vector read/load, AGP bypass, or fallback bypass is requested
- hostile social packets lose quarantine/appraisal/gate requirements
- useful criticism loses appraisal-before-memory/self-model requirements
- hardware normal is non-zero delta
- hardware low-power or below targets non-operational axes
- hardware prediction error loses diagnostic/operational-only status
- hardware polling tick creates a recursive concern loop
- speech pressure requests AGP/fallback bypass
- listening uncertainty relabels neutral input as hostile by itself
- imagination negative spiral loses scenario budget/cooldown/reality-check boundary
- memory/self update candidates lose appraisal/quarantine boundaries
- a single event targets all affect axes
- global synchrony is not blocked

## Compatibility

The bridge preserves the Round821-840 builder, Round801-820 proposal validator, Round701-720 emotion transition validator, Round721-740 gate, Round841-860 handoff, and Round741-760 dry-run apply plan distinctions. It calls the existing dry-run apply plan with `operator_authorized=False` only to prove compatibility; therefore the plan remains blocked for simulation authorization and live apply remains unavailable.

## Operator report

Run the compact read-only report with:

```bash
python scripts/operator_bridge_round861_880_affect_reviewed_payload_to_dryrun.py
```

The command prints compact JSON and does not write the suggested operator artifact path. The report contains exactly one next implementation recommendation: add an operator-authorized dry-run simulation acceptance fixture in a later round while keeping live apply permission false.
