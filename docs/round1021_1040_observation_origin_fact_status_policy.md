# Round1021-1040 read_only_observation_origin_and_fact_status_policy

## Track

`read_only_observation_origin_and_fact_status_policy`

## Scope

Round1021-1040 implements a strictly read-only policy for distinguishing the origin and fact status of observations. Before adding additional sensory or virtual world schemas, EVE must firmly demarcate external reality from virtual reality, imagination, simulation, memory replay, and internal states. This policy establishes the structural boundary to prevent memory contamination and hallucinatory fact assertion.

## Supported origins

- `external_visual`
- `external_audio`
- `screen_visual`
- `tool_state`
- `internal_state`
- `virtual_world_visual`
- `memory_replay`
- `imagination`
- `simulation`
- `dream_dmn`

## Supported fact statuses

- `observed_external`
- `observed_internal_virtual`
- `reconstructed_memory`
- `imagined_candidate`
- `simulated_future`
- `symbolic_visualization`

## Guards enforced

- `external_fact_assertion_guard`: Blocks asserting external facts from internal origins like imagination or memory.
- `virtual_world_reality_boundary_guard`: Ensures virtual world observations are properly classified as internal virtual facts.
- `memory_replay_provenance_guard`: Enforces reconstructed memory status for memory replays.
- `imagination_memory_contamination_guard`: Ensures imaginations are candidate only.
- `simulation_result_boundary_guard`: Ensures simulations are treated as simulated futures.
- `dream_dmn_boundary_guard`: Treats DMN traces as symbolic visualizations.

## Non-permissions

The schema preserves the read-only invariants and ensures the following remain strictly `False` for any origin and fact status combination:

- `memory_write_performed`
- `self_model_update_allowed`
- `affect_transition_allowed`
- `hormone_transition_allowed`
- `runtime_mutation_performed`
- `persistence_write_performed`
- `vector_read_performed`
- `vector_load_performed`
- `artifact_created_or_staged`
- `agp_bypass_allowed`
- `fallback_bypass_allowed`

Additionally, `external_fact_asserted` is explicitly kept `False` unless the origin and fact status legally permit it (e.g., `external_visual` + `observed_external`).

## Operator command

```bash
python scripts/operator_report_round1021_1040_observation_origin_fact_status_policy.py
```

The command evaluates the guards, enforces invariant non-permissions, checks for forbidden artifact creation via Git, and emits a compact JSON proof payload.

## Next implementation recommendation

Add a read-only virtual visual observation schema to support internal avatar states without confusing them with real physical states.
