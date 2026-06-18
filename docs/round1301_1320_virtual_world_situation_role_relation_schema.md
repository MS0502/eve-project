# Round1301-1320: Read-Only Virtual World Situation Role Relation Schema

## Purpose

This round adds `read_only_virtual_world_situation_role_relation_schema`, a deterministic schema for temporary semantic roles and relations inside a non-visual virtual situation. The schema describes internal candidates such as an agent waiting for an event, searching for a target, communicating with another agent, teaching, learning, or an entity participating in a situation.

A temporary semantic relation is not a persistent relationship state. It is only a candidate relation inside a virtual situation and does not assert that a real-world relationship exists, changed, or should be stored.

## Supported Relation Types

- `entity_participates_in_situation`
- `agent_performs_activity`
- `agent_waits_for_event`
- `agent_searches_for_target`
- `agent_observes_entity_candidate`
- `agent_communicates_with_agent`
- `agent_cooperates_with_agent`
- `agent_competes_with_agent`
- `agent_teaches_agent`
- `agent_learns_from_agent`
- `agent_uses_object_candidate`
- `entity_associated_with_entity_candidate`
- `symbolic_relation_candidate`
- `dmn_relation_candidate`
- `simulation_relation_candidate`
- `dream_relation_candidate`
- `mixed_unknown_relation_candidate`

Symbolic, DMN, simulation, dream, and mixed relations remain candidate-only internal relations. Mixed virtual/external boundaries explicitly block external assertions.

## Entity Structure

Each subject and object entity must be a dictionary with non-empty string fields:

```json
{
  "entity_id": "non-empty string",
  "entity_type": "non-empty string",
  "label": "non-empty string"
}
```

Literal Korean labels are preserved, including `민석`, but labels never become verified real-person identity assertions.

## Fail-Closed Behavior

The builder and validator fail closed for missing or unknown relation types, unknown boundary classes, unknown confidence states, malformed subject or object entities, malformed situation context, same-entity two-agent relations, and forbidden mutation or assertion requests.

Waiting and searching never assert future outcomes. Observing never asserts that an external observation occurred. Communicating, cooperating, and competing never update persistent relationship state. Teaching never asserts completed knowledge transfer, and learning never writes memory.

## Deterministic ID Generation

`relation_id` is built from normalized semantic input: relation type, boundary classification, confidence state, subject entity, object entity, situation context, metadata, and schema version. The implementation serializes canonical JSON with sorted keys and compact separators, preserves Korean text with `ensure_ascii=False`, then hashes the bytes with `hashlib.sha256`.

The same semantic input produces the same ID. Reordered dictionary keys do not change the ID. Different semantic input changes the ID. Validator recomputation detects tampering.

The schema does not use UUIDs, random sampling, current time, object identity, `hash()`, process state, external models, devices, vectors, or persistence.

## Immutable Read-Only Invariants

The relation is always candidate-only and situation-only. It never asserts external facts, current external facts, real-world state, identity, persistent relationship state, user emotion, user intent, or memory facts.

It never performs or allows memory writes, quarantine promotion, self-model updates, affect transitions, hormone transitions, world-state mutation, transition application, planning execution, tool execution, runtime mutation, persistence writes, vector reads or loads, model loading, device activation, artifact creation, AGP bypass, or fallback bypass.

Required review and gate flags remain true: situation, snapshot, transition, memory, quarantine, origin/fact status, appraisal, and AGP input review are all required.

## Candidate-Only Downstream Plans

Plan builders produce candidate-only, read-only plans for situation, snapshot, transition preflight, memory candidate, appraisal, and AGP input handoff. These plans do not mutate memory, relationships, self-model, affect, hormones, world state, persistence, vectors, models, devices, tools, AGP, fallback, or artifacts.

## Prohibited Side Effects

This round adds no runtime mutation path, no memory write, no self-model update, no persistent relationship update, no affect or hormone transition, no persistence enablement, no vector/vocab/subset loading, no OCR/STT/model/camera/microphone/device activation, no planning or tool execution, and no AGP or fallback bypass.

## Recommended Next Implementation Step

`read_only_virtual_world_situation_temporal_context_schema`
