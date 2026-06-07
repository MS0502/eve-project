# Round1081-1100: Read-Only Multimodal Event Candidate Contract

## Overview
This round introduces the **Read-Only Multimodal Event Candidate Contract**. The purpose of this contract is to group different observations into cross-modal event candidates, representing possible multi-modal events without modifying global state or enforcing reality assertions.

This is a **contract/schema/design/test-surface only** round. It does not implement actual cross-modal binding, resolve identity, merge observations into memory, or update the world state.

## Supported Observation Modalities
- `sensory`
- `visual_external`
- `visual_internal_virtual`
- `auditory`
- `tool_state`
- `internal_state`

## Supported Candidate Link Types
- `temporal_cooccurrence_candidate`
- `spatial_context_candidate`
- `semantic_relatedness_candidate`
- `source_continuity_candidate`
- `attention_focus_candidate`
- `transcript_visual_reference_candidate`
- `audio_visual_context_candidate`
- `internal_symbolic_reference_candidate`

## Supported Candidate Types
- `external_event_candidate`
- `internal_virtual_event_candidate`
- `memory_replay_event_candidate`
- `imagination_event_candidate`
- `simulation_event_candidate`
- `dmn_symbolic_event_candidate`
- `mixed_boundary_candidate`

## Key Behaviors and Safeguards
- Empty observations fail closed.
- Unknown modality, link type, or candidate type fail closed.
- `visual_external` + `auditory` may only become candidate, not asserted fact.
- `visual_internal_virtual` + `auditory` must become `mixed_boundary_candidate` unless explicitly separated.
- DMN inner voice + `visual_internal_virtual` must remain `dmn_symbolic_event_candidate`.
- Memory replay must not become external fact.
- Imagination must not become memory fact.
- Simulation must not become current fact.
- Mixed boundary creates uncertainty and boundary flags.
- **Assertion Flags**: All assertions (identity, user emotion, relationships, memory facts, cross-modal bindings) must remain `False`.
- **Mutation Flags**: All hardware activations, model loads, state mutations, memory writes, persistence writes, etc. must remain `False`.
- **AGP/Fallback**: All bypasses remain `False`.
- One multimodal candidate cannot produce all-axis affect/hormone activation.
- Global synchrony remains blocked.

## Next Implementation Recommendation
`read_only_cross_modal_binding_preflight_schema`
