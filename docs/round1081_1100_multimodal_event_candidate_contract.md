# Round1081-1100: Read-Only Multimodal Event Candidate Contract

## 1. Overview
EVE v3.1 requires a structured, safe boundary for processing multiple observations together (e.g. visual + auditory) to build a multimodal event candidate. This contract defines how observations from different modalities are safely gathered into a candidate event, enforcing read-only boundaries and failing closed on unsafe combinations.

This round is pure data, schema, and design testing only. **It does not** implement actual cross-modal binding, identity resolution, state mutations, memory writes, or vector loads.

## 2. Supported Modalities
- sensory
- visual_external
- visual_internal_virtual
- auditory
- tool_state
- internal_state

## 3. Supported Candidate Link Types
- temporal_cooccurrence_candidate
- spatial_context_candidate
- semantic_relatedness_candidate
- source_continuity_candidate
- attention_focus_candidate
- transcript_visual_reference_candidate
- audio_visual_context_candidate
- internal_symbolic_reference_candidate

## 4. Supported Event Candidate Types
- external_event_candidate
- internal_virtual_event_candidate
- memory_replay_event_candidate
- imagination_event_candidate
- simulation_event_candidate
- dmn_symbolic_event_candidate
- mixed_boundary_candidate

## 5. Key Invariants & Safeguards
1. **Empty / Unknown Check**: Empty observations, unknown modalities, and unknown candidate link types will immediately fail-closed and will not construct a candidate.
2. **Boundary Restrictions**:
   - External visual + auditory combinations only become `external_event_candidate`, and do NOT assert as fact.
   - Virtual visual + auditory external, or other cross-world boundaries, are immediately flagged as `mixed_boundary_candidate` and are tagged with uncertainty and isolation flags.
   - Internal boundaries (memory_replay, imagination, simulation) map safely to their respective candidate boundaries and are NEVER asserted as external, memory facts, or current reality.
3. **Identity Assertions are Always False**: `identity_asserted`, `speaker_identity_asserted`, and `person_identity_asserted` are strictly enforced to `False`.
4. **No Mutation**: `memory_write_performed`, `affect_transition_allowed`, `hormone_transition_allowed`, `vector_load_performed`, and `agp_bypass_allowed` remain `False`.
5. **No Actual Processing**: `cross_modal_binding_performed`, `identity_resolution_performed`, `ocr_performed`, and `model_loaded` are `False`. This schema represents the candidate plan to be processed later.
