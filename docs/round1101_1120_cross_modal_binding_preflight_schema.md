# Round1101-1120: Read-Only Cross-Modal Binding Preflight Schema

## Overview
This document introduces the **Read-Only Cross-Modal Binding Preflight Schema**. The purpose of this schema is to evaluate whether a multimodal event candidate is eligible for a future cross-modal binding attempt. This step ensures that candidates with uncertain origins, privacy risks, confidence issues, or mismatched boundaries are correctly halted before actual binding or memory adjustments occur.

This is a **preflight/schema/design/test-surface only** round. It explicitly does not:
- Implement actual cross-modal binding.
- Resolve identity.
- Merge observations into memory.
- Update world states or global states.
- Create final events or assert facts.
- Perform any mutations or external actions.

## Supported Binding Hypothesis Types
- `temporal_alignment_hypothesis`
- `spatial_alignment_hypothesis`
- `semantic_alignment_hypothesis`
- `source_continuity_hypothesis`
- `transcript_reference_hypothesis`
- `audio_visual_context_hypothesis`
- `internal_symbolic_alignment_hypothesis`
- `memory_replay_alignment_hypothesis`

## Supported Preflight Decisions
- `eligible_for_future_binding`
- `blocked_mixed_boundary`
- `blocked_identity_resolution_required`
- `blocked_low_confidence`
- `blocked_privacy_risk`
- `blocked_origin_conflict`
- `blocked_fact_status_conflict`
- `blocked_memory_replay_as_external`
- `blocked_virtual_as_external`
- `blocked_imagination_as_fact`
- `blocked_simulation_as_current_fact`
- `blocked_unknown_candidate`

## Key Behaviors and Safeguards
- Unknown or empty candidates fail closed.
- Unknown hypothesis types fail closed.
- Mixed-boundary candidates are blocked unless explicitly mapped as `future_separated`.
- Cases requiring identity resolution block binding.
- Memory replays, internal virtual states, imagination, simulations, and DMN contents are aggressively blocked from becoming asserted external facts.
- Privacy risk and low-confidence conditions block future binding attempts.
- **Assertion and Mutation Flags**: All flags regarding bindings performed, identities asserted, models loaded, memory mutated, or external devices activated *must* strictly remain `False`.

## Next Implementation Recommendation
`read_only_memory_replay_observation_schema`