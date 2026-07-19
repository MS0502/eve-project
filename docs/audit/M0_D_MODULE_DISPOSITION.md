# M0-D Module Disposition

Baseline: `main` at `fe10cd954bdf445400ea6aa9708dd214ed761114`

Status: recommendations only. No module is deleted, deprecated in code, wrapped, rewritten, activated, or retired by M0-D.

## Regeneration

```bash
python scripts/audit/m0_d_component_inventory.py --pretty
```

The canonical JSON assigns every scanned runtime module exactly one of `KEEP`, `WRAP`, `REWRITE`, `EXPERIMENTAL`, `DEPRECATE`, or `REMOVE`. Each row below includes a repository-relative `file:line` citation to M0-A/B/C evidence or new M0-D AST evidence, plus confidence and unresolved status. A recommendation is not an implementation action.

## Disposition policy

- `KEEP`: retain current behavior/evidence while v4 ownership and contracts are assigned.
- `WRAP`: preserve behind explicit capability, lifecycle, provenance, validation, compatibility, or rollback boundaries.
- `REWRITE`: preserve required capability and tests while replacing architecture that directly conflicts with v4.
- `EXPERIMENTAL`: retain as non-authoritative evidence with no production promotion or default activation.
- `DEPRECATE`: preserve historical/migration evidence while excluding future runtime authority.
- `REMOVE`: recommend deletion only with positive high-confidence evidence and reviewer approval. Lack of reachability, a name, or a docstring is insufficient.

## Validated totals

```text
KEEP: 30
WRAP: 78
REWRITE: 6
EXPERIMENTAL: 172
DEPRECATE: 2
REMOVE: 0
TOTAL: 288
resolved recommendations: 10
unresolved recommendations requiring reviewer ruling: 278
```

The initial mechanical pass produced 63 `REWRITE` recommendations because any reachable hormone reference triggered rewrite. That was rejected as over-aggressive. Automatic hormone coupling now yields `WRAP/unresolved`; only six manually evidenced architecture conflicts remain `REWRITE`. `REMOVE` remains empty.

## Complete evidence matrix

| Module | Disposition | Primary evidence | Confidence | UNRESOLVED |
|---|---|---|---|---|
| `active_inference.py` | `WRAP` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `active_inference.py:108`; `mutation`; attribute_assignment | `medium` | `YES` |
| `adapters/__init__.py` | `EXPERIMENTAL` | `adapters/__init__.py:1`; tracked Python module, AST parse/reachability result `reachable_from_active_root=False; references=0; component_evidence=0` | `low` | `YES` |
| `adapters/activation_adapter.py` | `WRAP` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/activation_adapter.py:29`; `mutation`; attribute_assignment | `medium` | `YES` |
| `adapters/affect_dryrun_trace_operator_decision_packet.py` | `EXPERIMENTAL` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/affect_dryrun_trace_operator_decision_packet.py:180`; `mutation`; mutation_method=set | `medium` | `YES` |
| `adapters/affect_event_proposal_validator.py` | `EXPERIMENTAL` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/affect_event_proposal_validator.py:89`; `mutation`; mutation_method=set | `medium` | `YES` |
| `adapters/affect_event_to_axis_proposal_map.py` | `EXPERIMENTAL` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/affect_event_to_axis_proposal_map.py:95`; `mutation`; mutation_method=set | `medium` | `YES` |
| `adapters/affect_future_dryrun_simulation_request_packet.py` | `EXPERIMENTAL` | `m0_d_component_inventory.py` → `adapters/affect_future_dryrun_simulation_request_packet.py:45`; `Assign`; representation_state_assignment=FEATURE_TRACK tokens=feature | `medium` | `YES` |
| `adapters/affect_future_dryrun_simulation_runner_preflight.py` | `EXPERIMENTAL` | `m0_d_component_inventory.py` → `adapters/affect_future_dryrun_simulation_runner_preflight.py:43`; `Assign`; representation_state_assignment=FEATURE_TRACK tokens=feature | `medium` | `YES` |
| `adapters/affect_hormone_interaction_matrix.py` | `EXPERIMENTAL` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/affect_hormone_interaction_matrix.py:116`; `mutation`; mutation_method=base.update | `medium` | `YES` |
| `adapters/affect_hormone_neural_rhythm_registry.py` | `EXPERIMENTAL` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/affect_hormone_neural_rhythm_registry.py:250`; `mutation`; subscript_assignment | `medium` | `YES` |
| `adapters/affect_proposal_transition_payload_builder.py` | `EXPERIMENTAL` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/affect_proposal_transition_payload_builder.py:87`; `mutation`; mutation_method=groups.append | `medium` | `YES` |
| `adapters/affect_reviewed_payload_dryrun_bridge.py` | `EXPERIMENTAL` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/affect_reviewed_payload_dryrun_bridge.py:90`; `mutation`; mutation_method=set | `medium` | `YES` |
| `adapters/affect_transition_checkpoint_rollback_dryrun_trace.py` | `EXPERIMENTAL` | `m0_d_component_inventory.py` → `adapters/affect_transition_checkpoint_rollback_dryrun_trace.py:42`; `Assign`; representation_state_assignment=FEATURE_TRACK tokens=feature | `medium` | `YES` |
| `adapters/affect_transition_checkpoint_rollback_plan.py` | `EXPERIMENTAL` | `m0_d_component_inventory.py` → `adapters/affect_transition_checkpoint_rollback_plan.py:42`; `Assign`; representation_state_assignment=FEATURE_TRACK tokens=feature | `medium` | `YES` |
| `adapters/affect_transition_payload_operator_handoff.py` | `EXPERIMENTAL` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/affect_transition_payload_operator_handoff.py:81`; `mutation`; mutation_method=set | `medium` | `YES` |
| `adapters/agency_adapter.py` | `WRAP` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/agency_adapter.py:29`; `mutation`; attribute_assignment | `medium` | `YES` |
| `adapters/agp_adapter.py` | `WRAP` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/agp_adapter.py:102`; `mutation`; attribute_assignment | `medium` | `YES` |
| `adapters/agp_operational_snapshot.py` | `EXPERIMENTAL` | `M0_C_PERSISTENCE_AND_STATE_MAP.md` → `adapters/agp_operational_snapshot.py:78`; `state_domain`; state_symbol=hormone_threshold;domain=affect_hormones | `low` | `YES` |
| `adapters/agp_proof_object_expansion.py` | `EXPERIMENTAL` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/agp_proof_object_expansion.py:39`; `mutation`; mutation_method=row_reasons.append | `low` | `YES` |
| `adapters/agp_threshold_decision.py` | `EXPERIMENTAL` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/agp_threshold_decision.py:96`; `mutation`; subscript_assignment | `low` | `YES` |
| `adapters/agp_trace_analyzer.py` | `WRAP` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/agp_trace_analyzer.py:47`; `mutation`; subscript_assignment | `medium` | `YES` |
| `adapters/ai_adapter.py` | `EXPERIMENTAL` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/ai_adapter.py:31`; `mutation`; attribute_assignment | `medium` | `YES` |
| `adapters/allostatic_adapter.py` | `WRAP` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/allostatic_adapter.py:28`; `mutation`; attribute_assignment | `high` | `NO` |
| `adapters/analogy_adapter.py` | `WRAP` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/analogy_adapter.py:26`; `mutation`; attribute_assignment | `medium` | `YES` |
| `adapters/apc_adapter.py` | `WRAP` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/apc_adapter.py:28`; `mutation`; attribute_assignment | `medium` | `YES` |
| `adapters/appraisal_classifier.py` | `WRAP` | `M0_C_PERSISTENCE_AND_STATE_MAP.md` → `adapters/appraisal_classifier.py:27`; `state_domain`; state_symbol=LABEL_EXTERNAL_AFFECTIVE_TONE;domain=affect_hormones | `medium` | `YES` |
| `adapters/attention_analyzer.py` | `WRAP` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/attention_analyzer.py:86`; `mutation`; attribute_assignment | `medium` | `YES` |
| `adapters/auditory_observation_schema.py` | `EXPERIMENTAL` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/auditory_observation_schema.py:122`; `mutation`; mutation_method=blocked_reasons.append | `medium` | `YES` |
| `adapters/autonomy_adapter.py` | `KEEP` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/autonomy_adapter.py:40`; `mutation`; attribute_assignment | `medium` | `YES` |
| `adapters/character_adapter.py` | `WRAP` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/character_adapter.py:67`; `mutation`; attribute_assignment | `medium` | `YES` |
| `adapters/compositor_adapter.py` | `WRAP` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/compositor_adapter.py:80`; `mutation`; attribute_assignment | `medium` | `YES` |
| `adapters/concept_memory_adapter.py` | `EXPERIMENTAL` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/concept_memory_adapter.py:44`; `mutation`; attribute_assignment | `medium` | `YES` |
| `adapters/concept_runtime_mapping_diagnostics.py` | `EXPERIMENTAL` | `M0_C_PERSISTENCE_AND_STATE_MAP.md` → `adapters/concept_runtime_mapping_diagnostics.py:50`; `state_domain`; state_symbol=vectors_written;domain=vectors | `low` | `YES` |
| `adapters/context_seed.py` | `WRAP` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/context_seed.py:29`; `mutation`; attribute_assignment | `medium` | `YES` |
| `adapters/continual_adapter.py` | `WRAP` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/continual_adapter.py:38`; `mutation`; attribute_assignment | `medium` | `YES` |
| `adapters/corpus_adapter.py` | `WRAP` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/corpus_adapter.py:31`; `mutation`; attribute_assignment | `medium` | `YES` |
| `adapters/counterfactual_adapter.py` | `WRAP` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/counterfactual_adapter.py:50`; `mutation`; attribute_assignment | `medium` | `YES` |
| `adapters/creative_adapter.py` | `WRAP` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/creative_adapter.py:29`; `mutation`; attribute_assignment | `medium` | `YES` |
| `adapters/cross_modal_binding_preflight_schema.py` | `EXPERIMENTAL` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/cross_modal_binding_preflight_schema.py:65`; `mutation`; mutation_method=blocked_reasons.append | `low` | `YES` |
| `adapters/curiosity_adapter.py` | `WRAP` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/curiosity_adapter.py:30`; `mutation`; attribute_assignment | `medium` | `YES` |
| `adapters/deep_reasoning_adapter.py` | `WRAP` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/deep_reasoning_adapter.py:23`; `mutation`; attribute_assignment | `medium` | `YES` |
| `adapters/dialogue_context_adapter.py` | `KEEP` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/dialogue_context_adapter.py:48`; `mutation`; attribute_assignment | `medium` | `YES` |
| `adapters/digital_somatic_adapter.py` | `WRAP` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/digital_somatic_adapter.py:36`; `mutation`; attribute_assignment | `medium` | `YES` |
| `adapters/dmn_adapter.py` | `WRAP` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/dmn_adapter.py:32`; `mutation`; attribute_assignment | `medium` | `YES` |
| `adapters/embedding_wrapper.py` | `EXPERIMENTAL` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/embedding_wrapper.py:28`; `mutation`; attribute_assignment | `medium` | `YES` |
| `adapters/emotion_cause_adapter.py` | `WRAP` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/emotion_cause_adapter.py:74`; `mutation`; attribute_assignment | `medium` | `YES` |
| `adapters/emotion_regulation_adapter.py` | `WRAP` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/emotion_regulation_adapter.py:31`; `mutation`; attribute_assignment | `medium` | `YES` |
| `adapters/emotion_state_transition_contract.py` | `EXPERIMENTAL` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/emotion_state_transition_contract.py:271`; `mutation`; mutation_method=set | `medium` | `YES` |
| `adapters/emotion_transition_dryrun_apply_plan.py` | `EXPERIMENTAL` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/emotion_transition_dryrun_apply_plan.py:109`; `mutation`; mutation_method=reasons.append | `medium` | `YES` |
| `adapters/emotion_transition_gate.py` | `EXPERIMENTAL` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/emotion_transition_gate.py:86`; `mutation`; mutation_method=merged.extend | `medium` | `YES` |
| `adapters/emotion_transition_validator.py` | `EXPERIMENTAL` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/emotion_transition_validator.py:78-89`; `mutation`; mutation_method=categories.setdefault | `medium` | `YES` |
| `adapters/enhancer_adapter.py` | `KEEP` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/enhancer_adapter.py:35`; `mutation`; attribute_assignment | `medium` | `YES` |
| `adapters/env_adapter.py` | `WRAP` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/env_adapter.py:39`; `mutation`; attribute_assignment | `medium` | `YES` |
| `adapters/eve_self_learning_adapter.py` | `WRAP` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/eve_self_learning_adapter.py:140`; `mutation`; attribute_assignment | `medium` | `YES` |
| `adapters/eve_vector_store.py` | `EXPERIMENTAL` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/eve_vector_store.py:35`; `mutation`; attribute_assignment | `medium` | `YES` |
| `adapters/eve_vocab_tracker.py` | `EXPERIMENTAL` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/eve_vocab_tracker.py:28`; `mutation`; attribute_assignment | `medium` | `YES` |
| `adapters/explicit_load_guard.py` | `EXPERIMENTAL` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/explicit_load_guard.py:136`; `mutation`; subscript_assignment | `medium` | `YES` |
| `adapters/external_seed_manifest.py` | `WRAP` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/external_seed_manifest.py:213`; `mutation`; mutation_method=errors.extend | `medium` | `YES` |
| `adapters/fasttext_embedding_adapter.py` | `WRAP` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/fasttext_embedding_adapter.py:116`; `mutation`; attribute_assignment | `medium` | `YES` |
| `adapters/fasttext_loader.py` | `EXPERIMENTAL` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/fasttext_loader.py:112`; `dependency_construction`; constructor_call=Path | `medium` | `YES` |
| `adapters/frame_adapter.py` | `WRAP` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/frame_adapter.py:28`; `mutation`; attribute_assignment | `medium` | `YES` |
| `adapters/goal_adapter.py` | `WRAP` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/goal_adapter.py:37`; `mutation`; attribute_assignment | `medium` | `YES` |
| `adapters/hormone_adapter.py` | `REWRITE` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/hormone_adapter.py:25`; `mutation`; attribute_assignment | `high` | `NO` |
| `adapters/humor_adapter.py` | `WRAP` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/humor_adapter.py:27`; `mutation`; attribute_assignment | `medium` | `YES` |
| `adapters/hypergraph_adapter.py` | `WRAP` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/hypergraph_adapter.py:26`; `mutation`; attribute_assignment | `medium` | `YES` |
| `adapters/integrated_self_adapter.py` | `KEEP` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/integrated_self_adapter.py:29`; `mutation`; attribute_assignment | `medium` | `YES` |
| `adapters/intent_decider.py` | `WRAP` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/intent_decider.py:38`; `mutation`; attribute_assignment | `medium` | `YES` |
| `adapters/korean/__init__.py` | `EXPERIMENTAL` | `adapters/korean/__init__.py:1`; tracked Python module, AST parse/reachability result `reachable_from_active_root=False; references=0; component_evidence=0` | `low` | `YES` |
| `adapters/korean/morph_analyzer.py` | `KEEP` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/korean/morph_analyzer.py:301`; `mutation`; attribute_assignment | `medium` | `YES` |
| `adapters/korean/sentence_structure.py` | `WRAP` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/korean/sentence_structure.py:136`; `mutation`; attribute_assignment | `medium` | `YES` |
| `adapters/korean_language_adapter.py` | `KEEP` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/korean_language_adapter.py:19`; `mutation`; attribute_assignment | `medium` | `YES` |
| `adapters/lex_concept_mapping_adapter.py` | `WRAP` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/lex_concept_mapping_adapter.py:31`; `mutation`; attribute_assignment | `medium` | `YES` |
| `adapters/live_loop.py` | `REWRITE` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/live_loop.py:36`; `mutation`; attribute_assignment | `high` | `NO` |
| `adapters/medium30k_runtime_load_integration.py` | `EXPERIMENTAL` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/medium30k_runtime_load_integration.py:60`; `mutation`; mutation_method=blockers.append | `medium` | `YES` |
| `adapters/medium_vector_manual_validation.py` | `EXPERIMENTAL` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/medium_vector_manual_validation.py:32`; `mutation`; mutation_method=data.setdefault | `medium` | `YES` |
| `adapters/medium_vector_release_restore.py` | `KEEP` | `m0_d_component_inventory.py` → `adapters/medium_vector_release_restore.py:40`; `Assign`; representation_state_assignment=INTERNAL_VECTORS_PATH tokens=vectors | `medium` | `YES` |
| `adapters/medium_vector_restoration.py` | `KEEP` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/medium_vector_restoration.py:136`; `mutation`; subscript_assignment | `medium` | `YES` |
| `adapters/memory_adapter.py` | `WRAP` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/memory_adapter.py:29`; `mutation`; attribute_assignment | `medium` | `YES` |
| `adapters/memory_provenance_quarantine_preflight_schema.py` | `EXPERIMENTAL` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/memory_provenance_quarantine_preflight_schema.py:83`; `mutation`; mutation_method=blocked_reasons.append | `low` | `YES` |
| `adapters/memory_replay_observation_schema.py` | `EXPERIMENTAL` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/memory_replay_observation_schema.py:69`; `mutation`; mutation_method=blocked_reasons.append | `low` | `YES` |
| `adapters/metacognition_adapter.py` | `WRAP` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/metacognition_adapter.py:33`; `mutation`; attribute_assignment | `medium` | `YES` |
| `adapters/module_learning_adapter.py` | `EXPERIMENTAL` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/module_learning_adapter.py:47`; `mutation`; attribute_assignment | `medium` | `YES` |
| `adapters/mood_adapter.py` | `WRAP` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/mood_adapter.py:59`; `mutation`; attribute_assignment | `medium` | `YES` |
| `adapters/multi_stream_adapter.py` | `WRAP` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/multi_stream_adapter.py:34`; `mutation`; attribute_assignment | `medium` | `YES` |
| `adapters/multimodal_event_candidate_contract.py` | `EXPERIMENTAL` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/multimodal_event_candidate_contract.py:63`; `mutation`; mutation_method=blocked_reasons.append | `low` | `YES` |
| `adapters/narrative_adapter.py` | `WRAP` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/narrative_adapter.py:30`; `mutation`; attribute_assignment | `medium` | `YES` |
| `adapters/nl_adapter.py` | `WRAP` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/nl_adapter.py:45`; `mutation`; attribute_assignment | `medium` | `YES` |
| `adapters/norm_adapter.py` | `WRAP` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/norm_adapter.py:32`; `mutation`; attribute_assignment | `medium` | `YES` |
| `adapters/observation_origin_fact_status_policy.py` | `EXPERIMENTAL` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/observation_origin_fact_status_policy.py:74`; `mutation`; subscript_assignment | `medium` | `YES` |
| `adapters/openai_server_adapter.py` | `WRAP` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/openai_server_adapter.py:67`; `mutation`; attribute_assignment | `medium` | `YES` |
| `adapters/operator_artifact_verification.py` | `WRAP` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/operator_artifact_verification.py:139`; `mutation`; mutation_method=blockers.append | `medium` | `YES` |
| `adapters/operator_verified_artifact_evidence.py` | `EXPERIMENTAL` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/operator_verified_artifact_evidence.py:65`; `mutation`; mutation_method=files_exist.update | `medium` | `YES` |
| `adapters/orchestrator_adapter.py` | `WRAP` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/orchestrator_adapter.py:295`; `mutation`; attribute_assignment | `medium` | `YES` |
| `adapters/persistence_adapter.py` | `REWRITE` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/persistence_adapter.py:31`; `mutation`; attribute_assignment | `high` | `NO` |
| `adapters/proactive_adapter.py` | `KEEP` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/proactive_adapter.py:28`; `mutation`; attribute_assignment | `medium` | `YES` |
| `adapters/runtime_mapping_import_blocker_recovery.py` | `EXPERIMENTAL` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/runtime_mapping_import_blocker_recovery.py:32`; `direct_write`; write_call=out.parent.mkdir | `low` | `YES` |
| `adapters/runtime_mapping_limited_persistence_sandbox.py` | `EXPERIMENTAL` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/runtime_mapping_limited_persistence_sandbox.py:59`; `direct_write`; write_call=out.parent.mkdir | `low` | `YES` |
| `adapters/runtime_mapping_persistence_activation_candidate.py` | `EXPERIMENTAL` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/runtime_mapping_persistence_activation_candidate.py:40`; `direct_write`; write_call=path.parent.mkdir | `low` | `YES` |
| `adapters/runtime_mapping_persistence_activation_dryrun.py` | `EXPERIMENTAL` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/runtime_mapping_persistence_activation_dryrun.py:399`; `direct_write`; write_call=out.parent.mkdir | `medium` | `YES` |
| `adapters/runtime_mapping_persistence_approval.py` | `EXPERIMENTAL` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/runtime_mapping_persistence_approval.py:31`; `mutation`; mutation_method=reasons.append | `low` | `YES` |
| `adapters/runtime_mapping_persistence_approval_fixture.py` | `EXPERIMENTAL` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/runtime_mapping_persistence_approval_fixture.py:51`; `direct_write`; write_call=out.parent.mkdir | `low` | `YES` |
| `adapters/runtime_mapping_persistence_decision.py` | `EXPERIMENTAL` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/runtime_mapping_persistence_decision.py:30`; `mutation`; mutation_method=reasons.append | `low` | `YES` |
| `adapters/runtime_mapping_production_readiness.py` | `EXPERIMENTAL` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/runtime_mapping_production_readiness.py:50`; `mutation`; mutation_method=forbidden.append | `medium` | `YES` |
| `adapters/runtime_smoke_runner.py` | `EXPERIMENTAL` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/runtime_smoke_runner.py:36`; `mutation`; mutation_method=texts.append | `medium` | `YES` |
| `adapters/safety_adapter.py` | `KEEP` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/safety_adapter.py:88`; `mutation`; attribute_assignment | `medium` | `YES` |
| `adapters/salience_adapter.py` | `KEEP` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/salience_adapter.py:46`; `mutation`; attribute_assignment | `medium` | `YES` |
| `adapters/sd_adapter.py` | `WRAP` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/sd_adapter.py:40`; `mutation`; attribute_assignment | `medium` | `YES` |
| `adapters/seed_vector_artifact_readiness.py` | `EXPERIMENTAL` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/seed_vector_artifact_readiness.py:58`; `mutation`; mutation_method=set | `medium` | `YES` |
| `adapters/seed_vector_restore_contract.py` | `EXPERIMENTAL` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/seed_vector_restore_contract.py:68`; `mutation`; subscript_assignment | `medium` | `YES` |
| `adapters/self_embedding_adapter.py` | `WRAP` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/self_embedding_adapter.py:92`; `mutation`; mutation_method=tokens.append | `medium` | `YES` |
| `adapters/self_state.py` | `KEEP` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/self_state.py:25`; `mutation`; attribute_assignment | `medium` | `YES` |
| `adapters/semantic_distance_adapter.py` | `WRAP` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/semantic_distance_adapter.py:31`; `mutation`; attribute_assignment | `medium` | `YES` |
| `adapters/sensory_adapter.py` | `WRAP` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/sensory_adapter.py:31`; `mutation`; attribute_assignment | `medium` | `YES` |
| `adapters/sensory_observation_contract.py` | `EXPERIMENTAL` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/sensory_observation_contract.py:232`; `mutation`; mutation_method=items.append | `medium` | `YES` |
| `adapters/situation_responder.py` | `WRAP` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/situation_responder.py:32`; `mutation`; attribute_assignment | `medium` | `YES` |
| `adapters/smoke_data_analyzer.py` | `EXPERIMENTAL` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/smoke_data_analyzer.py:35`; `mutation`; mutation_method=rows.append | `medium` | `YES` |
| `adapters/social_env_adapter.py` | `KEEP` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/social_env_adapter.py:39`; `mutation`; attribute_assignment | `medium` | `YES` |
| `adapters/speech_hub.py` | `WRAP` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/speech_hub.py:259`; `mutation`; attribute_assignment | `medium` | `YES` |
| `adapters/state_debug_adapter.py` | `WRAP` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/state_debug_adapter.py:15`; `mutation`; attribute_assignment | `medium` | `YES` |
| `adapters/suffering_adapter.py` | `WRAP` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/suffering_adapter.py:41`; `mutation`; attribute_assignment | `medium` | `YES` |
| `adapters/teaching_adapter.py` | `WRAP` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/teaching_adapter.py:169`; `mutation`; attribute_assignment | `medium` | `YES` |
| `adapters/temporal_adapter.py` | `WRAP` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/temporal_adapter.py:31`; `mutation`; attribute_assignment | `medium` | `YES` |
| `adapters/thought_chain_adapter.py` | `EXPERIMENTAL` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/thought_chain_adapter.py:72`; `mutation`; attribute_assignment | `medium` | `YES` |
| `adapters/urge_adapter.py` | `WRAP` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/urge_adapter.py:35`; `mutation`; attribute_assignment | `high` | `NO` |
| `adapters/user_instruction_adapter.py` | `WRAP` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/user_instruction_adapter.py:41`; `mutation`; attribute_assignment | `medium` | `YES` |
| `adapters/user_presence_adapter.py` | `KEEP` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/user_presence_adapter.py:34`; `mutation`; attribute_assignment | `medium` | `YES` |
| `adapters/virtual_visual_observation_schema.py` | `EXPERIMENTAL` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/virtual_visual_observation_schema.py:165`; `mutation`; subscript_assignment | `medium` | `YES` |
| `adapters/virtual_world_consistency_audit_schema.py` | `EXPERIMENTAL` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/virtual_world_consistency_audit_schema.py:150`; `mutation`; subscript_assignment | `medium` | `YES` |
| `adapters/virtual_world_non_visual_situation_schema.py` | `EXPERIMENTAL` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/virtual_world_non_visual_situation_schema.py:65`; `mutation`; subscript_assignment | `medium` | `YES` |
| `adapters/virtual_world_observation_contract.py` | `EXPERIMENTAL` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/virtual_world_observation_contract.py:109`; `mutation`; mutation_method=virtual_boundary_flags.append | `low` | `YES` |
| `adapters/virtual_world_operator_intent_boundary_schema.py` | `EXPERIMENTAL` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/virtual_world_operator_intent_boundary_schema.py:150`; `mutation`; mutation_method=append | `low` | `YES` |
| `adapters/virtual_world_policy_gate_schema.py` | `EXPERIMENTAL` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/virtual_world_policy_gate_schema.py:135`; `mutation`; mutation_method=append | `low` | `YES` |
| `adapters/virtual_world_situation_causal_context_schema.py` | `EXPERIMENTAL` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/virtual_world_situation_causal_context_schema.py:121`; `mutation`; mutation_method=set | `medium` | `YES` |
| `adapters/virtual_world_situation_constraint_context_schema.py` | `EXPERIMENTAL` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/virtual_world_situation_constraint_context_schema.py:43`; `mutation`; mutation_method=set | `medium` | `YES` |
| `adapters/virtual_world_situation_evidence_context_schema.py` | `EXPERIMENTAL` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/virtual_world_situation_evidence_context_schema.py:48`; `mutation`; mutation_method=set | `medium` | `YES` |
| `adapters/virtual_world_situation_hypothesis_context_schema.py` | `EXPERIMENTAL` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/virtual_world_situation_hypothesis_context_schema.py:40`; `mutation`; mutation_method=set | `medium` | `YES` |
| `adapters/virtual_world_situation_inference_context_schema.py` | `EXPERIMENTAL` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/virtual_world_situation_inference_context_schema.py:47`; `mutation`; mutation_method=set | `medium` | `YES` |
| `adapters/virtual_world_situation_role_relation_schema.py` | `EXPERIMENTAL` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/virtual_world_situation_role_relation_schema.py:221`; `mutation`; subscript_assignment | `medium` | `YES` |
| `adapters/virtual_world_situation_temporal_context_schema.py` | `EXPERIMENTAL` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/virtual_world_situation_temporal_context_schema.py:212`; `mutation`; mutation_method=set | `medium` | `YES` |
| `adapters/virtual_world_situation_uncertainty_context_schema.py` | `EXPERIMENTAL` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/virtual_world_situation_uncertainty_context_schema.py:114`; `mutation`; mutation_method=set | `medium` | `YES` |
| `adapters/virtual_world_state_snapshot_schema.py` | `EXPERIMENTAL` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/virtual_world_state_snapshot_schema.py:123`; `mutation`; subscript_assignment | `low` | `YES` |
| `adapters/virtual_world_transition_preflight_schema.py` | `EXPERIMENTAL` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/virtual_world_transition_preflight_schema.py:147`; `mutation`; subscript_assignment | `low` | `YES` |
| `adapters/vision_adapter.py` | `WRAP` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/vision_adapter.py:42`; `mutation`; attribute_assignment | `medium` | `YES` |
| `adapters/visual_observation_schema.py` | `EXPERIMENTAL` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/visual_observation_schema.py:138`; `mutation`; mutation_method=items.append | `medium` | `YES` |
| `adapters/visualizer_server.py` | `WRAP` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/visualizer_server.py:78-82`; `mutation`; subscript_assignment | `medium` | `YES` |
| `adapters/voice_loop_adapter.py` | `WRAP` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/voice_loop_adapter.py:31`; `mutation`; attribute_assignment | `medium` | `YES` |
| `adapters/vsa_adapter.py` | `WRAP` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/vsa_adapter.py:28`; `mutation`; attribute_assignment | `medium` | `YES` |
| `adapters/web_learning_adapter.py` | `WRAP` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/web_learning_adapter.py:51`; `mutation`; attribute_assignment | `medium` | `YES` |
| `adapters/world_model_adapter.py` | `WRAP` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/world_model_adapter.py:69`; `mutation`; attribute_assignment | `medium` | `YES` |
| `agent/__init__.py` | `EXPERIMENTAL` | `agent/__init__.py:1`; tracked Python module, AST parse/reachability result `reachable_from_active_root=False; references=0; component_evidence=0` | `low` | `YES` |
| `agent/tools.py` | `KEEP` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `agent/tools.py:29-31`; `entrypoint`; callable_name=run | `medium` | `YES` |
| `blueprint.py` | `KEEP` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `blueprint.py:59`; `mutation`; attribute_assignment | `medium` | `YES` |
| `cleanup_drive.py` | `EXPERIMENTAL` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `cleanup_drive.py:107`; `mutation`; mutation_method=to_archive.append | `low` | `YES` |
| `cognition/__init__.py` | `EXPERIMENTAL` | `cognition/__init__.py:1`; tracked Python module, AST parse/reachability result `reachable_from_active_root=False; references=0; component_evidence=0` | `low` | `YES` |
| `cognition/meaning_graph.py` | `KEEP` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `cognition/meaning_graph.py:15`; `mutation`; attribute_assignment | `medium` | `YES` |
| `core/__init__.py` | `EXPERIMENTAL` | `core/__init__.py:1`; tracked Python module, AST parse/reachability result `reachable_from_active_root=False; references=0; component_evidence=0` | `low` | `YES` |
| `core/autonomous.py` | `REWRITE` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `core/autonomous.py:46`; `mutation`; mutation_method=needs.append | `high` | `NO` |
| `core/length_decider.py` | `WRAP` | `M0_C_PERSISTENCE_AND_STATE_MAP.md` → `core/length_decider.py:39`; `hormone_state`; hormone_symbol=emotion_max | `medium` | `YES` |
| `core/reasoning.py` | `WRAP` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `core/reasoning.py:45`; `mutation`; attribute_assignment | `medium` | `YES` |
| `core/simulation.py` | `EXPERIMENTAL` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `core/simulation.py:31`; `mutation`; attribute_assignment | `medium` | `YES` |
| `core/system1.py` | `WRAP` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `core/system1.py:36`; `mutation`; attribute_assignment | `medium` | `YES` |
| `core/system2.py` | `KEEP` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `core/system2.py:17`; `mutation`; attribute_assignment | `medium` | `YES` |
| `core/workspace.py` | `WRAP` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `core/workspace.py:22`; `mutation`; attribute_assignment | `medium` | `YES` |
| `day6_integration.py` | `EXPERIMENTAL` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `day6_integration.py:7`; `mutation`; mutation_method=sys.path.insert | `medium` | `YES` |
| `day8_hybrid.py` | `EXPERIMENTAL` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `day8_hybrid.py:20`; `mutation`; mutation_method=sys.path.insert | `medium` | `YES` |
| `day8_hybrid_v2.py` | `EXPERIMENTAL` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `day8_hybrid_v2.py:11`; `mutation`; mutation_method=sys.path.insert | `medium` | `YES` |
| `day8_hybrid_v3.py` | `EXPERIMENTAL` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `day8_hybrid_v3.py:18`; `mutation`; mutation_method=sys.path.insert | `medium` | `YES` |
| `day8_hybrid_v4.py` | `EXPERIMENTAL` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `day8_hybrid_v4.py:13`; `mutation`; mutation_method=sys.path.insert | `medium` | `YES` |
| `day8_layer1.py` | `EXPERIMENTAL` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `day8_layer1.py:9`; `mutation`; mutation_method=sys.path.insert | `medium` | `YES` |
| `day8_layer1_v2.py` | `EXPERIMENTAL` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `day8_layer1_v2.py:9`; `mutation`; mutation_method=sys.path.insert | `medium` | `YES` |
| `day8_layer1_v3.py` | `EXPERIMENTAL` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `day8_layer1_v3.py:10`; `mutation`; mutation_method=sys.path.insert | `medium` | `YES` |
| `day8_option_b.py` | `EXPERIMENTAL` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `day8_option_b.py:17`; `mutation`; mutation_method=sys.path.insert | `medium` | `YES` |
| `day8_option_b_v3.py` | `EXPERIMENTAL` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `day8_option_b_v3.py:10`; `mutation`; mutation_method=sys.path.insert | `medium` | `YES` |
| `day8_vogels.py` | `EXPERIMENTAL` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `day8_vogels.py:18`; `mutation`; mutation_method=sys.path.insert | `medium` | `YES` |
| `digital_somatic.py` | `KEEP` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `digital_somatic.py:9`; `import`; from __future__ import annotations | `medium` | `YES` |
| `dmn.py` | `KEEP` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `dmn.py:9`; `import`; from __future__ import annotations | `medium` | `YES` |
| `episodic.py` | `WRAP` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `episodic.py:103`; `mutation`; attribute_assignment | `medium` | `YES` |
| `eve_ai2thor_colab.py` | `EXPERIMENTAL` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `eve_ai2thor_colab.py:37`; `mutation`; subscript_assignment | `low` | `YES` |
| `eve_all_in_one.py` | `EXPERIMENTAL` | `M0_B_GATE_FAILURE_CLOCK_CONCURRENCY_MAP.md` → `eve_all_in_one.py:26`; `output`; output_call=print | `low` | `YES` |
| `eve_foundation_v10_2.py` | `DEPRECATE` | `m0_d_component_inventory.py` → `eve_foundation_v10_2.py:11557`; `ast.parse`; '[' was never closed (eve_foundation_v10_2.py, line 11557) | `high` | `NO` |
| `eve_foundation_v12_0.py` | `DEPRECATE` | `m0_d_component_inventory.py` → `eve_foundation_v12_0.py:11542`; `ast.parse`; '[' was never closed (eve_foundation_v12_0.py, line 11542) | `high` | `NO` |
| `eve_main_ab.py` | `EXPERIMENTAL` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `eve_main_ab.py:37`; `mutation`; mutation_method=sys.path.insert | `medium` | `YES` |
| `eve_main_abc.py` | `EXPERIMENTAL` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `eve_main_abc.py:38`; `mutation`; mutation_method=sys.path.insert | `medium` | `YES` |
| `eve_optuna_tune.py` | `KEEP` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `eve_optuna_tune.py:65`; `mutation`; subscript_assignment | `medium` | `YES` |
| `eve_tune_v2.py` | `EXPERIMENTAL` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `eve_tune_v2.py:26`; `mutation`; subscript_assignment | `medium` | `YES` |
| `eve_tune_v3_stable.py` | `EXPERIMENTAL` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `eve_tune_v3_stable.py:47`; `mutation`; subscript_assignment | `medium` | `YES` |
| `eve_v12_massive.py` | `EXPERIMENTAL` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `eve_v12_massive.py:28`; `mutation`; mutation_method=sys.path.insert | `medium` | `YES` |
| `eve_v15_synaptic.py` | `EXPERIMENTAL` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `eve_v15_synaptic.py:39`; `mutation`; attribute_assignment | `medium` | `YES` |
| `eve_v18_complete.py` | `EXPERIMENTAL` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `eve_v18_complete.py:64`; `mutation`; attribute_assignment | `low` | `YES` |
| `eve_v19_humanlike.py` | `EXPERIMENTAL` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `eve_v19_humanlike.py:33`; `mutation`; attribute_assignment | `medium` | `YES` |
| `eve_v20_safe.py` | `EXPERIMENTAL` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `eve_v20_safe.py:28`; `mutation`; attribute_assignment | `medium` | `YES` |
| `eve_v21_real.py` | `EXPERIMENTAL` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `eve_v21_real.py:44`; `mutation`; attribute_assignment | `medium` | `YES` |
| `eve_v22_meaning.py` | `EXPERIMENTAL` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `eve_v22_meaning.py:41`; `mutation`; attribute_assignment | `medium` | `YES` |
| `eve_v23_auto.py` | `EXPERIMENTAL` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `eve_v23_auto.py:30`; `mutation`; mutation_method=set | `medium` | `YES` |
| `eve_v24_12hours.py` | `EXPERIMENTAL` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `eve_v24_12hours.py:34`; `mutation`; mutation_method=set | `medium` | `YES` |
| `eve_virtual_learn_100.py` | `EXPERIMENTAL` | `M0_B_GATE_FAILURE_CLOCK_CONCURRENCY_MAP.md` → `eve_virtual_learn_100.py:7`; `output`; output_call=print | `low` | `YES` |
| `hormone_system.py` | `WRAP` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `hormone_system.py:56`; `mutation`; attribute_assignment | `medium` | `YES` |
| `language/__init__.py` | `EXPERIMENTAL` | `language/__init__.py:1`; tracked Python module, AST parse/reachability result `reachable_from_active_root=False; references=0; component_evidence=0` | `low` | `YES` |
| `language/generator.py` | `KEEP` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `language/generator.py:12`; `import`; from utils.types import ResponsePlan | `medium` | `YES` |
| `language/planner.py` | `WRAP` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `language/planner.py:55`; `mutation`; attribute_assignment | `medium` | `YES` |
| `language/streaming.py` | `REWRITE` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `language/streaming.py:89`; `mutation`; attribute_assignment | `high` | `NO` |
| `language/understanding.py` | `WRAP` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `language/understanding.py:86`; `mutation`; attribute_assignment | `medium` | `YES` |
| `learning/__init__.py` | `EXPERIMENTAL` | `learning/__init__.py:1`; tracked Python module, AST parse/reachability result `reachable_from_active_root=False; references=0; component_evidence=0` | `low` | `YES` |
| `learning/code_synthesis.py` | `KEEP` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `learning/code_synthesis.py:22`; `mutation`; attribute_assignment | `medium` | `YES` |
| `legacy/eve_main_abc.py` | `EXPERIMENTAL` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `legacy/eve_main_abc.py:52`; `mutation`; mutation_method=sys.path.insert | `medium` | `YES` |
| `legacy/eve_modules/active_inference.py` | `EXPERIMENTAL` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `legacy/eve_modules/active_inference.py:108`; `mutation`; attribute_assignment | `medium` | `YES` |
| `legacy/eve_modules/agency.py` | `EXPERIMENTAL` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `legacy/eve_modules/agency.py:109`; `mutation`; attribute_assignment | `medium` | `YES` |
| `legacy/eve_modules/allostatic_learn.py` | `EXPERIMENTAL` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `legacy/eve_modules/allostatic_learn.py:106`; `mutation`; attribute_assignment | `medium` | `YES` |
| `legacy/eve_modules/analogy.py` | `EXPERIMENTAL` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `legacy/eve_modules/analogy.py:94`; `mutation`; attribute_assignment | `medium` | `YES` |
| `legacy/eve_modules/apc_learner.py` | `EXPERIMENTAL` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `legacy/eve_modules/apc_learner.py:88`; `mutation`; attribute_assignment | `medium` | `YES` |
| `legacy/eve_modules/causal_graph.py` | `EXPERIMENTAL` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `legacy/eve_modules/causal_graph.py:100`; `mutation`; attribute_assignment | `medium` | `YES` |
| `legacy/eve_modules/continual_rehearsal.py` | `EXPERIMENTAL` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `legacy/eve_modules/continual_rehearsal.py:81`; `mutation`; attribute_assignment | `medium` | `YES` |
| `legacy/eve_modules/corpus_learner.py` | `EXPERIMENTAL` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `legacy/eve_modules/corpus_learner.py:93`; `mutation`; attribute_assignment | `medium` | `YES` |
| `legacy/eve_modules/counterfactual.py` | `EXPERIMENTAL` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `legacy/eve_modules/counterfactual.py:101`; `mutation`; attribute_assignment | `medium` | `YES` |
| `legacy/eve_modules/creative.py` | `EXPERIMENTAL` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `legacy/eve_modules/creative.py:72`; `mutation`; attribute_assignment | `medium` | `YES` |
| `legacy/eve_modules/creative_advanced.py` | `EXPERIMENTAL` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `legacy/eve_modules/creative_advanced.py:68`; `mutation`; attribute_assignment | `medium` | `YES` |
| `legacy/eve_modules/deep_reasoning.py` | `EXPERIMENTAL` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `legacy/eve_modules/deep_reasoning.py:88`; `mutation`; attribute_assignment | `medium` | `YES` |
| `legacy/eve_modules/digital_somatic.py` | `WRAP` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `legacy/eve_modules/digital_somatic.py:68`; `mutation`; attribute_assignment | `medium` | `YES` |
| `legacy/eve_modules/dmn.py` | `WRAP` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `legacy/eve_modules/dmn.py:57`; `mutation`; attribute_assignment | `medium` | `YES` |
| `legacy/eve_modules/emotion_regulation.py` | `EXPERIMENTAL` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `legacy/eve_modules/emotion_regulation.py:114`; `mutation`; attribute_assignment | `medium` | `YES` |
| `legacy/eve_modules/episodic.py` | `EXPERIMENTAL` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `legacy/eve_modules/episodic.py:103`; `mutation`; attribute_assignment | `medium` | `YES` |
| `legacy/eve_modules/frame_semantics.py` | `EXPERIMENTAL` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `legacy/eve_modules/frame_semantics.py:138`; `mutation`; attribute_assignment | `low` | `YES` |
| `legacy/eve_modules/goal_management.py` | `EXPERIMENTAL` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `legacy/eve_modules/goal_management.py:140`; `mutation`; attribute_assignment | `medium` | `YES` |
| `legacy/eve_modules/hormone_system.py` | `EXPERIMENTAL` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `legacy/eve_modules/hormone_system.py:87`; `mutation`; attribute_assignment | `medium` | `YES` |
| `legacy/eve_modules/humor.py` | `EXPERIMENTAL` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `legacy/eve_modules/humor.py:77`; `mutation`; attribute_assignment | `medium` | `YES` |
| `legacy/eve_modules/hypergraph.py` | `EXPERIMENTAL` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `legacy/eve_modules/hypergraph.py:118`; `mutation`; attribute_assignment | `medium` | `YES` |
| `legacy/eve_modules/metacognition.py` | `EXPERIMENTAL` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `legacy/eve_modules/metacognition.py:102`; `mutation`; attribute_assignment | `medium` | `YES` |
| `legacy/eve_modules/multi_stream.py` | `EXPERIMENTAL` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `legacy/eve_modules/multi_stream.py:54`; `mutation`; attribute_assignment | `medium` | `YES` |
| `legacy/eve_modules/narrative_self.py` | `EXPERIMENTAL` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `legacy/eve_modules/narrative_self.py:70`; `mutation`; attribute_assignment | `medium` | `YES` |
| `legacy/eve_modules/natural_lang.py` | `EXPERIMENTAL` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `legacy/eve_modules/natural_lang.py:163`; `mutation`; attribute_assignment | `medium` | `YES` |
| `legacy/eve_modules/norm_internal.py` | `EXPERIMENTAL` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `legacy/eve_modules/norm_internal.py:116`; `mutation`; attribute_assignment | `medium` | `YES` |
| `legacy/eve_modules/self_doubt.py` | `EXPERIMENTAL` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `legacy/eve_modules/self_doubt.py:111`; `mutation`; attribute_assignment | `medium` | `YES` |
| `legacy/eve_modules/semantic_distance.py` | `EXPERIMENTAL` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `legacy/eve_modules/semantic_distance.py:76`; `mutation`; attribute_assignment | `low` | `YES` |
| `legacy/eve_modules/spreading_activation.py` | `WRAP` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `legacy/eve_modules/spreading_activation.py:92`; `mutation`; attribute_assignment | `medium` | `YES` |
| `legacy/eve_modules/suffering.py` | `EXPERIMENTAL` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `legacy/eve_modules/suffering.py:77`; `mutation`; attribute_assignment | `medium` | `YES` |
| `legacy/eve_modules/temporal.py` | `EXPERIMENTAL` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `legacy/eve_modules/temporal.py:88`; `mutation`; attribute_assignment | `medium` | `YES` |
| `legacy/eve_modules/tool_use.py` | `EXPERIMENTAL` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `legacy/eve_modules/tool_use.py:60`; `mutation`; attribute_assignment | `medium` | `YES` |
| `legacy/eve_modules/vsa_binding.py` | `EXPERIMENTAL` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `legacy/eve_modules/vsa_binding.py:86`; `mutation`; attribute_assignment | `medium` | `YES` |
| `legacy/eve_modules/working_memory.py` | `WRAP` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `legacy/eve_modules/working_memory.py:51`; `mutation`; attribute_assignment | `medium` | `YES` |
| `legacy/eve_modules/world_model.py` | `EXPERIMENTAL` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `legacy/eve_modules/world_model.py:105`; `mutation`; attribute_assignment | `medium` | `YES` |
| `legacy/v36_modules/airi_adapter.py` | `EXPERIMENTAL` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `legacy/v36_modules/airi_adapter.py:96`; `mutation`; subscript_assignment | `medium` | `YES` |
| `legacy/v36_modules/airi_server.py` | `KEEP` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `legacy/v36_modules/airi_server.py:50`; `mutation`; mutation_method=sys.path.insert | `medium` | `YES` |
| `legacy/v36_modules/broca_qwen.py` | `EXPERIMENTAL` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `legacy/v36_modules/broca_qwen.py:99`; `mutation`; attribute_assignment | `medium` | `YES` |
| `legacy/v36_modules/commonsense_seed.py` | `EXPERIMENTAL` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `legacy/v36_modules/commonsense_seed.py:286`; `mutation`; mutation_method=set | `medium` | `YES` |
| `legacy/v36_modules/dashboard.py` | `KEEP` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `legacy/v36_modules/dashboard.py:34`; `mutation`; mutation_method=sys.path.insert | `medium` | `YES` |
| `legacy/v36_modules/dashboard_data.py` | `EXPERIMENTAL` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `legacy/v36_modules/dashboard_data.py:125-131`; `mutation`; subscript_assignment | `medium` | `YES` |
| `legacy/v36_modules/decide_action.py` | `EXPERIMENTAL` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `legacy/v36_modules/decide_action.py:123`; `mutation`; attribute_assignment | `medium` | `YES` |
| `legacy/v36_modules/env_adapter.py` | `EXPERIMENTAL` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `legacy/v36_modules/env_adapter.py:50`; `mutation`; attribute_assignment | `medium` | `YES` |
| `legacy/v36_modules/eve_room.py` | `EXPERIMENTAL` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `legacy/v36_modules/eve_room.py:79`; `mutation`; subscript_assignment | `low` | `YES` |
| `legacy/v36_modules/eve_v36.py` | `EXPERIMENTAL` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `legacy/v36_modules/eve_v36.py:24`; `mutation`; mutation_method=sys.path.insert | `medium` | `YES` |
| `legacy/v36_modules/eve_v39.py` | `EXPERIMENTAL` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `legacy/v36_modules/eve_v39.py:32`; `mutation`; mutation_method=sys.path.insert | `medium` | `YES` |
| `legacy/v36_modules/integrated_self.py` | `EXPERIMENTAL` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `legacy/v36_modules/integrated_self.py:97`; `mutation`; mutation_method=bits.append | `low` | `YES` |
| `legacy/v36_modules/persistence.py` | `EXPERIMENTAL` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `legacy/v36_modules/persistence.py:136-144`; `mutation`; subscript_assignment | `low` | `YES` |
| `legacy/v36_modules/proactive.py` | `EXPERIMENTAL` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `legacy/v36_modules/proactive.py:62`; `mutation`; attribute_assignment | `low` | `YES` |
| `legacy/v36_modules/response_enhancer.py` | `EXPERIMENTAL` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `legacy/v36_modules/response_enhancer.py:183`; `mutation`; attribute_assignment | `low` | `YES` |
| `legacy/v36_modules/social_env.py` | `EXPERIMENTAL` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `legacy/v36_modules/social_env.py:86`; `mutation`; attribute_assignment | `medium` | `YES` |
| `legacy/v36_modules/symbolic_env.py` | `EXPERIMENTAL` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `legacy/v36_modules/symbolic_env.py:96`; `mutation`; subscript_assignment | `medium` | `YES` |
| `legacy/v36_modules/user_presence.py` | `EXPERIMENTAL` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `legacy/v36_modules/user_presence.py:48`; `mutation`; attribute_assignment | `medium` | `YES` |
| `main.py` | `REWRITE` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `main.py:20-356`; `entrypoint`; callable_name=build_full_engine | `high` | `NO` |
| `memory/__init__.py` | `EXPERIMENTAL` | `memory/__init__.py:1`; tracked Python module, AST parse/reachability result `reachable_from_active_root=False; references=0; component_evidence=0` | `low` | `YES` |
| `natural_lang.py` | `WRAP` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `natural_lang.py:108`; `mutation`; attribute_assignment | `medium` | `YES` |
| `natural_lang1.py` | `EXPERIMENTAL` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `natural_lang1.py:131`; `mutation`; attribute_assignment | `medium` | `YES` |
| `round71_full_suite_runner.py` | `EXPERIMENTAL` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `round71_full_suite_runner.py:11`; `direct_write`; write_call=status_path.write_text | `medium` | `YES` |
| `round72_split_suite_runner.py` | `EXPERIMENTAL` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `round72_split_suite_runner.py:13`; `direct_write`; write_call=status_path.write_text | `medium` | `YES` |
| `round77_split_suite_runner.py` | `EXPERIMENTAL` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `round77_split_suite_runner.py:10`; `direct_write`; write_call=chunks_path.write_text | `low` | `YES` |
| `self_doubt.py` | `WRAP` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `self_doubt.py:111`; `mutation`; attribute_assignment | `medium` | `YES` |
| `spreading_activation.py` | `KEEP` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `spreading_activation.py:9`; `import`; from __future__ import annotations | `medium` | `YES` |
| `synaptic_scaling.py` | `EXPERIMENTAL` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `synaptic_scaling.py:10`; `mutation`; mutation_method=sys.path.insert | `medium` | `YES` |
| `synaptic_scaling_v2.py` | `EXPERIMENTAL` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `synaptic_scaling_v2.py:9`; `mutation`; mutation_method=sys.path.insert | `medium` | `YES` |
| `utils/__init__.py` | `EXPERIMENTAL` | `utils/__init__.py:1`; tracked Python module, AST parse/reachability result `reachable_from_active_root=False; references=0; component_evidence=0` | `low` | `YES` |
| `utils/korean_endings.py` | `KEEP` | `utils/korean_endings.py:1`; tracked Python module, AST parse/reachability result `reachable_from_active_root=True; references=0; component_evidence=0` | `medium` | `YES` |
| `utils/korean_particles.py` | `KEEP` | `utils/korean_particles.py:1`; tracked Python module, AST parse/reachability result `reachable_from_active_root=True; references=0; component_evidence=0` | `medium` | `YES` |
| `utils/legacy_path.py` | `KEEP` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `utils/legacy_path.py:25`; `mutation`; mutation_method=sys.path.insert | `medium` | `YES` |
| `utils/mock_eve.py` | `WRAP` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `utils/mock_eve.py:20`; `mutation`; attribute_assignment | `medium` | `YES` |
| `utils/speech_style.py` | `EXPERIMENTAL` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `utils/speech_style.py:174`; `mutation`; mutation_method=keywords.append | `medium` | `YES` |
| `utils/types.py` | `WRAP` | `M0_C_PERSISTENCE_AND_STATE_MAP.md` → `utils/types.py:18`; `state_domain`; state_symbol=emotions;domain=affect_hormones | `medium` | `YES` |
| `v2/__init__.py` | `EXPERIMENTAL` | `v2/__init__.py:1`; tracked Python module, AST parse/reachability result `reachable_from_active_root=False; references=0; component_evidence=0` | `low` | `YES` |
| `v2/core/__init__.py` | `EXPERIMENTAL` | `v2/core/__init__.py:1`; tracked Python module, AST parse/reachability result `reachable_from_active_root=False; references=0; component_evidence=0` | `low` | `YES` |
| `v2/development/__init__.py` | `EXPERIMENTAL` | `v2/development/__init__.py:1`; tracked Python module, AST parse/reachability result `reachable_from_active_root=False; references=0; component_evidence=0` | `low` | `YES` |
| `v2/environment/__init__.py` | `EXPERIMENTAL` | `v2/environment/__init__.py:1`; tracked Python module, AST parse/reachability result `reachable_from_active_root=False; references=0; component_evidence=0` | `low` | `YES` |
| `v2/innate/__init__.py` | `EXPERIMENTAL` | `v2/innate/__init__.py:1`; tracked Python module, AST parse/reachability result `reachable_from_active_root=False; references=0; component_evidence=0` | `low` | `YES` |
| `v2/language/__init__.py` | `EXPERIMENTAL` | `v2/language/__init__.py:1`; tracked Python module, AST parse/reachability result `reachable_from_active_root=False; references=0; component_evidence=0` | `low` | `YES` |
| `v2/spatial_temporal/__init__.py` | `EXPERIMENTAL` | `v2/spatial_temporal/__init__.py:1`; tracked Python module, AST parse/reachability result `reachable_from_active_root=False; references=0; component_evidence=0` | `low` | `YES` |
| `v2/storage/__init__.py` | `EXPERIMENTAL` | `v2/storage/__init__.py:1`; tracked Python module, AST parse/reachability result `reachable_from_active_root=False; references=0; component_evidence=0` | `low` | `YES` |
| `v2/tests/__init__.py` | `EXPERIMENTAL` | `v2/tests/__init__.py:1`; tracked Python module, AST parse/reachability result `reachable_from_active_root=False; references=0; component_evidence=0` | `low` | `YES` |
| `v2/utils/__init__.py` | `EXPERIMENTAL` | `v2/utils/__init__.py:1`; tracked Python module, AST parse/reachability result `reachable_from_active_root=False; references=0; component_evidence=0` | `low` | `YES` |
| `v34_module1_snn.py` | `EXPERIMENTAL` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `v34_module1_snn.py:96`; `mutation`; attribute_assignment | `low` | `YES` |
| `working_memory.py` | `KEEP` | `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `working_memory.py:9`; `import`; from __future__ import annotations | `medium` | `YES` |

The canonical JSON contains all additional evidence references and full reasons. The table uses one primary citation per module to remain reviewable; it does not discard the remaining references.

## Full REMOVE and DEPRECATE lists

### REMOVE (0)

- _None._ No module is recommended for removal.

### DEPRECATE (2)

- `eve_foundation_v10_2.py` — Tracked legacy foundation/snapshot module is not safely analyzable or is explicitly versioned legacy; preserve for historical/migration evidence but exclude from future runtime authority. Evidence: `m0_d_component_inventory.py` → `eve_foundation_v10_2.py:11557`; `ast.parse`; '[' was never closed (eve_foundation_v10_2.py, line 11557)
- `eve_foundation_v12_0.py` — Tracked legacy foundation/snapshot module is not safely analyzable or is explicitly versioned legacy; preserve for historical/migration evidence but exclude from future runtime authority. Evidence: `m0_d_component_inventory.py` → `eve_foundation_v12_0.py:11542`; `ast.parse`; '[' was never closed (eve_foundation_v12_0.py, line 11542)

These are historical/migration recommendations, not deletion actions. Both files remain tracked and untouched.

## High-impact manual recommendations

| Module | Recommendation | Evidence-backed reason |
|---|---|---|
| `adapters/allostatic_adapter.py` | `WRAP` | Allostatic coupling is potentially reusable as a bounded Vital-loop compatibility projection but must not remain an implicit hormone-to-agency bridge. Primary evidence: `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/allostatic_adapter.py:28`; `mutation`; attribute_assignment |
| `adapters/hormone_adapter.py` | `REWRITE` | Current hormone representation is a legacy affect substrate; v4 requires core drives, appraisal, and derived emotion with continuity-preserving migration. Primary evidence: `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/hormone_adapter.py:25`; `mutation`; attribute_assignment |
| `adapters/live_loop.py` | `REWRITE` | Active daemon loop combines clocks, hormone mutation, autonomy, proactive output, queues, and autosave; v4 requires explicit loop taxonomy, event provenance, lifecycle ownership, and isolated persistence. Primary evidence: `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/live_loop.py:36`; `mutation`; attribute_assignment |
| `adapters/persistence_adapter.py` | `REWRITE` | Current legacy persistence plus gzip/pickle sidecar conflicts with the future append-only SQLite event log and validated snapshot architecture. Primary evidence: `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/persistence_adapter.py:31`; `mutation`; attribute_assignment |
| `adapters/urge_adapter.py` | `WRAP` | Urge computation may inform v4 drives but currently consumes hormone state and feeds proactive behavior; retain only behind an explicit compatibility boundary. Primary evidence: `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/urge_adapter.py:35`; `mutation`; attribute_assignment |
| `core/autonomous.py` | `REWRITE` | Active autonomous step combines need detection, state transition, environment mutation, curiosity, history, and proactive expression; it must be decomposed into v4 goal/activity/learning/expression loops. Primary evidence: `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `core/autonomous.py:46`; `mutation`; mutation_method=needs.append |
| `language/streaming.py` | `REWRITE` | Active chat funnel combines raw input handling, state mutation, learning, history, and expression; v4 requires structural separation between quarantined source text, cognition, and expression. Primary evidence: `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `language/streaming.py:89`; `mutation`; attribute_assignment |
| `main.py` | `REWRITE` | Active composition root mixes construction, command dispatch, automatic background start, and persistence boundaries that must be separated into v4 capabilities and event-driven services. Primary evidence: `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `main.py:20-356`; `entrypoint`; callable_name=build_full_engine |

## Frozen PR recommendations

Recommendations only. M0-D did not close, comment on, rebase, modify, merge, or reuse any frozen PR.

### PR #109 — `REWRITE-AS-V4-CONTRACT`

Deterministic read-only conclusion-candidate evidence is useful, but the branch was authored against the superseded v3 schema ladder and an M0-prebaseline main. Preserve strict-JSON, canonical-ID, forbidden-field, tamper, and no-side-effect tests, then restate the contract under v4 observation/provenance, quarantine, and expression boundaries.

Preserve before any later close: Adapter validation matrix, canonical-ID rules, hostile-input tests, recursive forbidden-field tests, tamper tests, and downstream read-only plan invariants.

### PR #97 — `CLOSE-PRESERVE-EVIDENCE`

The branch is an obsolete pre-fix duplicate of non-visual virtual-situation work later merged and corrected on main, including deterministic ID repair. Merging it would reintroduce an older authority and duplicate files.

Preserve before any later close: Fail-closed situation validation cases, entity/relationship fixtures, read-only handoff-plan assertions, and historical evidence of the deterministic-ID defect.

### PR #86 — `REWRITE-AS-V4-CONTRACT`

Memory replay provenance and boundary tests remain relevant, but the contract uses v3 memory/fact/hormone terminology and predates the v4 event, source-claim, and forgetting architecture.

Preserve before any later close: Replay source classes, confidence/boundary matrices, no-mutation assertions, origin/fact-status tests, and operator-report fixtures.

### PR #84 — `REWRITE-AS-V4-CONTRACT`

Cross-modal preflight evidence is compatible with v4's candidate-only observation boundary, but the branch predates v4 provenance, model-version, capability, and quarantine requirements.

Preserve before any later close: Modality compatibility matrix, identity-resolution prohibition, fail-closed cases, deterministic behavior, and no-side-effect tests.

### PR #82 — `REWRITE-AS-V4-CONTRACT`

The multimodal event candidate contract is useful source-boundary evidence but is governed by v3.1 assumptions and predates v4's explicit source store, model/tool version provenance, and expression isolation.

Preserve before any later close: Supported modality/event matrices, mixed-boundary cases, no-fact/no-identity assertions, and quarantine/AGP/fallback safety tests.

### PR #11 — `ABSORB-INTO-M1`

The branch proposes operator-controlled persistence activation and ephemeral runtime mutation, which cannot merge during M0 and conflicts with the v4 event-store direction. Its safety proofs should become M1 requirements without preserving the activation implementation.

Preserve before any later close: Operator approval guard, checkpoint-before-mutation ordering, before/after debug evidence, rollback verification, and protected-state non-mutation tests.

### PR #7 — `ABSORB-INTO-M1`

The branch combines vector restoration, manual validation, AGP proof expansion, and persistence approval/decision logic on an obsolete baseline. Preserve its controls as M1 bounded learned-subsystem and persistence requirements; do not merge the mixed bundle.

Preserve before any later close: Vector manifest/checksum/shape/dtype gates, ignored-artifact boundary, approval decision records, AGP proof tests, and fail-closed hard-stop behavior.

### PR #4 — `ABSORB-INTO-M1`

The read-only medium-vector restoration audit contains useful external-seed provenance and artifact validation evidence, but its fixed 30k fastText workflow is a v3 implementation decision rather than a v4 bounded learned-subsystem contract.

Preserve before any later close: Manifest provenance, SHA-256, shape/dtype verification, no-download/no-copy default, no-binary-commit policy, and missing-artifact fail-closed tests.

### PR #1 — `CLOSE-PRESERVE-EVIDENCE`

The split Round96 source-package restore workflow addresses a historical checkout artifact that no longer governs the current repository. Retain hash-verification and hard-stop evidence in history, not in v4 runtime or governance.

Preserve before any later close: Manifest hash verification, missing-part failure behavior, safe extraction checks, and reports documenting the historical package blocker.

Actual closing is a separate post-M0-D reviewer action.

## v4.0 assumptions vs runtime reality

1. **event-log-vs-direct-mutation** — v4.0 assumption: Meaningful state transitions are represented by replayable events with causal provenance. Runtime reality: The active chat, live, autonomous, and persistence funnels perform distributed direct mutation and writes without one event-kernel boundary. Evidence: `docs/EVE_DESIGN_v4.md:29-35`, `docs/audit/M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md:78-86`.
2. **sqlite-event-store-vs-pickle-sidecar** — v4.0 assumption: M1/M2 persistence uses append-only SQLite events and validated snapshots. Runtime reality: Current active persistence combines legacy state with gzip/pickle sidecars, explicit operator save/load, and automatic autosave. Evidence: `docs/EVE_DESIGN_v4.md:37-39`, `docs/audit/M0_C_PERSISTENCE_AND_STATE_MAP.md:126-147`.
3. **affect-migration-plan-missing** — v4.0 assumption: M0 proposes migration from 26 hormones toward core drives, appraisal, and derived emotion while preserving continuity. Runtime reality: M0-C inventories 1,777 hormone/affect sites, 386 drive/need sites, and 54 bridge candidates, but contains no concrete migration phases, state mapping, compatibility projection, event/snapshot migration, rollback, or acceptance criteria. Evidence: `docs/EVE_DESIGN_v4.md:45-47`, `docs/audit/M0_C_PERSISTENCE_AND_STATE_MAP.md:157-176`, `docs/audit/M0_C_PERSISTENCE_AND_STATE_MAP.md:191-203`.
4. **speech-output-vs-life-continuity** — v4.0 assumption: Timer ticks, hormone decay, and proactive speech are not proof of life; continuity depends on state, goals, learning, and resumable activity. Runtime reality: Current LiveLoop, AutonomousLoop, and DMN/proactive paths converge heavily on timed output and speech while lifecycle/state ownership is distributed. Evidence: `docs/EVE_DESIGN_v4.md:57-59`, `docs/audit/M0_B_GATE_FAILURE_CLOCK_CONCURRENCY_MAP.md:90-96`, `docs/audit/M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md:80-86`.
5. **source-quarantine-vs-chat-funnel** — v4.0 assumption: Raw external text is confined to a quarantined source store and expression cannot read it. Runtime reality: The active StreamingEngine chat funnel receives input, mutates context/learning/history, and produces expression inside one module boundary; structural source-store isolation is not demonstrated by M0 evidence. Evidence: `docs/EVE_DESIGN_v4.md:25-27`, `docs/audit/M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md:78-79`.
6. **bounded-learned-subsystems-vs-distributed-numeric-state** — v4.0 assumption: Learned subsystems require provenance, confidence, capability, evaluation, versioning, rollback, and default no-load. Runtime reality: Numeric/vector/adaptive state and update methods are distributed across many modules and artifact formats; a single bounded activation/version/rollback contract is not present. Evidence: `docs/EVE_DESIGN_v4.md:17-19`, `docs/audit/M0_C_PERSISTENCE_AND_STATE_MAP.md:62-80`, `docs/audit/M0_C_PERSISTENCE_AND_STATE_MAP.md:82-124`.

This list is direct input to a later human-reviewed v4.1 revision. M0-D does not draft replacement constitutional text.

## Prominent UNRESOLVED items

### Constitutional blocker

- `M0_C_REQUIRED_MIGRATION_PLAN_ABSENT` — M0-C inventories hormone/affect, drive/need, and bridge candidates but provides no concrete migration phases, state mapping, compatibility projection, event/snapshot migration, rollback, or acceptance criteria required by EVE v4.

### Parse blockers

- `eve_foundation_v10_2.py:11557` — `[` was never closed; retained as `DEPRECATE`.
- `eve_foundation_v12_0.py:11542` — `[` was never closed; retained as `DEPRECATE`.

### Unresolved module recommendations (278)

These paths already have one provisional category, but the category remains unresolved pending reviewer ruling:

- `active_inference.py` — `WRAP`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `active_inference.py:108`; `mutation`; attribute_assignment
- `adapters/__init__.py` — `EXPERIMENTAL`; `adapters/__init__.py:1`; tracked Python module, AST parse/reachability result `reachable_from_active_root=False; references=0; component_evidence=0`
- `adapters/activation_adapter.py` — `WRAP`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/activation_adapter.py:29`; `mutation`; attribute_assignment
- `adapters/affect_dryrun_trace_operator_decision_packet.py` — `EXPERIMENTAL`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/affect_dryrun_trace_operator_decision_packet.py:180`; `mutation`; mutation_method=set
- `adapters/affect_event_proposal_validator.py` — `EXPERIMENTAL`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/affect_event_proposal_validator.py:89`; `mutation`; mutation_method=set
- `adapters/affect_event_to_axis_proposal_map.py` — `EXPERIMENTAL`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/affect_event_to_axis_proposal_map.py:95`; `mutation`; mutation_method=set
- `adapters/affect_future_dryrun_simulation_request_packet.py` — `EXPERIMENTAL`; `m0_d_component_inventory.py` → `adapters/affect_future_dryrun_simulation_request_packet.py:45`; `Assign`; representation_state_assignment=FEATURE_TRACK tokens=feature
- `adapters/affect_future_dryrun_simulation_runner_preflight.py` — `EXPERIMENTAL`; `m0_d_component_inventory.py` → `adapters/affect_future_dryrun_simulation_runner_preflight.py:43`; `Assign`; representation_state_assignment=FEATURE_TRACK tokens=feature
- `adapters/affect_hormone_interaction_matrix.py` — `EXPERIMENTAL`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/affect_hormone_interaction_matrix.py:116`; `mutation`; mutation_method=base.update
- `adapters/affect_hormone_neural_rhythm_registry.py` — `EXPERIMENTAL`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/affect_hormone_neural_rhythm_registry.py:250`; `mutation`; subscript_assignment
- `adapters/affect_proposal_transition_payload_builder.py` — `EXPERIMENTAL`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/affect_proposal_transition_payload_builder.py:87`; `mutation`; mutation_method=groups.append
- `adapters/affect_reviewed_payload_dryrun_bridge.py` — `EXPERIMENTAL`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/affect_reviewed_payload_dryrun_bridge.py:90`; `mutation`; mutation_method=set
- `adapters/affect_transition_checkpoint_rollback_dryrun_trace.py` — `EXPERIMENTAL`; `m0_d_component_inventory.py` → `adapters/affect_transition_checkpoint_rollback_dryrun_trace.py:42`; `Assign`; representation_state_assignment=FEATURE_TRACK tokens=feature
- `adapters/affect_transition_checkpoint_rollback_plan.py` — `EXPERIMENTAL`; `m0_d_component_inventory.py` → `adapters/affect_transition_checkpoint_rollback_plan.py:42`; `Assign`; representation_state_assignment=FEATURE_TRACK tokens=feature
- `adapters/affect_transition_payload_operator_handoff.py` — `EXPERIMENTAL`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/affect_transition_payload_operator_handoff.py:81`; `mutation`; mutation_method=set
- `adapters/agency_adapter.py` — `WRAP`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/agency_adapter.py:29`; `mutation`; attribute_assignment
- `adapters/agp_adapter.py` — `WRAP`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/agp_adapter.py:102`; `mutation`; attribute_assignment
- `adapters/agp_operational_snapshot.py` — `EXPERIMENTAL`; `M0_C_PERSISTENCE_AND_STATE_MAP.md` → `adapters/agp_operational_snapshot.py:78`; `state_domain`; state_symbol=hormone_threshold;domain=affect_hormones
- `adapters/agp_proof_object_expansion.py` — `EXPERIMENTAL`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/agp_proof_object_expansion.py:39`; `mutation`; mutation_method=row_reasons.append
- `adapters/agp_threshold_decision.py` — `EXPERIMENTAL`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/agp_threshold_decision.py:96`; `mutation`; subscript_assignment
- `adapters/agp_trace_analyzer.py` — `WRAP`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/agp_trace_analyzer.py:47`; `mutation`; subscript_assignment
- `adapters/ai_adapter.py` — `EXPERIMENTAL`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/ai_adapter.py:31`; `mutation`; attribute_assignment
- `adapters/analogy_adapter.py` — `WRAP`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/analogy_adapter.py:26`; `mutation`; attribute_assignment
- `adapters/apc_adapter.py` — `WRAP`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/apc_adapter.py:28`; `mutation`; attribute_assignment
- `adapters/appraisal_classifier.py` — `WRAP`; `M0_C_PERSISTENCE_AND_STATE_MAP.md` → `adapters/appraisal_classifier.py:27`; `state_domain`; state_symbol=LABEL_EXTERNAL_AFFECTIVE_TONE;domain=affect_hormones
- `adapters/attention_analyzer.py` — `WRAP`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/attention_analyzer.py:86`; `mutation`; attribute_assignment
- `adapters/auditory_observation_schema.py` — `EXPERIMENTAL`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/auditory_observation_schema.py:122`; `mutation`; mutation_method=blocked_reasons.append
- `adapters/autonomy_adapter.py` — `KEEP`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/autonomy_adapter.py:40`; `mutation`; attribute_assignment
- `adapters/character_adapter.py` — `WRAP`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/character_adapter.py:67`; `mutation`; attribute_assignment
- `adapters/compositor_adapter.py` — `WRAP`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/compositor_adapter.py:80`; `mutation`; attribute_assignment
- `adapters/concept_memory_adapter.py` — `EXPERIMENTAL`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/concept_memory_adapter.py:44`; `mutation`; attribute_assignment
- `adapters/concept_runtime_mapping_diagnostics.py` — `EXPERIMENTAL`; `M0_C_PERSISTENCE_AND_STATE_MAP.md` → `adapters/concept_runtime_mapping_diagnostics.py:50`; `state_domain`; state_symbol=vectors_written;domain=vectors
- `adapters/context_seed.py` — `WRAP`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/context_seed.py:29`; `mutation`; attribute_assignment
- `adapters/continual_adapter.py` — `WRAP`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/continual_adapter.py:38`; `mutation`; attribute_assignment
- `adapters/corpus_adapter.py` — `WRAP`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/corpus_adapter.py:31`; `mutation`; attribute_assignment
- `adapters/counterfactual_adapter.py` — `WRAP`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/counterfactual_adapter.py:50`; `mutation`; attribute_assignment
- `adapters/creative_adapter.py` — `WRAP`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/creative_adapter.py:29`; `mutation`; attribute_assignment
- `adapters/cross_modal_binding_preflight_schema.py` — `EXPERIMENTAL`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/cross_modal_binding_preflight_schema.py:65`; `mutation`; mutation_method=blocked_reasons.append
- `adapters/curiosity_adapter.py` — `WRAP`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/curiosity_adapter.py:30`; `mutation`; attribute_assignment
- `adapters/deep_reasoning_adapter.py` — `WRAP`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/deep_reasoning_adapter.py:23`; `mutation`; attribute_assignment
- `adapters/dialogue_context_adapter.py` — `KEEP`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/dialogue_context_adapter.py:48`; `mutation`; attribute_assignment
- `adapters/digital_somatic_adapter.py` — `WRAP`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/digital_somatic_adapter.py:36`; `mutation`; attribute_assignment
- `adapters/dmn_adapter.py` — `WRAP`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/dmn_adapter.py:32`; `mutation`; attribute_assignment
- `adapters/embedding_wrapper.py` — `EXPERIMENTAL`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/embedding_wrapper.py:28`; `mutation`; attribute_assignment
- `adapters/emotion_cause_adapter.py` — `WRAP`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/emotion_cause_adapter.py:74`; `mutation`; attribute_assignment
- `adapters/emotion_regulation_adapter.py` — `WRAP`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/emotion_regulation_adapter.py:31`; `mutation`; attribute_assignment
- `adapters/emotion_state_transition_contract.py` — `EXPERIMENTAL`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/emotion_state_transition_contract.py:271`; `mutation`; mutation_method=set
- `adapters/emotion_transition_dryrun_apply_plan.py` — `EXPERIMENTAL`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/emotion_transition_dryrun_apply_plan.py:109`; `mutation`; mutation_method=reasons.append
- `adapters/emotion_transition_gate.py` — `EXPERIMENTAL`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/emotion_transition_gate.py:86`; `mutation`; mutation_method=merged.extend
- `adapters/emotion_transition_validator.py` — `EXPERIMENTAL`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/emotion_transition_validator.py:78-89`; `mutation`; mutation_method=categories.setdefault
- `adapters/enhancer_adapter.py` — `KEEP`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/enhancer_adapter.py:35`; `mutation`; attribute_assignment
- `adapters/env_adapter.py` — `WRAP`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/env_adapter.py:39`; `mutation`; attribute_assignment
- `adapters/eve_self_learning_adapter.py` — `WRAP`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/eve_self_learning_adapter.py:140`; `mutation`; attribute_assignment
- `adapters/eve_vector_store.py` — `EXPERIMENTAL`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/eve_vector_store.py:35`; `mutation`; attribute_assignment
- `adapters/eve_vocab_tracker.py` — `EXPERIMENTAL`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/eve_vocab_tracker.py:28`; `mutation`; attribute_assignment
- `adapters/explicit_load_guard.py` — `EXPERIMENTAL`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/explicit_load_guard.py:136`; `mutation`; subscript_assignment
- `adapters/external_seed_manifest.py` — `WRAP`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/external_seed_manifest.py:213`; `mutation`; mutation_method=errors.extend
- `adapters/fasttext_embedding_adapter.py` — `WRAP`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/fasttext_embedding_adapter.py:116`; `mutation`; attribute_assignment
- `adapters/fasttext_loader.py` — `EXPERIMENTAL`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/fasttext_loader.py:112`; `dependency_construction`; constructor_call=Path
- `adapters/frame_adapter.py` — `WRAP`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/frame_adapter.py:28`; `mutation`; attribute_assignment
- `adapters/goal_adapter.py` — `WRAP`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/goal_adapter.py:37`; `mutation`; attribute_assignment
- `adapters/humor_adapter.py` — `WRAP`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/humor_adapter.py:27`; `mutation`; attribute_assignment
- `adapters/hypergraph_adapter.py` — `WRAP`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/hypergraph_adapter.py:26`; `mutation`; attribute_assignment
- `adapters/integrated_self_adapter.py` — `KEEP`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/integrated_self_adapter.py:29`; `mutation`; attribute_assignment
- `adapters/intent_decider.py` — `WRAP`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/intent_decider.py:38`; `mutation`; attribute_assignment
- `adapters/korean/__init__.py` — `EXPERIMENTAL`; `adapters/korean/__init__.py:1`; tracked Python module, AST parse/reachability result `reachable_from_active_root=False; references=0; component_evidence=0`
- `adapters/korean/morph_analyzer.py` — `KEEP`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/korean/morph_analyzer.py:301`; `mutation`; attribute_assignment
- `adapters/korean/sentence_structure.py` — `WRAP`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/korean/sentence_structure.py:136`; `mutation`; attribute_assignment
- `adapters/korean_language_adapter.py` — `KEEP`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/korean_language_adapter.py:19`; `mutation`; attribute_assignment
- `adapters/lex_concept_mapping_adapter.py` — `WRAP`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/lex_concept_mapping_adapter.py:31`; `mutation`; attribute_assignment
- `adapters/medium30k_runtime_load_integration.py` — `EXPERIMENTAL`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/medium30k_runtime_load_integration.py:60`; `mutation`; mutation_method=blockers.append
- `adapters/medium_vector_manual_validation.py` — `EXPERIMENTAL`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/medium_vector_manual_validation.py:32`; `mutation`; mutation_method=data.setdefault
- `adapters/medium_vector_release_restore.py` — `KEEP`; `m0_d_component_inventory.py` → `adapters/medium_vector_release_restore.py:40`; `Assign`; representation_state_assignment=INTERNAL_VECTORS_PATH tokens=vectors
- `adapters/medium_vector_restoration.py` — `KEEP`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/medium_vector_restoration.py:136`; `mutation`; subscript_assignment
- `adapters/memory_adapter.py` — `WRAP`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/memory_adapter.py:29`; `mutation`; attribute_assignment
- `adapters/memory_provenance_quarantine_preflight_schema.py` — `EXPERIMENTAL`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/memory_provenance_quarantine_preflight_schema.py:83`; `mutation`; mutation_method=blocked_reasons.append
- `adapters/memory_replay_observation_schema.py` — `EXPERIMENTAL`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/memory_replay_observation_schema.py:69`; `mutation`; mutation_method=blocked_reasons.append
- `adapters/metacognition_adapter.py` — `WRAP`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/metacognition_adapter.py:33`; `mutation`; attribute_assignment
- `adapters/module_learning_adapter.py` — `EXPERIMENTAL`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/module_learning_adapter.py:47`; `mutation`; attribute_assignment
- `adapters/mood_adapter.py` — `WRAP`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/mood_adapter.py:59`; `mutation`; attribute_assignment
- `adapters/multi_stream_adapter.py` — `WRAP`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/multi_stream_adapter.py:34`; `mutation`; attribute_assignment
- `adapters/multimodal_event_candidate_contract.py` — `EXPERIMENTAL`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/multimodal_event_candidate_contract.py:63`; `mutation`; mutation_method=blocked_reasons.append
- `adapters/narrative_adapter.py` — `WRAP`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/narrative_adapter.py:30`; `mutation`; attribute_assignment
- `adapters/nl_adapter.py` — `WRAP`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/nl_adapter.py:45`; `mutation`; attribute_assignment
- `adapters/norm_adapter.py` — `WRAP`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/norm_adapter.py:32`; `mutation`; attribute_assignment
- `adapters/observation_origin_fact_status_policy.py` — `EXPERIMENTAL`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/observation_origin_fact_status_policy.py:74`; `mutation`; subscript_assignment
- `adapters/openai_server_adapter.py` — `WRAP`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/openai_server_adapter.py:67`; `mutation`; attribute_assignment
- `adapters/operator_artifact_verification.py` — `WRAP`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/operator_artifact_verification.py:139`; `mutation`; mutation_method=blockers.append
- `adapters/operator_verified_artifact_evidence.py` — `EXPERIMENTAL`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/operator_verified_artifact_evidence.py:65`; `mutation`; mutation_method=files_exist.update
- `adapters/orchestrator_adapter.py` — `WRAP`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/orchestrator_adapter.py:295`; `mutation`; attribute_assignment
- `adapters/proactive_adapter.py` — `KEEP`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/proactive_adapter.py:28`; `mutation`; attribute_assignment
- `adapters/runtime_mapping_import_blocker_recovery.py` — `EXPERIMENTAL`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/runtime_mapping_import_blocker_recovery.py:32`; `direct_write`; write_call=out.parent.mkdir
- `adapters/runtime_mapping_limited_persistence_sandbox.py` — `EXPERIMENTAL`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/runtime_mapping_limited_persistence_sandbox.py:59`; `direct_write`; write_call=out.parent.mkdir
- `adapters/runtime_mapping_persistence_activation_candidate.py` — `EXPERIMENTAL`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/runtime_mapping_persistence_activation_candidate.py:40`; `direct_write`; write_call=path.parent.mkdir
- `adapters/runtime_mapping_persistence_activation_dryrun.py` — `EXPERIMENTAL`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/runtime_mapping_persistence_activation_dryrun.py:399`; `direct_write`; write_call=out.parent.mkdir
- `adapters/runtime_mapping_persistence_approval.py` — `EXPERIMENTAL`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/runtime_mapping_persistence_approval.py:31`; `mutation`; mutation_method=reasons.append
- `adapters/runtime_mapping_persistence_approval_fixture.py` — `EXPERIMENTAL`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/runtime_mapping_persistence_approval_fixture.py:51`; `direct_write`; write_call=out.parent.mkdir
- `adapters/runtime_mapping_persistence_decision.py` — `EXPERIMENTAL`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/runtime_mapping_persistence_decision.py:30`; `mutation`; mutation_method=reasons.append
- `adapters/runtime_mapping_production_readiness.py` — `EXPERIMENTAL`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/runtime_mapping_production_readiness.py:50`; `mutation`; mutation_method=forbidden.append
- `adapters/runtime_smoke_runner.py` — `EXPERIMENTAL`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/runtime_smoke_runner.py:36`; `mutation`; mutation_method=texts.append
- `adapters/safety_adapter.py` — `KEEP`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/safety_adapter.py:88`; `mutation`; attribute_assignment
- `adapters/salience_adapter.py` — `KEEP`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/salience_adapter.py:46`; `mutation`; attribute_assignment
- `adapters/sd_adapter.py` — `WRAP`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/sd_adapter.py:40`; `mutation`; attribute_assignment
- `adapters/seed_vector_artifact_readiness.py` — `EXPERIMENTAL`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/seed_vector_artifact_readiness.py:58`; `mutation`; mutation_method=set
- `adapters/seed_vector_restore_contract.py` — `EXPERIMENTAL`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/seed_vector_restore_contract.py:68`; `mutation`; subscript_assignment
- `adapters/self_embedding_adapter.py` — `WRAP`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/self_embedding_adapter.py:92`; `mutation`; mutation_method=tokens.append
- `adapters/self_state.py` — `KEEP`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/self_state.py:25`; `mutation`; attribute_assignment
- `adapters/semantic_distance_adapter.py` — `WRAP`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/semantic_distance_adapter.py:31`; `mutation`; attribute_assignment
- `adapters/sensory_adapter.py` — `WRAP`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/sensory_adapter.py:31`; `mutation`; attribute_assignment
- `adapters/sensory_observation_contract.py` — `EXPERIMENTAL`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/sensory_observation_contract.py:232`; `mutation`; mutation_method=items.append
- `adapters/situation_responder.py` — `WRAP`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/situation_responder.py:32`; `mutation`; attribute_assignment
- `adapters/smoke_data_analyzer.py` — `EXPERIMENTAL`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/smoke_data_analyzer.py:35`; `mutation`; mutation_method=rows.append
- `adapters/social_env_adapter.py` — `KEEP`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/social_env_adapter.py:39`; `mutation`; attribute_assignment
- `adapters/speech_hub.py` — `WRAP`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/speech_hub.py:259`; `mutation`; attribute_assignment
- `adapters/state_debug_adapter.py` — `WRAP`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/state_debug_adapter.py:15`; `mutation`; attribute_assignment
- `adapters/suffering_adapter.py` — `WRAP`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/suffering_adapter.py:41`; `mutation`; attribute_assignment
- `adapters/teaching_adapter.py` — `WRAP`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/teaching_adapter.py:169`; `mutation`; attribute_assignment
- `adapters/temporal_adapter.py` — `WRAP`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/temporal_adapter.py:31`; `mutation`; attribute_assignment
- `adapters/thought_chain_adapter.py` — `EXPERIMENTAL`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/thought_chain_adapter.py:72`; `mutation`; attribute_assignment
- `adapters/user_instruction_adapter.py` — `WRAP`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/user_instruction_adapter.py:41`; `mutation`; attribute_assignment
- `adapters/user_presence_adapter.py` — `KEEP`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/user_presence_adapter.py:34`; `mutation`; attribute_assignment
- `adapters/virtual_visual_observation_schema.py` — `EXPERIMENTAL`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/virtual_visual_observation_schema.py:165`; `mutation`; subscript_assignment
- `adapters/virtual_world_consistency_audit_schema.py` — `EXPERIMENTAL`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/virtual_world_consistency_audit_schema.py:150`; `mutation`; subscript_assignment
- `adapters/virtual_world_non_visual_situation_schema.py` — `EXPERIMENTAL`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/virtual_world_non_visual_situation_schema.py:65`; `mutation`; subscript_assignment
- `adapters/virtual_world_observation_contract.py` — `EXPERIMENTAL`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/virtual_world_observation_contract.py:109`; `mutation`; mutation_method=virtual_boundary_flags.append
- `adapters/virtual_world_operator_intent_boundary_schema.py` — `EXPERIMENTAL`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/virtual_world_operator_intent_boundary_schema.py:150`; `mutation`; mutation_method=append
- `adapters/virtual_world_policy_gate_schema.py` — `EXPERIMENTAL`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/virtual_world_policy_gate_schema.py:135`; `mutation`; mutation_method=append
- `adapters/virtual_world_situation_causal_context_schema.py` — `EXPERIMENTAL`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/virtual_world_situation_causal_context_schema.py:121`; `mutation`; mutation_method=set
- `adapters/virtual_world_situation_constraint_context_schema.py` — `EXPERIMENTAL`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/virtual_world_situation_constraint_context_schema.py:43`; `mutation`; mutation_method=set
- `adapters/virtual_world_situation_evidence_context_schema.py` — `EXPERIMENTAL`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/virtual_world_situation_evidence_context_schema.py:48`; `mutation`; mutation_method=set
- `adapters/virtual_world_situation_hypothesis_context_schema.py` — `EXPERIMENTAL`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/virtual_world_situation_hypothesis_context_schema.py:40`; `mutation`; mutation_method=set
- `adapters/virtual_world_situation_inference_context_schema.py` — `EXPERIMENTAL`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/virtual_world_situation_inference_context_schema.py:47`; `mutation`; mutation_method=set
- `adapters/virtual_world_situation_role_relation_schema.py` — `EXPERIMENTAL`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/virtual_world_situation_role_relation_schema.py:221`; `mutation`; subscript_assignment
- `adapters/virtual_world_situation_temporal_context_schema.py` — `EXPERIMENTAL`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/virtual_world_situation_temporal_context_schema.py:212`; `mutation`; mutation_method=set
- `adapters/virtual_world_situation_uncertainty_context_schema.py` — `EXPERIMENTAL`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/virtual_world_situation_uncertainty_context_schema.py:114`; `mutation`; mutation_method=set
- `adapters/virtual_world_state_snapshot_schema.py` — `EXPERIMENTAL`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/virtual_world_state_snapshot_schema.py:123`; `mutation`; subscript_assignment
- `adapters/virtual_world_transition_preflight_schema.py` — `EXPERIMENTAL`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/virtual_world_transition_preflight_schema.py:147`; `mutation`; subscript_assignment
- `adapters/vision_adapter.py` — `WRAP`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/vision_adapter.py:42`; `mutation`; attribute_assignment
- `adapters/visual_observation_schema.py` — `EXPERIMENTAL`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/visual_observation_schema.py:138`; `mutation`; mutation_method=items.append
- `adapters/visualizer_server.py` — `WRAP`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/visualizer_server.py:78-82`; `mutation`; subscript_assignment
- `adapters/voice_loop_adapter.py` — `WRAP`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/voice_loop_adapter.py:31`; `mutation`; attribute_assignment
- `adapters/vsa_adapter.py` — `WRAP`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/vsa_adapter.py:28`; `mutation`; attribute_assignment
- `adapters/web_learning_adapter.py` — `WRAP`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/web_learning_adapter.py:51`; `mutation`; attribute_assignment
- `adapters/world_model_adapter.py` — `WRAP`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `adapters/world_model_adapter.py:69`; `mutation`; attribute_assignment
- `agent/__init__.py` — `EXPERIMENTAL`; `agent/__init__.py:1`; tracked Python module, AST parse/reachability result `reachable_from_active_root=False; references=0; component_evidence=0`
- `agent/tools.py` — `KEEP`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `agent/tools.py:29-31`; `entrypoint`; callable_name=run
- `blueprint.py` — `KEEP`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `blueprint.py:59`; `mutation`; attribute_assignment
- `cleanup_drive.py` — `EXPERIMENTAL`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `cleanup_drive.py:107`; `mutation`; mutation_method=to_archive.append
- `cognition/__init__.py` — `EXPERIMENTAL`; `cognition/__init__.py:1`; tracked Python module, AST parse/reachability result `reachable_from_active_root=False; references=0; component_evidence=0`
- `cognition/meaning_graph.py` — `KEEP`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `cognition/meaning_graph.py:15`; `mutation`; attribute_assignment
- `core/__init__.py` — `EXPERIMENTAL`; `core/__init__.py:1`; tracked Python module, AST parse/reachability result `reachable_from_active_root=False; references=0; component_evidence=0`
- `core/length_decider.py` — `WRAP`; `M0_C_PERSISTENCE_AND_STATE_MAP.md` → `core/length_decider.py:39`; `hormone_state`; hormone_symbol=emotion_max
- `core/reasoning.py` — `WRAP`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `core/reasoning.py:45`; `mutation`; attribute_assignment
- `core/simulation.py` — `EXPERIMENTAL`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `core/simulation.py:31`; `mutation`; attribute_assignment
- `core/system1.py` — `WRAP`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `core/system1.py:36`; `mutation`; attribute_assignment
- `core/system2.py` — `KEEP`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `core/system2.py:17`; `mutation`; attribute_assignment
- `core/workspace.py` — `WRAP`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `core/workspace.py:22`; `mutation`; attribute_assignment
- `day6_integration.py` — `EXPERIMENTAL`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `day6_integration.py:7`; `mutation`; mutation_method=sys.path.insert
- `day8_hybrid.py` — `EXPERIMENTAL`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `day8_hybrid.py:20`; `mutation`; mutation_method=sys.path.insert
- `day8_hybrid_v2.py` — `EXPERIMENTAL`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `day8_hybrid_v2.py:11`; `mutation`; mutation_method=sys.path.insert
- `day8_hybrid_v3.py` — `EXPERIMENTAL`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `day8_hybrid_v3.py:18`; `mutation`; mutation_method=sys.path.insert
- `day8_hybrid_v4.py` — `EXPERIMENTAL`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `day8_hybrid_v4.py:13`; `mutation`; mutation_method=sys.path.insert
- `day8_layer1.py` — `EXPERIMENTAL`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `day8_layer1.py:9`; `mutation`; mutation_method=sys.path.insert
- `day8_layer1_v2.py` — `EXPERIMENTAL`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `day8_layer1_v2.py:9`; `mutation`; mutation_method=sys.path.insert
- `day8_layer1_v3.py` — `EXPERIMENTAL`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `day8_layer1_v3.py:10`; `mutation`; mutation_method=sys.path.insert
- `day8_option_b.py` — `EXPERIMENTAL`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `day8_option_b.py:17`; `mutation`; mutation_method=sys.path.insert
- `day8_option_b_v3.py` — `EXPERIMENTAL`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `day8_option_b_v3.py:10`; `mutation`; mutation_method=sys.path.insert
- `day8_vogels.py` — `EXPERIMENTAL`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `day8_vogels.py:18`; `mutation`; mutation_method=sys.path.insert
- `digital_somatic.py` — `KEEP`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `digital_somatic.py:9`; `import`; from __future__ import annotations
- `dmn.py` — `KEEP`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `dmn.py:9`; `import`; from __future__ import annotations
- `episodic.py` — `WRAP`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `episodic.py:103`; `mutation`; attribute_assignment
- `eve_ai2thor_colab.py` — `EXPERIMENTAL`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `eve_ai2thor_colab.py:37`; `mutation`; subscript_assignment
- `eve_all_in_one.py` — `EXPERIMENTAL`; `M0_B_GATE_FAILURE_CLOCK_CONCURRENCY_MAP.md` → `eve_all_in_one.py:26`; `output`; output_call=print
- `eve_main_ab.py` — `EXPERIMENTAL`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `eve_main_ab.py:37`; `mutation`; mutation_method=sys.path.insert
- `eve_main_abc.py` — `EXPERIMENTAL`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `eve_main_abc.py:38`; `mutation`; mutation_method=sys.path.insert
- `eve_optuna_tune.py` — `KEEP`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `eve_optuna_tune.py:65`; `mutation`; subscript_assignment
- `eve_tune_v2.py` — `EXPERIMENTAL`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `eve_tune_v2.py:26`; `mutation`; subscript_assignment
- `eve_tune_v3_stable.py` — `EXPERIMENTAL`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `eve_tune_v3_stable.py:47`; `mutation`; subscript_assignment
- `eve_v12_massive.py` — `EXPERIMENTAL`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `eve_v12_massive.py:28`; `mutation`; mutation_method=sys.path.insert
- `eve_v15_synaptic.py` — `EXPERIMENTAL`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `eve_v15_synaptic.py:39`; `mutation`; attribute_assignment
- `eve_v18_complete.py` — `EXPERIMENTAL`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `eve_v18_complete.py:64`; `mutation`; attribute_assignment
- `eve_v19_humanlike.py` — `EXPERIMENTAL`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `eve_v19_humanlike.py:33`; `mutation`; attribute_assignment
- `eve_v20_safe.py` — `EXPERIMENTAL`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `eve_v20_safe.py:28`; `mutation`; attribute_assignment
- `eve_v21_real.py` — `EXPERIMENTAL`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `eve_v21_real.py:44`; `mutation`; attribute_assignment
- `eve_v22_meaning.py` — `EXPERIMENTAL`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `eve_v22_meaning.py:41`; `mutation`; attribute_assignment
- `eve_v23_auto.py` — `EXPERIMENTAL`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `eve_v23_auto.py:30`; `mutation`; mutation_method=set
- `eve_v24_12hours.py` — `EXPERIMENTAL`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `eve_v24_12hours.py:34`; `mutation`; mutation_method=set
- `eve_virtual_learn_100.py` — `EXPERIMENTAL`; `M0_B_GATE_FAILURE_CLOCK_CONCURRENCY_MAP.md` → `eve_virtual_learn_100.py:7`; `output`; output_call=print
- `hormone_system.py` — `WRAP`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `hormone_system.py:56`; `mutation`; attribute_assignment
- `language/__init__.py` — `EXPERIMENTAL`; `language/__init__.py:1`; tracked Python module, AST parse/reachability result `reachable_from_active_root=False; references=0; component_evidence=0`
- `language/generator.py` — `KEEP`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `language/generator.py:12`; `import`; from utils.types import ResponsePlan
- `language/planner.py` — `WRAP`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `language/planner.py:55`; `mutation`; attribute_assignment
- `language/understanding.py` — `WRAP`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `language/understanding.py:86`; `mutation`; attribute_assignment
- `learning/__init__.py` — `EXPERIMENTAL`; `learning/__init__.py:1`; tracked Python module, AST parse/reachability result `reachable_from_active_root=False; references=0; component_evidence=0`
- `learning/code_synthesis.py` — `KEEP`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `learning/code_synthesis.py:22`; `mutation`; attribute_assignment
- `legacy/eve_main_abc.py` — `EXPERIMENTAL`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `legacy/eve_main_abc.py:52`; `mutation`; mutation_method=sys.path.insert
- `legacy/eve_modules/active_inference.py` — `EXPERIMENTAL`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `legacy/eve_modules/active_inference.py:108`; `mutation`; attribute_assignment
- `legacy/eve_modules/agency.py` — `EXPERIMENTAL`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `legacy/eve_modules/agency.py:109`; `mutation`; attribute_assignment
- `legacy/eve_modules/allostatic_learn.py` — `EXPERIMENTAL`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `legacy/eve_modules/allostatic_learn.py:106`; `mutation`; attribute_assignment
- `legacy/eve_modules/analogy.py` — `EXPERIMENTAL`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `legacy/eve_modules/analogy.py:94`; `mutation`; attribute_assignment
- `legacy/eve_modules/apc_learner.py` — `EXPERIMENTAL`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `legacy/eve_modules/apc_learner.py:88`; `mutation`; attribute_assignment
- `legacy/eve_modules/causal_graph.py` — `EXPERIMENTAL`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `legacy/eve_modules/causal_graph.py:100`; `mutation`; attribute_assignment
- `legacy/eve_modules/continual_rehearsal.py` — `EXPERIMENTAL`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `legacy/eve_modules/continual_rehearsal.py:81`; `mutation`; attribute_assignment
- `legacy/eve_modules/corpus_learner.py` — `EXPERIMENTAL`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `legacy/eve_modules/corpus_learner.py:93`; `mutation`; attribute_assignment
- `legacy/eve_modules/counterfactual.py` — `EXPERIMENTAL`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `legacy/eve_modules/counterfactual.py:101`; `mutation`; attribute_assignment
- `legacy/eve_modules/creative.py` — `EXPERIMENTAL`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `legacy/eve_modules/creative.py:72`; `mutation`; attribute_assignment
- `legacy/eve_modules/creative_advanced.py` — `EXPERIMENTAL`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `legacy/eve_modules/creative_advanced.py:68`; `mutation`; attribute_assignment
- `legacy/eve_modules/deep_reasoning.py` — `EXPERIMENTAL`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `legacy/eve_modules/deep_reasoning.py:88`; `mutation`; attribute_assignment
- `legacy/eve_modules/digital_somatic.py` — `WRAP`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `legacy/eve_modules/digital_somatic.py:68`; `mutation`; attribute_assignment
- `legacy/eve_modules/dmn.py` — `WRAP`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `legacy/eve_modules/dmn.py:57`; `mutation`; attribute_assignment
- `legacy/eve_modules/emotion_regulation.py` — `EXPERIMENTAL`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `legacy/eve_modules/emotion_regulation.py:114`; `mutation`; attribute_assignment
- `legacy/eve_modules/episodic.py` — `EXPERIMENTAL`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `legacy/eve_modules/episodic.py:103`; `mutation`; attribute_assignment
- `legacy/eve_modules/frame_semantics.py` — `EXPERIMENTAL`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `legacy/eve_modules/frame_semantics.py:138`; `mutation`; attribute_assignment
- `legacy/eve_modules/goal_management.py` — `EXPERIMENTAL`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `legacy/eve_modules/goal_management.py:140`; `mutation`; attribute_assignment
- `legacy/eve_modules/hormone_system.py` — `EXPERIMENTAL`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `legacy/eve_modules/hormone_system.py:87`; `mutation`; attribute_assignment
- `legacy/eve_modules/humor.py` — `EXPERIMENTAL`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `legacy/eve_modules/humor.py:77`; `mutation`; attribute_assignment
- `legacy/eve_modules/hypergraph.py` — `EXPERIMENTAL`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `legacy/eve_modules/hypergraph.py:118`; `mutation`; attribute_assignment
- `legacy/eve_modules/metacognition.py` — `EXPERIMENTAL`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `legacy/eve_modules/metacognition.py:102`; `mutation`; attribute_assignment
- `legacy/eve_modules/multi_stream.py` — `EXPERIMENTAL`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `legacy/eve_modules/multi_stream.py:54`; `mutation`; attribute_assignment
- `legacy/eve_modules/narrative_self.py` — `EXPERIMENTAL`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `legacy/eve_modules/narrative_self.py:70`; `mutation`; attribute_assignment
- `legacy/eve_modules/natural_lang.py` — `EXPERIMENTAL`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `legacy/eve_modules/natural_lang.py:163`; `mutation`; attribute_assignment
- `legacy/eve_modules/norm_internal.py` — `EXPERIMENTAL`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `legacy/eve_modules/norm_internal.py:116`; `mutation`; attribute_assignment
- `legacy/eve_modules/self_doubt.py` — `EXPERIMENTAL`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `legacy/eve_modules/self_doubt.py:111`; `mutation`; attribute_assignment
- `legacy/eve_modules/semantic_distance.py` — `EXPERIMENTAL`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `legacy/eve_modules/semantic_distance.py:76`; `mutation`; attribute_assignment
- `legacy/eve_modules/spreading_activation.py` — `WRAP`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `legacy/eve_modules/spreading_activation.py:92`; `mutation`; attribute_assignment
- `legacy/eve_modules/suffering.py` — `EXPERIMENTAL`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `legacy/eve_modules/suffering.py:77`; `mutation`; attribute_assignment
- `legacy/eve_modules/temporal.py` — `EXPERIMENTAL`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `legacy/eve_modules/temporal.py:88`; `mutation`; attribute_assignment
- `legacy/eve_modules/tool_use.py` — `EXPERIMENTAL`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `legacy/eve_modules/tool_use.py:60`; `mutation`; attribute_assignment
- `legacy/eve_modules/vsa_binding.py` — `EXPERIMENTAL`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `legacy/eve_modules/vsa_binding.py:86`; `mutation`; attribute_assignment
- `legacy/eve_modules/working_memory.py` — `WRAP`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `legacy/eve_modules/working_memory.py:51`; `mutation`; attribute_assignment
- `legacy/eve_modules/world_model.py` — `EXPERIMENTAL`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `legacy/eve_modules/world_model.py:105`; `mutation`; attribute_assignment
- `legacy/v36_modules/airi_adapter.py` — `EXPERIMENTAL`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `legacy/v36_modules/airi_adapter.py:96`; `mutation`; subscript_assignment
- `legacy/v36_modules/airi_server.py` — `KEEP`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `legacy/v36_modules/airi_server.py:50`; `mutation`; mutation_method=sys.path.insert
- `legacy/v36_modules/broca_qwen.py` — `EXPERIMENTAL`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `legacy/v36_modules/broca_qwen.py:99`; `mutation`; attribute_assignment
- `legacy/v36_modules/commonsense_seed.py` — `EXPERIMENTAL`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `legacy/v36_modules/commonsense_seed.py:286`; `mutation`; mutation_method=set
- `legacy/v36_modules/dashboard.py` — `KEEP`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `legacy/v36_modules/dashboard.py:34`; `mutation`; mutation_method=sys.path.insert
- `legacy/v36_modules/dashboard_data.py` — `EXPERIMENTAL`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `legacy/v36_modules/dashboard_data.py:125-131`; `mutation`; subscript_assignment
- `legacy/v36_modules/decide_action.py` — `EXPERIMENTAL`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `legacy/v36_modules/decide_action.py:123`; `mutation`; attribute_assignment
- `legacy/v36_modules/env_adapter.py` — `EXPERIMENTAL`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `legacy/v36_modules/env_adapter.py:50`; `mutation`; attribute_assignment
- `legacy/v36_modules/eve_room.py` — `EXPERIMENTAL`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `legacy/v36_modules/eve_room.py:79`; `mutation`; subscript_assignment
- `legacy/v36_modules/eve_v36.py` — `EXPERIMENTAL`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `legacy/v36_modules/eve_v36.py:24`; `mutation`; mutation_method=sys.path.insert
- `legacy/v36_modules/eve_v39.py` — `EXPERIMENTAL`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `legacy/v36_modules/eve_v39.py:32`; `mutation`; mutation_method=sys.path.insert
- `legacy/v36_modules/integrated_self.py` — `EXPERIMENTAL`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `legacy/v36_modules/integrated_self.py:97`; `mutation`; mutation_method=bits.append
- `legacy/v36_modules/persistence.py` — `EXPERIMENTAL`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `legacy/v36_modules/persistence.py:136-144`; `mutation`; subscript_assignment
- `legacy/v36_modules/proactive.py` — `EXPERIMENTAL`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `legacy/v36_modules/proactive.py:62`; `mutation`; attribute_assignment
- `legacy/v36_modules/response_enhancer.py` — `EXPERIMENTAL`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `legacy/v36_modules/response_enhancer.py:183`; `mutation`; attribute_assignment
- `legacy/v36_modules/social_env.py` — `EXPERIMENTAL`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `legacy/v36_modules/social_env.py:86`; `mutation`; attribute_assignment
- `legacy/v36_modules/symbolic_env.py` — `EXPERIMENTAL`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `legacy/v36_modules/symbolic_env.py:96`; `mutation`; subscript_assignment
- `legacy/v36_modules/user_presence.py` — `EXPERIMENTAL`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `legacy/v36_modules/user_presence.py:48`; `mutation`; attribute_assignment
- `memory/__init__.py` — `EXPERIMENTAL`; `memory/__init__.py:1`; tracked Python module, AST parse/reachability result `reachable_from_active_root=False; references=0; component_evidence=0`
- `natural_lang.py` — `WRAP`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `natural_lang.py:108`; `mutation`; attribute_assignment
- `natural_lang1.py` — `EXPERIMENTAL`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `natural_lang1.py:131`; `mutation`; attribute_assignment
- `round71_full_suite_runner.py` — `EXPERIMENTAL`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `round71_full_suite_runner.py:11`; `direct_write`; write_call=status_path.write_text
- `round72_split_suite_runner.py` — `EXPERIMENTAL`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `round72_split_suite_runner.py:13`; `direct_write`; write_call=status_path.write_text
- `round77_split_suite_runner.py` — `EXPERIMENTAL`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `round77_split_suite_runner.py:10`; `direct_write`; write_call=chunks_path.write_text
- `self_doubt.py` — `WRAP`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `self_doubt.py:111`; `mutation`; attribute_assignment
- `spreading_activation.py` — `KEEP`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `spreading_activation.py:9`; `import`; from __future__ import annotations
- `synaptic_scaling.py` — `EXPERIMENTAL`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `synaptic_scaling.py:10`; `mutation`; mutation_method=sys.path.insert
- `synaptic_scaling_v2.py` — `EXPERIMENTAL`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `synaptic_scaling_v2.py:9`; `mutation`; mutation_method=sys.path.insert
- `utils/__init__.py` — `EXPERIMENTAL`; `utils/__init__.py:1`; tracked Python module, AST parse/reachability result `reachable_from_active_root=False; references=0; component_evidence=0`
- `utils/korean_endings.py` — `KEEP`; `utils/korean_endings.py:1`; tracked Python module, AST parse/reachability result `reachable_from_active_root=True; references=0; component_evidence=0`
- `utils/korean_particles.py` — `KEEP`; `utils/korean_particles.py:1`; tracked Python module, AST parse/reachability result `reachable_from_active_root=True; references=0; component_evidence=0`
- `utils/legacy_path.py` — `KEEP`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `utils/legacy_path.py:25`; `mutation`; mutation_method=sys.path.insert
- `utils/mock_eve.py` — `WRAP`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `utils/mock_eve.py:20`; `mutation`; attribute_assignment
- `utils/speech_style.py` — `EXPERIMENTAL`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `utils/speech_style.py:174`; `mutation`; mutation_method=keywords.append
- `utils/types.py` — `WRAP`; `M0_C_PERSISTENCE_AND_STATE_MAP.md` → `utils/types.py:18`; `state_domain`; state_symbol=emotions;domain=affect_hormones
- `v2/__init__.py` — `EXPERIMENTAL`; `v2/__init__.py:1`; tracked Python module, AST parse/reachability result `reachable_from_active_root=False; references=0; component_evidence=0`
- `v2/core/__init__.py` — `EXPERIMENTAL`; `v2/core/__init__.py:1`; tracked Python module, AST parse/reachability result `reachable_from_active_root=False; references=0; component_evidence=0`
- `v2/development/__init__.py` — `EXPERIMENTAL`; `v2/development/__init__.py:1`; tracked Python module, AST parse/reachability result `reachable_from_active_root=False; references=0; component_evidence=0`
- `v2/environment/__init__.py` — `EXPERIMENTAL`; `v2/environment/__init__.py:1`; tracked Python module, AST parse/reachability result `reachable_from_active_root=False; references=0; component_evidence=0`
- `v2/innate/__init__.py` — `EXPERIMENTAL`; `v2/innate/__init__.py:1`; tracked Python module, AST parse/reachability result `reachable_from_active_root=False; references=0; component_evidence=0`
- `v2/language/__init__.py` — `EXPERIMENTAL`; `v2/language/__init__.py:1`; tracked Python module, AST parse/reachability result `reachable_from_active_root=False; references=0; component_evidence=0`
- `v2/spatial_temporal/__init__.py` — `EXPERIMENTAL`; `v2/spatial_temporal/__init__.py:1`; tracked Python module, AST parse/reachability result `reachable_from_active_root=False; references=0; component_evidence=0`
- `v2/storage/__init__.py` — `EXPERIMENTAL`; `v2/storage/__init__.py:1`; tracked Python module, AST parse/reachability result `reachable_from_active_root=False; references=0; component_evidence=0`
- `v2/tests/__init__.py` — `EXPERIMENTAL`; `v2/tests/__init__.py:1`; tracked Python module, AST parse/reachability result `reachable_from_active_root=False; references=0; component_evidence=0`
- `v2/utils/__init__.py` — `EXPERIMENTAL`; `v2/utils/__init__.py:1`; tracked Python module, AST parse/reachability result `reachable_from_active_root=False; references=0; component_evidence=0`
- `v34_module1_snn.py` — `EXPERIMENTAL`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `v34_module1_snn.py:96`; `mutation`; attribute_assignment
- `working_memory.py` — `KEEP`; `M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md` → `working_memory.py:9`; `import`; from __future__ import annotations

## Validation

```text
compileall: PASS
deterministic double-run: byte-identical
focused M0-D tests: 11 passed in 42.37s
collection: 2,584 tests collected in 5.16s
full suite: 2,584 passed in 84.66s
temporary six-file scope: PASS
```

Validation run `29680010246`, artifact `eve-m0-d-validation`, SHA-256 `7aca00e64dd30151044131f1726dfa28461ea7ae99698bcf7a789fe09f36c9d1`.

## Decision boundary

No recommendation is implemented here. Reviewer approval is required for unresolved rulings, all `REWRITE`/`DEPRECATE` recommendations, and any later frozen-PR close action.
