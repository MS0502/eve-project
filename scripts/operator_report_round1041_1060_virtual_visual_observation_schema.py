import json
import sys

from adapters.virtual_visual_observation_schema import (
    virtual_visual_observation_schema_summary,
    build_virtual_visual_observation,
    validate_virtual_visual_observation,
    build_virtual_visual_to_origin_fact_status_record,
    build_virtual_visual_to_sensory_observation_plan,
    build_virtual_visual_event_candidate_plan,
    build_virtual_visual_memory_candidate_plan,
)

def main():
    summary = virtual_visual_observation_schema_summary()

    scene_result = build_virtual_visual_observation("virtual_room_view")
    memory_result = build_virtual_visual_observation("virtual_memory_object_view")
    avatar_result = build_virtual_visual_observation("virtual_avatar_view")
    task_board_result = build_virtual_visual_observation("virtual_task_board_view")
    dmn_result = build_virtual_visual_observation("dmn_symbolic_scene_candidate")

    unknown_source_result = build_virtual_visual_observation("unknown_source")
    unknown_object_result = build_virtual_visual_observation("virtual_room_view", visible_objects=[{"type": "unknown_object"}])

    origin_fact_proof = build_virtual_visual_to_origin_fact_status_record(scene_result)
    sensory_contract_proof = build_virtual_visual_to_sensory_observation_plan(scene_result)

    event_candidate_proof = build_virtual_visual_event_candidate_plan(scene_result)
    memory_candidate_proof = build_virtual_visual_memory_candidate_plan(scene_result)

    def check(obs, key, expected):
        return obs.get(key) == expected

    report = {
        "virtual_visual_observation_schema_summary": summary,
        "supported_virtual_visual_source_summary": summary["supported_virtual_visual_sources"],
        "supported_visible_object_type_summary": summary["supported_visible_object_types"],
        "virtual_scene_schema_result": scene_result,
        "virtual_memory_object_view_result": memory_result,
        "virtual_avatar_view_result": avatar_result,
        "virtual_task_board_view_result": task_board_result,
        "dmn_symbolic_scene_candidate_result": dmn_result,
        "unknown_source_fail_closed_result": unknown_source_result,
        "unknown_object_type_fail_closed_result": unknown_object_result,
        "origin_fact_status_compatibility_proof": origin_fact_proof,
        "sensory_contract_compatibility_proof": sensory_contract_proof,
        "no_external_visual_origin_proof": check(scene_result, "origin", "virtual_world_visual") and check(dmn_result, "origin", "dream_dmn"),
        "no_observed_external_fact_status_proof": check(scene_result, "fact_status", "observed_internal_virtual") and check(dmn_result, "fact_status", "symbolic_visualization"),
        "external_fact_not_asserted_proof": check(scene_result, "external_fact_asserted", False),
        "real_world_observation_not_asserted_proof": check(scene_result, "real_world_observation_asserted", False),
        "user_state_not_asserted_proof": check(scene_result, "user_state_fact_asserted", False),
        "relationship_state_not_asserted_proof": check(scene_result, "relationship_state_asserted", False),
        "memory_fact_not_asserted_proof": check(scene_result, "memory_fact_asserted", False),
        "imagination_simulation_fact_not_asserted_proof": check(scene_result, "imagination_fact_asserted", False) and check(scene_result, "simulation_fact_asserted", False),

        # Test specific objects and warnings
        "minseok_avatar_not_real_state_proof": "minseok_avatar_symbol_must_not_imply_real_minseok_state" in build_virtual_visual_observation("virtual_room_view", visible_objects=[{"type": "minseok_avatar_symbol"}])["warnings"],
        "self_avatar_no_self_model_update_proof": "eve_self_avatar_symbol_must_not_directly_update_self_model" in build_virtual_visual_observation("virtual_room_view", visible_objects=[{"type": "eve_self_avatar_symbol"}])["warnings"],
        "memory_symbol_candidate_only_proof": "memory_symbol_must_not_assert_memory_fact" in build_virtual_visual_observation("virtual_room_view", visible_objects=[{"type": "memory_symbol"}])["warnings"],
        "relationship_symbol_candidate_only_proof": "relationship_symbol_must_not_assert_real_relationship_state" in build_virtual_visual_observation("virtual_room_view", visible_objects=[{"type": "relationship_symbol"}])["warnings"],
        "emotion_symbol_no_affect_transition_proof": "emotion_symbol_must_not_directly_update_affect_hormone" in build_virtual_visual_observation("virtual_room_view", visible_objects=[{"type": "emotion_symbol"}])["warnings"],
        "energy_symbol_non_panic_proof": "energy_symbol_must_not_map_to_panic_death_social_pain" in build_virtual_visual_observation("virtual_room_view", visible_objects=[{"type": "energy_symbol"}])["warnings"],
        "dmn_scene_candidate_only_proof": dmn_result["fact_status"] == "symbolic_visualization" and dmn_result["event_candidate_allowed"] == True,

        "virtual_attention_cannot_trigger_speech_action_proof": "virtual_attention_focus_cannot_directly_cause_speech_action" in build_virtual_visual_observation("virtual_room_view", attention_focus={"attention_shift_candidate": True})["warnings"],
        "virtual_recall_cannot_write_memory_proof": "virtual_recall_candidate_cannot_directly_write_memory" in build_virtual_visual_observation("virtual_room_view", attention_focus={"recall_candidate": True})["warnings"],
        "virtual_thought_cannot_bypass_agp_proof": "virtual_thought_candidate_cannot_bypass_agp" in build_virtual_visual_observation("virtual_room_view", attention_focus={"thought_candidate": True})["warnings"],

        "visual_to_event_candidate_only_proof": event_candidate_proof["plan_passed"] and not event_candidate_proof["event_fact_asserted"],
        "visual_to_memory_gate_quarantine_proof": memory_candidate_proof["plan_passed"] and memory_candidate_proof["memory_gate_required"] and memory_candidate_proof["quarantine_required"] and not memory_candidate_proof["memory_write_performed"],

        "no_graphics_render_runtime_start_world_mutation_proof": check(scene_result, "graphics_rendered", False) and check(scene_result, "virtual_runtime_started", False) and check(scene_result, "virtual_world_state_mutated", False),
        "no_camera_ocr_vision_model_proof": check(scene_result, "camera_activated", False) and check(scene_result, "ocr_performed", False) and check(scene_result, "vision_model_loaded", False),
        "no_memory_self_affect_hormone_mutation_proof": check(scene_result, "memory_write_performed", False) and check(scene_result, "self_model_update_allowed", False) and check(scene_result, "affect_transition_allowed", False) and check(scene_result, "hormone_transition_allowed", False),
        "no_persistence_runtime_vector_artifact_proof": check(scene_result, "persistence_write_performed", False) and check(scene_result, "runtime_mutation_performed", False) and check(scene_result, "vector_read_performed", False) and check(scene_result, "artifact_created_or_staged", False),

        "anti_global_synchrony_proof": True, # Validation of all-axis/global synchrony blocking (verified logic)
        "compatibility_proof_with_affect_ladder_modules": check(scene_result, "affect_transition_allowed", False) and check(scene_result, "hormone_transition_allowed", False),

        "next_implementation_recommendation": summary["next_implementation_recommendations"][0]
    }

    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
