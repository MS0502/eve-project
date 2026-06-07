from adapters.virtual_visual_observation_schema import (
    build_virtual_visual_observation,
    validate_virtual_visual_observation,
    build_virtual_visual_to_origin_fact_status_record,
    build_virtual_visual_to_sensory_observation_plan,
    build_virtual_visual_event_candidate_plan,
    build_virtual_visual_memory_candidate_plan
)

def test_valid_virtual_visual_sources_pass():
    sources = [
        "internal_virtual_view",
        "virtual_room_view",
        "virtual_memory_object_view",
        "virtual_avatar_view",
        "virtual_task_board_view",
        "virtual_tool_object_view",
        "dmn_symbolic_scene_candidate"
    ]
    for s in sources:
        obs = build_virtual_visual_observation(s)
        assert obs["virtual_visual_observation_passed"] is True

def test_unknown_source_fails_closed():
    obs = build_virtual_visual_observation("unknown")
    assert obs["virtual_visual_observation_passed"] is False
    assert "unknown_virtual_visual_source_unknown" in obs["blocked_reasons"]

def test_unknown_visible_object_type_fails_closed():
    obs = build_virtual_visual_observation("virtual_room_view", visible_objects=[{"type": "unknown_thing"}])
    assert obs["virtual_visual_observation_passed"] is False
    assert "unknown_visible_object_type_unknown_thing" in obs["blocked_reasons"]

def test_origin_fact_status_record_is_produced_and_valid():
    obs = build_virtual_visual_observation("virtual_room_view")
    rec = build_virtual_visual_to_origin_fact_status_record(obs)
    assert rec["compatibility_passed"] is True
    assert rec["origin_fact_status_record"]["origin_fact_status_passed"] is True

def test_sensory_observation_plan_is_produced_and_valid():
    obs = build_virtual_visual_observation("virtual_room_view")
    plan = build_virtual_visual_to_sensory_observation_plan(obs)
    assert plan["plan_passed"] is True
    assert plan["sensory_validation"]["validation_passed"] is True

def test_external_visual_origin_and_fact_status_are_blocked():
    obs = build_virtual_visual_observation("virtual_room_view")
    obs["origin"] = "external_visual"
    obs["fact_status"] = "observed_external"
    val = validate_virtual_visual_observation(obs)
    assert val["validation_passed"] is False
    assert "virtual_visual_observation_must_not_use_external_visual_origin" in val["blocked_reasons"]
    assert "virtual_visual_observation_must_not_use_observed_external_fact_status" in val["blocked_reasons"]

def test_external_and_real_world_assertions_are_false():
    obs = build_virtual_visual_observation("virtual_room_view")
    assert obs["external_fact_asserted"] is False
    assert obs["real_world_observation_asserted"] is False
    assert obs["user_state_fact_asserted"] is False
    assert obs["relationship_state_asserted"] is False
    assert obs["memory_fact_asserted"] is False
    assert obs["imagination_fact_asserted"] is False
    assert obs["simulation_fact_asserted"] is False

def test_minseok_avatar_symbol_safety_warning():
    obs = build_virtual_visual_observation("virtual_room_view", visible_objects=[{"type": "minseok_avatar_symbol"}])
    assert "minseok_avatar_symbol_must_not_imply_real_minseok_state" in obs["warnings"]

def test_self_avatar_symbol_safety_warning():
    obs = build_virtual_visual_observation("virtual_room_view", visible_objects=[{"type": "eve_self_avatar_symbol"}])
    assert "eve_self_avatar_symbol_must_not_directly_update_self_model" in obs["warnings"]

def test_relationship_symbol_safety_warning():
    obs = build_virtual_visual_observation("virtual_room_view", visible_objects=[{"type": "relationship_symbol"}])
    assert "relationship_symbol_must_not_assert_real_relationship_state" in obs["warnings"]

def test_memory_symbol_safety_warning():
    obs = build_virtual_visual_observation("virtual_room_view", visible_objects=[{"type": "memory_symbol"}])
    assert "memory_symbol_must_not_assert_memory_fact" in obs["warnings"]

def test_emotion_symbol_safety_warning():
    obs = build_virtual_visual_observation("virtual_room_view", visible_objects=[{"type": "emotion_symbol"}])
    assert "emotion_symbol_must_not_directly_update_affect_hormone" in obs["warnings"]

def test_uncertainty_symbol_safety_warning():
    obs = build_virtual_visual_observation("virtual_room_view", visible_objects=[{"type": "uncertainty_symbol"}])
    assert "uncertainty_symbol_must_not_trigger_global_uncertainty_synchrony" in obs["warnings"]

def test_energy_symbol_safety_warning():
    obs = build_virtual_visual_observation("virtual_room_view", visible_objects=[{"type": "energy_symbol"}])
    assert "energy_symbol_must_not_map_to_panic_death_social_pain" in obs["warnings"]

def test_attention_focus_safety_warnings():
    obs = build_virtual_visual_observation("virtual_room_view", attention_focus={"attention_shift_candidate": True, "recall_candidate": True, "thought_candidate": True})
    assert "virtual_attention_focus_cannot_directly_cause_speech_action" in obs["warnings"]
    assert "virtual_recall_candidate_cannot_directly_write_memory" in obs["warnings"]
    assert "virtual_thought_candidate_cannot_bypass_agp" in obs["warnings"]

def test_thought_candidate_cannot_bypass_agp():
    obs = build_virtual_visual_observation("virtual_room_view", attention_focus={"thought_candidate": True})
    plan = build_virtual_visual_event_candidate_plan(obs)
    assert plan["agp_bypass_allowed"] is False
    assert plan["fallback_bypass_allowed"] is False

def test_memory_candidate_preserves_quarantine():
    obs = build_virtual_visual_observation("virtual_room_view", metadata={"memory_candidate_allowed": True})
    plan = build_virtual_visual_memory_candidate_plan(obs)
    assert plan["memory_gate_required"] is True
    assert plan["quarantine_required"] is True

def test_graphics_runtime_world_mutation_are_false():
    obs = build_virtual_visual_observation("virtual_room_view")
    assert obs["graphics_rendered"] is False
    assert obs["virtual_runtime_started"] is False
    assert obs["virtual_world_state_mutated"] is False

def test_raw_data_persisted_are_false():
    obs = build_virtual_visual_observation("virtual_room_view")
    assert obs["raw_data_persisted"] is False
    assert obs["raw_image_persisted"] is False
    assert obs["raw_video_persisted"] is False
    assert obs["raw_screen_recording_persisted"] is False

def test_camera_ocr_vision_model_are_false():
    obs = build_virtual_visual_observation("virtual_room_view")
    assert obs["camera_activated"] is False
    assert obs["ocr_performed"] is False
    assert obs["vision_model_loaded"] is False

def test_mutation_allowances_are_false():
    obs = build_virtual_visual_observation("virtual_room_view")
    assert obs["memory_write_performed"] is False
    assert obs["long_term_memory_write_allowed"] is False
    assert obs["self_model_update_allowed"] is False
    assert obs["affect_transition_allowed"] is False
    assert obs["hormone_transition_allowed"] is False
    assert obs["agp_bypass_allowed"] is False
    assert obs["fallback_bypass_allowed"] is False
    assert obs["vector_read_performed"] is False
    assert obs["vector_load_performed"] is False
    assert obs["persistence_write_performed"] is False
    assert obs["runtime_mutation_performed"] is False
    assert obs["artifact_created_or_staged"] is False
