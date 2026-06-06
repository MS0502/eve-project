from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from adapters.affect_hormone_neural_rhythm_registry import (
    AXIS_GROUPS,
    HARDWARE_FORBIDDEN_AFFECT_OUTCOMES,
    SOCIAL_SELF_IDENTITY_AXES,
    affect_hormone_axis_registry,
    anti_global_synchrony_policy,
    build_round761_780_registry_report,
    classify_battery_governor_band,
    hardware_governor_policy,
    neural_rhythm_schema,
    no_runtime_mutation_proof,
    rhythm_connection_map,
    validate_registry_invariants,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_all_required_affect_hormone_axes_exist() -> None:
    registry = affect_hormone_axis_registry()
    required_axes = [axis for axes in AXIS_GROUPS.values() for axis in axes]

    assert len(required_axes) == 37
    assert set(registry) == set(required_axes)
    assert validate_registry_invariants(registry)["passed"] is True


def test_each_axis_has_independent_decay_cycle_refractory_and_evidence_fields() -> None:
    registry = affect_hormone_axis_registry()

    for axis, entry in registry.items():
        assert entry["axis"] == axis
        assert isinstance(entry["decay_rate"], float)
        assert entry["decay_rate"] > 0
        assert entry["cycle_class"]
        assert entry["rhythm_class"]
        assert isinstance(entry["refractory_ticks"], int)
        assert entry["refractory_ticks"] > 0
        assert entry["evidence_required"]
        assert entry["activation_pattern_inputs"]


def test_all_axes_have_saturation_and_baseline_return_requirements() -> None:
    registry = affect_hormone_axis_registry()

    for entry in registry.values():
        assert 0.0 <= entry["min"] <= entry["baseline"] <= entry["max"] <= 1.0
        assert entry["default"] == entry["baseline"]
        assert entry["spike_limit"] > 0
        assert "baseline" in entry["saturation_policy"]
        assert "refractory" in entry["saturation_policy"]


def test_all_axes_forbid_agp_and_fallback_bypass() -> None:
    registry = affect_hormone_axis_registry()

    for entry in registry.values():
        assert entry["agp_bypass_allowed"] is False
        assert entry["fallback_bypass_allowed"] is False
        assert entry["panic_allowed"] is False
        assert entry["persistence_write_allowed"] is False
        assert entry["vector_read_allowed"] is False


def test_hardware_signal_cannot_directly_affect_social_self_identity_axes() -> None:
    registry = affect_hormone_axis_registry()

    assert SOCIAL_SELF_IDENTITY_AXES
    for axis in SOCIAL_SELF_IDENTITY_AXES:
        assert registry[axis]["hardware_direct_input_allowed"] is False


def test_normal_battery_range_has_zero_affect_impact() -> None:
    at_50 = classify_battery_governor_band(50)
    high = classify_battery_governor_band(87)
    light = classify_battery_governor_band(42)

    assert at_50["mode"] == "normal"
    assert at_50["affect_effect"] == "none"
    assert at_50["warning"] is None
    assert high["affect_effect"] == "none"
    assert light["mode"] == "light_conserve"
    assert light["affect_effect"] == "none"


def test_battery_drop_cannot_trigger_forbidden_existential_or_social_outcomes() -> None:
    for level in (95, 49, 34, 19, 9, 2):
        band = classify_battery_governor_band(level)
        assert band["panic_allowed"] is False
        assert band["death_fear_allowed"] is False
        assert band["identity_threat_allowed"] is False
        assert band["social_pain_allowed"] is False
        assert band["abandonment_fear_allowed"] is False
        assert band["self_worth_change_allowed"] is False

    policy = hardware_governor_policy()
    assert set(policy["battery_drop_alone_cannot_trigger"]) == set(HARDWARE_FORBIDDEN_AFFECT_OUTCOMES)


def test_hardware_governor_uses_hysteresis_debounce_trend_and_no_recursive_loops() -> None:
    policy = hardware_governor_policy()
    schema = neural_rhythm_schema()["hardware_tick"]

    assert policy["hysteresis_required"] is True
    assert policy["debounce_required"] is True
    assert policy["trend_rules_required"] is True
    assert policy["rate_limited"] is True
    assert policy["hardware_polling_can_create_recursive_concern_loops"] is False
    assert schema["hysteresis_required"] is True
    assert schema["debounce_required"] is True
    assert schema["trend_window_required"] is True
    assert schema["recursive_concern_loop_allowed"] is False


def test_hardware_state_cannot_write_memory_or_self_model_and_critical_is_graceful_only() -> None:
    policy = hardware_governor_policy()
    critical = classify_battery_governor_band(3)

    assert policy["hardware_state_can_directly_write_memory_or_self_model"] is False
    assert policy["critical_states_may_request_panic"] is False
    assert set(policy["critical_states_may_request"]) == {"graceful_pause", "checkpoint"}
    assert critical["memory_write_allowed"] is False
    assert critical["self_model_write_allowed"] is False
    assert critical["affect_effect"] == "no_panic_no_death_framing_graceful_pause_only"


def test_thought_imagination_speech_listening_memory_action_connected_to_activation_schema() -> None:
    connection_map = rhythm_connection_map()
    schema = neural_rhythm_schema()

    assert set(connection_map) == {"thought", "imagination", "speech", "listening", "memory", "action"}
    for surface, entry in connection_map.items():
        assert entry["activation_pattern_schema_connected"] is True
        for rhythm in entry["rhythms"]:
            assert rhythm in schema, surface


def test_speaking_cannot_bypass_agp_or_fallback_or_emit_from_affect_spike() -> None:
    speech = rhythm_connection_map()["speech"]

    assert speech["agp_required"] is True
    assert speech["fallback_required"] is True
    assert speech["can_bypass_agp"] is False
    assert speech["can_bypass_fallback"] is False
    assert speech["can_emit_from_affect_spike_directly"] is False


def test_listening_cannot_relabel_neutral_input_as_hostile_by_threat_pressure_alone() -> None:
    listening = rhythm_connection_map()["listening"]
    schema = neural_rhythm_schema()["listening_tick"]

    assert listening["can_request_extra_appraisal_under_uncertainty_or_threat"] is True
    assert listening["can_relabel_neutral_as_hostile_by_threat_pressure_alone"] is False
    assert "cannot_relabel_neutral_input_as_hostile_by_itself" in schema["safety_boundary"]


def test_imagination_has_scenario_budget_cooldown_and_reality_check_boundary() -> None:
    imagination = rhythm_connection_map()["imagination"]
    schema = neural_rhythm_schema()["imagination_tick"]

    assert imagination["scenario_budget_required"] is True
    assert imagination["cooldown_required"] is True
    assert imagination["reality_check_boundary_required"] is True
    assert schema["scenario_budget"] > 0
    assert schema["cooldown_ticks"] > 0
    assert "simulation" in schema["reality_check_boundary"]


def test_memory_requires_appraisal_and_quarantine_before_long_term_update_from_social_feedback() -> None:
    memory = rhythm_connection_map()["memory"]

    assert memory["requires_appraisal_for_social_feedback"] is True
    assert memory["requires_quarantine_for_social_feedback"] is True
    assert memory["can_write_long_term_memory_from_raw_feedback"] is False


def test_global_synchrony_runaway_is_forbidden() -> None:
    policy = anti_global_synchrony_policy()

    assert policy["global_synchrony_runaway_allowed"] is False
    assert policy["one_event_can_spike_all_axes"] is False
    assert policy["max_axis_groups_spiked_by_single_event"] < len(AXIS_GROUPS)
    assert policy["requires_saturation_and_baseline_return"] is True
    assert policy["activation_patterns_can_directly_mutate_live_state"] is False


def test_no_live_emotion_hormone_memory_mutation_or_persistence_enablement() -> None:
    proof = no_runtime_mutation_proof()
    report = build_round761_780_registry_report()

    assert proof["emotion_transition_applied"] is False
    assert proof["live_emotion_state_mutated"] is False
    assert proof["live_hormone_state_mutated"] is False
    assert proof["memory_written"] is False
    assert proof["persistence_written"] is False
    assert report["no_persistence_proof"]["production_persistence_enabled"] is False


def test_runtime_mapping_and_enforcement_defaults_remain_disabled() -> None:
    report = build_round761_780_registry_report()

    assert report["no_persistence_proof"]["runtime_mapping_enabled_default"] is False
    assert report["no_persistence_proof"]["enforcement_enabled_default"] is False
    assert report["no_runtime_mutation_proof"]["runtime_route_registered"] is False


def test_no_vector_content_read_or_load() -> None:
    report = build_round761_780_registry_report()

    assert report["no_vector_content_read_load_proof"]["vector_contents_read"] is False
    assert report["no_vector_content_read_load_proof"]["vectors_loaded"] is False
    assert report["no_vector_content_read_load_proof"]["vector_read_allowed_for_any_axis"] is False


def test_operator_report_is_compact_json_and_read_only_summary() -> None:
    proc = subprocess.run(
        [sys.executable, "scripts/operator_report_round761_780_affect_hormone_neural_rhythm_registry.py"],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    payload = json.loads(proc.stdout)

    assert payload["operator_command"] == "python scripts/operator_report_round761_780_affect_hormone_neural_rhythm_registry.py"
    assert payload["operator_report_path"] == "docs/round761_780_affect_hormone_neural_rhythm_registry.md"
    assert payload["full_affect_hormone_axis_registry_summary"]["axis_count"] == 37
    assert payload["registry_invariant_summary"]["passed"] is True
    assert payload["no_runtime_mutation_proof"]["hardware_polling_started"] is False
    assert payload["runtime_defaults_proof"]["default_runtime_remains_no_load"] is True
    assert payload["next_implementation_recommendation"] == (
        "add_a_read_only_affect_proposal_validator_against_this_registry_before_any_transition_apply_round"
    )


def test_no_forbidden_operator_artifacts_or_vectors_are_staged_or_tracked() -> None:
    proc = subprocess.run(
        ["git", "status", "--short"],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    forbidden_patterns = (
        "_operator_artifacts/",
        "vectors.npy",
        "vocab.txt",
        "subset_manifest.json",
        "seeds/subsets",
        ".zip",
        ".part",
    )

    assert not [line for line in proc.stdout.splitlines() if any(pattern in line for pattern in forbidden_patterns)]


def test_korean_fixtures_and_minseok_literal_remain_preserved() -> None:
    # This round does not edit fixture/runtime Korean text.  The preserved literal
    # remains present in existing source surfaces without adding vector/artifact reads.
    fixture_candidates = [
        REPO_ROOT / "adapters" / "speech_hub.py",
        REPO_ROOT / "adapters" / "situation_responder.py",
    ]

    assert any("민석" in path.read_text(encoding="utf-8") for path in fixture_candidates)
