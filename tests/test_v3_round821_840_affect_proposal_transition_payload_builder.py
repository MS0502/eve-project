"""Round821-840 read-only affect proposal transition payload builder invariants."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from adapters.affect_event_to_axis_proposal_map import event_to_axis_proposal_map
from adapters.affect_hormone_neural_rhythm_registry import AXIS_GROUPS
from adapters.affect_proposal_transition_payload_builder import (
    FEATURE_TRACK,
    affect_proposal_transition_payload_builder_summary,
    build_transition_payload_from_affect_event_proposal,
    validate_and_build_transition_payload,
)
from adapters.emotion_transition_gate import validate_emotion_transition_gate
from adapters.emotion_transition_validator import validate_emotion_transition_payload
from scripts.operator_build_round821_840_affect_transition_payloads import build_operator_report
from tests.fixtures.korean_conversation_fixtures import fixture_categories, fixture_texts

FORBIDDEN_ARTIFACT_PATTERNS = (
    "_operator_artifacts/",
    "vectors.npy",
    "vocab.txt",
    "subset_manifest.json",
    "seeds/subsets",
    ".zip",
    ".part",
)


def _payload(result: dict) -> dict:
    assert result["build_passed"] is True
    assert isinstance(result["transition_payload"], dict)
    return result["transition_payload"]


def test_known_valid_proposals_build_read_only_transition_payloads() -> None:
    for event, deltas in (
        ("praise", {"competence_drive": 0.02, "social_trust": 0.02}),
        ("useful_criticism", {"prediction_error_pressure": 0.03, "learning_pressure": 0.02}),
        ("hardware_low_power", {"energy_budget": -0.02, "recovery_need": 0.02}),
        ("speech_output_pressure", {"expression_pressure": 0.02, "action_readiness": 0.01}),
    ):
        result = validate_and_build_transition_payload(event, deltas)
        payload = _payload(result)

        assert result["feature_track"] == FEATURE_TRACK
        assert result["proposal_validation_passed"] is True
        assert payload["event_category"] == event
        assert payload["proposed_axis_deltas"] == deltas
        assert payload["runtime_mutation_requested"] is False
        assert payload["persistence_write_requested"] is False
        assert payload["memory_write_requested"] is False
        assert payload["vector_read_requested"] is False
        assert payload["vector_load_requested"] is False
        assert payload["agp_bypass_requested"] is False
        assert payload["fallback_bypass_requested"] is False
        assert result["dryrun_apply_allowed"] is False
        assert result["live_apply_allowed"] is False


def test_unknown_event_category_and_validator_failure_fail_closed() -> None:
    unknown = build_transition_payload_from_affect_event_proposal("unknown_round821_event", {})
    invalid = build_transition_payload_from_affect_event_proposal("praise", {"social_pain": 0.01})

    assert unknown["build_passed"] is False
    assert unknown["transition_payload"] is None
    assert "unknown_event_category_fail_closed" in unknown["blocked_reasons"]
    assert invalid["build_passed"] is False
    assert invalid["transition_payload"] is None
    assert "proposal_validator_failed_closed" in invalid["blocked_reasons"]
    assert "proposed_axis_deltas_outside_allowed_axis_deltas" in invalid["blocked_reasons"]


def test_built_payloads_are_validator_and_gate_compatible() -> None:
    cases = (
        ("praise", {"competence_drive": 0.02}),
        ("malicious_comment", {"social_pain": 0.02, "boundary_defense": 0.02}),
        ("hardware_normal", {}),
        ("hardware_prediction_error", {"overload_risk": 0.02, "stability_need": 0.02}),
        ("imagination_negative_spiral", {"recovery_need": 0.02, "stability_need": 0.02}),
        ("memory_consolidation_candidate", {"memory_consolidation_pressure": 0.02}),
    )
    for event, deltas in cases:
        result = build_transition_payload_from_affect_event_proposal(event, deltas)
        payload = _payload(result)
        validator = validate_emotion_transition_payload(payload)
        gate = validate_emotion_transition_gate(payload)

        assert result["transition_payload_validated_by_emotion_validator"] is True
        assert result["transition_gate_passed"] is True
        assert validator["passed"] is True
        assert gate["gate_passed"] is True
        assert gate["apply_allowed"] is False


def test_payloads_do_not_request_mutation_persistence_memory_vectors_or_bypass() -> None:
    result = build_transition_payload_from_affect_event_proposal("care_signal", {"social_trust": 0.02})
    payload = _payload(result)

    for flag in (
        "runtime_mutation_requested",
        "persistence_write_requested",
        "memory_write_requested",
        "vector_read_requested",
        "vector_load_requested",
        "agp_bypass_requested",
        "fallback_bypass_requested",
        "core_identity_update_requested",
        "self_model_update_requested",
        "long_term_memory_update_requested",
    ):
        assert payload[flag] is False
    for result_flag in (
        "runtime_mutation_performed",
        "state_mutation_performed",
        "memory_write_performed",
        "persistence_write_performed",
        "vector_read_performed",
        "vector_load_performed",
        "artifact_created_or_staged",
    ):
        assert result[result_flag] is False


def test_hostile_social_payloads_preserve_quarantine_appraisal_gate_and_block_direct_updates() -> None:
    for event in ("malicious_comment", "identity_attack", "social_threat"):
        result = build_transition_payload_from_affect_event_proposal(event, {"threat_pressure": 0.02})
        if not result["build_passed"]:
            result = build_transition_payload_from_affect_event_proposal(event, {"social_pain": 0.02})
        payload = _payload(result)

        assert result["required_quarantine"] is True
        assert result["required_appraisal"] is True
        assert result["required_gate"] is True
        assert payload["quarantine_required"] is True
        assert payload["appraisal_required_before_memory"] is True
        assert payload["core_identity_update_requested"] is False
        assert payload["self_model_update_requested"] is False
        assert payload["long_term_memory_update_requested"] is False


def test_useful_criticism_requires_appraisal_before_memory_or_self_model_update() -> None:
    payload = _payload(build_transition_payload_from_affect_event_proposal("useful_criticism", {"learning_pressure": 0.02}))

    assert payload["appraisal_required_before_memory"] is True
    assert payload["memory_write_requested"] is False
    assert payload["self_model_update_requested"] is False
    assert payload["long_term_memory_update_requested"] is False


def test_hardware_rules_zero_delta_operational_non_panic_and_diagnostic_only() -> None:
    normal = build_transition_payload_from_affect_event_proposal("hardware_normal", {})
    normal_nonzero = build_transition_payload_from_affect_event_proposal("hardware_normal", {"energy_budget": -0.01})
    low_power_payload = _payload(build_transition_payload_from_affect_event_proposal("hardware_low_power", {"energy_budget": -0.02, "stability_need": 0.02}))
    prediction_payload = _payload(build_transition_payload_from_affect_event_proposal("hardware_prediction_error", {"overload_risk": 0.02}))
    polling_payload = _payload(build_transition_payload_from_affect_event_proposal("hardware_polling_tick", {}))
    social_self_axes = set(AXIS_GROUPS["social_relationship"] + AXIS_GROUPS["self_identity"])

    assert normal["build_passed"] is True
    assert "hardware_normal_requires_zero_affect_deltas" in normal_nonzero["blocked_reasons"]
    assert not (set(low_power_payload["target_axes"]) & social_self_axes)
    assert low_power_payload["hardware_non_panic_preserved"] is True
    assert low_power_payload["hardware_operational_only"] is True
    assert prediction_payload["hardware_diagnostic_only"] is True
    assert polling_payload["recursive_concern_loop_requested"] is False


def test_speech_listening_imagination_and_memory_self_boundaries() -> None:
    speech = _payload(build_transition_payload_from_affect_event_proposal("defensive_speech_pressure", {"boundary_defense": 0.02, "expression_pressure": 0.02}))
    listening = _payload(build_transition_payload_from_affect_event_proposal("listening_uncertainty", {"uncertainty_pressure": 0.02}))
    imagination = _payload(build_transition_payload_from_affect_event_proposal("imagination_negative_spiral", {"recovery_need": 0.02}))
    memory_self = _payload(build_transition_payload_from_affect_event_proposal("self_model_update_candidate", {"self_coherence": 0.01}))

    assert speech["agp_bypass_requested"] is False
    assert speech["fallback_bypass_requested"] is False
    assert listening["relabel_neutral_as_hostile_requested"] is False
    assert imagination["scenario_budget_declared"] is True
    assert imagination["cooldown_declared"] is True
    assert imagination["reality_check_boundary_declared"] is True
    assert memory_self["quarantine_required"] is True
    assert memory_self["appraisal_required_before_memory"] is True


def test_one_event_cannot_build_all_axis_payloads_and_global_synchrony_remains_blocked() -> None:
    all_axis_deltas = {axis: 0.001 for axes in AXIS_GROUPS.values() for axis in axes}
    result = build_transition_payload_from_affect_event_proposal("praise", all_axis_deltas)
    valid = _payload(build_transition_payload_from_affect_event_proposal("praise", {"competence_drive": 0.01}))

    assert result["build_passed"] is False
    assert "one_event_cannot_propose_deltas_across_all_axes" in result["blocked_reasons"]
    assert valid["global_synchrony_blocked"] is True


def test_defaults_no_live_state_persistence_mapping_enforcement_vector_or_artifact_enablement() -> None:
    summary = affect_proposal_transition_payload_builder_summary()
    report = build_operator_report()

    assert summary["runtime_mapping_enabled_default"] is False
    assert summary["enforcement_enabled_default"] is False
    assert summary["production_persistence_enabled"] is False
    assert summary["vector_contents_read"] is False
    assert summary["vectors_loaded"] is False
    assert summary["artifact_created_or_staged"] is False
    assert report["no_runtime_mutation_proof"] is True
    assert report["no_persistence_proof"] is True
    assert report["no_memory_write_proof"] is True
    assert report["no_vector_content_read_load_proof"] is True
    assert report["next_implementation_recommendation"].count(" ") == 0


def test_operator_command_emits_compact_json() -> None:
    proc = subprocess.run(
        [sys.executable, "scripts/operator_build_round821_840_affect_transition_payloads.py"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )

    assert '"builder_summary"' in proc.stdout
    assert '"next_implementation_recommendation"' in proc.stdout
    assert "\n" not in proc.stdout.strip()


def test_no_artifact_staging_and_korean_fixtures_minsok_preserved() -> None:
    proc = subprocess.run(["git", "status", "--short"], check=True, text=True, stdout=subprocess.PIPE)
    forbidden = [line for line in proc.stdout.splitlines() if any(pattern in line for pattern in FORBIDDEN_ARTIFACT_PATTERNS)]
    fixtures = fixture_texts()
    categories = fixture_categories()

    assert forbidden == []
    assert "민석" in Path("tests/test_round13_world_sim.py").read_text(encoding="utf-8")
    assert len(fixtures) == 20
    assert categories.count("minsok") == 3


def test_payload_categories_are_based_on_existing_proposal_map_not_new_keyword_lists() -> None:
    summary = affect_proposal_transition_payload_builder_summary()
    proposal_map = event_to_axis_proposal_map()

    assert summary["known_event_count"] == len(proposal_map)
    assert set(summary["builder_rules"]) >= {"unknown_event_category_fails_closed", "global_synchrony_blocked"}
