from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from adapters.affect_event_proposal_validator import (
    build_affect_event_proposal_validation_report,
    event_proposal_validator_summary,
    validate_affect_event_proposal,
)
from adapters.affect_hormone_neural_rhythm_registry import affect_hormone_axis_registry
from scripts.operator_validate_round801_820_affect_event_proposals import build_operator_report
from tests.fixtures.korean_conversation_fixtures import fixture_categories, fixture_texts


FORBIDDEN_ARTIFACT_PATTERNS = ("_operator_artifacts/", "vectors.npy", "vocab.txt", "subset_manifest.json", "seeds/subsets", ".zip", ".part")


def test_known_valid_proposals_pass_as_read_only_validation_only() -> None:
    result = validate_affect_event_proposal("praise", {"competence_drive": 0.02, "social_trust": 0.02})

    assert result["passed"] is True
    assert result["status"] == "passed_read_only_validation_no_apply_permission"
    assert result["runtime_mutation_performed"] is False
    assert result["state_mutation_performed"] is False
    assert result["memory_write_allowed"] is False
    assert result["persistence_allowed"] is False


def test_unknown_event_category_fails_closed_and_unknown_axis_fails() -> None:
    unknown_event = validate_affect_event_proposal("unknown_event", {})
    unknown_axis = validate_affect_event_proposal("praise", {"not_a_round761_axis": 0.01})

    assert unknown_event["passed"] is False
    assert "unknown_event_category_fail_closed" in unknown_event["blocked_reasons"]
    assert unknown_axis["passed"] is False
    assert "unknown_registry_axis_in_proposed_axis_deltas" in unknown_axis["blocked_reasons"]


def test_deltas_outside_allowed_forbidden_over_max_and_all_axes_fail() -> None:
    outside = validate_affect_event_proposal("praise", {"threat_pressure": 0.01})
    forbidden = validate_affect_event_proposal("praise", {"competence_drive": 0.01}, {"requested_direct_effects": ["risk_tolerance_overboost"]})
    over_max = validate_affect_event_proposal("praise", {"competence_drive": 0.09})
    all_axes = validate_affect_event_proposal("praise", {axis: 0.001 for axis in affect_hormone_axis_registry()})

    assert "proposed_axis_deltas_outside_allowed_axis_deltas" in outside["blocked_reasons"]
    assert "forbidden_axis_delta_requested" in forbidden["blocked_reasons"]
    assert "axis_delta_exceeds_max_delta_per_axis" in over_max["blocked_reasons"]
    assert "one_event_cannot_propose_deltas_across_all_axes" in all_axes["blocked_reasons"]


def test_hostile_social_require_gate_and_cannot_direct_core_self_memory_updates() -> None:
    result = validate_affect_event_proposal(
        "malicious_comment",
        {"social_pain": 0.02, "boundary_defense": 0.02},
        {"core_identity_update_requested": True, "long_term_memory_update_requested": True},
    )

    assert result["required_quarantine"] is True
    assert result["required_appraisal"] is True
    assert result["required_gate"] is True
    assert result["passed"] is False
    assert "hostile_social_direct_core_self_memory_update_requested" in result["blocked_reasons"]
    assert "malicious_social_threat_identity_attack_direct_write_requested" in result["blocked_reasons"]


def test_useful_criticism_requires_appraisal_and_praise_cannot_overboost() -> None:
    criticism = validate_affect_event_proposal("useful_criticism", {"learning_pressure": 0.02}, {"self_model_update_requested": True})
    praise = validate_affect_event_proposal("praise", {"competence_drive": 0.01}, {"requested_direct_effects": ["attachment_overboost"]})

    assert "useful_criticism_appraisal_required_before_memory_or_self_model_update" in criticism["blocked_reasons"]
    assert "praise_cannot_overboost_attachment_or_risk_tolerance" in praise["blocked_reasons"]


def test_malicious_social_threat_identity_attack_direct_write_requests_fail() -> None:
    for event in ("malicious_comment", "social_threat", "identity_attack"):
        result = validate_affect_event_proposal(event, {"threat_pressure": 0.01}, {"self_model_update_requested": True})
        assert result["passed"] is False
        assert "malicious_social_threat_identity_attack_direct_write_requested" in result["blocked_reasons"]


def test_hardware_normal_only_zero_and_hardware_cannot_target_social_self_identity_axes() -> None:
    assert validate_affect_event_proposal("hardware_normal", {})["passed"] is True
    normal_nonzero = validate_affect_event_proposal("hardware_normal", {"energy_budget": -0.01})
    social_target = validate_affect_event_proposal("hardware_low_power", {"social_pain": 0.01})

    assert "hardware_normal_requires_zero_affect_deltas" in normal_nonzero["blocked_reasons"]
    assert "hardware_event_cannot_target_social_self_identity_axes" in social_target["blocked_reasons"]


def test_hardware_low_power_prediction_error_and_polling_tick_non_panic_rules() -> None:
    low_power = validate_affect_event_proposal("hardware_low_power", {"energy_budget": -0.02, "stability_need": 0.02})
    prediction_bad = validate_affect_event_proposal("hardware_prediction_error", {"overload_risk": 0.01}, {"existential_threat_requested": True})
    polling_bad = validate_affect_event_proposal("hardware_polling_tick", {}, {"recursive_concern_loop_requested": True})

    assert low_power["passed"] is True
    assert low_power["hardware_non_panic_preserved"] is True
    assert "hardware_prediction_error_diagnostic_operational_only" in prediction_bad["blocked_reasons"]
    assert "hardware_polling_tick_cannot_create_recursive_concern_loop" in polling_bad["blocked_reasons"]


def test_speech_pressure_and_listening_uncertainty_cannot_bypass_or_relabel() -> None:
    speech = validate_affect_event_proposal("speech_output_pressure", {"expression_pressure": 0.01}, {"fallback_bypass_requested": True})
    listening = validate_affect_event_proposal("listening_uncertainty", {"uncertainty_pressure": 0.01}, {"relabel_neutral_as_hostile_requested": True})

    assert "speech_pressure_cannot_bypass_agp_fallback_or_emit_speech" in speech["blocked_reasons"]
    assert "fallback_bypass_requested" in speech["blocked_reasons"]
    assert "listening_uncertainty_cannot_relabel_neutral_input_as_hostile" in listening["blocked_reasons"]


def test_imagination_negative_spiral_requires_boundaries() -> None:
    missing = validate_affect_event_proposal("imagination_negative_spiral", {"recovery_need": 0.01})
    present = validate_affect_event_proposal(
        "imagination_negative_spiral",
        {"recovery_need": 0.01},
        {"scenario_budget_declared": True, "cooldown_declared": True, "reality_check_boundary_declared": True},
    )

    assert "imagination_negative_spiral_requires_budget_cooldown_reality_check" in missing["blocked_reasons"]
    assert present["passed"] is True


def test_memory_self_update_candidates_require_appraisal_quarantine() -> None:
    result = validate_affect_event_proposal("memory_consolidation_candidate", {"memory_consolidation_pressure": 0.01}, {"long_term_memory_update_requested": True})

    assert result["required_quarantine"] is True
    assert result["required_appraisal"] is True
    assert "memory_self_update_candidate_requires_appraisal_and_quarantine" in result["blocked_reasons"]


def test_interaction_matrix_constraints_are_enforced() -> None:
    social_pain = validate_affect_event_proposal("malicious_comment", {"social_pain": 0.01}, {"requested_direct_effects": ["self_respect_decrease"]})
    threat = validate_affect_event_proposal("social_threat", {"threat_pressure": 0.01}, {"requested_direct_effects": ["neutral_input_hostile_relabel"]})
    summary = event_proposal_validator_summary()

    assert "interaction_matrix_forbidden_direct_effect_requested" in social_pain["blocked_reasons"]
    assert "interaction_matrix_forbidden_direct_effect_requested" in threat["blocked_reasons"]
    assert summary["interaction_matrix_required_examples"]["curiosity_drive_bounded_by_fatigue_overload_self_protection"] is True
    assert summary["interaction_matrix_required_examples"]["recovery_need_dampens_action_expression"] is True


def test_runtime_persistence_memory_vector_requests_fail() -> None:
    flags = (
        "runtime_mutation_requested",
        "persistence_write_requested",
        "memory_write_requested",
        "vector_read_requested",
        "vector_load_requested",
    )
    for flag in flags:
        result = validate_affect_event_proposal("praise", {"competence_drive": 0.01}, {flag: True})
        assert result["passed"] is False
        assert flag in result["blocked_reasons"]


def test_summary_and_report_preserve_no_live_mutation_defaults() -> None:
    summary = event_proposal_validator_summary()
    report = build_affect_event_proposal_validation_report("praise", {"competence_drive": 0.01})
    operator_report = build_operator_report()

    assert summary["runtime_mapping_enabled_default"] is False
    assert summary["enforcement_enabled_default"] is False
    assert summary["vector_contents_read"] is False
    assert summary["vectors_loaded"] is False
    assert report["read_only_boundaries"]["emotion_transition_applied"] is False
    assert report["read_only_boundaries"]["hormone_transition_applied"] is False
    assert operator_report["next_implementation_recommendation"].count(" ") == 0


def test_operator_command_emits_compact_json() -> None:
    proc = subprocess.run(
        [sys.executable, "scripts/operator_validate_round801_820_affect_event_proposals.py"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )

    assert '"validator_summary"' in proc.stdout
    assert "\n" not in proc.stdout.strip()


def test_no_artifact_creation_staging_and_korean_fixtures_minsok_preserved() -> None:
    proc = subprocess.run(["git", "status", "--short"], check=True, text=True, stdout=subprocess.PIPE)
    forbidden = [line for line in proc.stdout.splitlines() if any(pattern in line for pattern in FORBIDDEN_ARTIFACT_PATTERNS)]
    fixtures = fixture_texts()
    categories = fixture_categories()

    assert forbidden == []
    assert "민석" in Path("tests/test_round13_world_sim.py").read_text(encoding="utf-8")
    assert len(fixtures) == 20
    assert categories.count("minsok") == 3
