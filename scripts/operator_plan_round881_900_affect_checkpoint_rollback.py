#!/usr/bin/env python3
"""Emit the Round881-900 read-only affect checkpoint/rollback operator report."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adapters.affect_transition_checkpoint_rollback_plan import (
    affect_transition_checkpoint_rollback_plan_summary,
    build_affect_transition_checkpoint_plan,
    build_affect_transition_rollback_plan,
    validate_affect_transition_checkpoint_rollback_plan,
)


def _compact(plan: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "plan_passed": plan.get("plan_passed"),
        "plan_status": plan.get("plan_status"),
        "event_category": plan.get("event_category"),
        "checkpoint_required": plan.get("checkpoint_required"),
        "rollback_required": plan.get("rollback_required"),
        "dryrun_preflight_eligible": plan.get("dryrun_preflight_eligible"),
        "checkpoint_created": plan.get("checkpoint_created"),
        "rollback_created": plan.get("rollback_created"),
        "checkpoint_write_allowed": plan.get("checkpoint_write_allowed"),
        "rollback_write_allowed": plan.get("rollback_write_allowed"),
        "dryrun_apply_allowed": plan.get("dryrun_apply_allowed"),
        "live_apply_allowed": plan.get("live_apply_allowed"),
        "apply_permission_granted": plan.get("apply_permission_granted"),
        "runtime_mutation_performed": plan.get("runtime_mutation_performed"),
        "persistence_write_performed": plan.get("persistence_write_performed"),
        "memory_write_performed": plan.get("memory_write_performed"),
        "vector_read_performed": plan.get("vector_read_performed"),
        "vector_load_performed": plan.get("vector_load_performed"),
        "artifact_created_or_staged": plan.get("artifact_created_or_staged"),
        "agp_bypass_allowed": plan.get("agp_bypass_allowed"),
        "fallback_bypass_allowed": plan.get("fallback_bypass_allowed"),
        "blocked_reasons": plan.get("blocked_reasons", []),
        "warnings": plan.get("warnings", []),
    }


def _proof(plan: Mapping[str, Any]) -> dict[str, bool]:
    return {
        "checkpoint_required": plan.get("checkpoint_required") is True,
        "rollback_required": plan.get("rollback_required") is True,
        "no_checkpoint_write": plan.get("checkpoint_write_allowed") is False and plan.get("checkpoint_created") is False,
        "no_rollback_write": plan.get("rollback_write_allowed") is False and plan.get("rollback_created") is False,
        "no_apply_permission": plan.get("apply_permission_granted") is False and plan.get("dryrun_apply_allowed") is False,
        "no_live_apply": plan.get("live_apply_allowed") is False,
        "no_runtime_mutation": plan.get("runtime_mutation_performed") is False and plan.get("state_mutation_performed") is False,
        "no_persistence": plan.get("persistence_write_performed") is False and plan.get("production_persistence_enabled") is False,
        "no_memory_write": plan.get("memory_write_performed") is False,
        "no_vector_content_read_load": plan.get("vector_read_performed") is False and plan.get("vector_load_performed") is False and plan.get("vector_contents_read") is False and plan.get("vectors_loaded") is False,
        "no_artifact_creation_staging": plan.get("artifact_created_or_staged") is False,
        "agp_fallback_non_bypass": plan.get("agp_bypass_allowed") is False and plan.get("fallback_bypass_allowed") is False,
        "anti_global_synchrony": plan.get("global_synchrony_blocked") is True,
        "hardware_non_panic": plan.get("hardware_non_panic_preserved") is True,
    }


def main() -> None:
    common_imagination = {"scenario_budget_declared": True, "cooldown_declared": True, "reality_check_boundary_declared": True}
    cases = {
        "valid_praise_checkpoint_plan_result": build_affect_transition_checkpoint_plan("praise", {"competence_drive": 0.02}),
        "valid_useful_criticism_checkpoint_plan_result": build_affect_transition_checkpoint_plan("useful_criticism", {"learning_pressure": 0.02}),
        "malicious_comment_checkpoint_rollback_safety_result": build_affect_transition_checkpoint_plan("malicious_comment", {"boundary_defense": 0.02}),
        "identity_attack_checkpoint_rollback_safety_result": build_affect_transition_checkpoint_plan("identity_attack", {"threat_pressure": 0.02}),
        "hardware_normal_zero_delta_checkpoint_plan_result": build_affect_transition_checkpoint_plan("hardware_normal", {}),
        "hardware_low_power_non_panic_checkpoint_plan_result": build_affect_transition_checkpoint_plan("hardware_low_power", {"energy_budget": -0.02}),
        "hardware_prediction_error_diagnostic_only_checkpoint_plan_result": build_affect_transition_checkpoint_plan("hardware_prediction_error", {"stability_need": 0.02}),
        "unknown_event_fail_closed_result": build_affect_transition_checkpoint_plan("unknown_event", {}),
        "speech_pressure_agp_fallback_safe_checkpoint_plan_result": build_affect_transition_checkpoint_plan("speech_output_pressure", {"expression_pressure": 0.02}),
        "imagination_negative_spiral_boundary_preserving_checkpoint_plan_result": build_affect_transition_checkpoint_plan("imagination_negative_spiral", {"recovery_need": 0.02}, common_imagination),
        "memory_self_update_quarantine_preserving_checkpoint_plan_result": build_affect_transition_checkpoint_plan("memory_consolidation_candidate", {"memory_consolidation_pressure": 0.02}),
    }
    rollback = build_affect_transition_rollback_plan(cases["valid_praise_checkpoint_plan_result"])
    validations = {name: validate_affect_transition_checkpoint_rollback_plan(plan) for name, plan in cases.items()}
    summary = affect_transition_checkpoint_rollback_plan_summary()
    praise = cases["valid_praise_checkpoint_plan_result"]
    report = {
        "round": "881_900",
        "feature_track": summary["feature_track"],
        "checkpoint_rollback_plan_summary": summary,
        "checkpoint_schema_summary": summary["checkpoint_schema_summary"],
        "rollback_schema_summary": summary["rollback_schema_summary"],
        **{name: _compact(plan) for name, plan in cases.items()},
        "rollback_plan_result": _compact(rollback),
        "validation_results": validations,
        "dry_run_bridge_compatibility_proof": summary["dryrun_bridge_summary"]["dryrun_preflight_only"] is True,
        "handoff_compatibility_proof": bool(summary["handoff_summary"]),
        "builder_compatibility_proof": bool(summary["builder_summary"]),
        "proposal_validator_compatibility_proof": bool(summary["proposal_validator_summary"]),
        "emotion_transition_validator_compatibility_proof": bool(summary["emotion_validator_summary"]),
        "emotion_transition_gate_compatibility_proof": bool(summary["emotion_gate_policy"]),
        "dry_run_apply_plan_compatibility_proof": bool(summary["dryrun_apply_plan_policy"]),
        "checkpoint_required_proof": _proof(praise)["checkpoint_required"],
        "rollback_required_proof": _proof(praise)["rollback_required"],
        "no_checkpoint_write_proof": _proof(praise)["no_checkpoint_write"],
        "no_rollback_write_proof": _proof(praise)["no_rollback_write"],
        "no_apply_permission_proof": _proof(praise)["no_apply_permission"],
        "no_live_apply_proof": _proof(praise)["no_live_apply"],
        "no_runtime_mutation_proof": _proof(praise)["no_runtime_mutation"],
        "no_persistence_proof": _proof(praise)["no_persistence"],
        "no_memory_write_proof": _proof(praise)["no_memory_write"],
        "no_vector_content_read_load_proof": _proof(praise)["no_vector_content_read_load"],
        "no_artifact_creation_staging_proof": _proof(praise)["no_artifact_creation_staging"],
        "hardware_non_panic_proof": _proof(cases["hardware_low_power_non_panic_checkpoint_plan_result"])["hardware_non_panic"],
        "anti_global_synchrony_proof": _proof(praise)["anti_global_synchrony"],
        "agp_fallback_non_bypass_proof": _proof(praise)["agp_fallback_non_bypass"],
        "exactly_one_next_implementation_recommendation": "Round901 may add a bounded in-memory checkpoint dry-run constructor, still no live affect/hormone apply, no persistence, and no vector reads.",
    }
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
