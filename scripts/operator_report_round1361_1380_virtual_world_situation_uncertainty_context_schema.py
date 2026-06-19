"""Operator report for round1361-1380 uncertainty context schema."""

import copy
import json
import sys

from adapters.virtual_world_situation_uncertainty_context_schema import (
    build_virtual_world_situation_uncertainty_context,
    build_virtual_world_situation_uncertainty_context_schema_summary,
    validate_virtual_world_situation_uncertainty_context,
)


def factor(fid="f1", kind="missing_information", pol="neutral_unknown", sid="sit-민석", **extra):
    data = {"factor_id": fid, "situation_id": sid, "factor_kind": kind, "polarity": pol}
    data.update(extra)
    return data


def main():
    valid = build_virtual_world_situation_uncertainty_context("missing_information_candidate", "sit-민석", [factor()], {})
    invalid = build_virtual_world_situation_uncertainty_context("bad", "sit-민석", [factor()], {})
    same_a = build_virtual_world_situation_uncertainty_context("missing_information_candidate", "sit-민석", [factor("f2"), factor("f1", related_context_id="r")], {"b":2,"a":1})
    same_b = build_virtual_world_situation_uncertainty_context("missing_information_candidate", "sit-민석", [factor("f1", related_context_id="r"), factor("f2")], {"a":1,"b":2})
    tampered = copy.deepcopy(valid); tampered["uncertainty_context_id"] = "tampered"
    conflict = build_virtual_world_situation_uncertainty_context("conflicting_evidence_candidate", "sit-민석", [factor("f1", "supporting_evidence", "support_candidate"), factor("f2", "challenging_evidence", "challenge_candidate")], {})
    ambiguity = build_virtual_world_situation_uncertainty_context("ambiguity_candidate", "sit-민석", [factor("f1", "neutral_unknown"), factor("f2", "temporal_ambiguity")], {})
    mixed = build_virtual_world_situation_uncertainty_context("mixed_unknown_uncertainty_candidate", "sit-민석", [factor("f1", "missing_information"), factor("f2", "unknown_origin")], {})
    type_tampered = copy.deepcopy(valid); type_tampered["read_only"] = 1
    enum_malformed = build_virtual_world_situation_uncertainty_context("missing_information_candidate", "sit-민석", [factor()], {"uncertainty_boundary_classification": None})
    enum_unknown = build_virtual_world_situation_uncertainty_context("missing_information_candidate", "sit-민석", [factor()], {"uncertainty_boundary_classification": "None"})
    reordered_a = build_virtual_world_situation_uncertainty_context("missing_information_candidate", "sit-민석", [factor("f1", memory_write_requested=True), factor("f2", timer_requested=True)], {})
    reordered_b = build_virtual_world_situation_uncertainty_context("missing_information_candidate", "sit-민석", [factor("f2", timer_requested=True), factor("f1", memory_write_requested=True)], {})
    checks = {
        "valid_case_passed": validate_virtual_world_situation_uncertainty_context(valid),
        "invalid_case_passed": invalid["blocked_reasons"] == ["unknown_uncertainty_type"] and not validate_virtual_world_situation_uncertainty_context(invalid),
        "deterministic_id_passed": same_a["uncertainty_context_id"] == same_b["uncertainty_context_id"],
        "tamper_detection_passed": not validate_virtual_world_situation_uncertainty_context(tampered),
        "type_exact_tamper_detection_passed": not validate_virtual_world_situation_uncertainty_context(type_tampered),
        "malformed_unknown_enum_distinction_passed": enum_malformed["blocked_reasons"] == ["malformed_uncertainty_boundary_class"] and enum_unknown["blocked_reasons"] == ["unknown_uncertainty_boundary_class"],
        "invalid_factor_reordering_deterministic_passed": reordered_a == reordered_b and reordered_a["blocked_reasons"] == reordered_b["blocked_reasons"] and reordered_a["situation_uncertainty_context_passed"] is False,
        "factor_compatibility_passed": build_virtual_world_situation_uncertainty_context("temporal_uncertainty_candidate", "sit-민석", [factor("f1", "causal_ambiguity")], {})["blocked_reasons"] == ["incompatible_uncertainty_type_factor_kind"],
        "polarity_compatibility_passed": build_virtual_world_situation_uncertainty_context("missing_information_candidate", "sit-민석", [factor("f1", "missing_information", "support_candidate")], {})["blocked_reasons"] == ["incompatible_factor_kind_polarity"],
        "factor_situation_coherence_passed": build_virtual_world_situation_uncertainty_context("missing_information_candidate", "sit-민석", [factor(sid="other")], {})["blocked_reasons"] == ["uncertainty_factor_situation_mismatch"],
        "duplicate_semantic_factor_passed": build_virtual_world_situation_uncertainty_context("missing_information_candidate", "sit-민석", [factor("f1"), factor("f2")], {})["blocked_reasons"] == ["duplicate_semantic_uncertainty_factor"],
        "conflicting_evidence_validation_passed": validate_virtual_world_situation_uncertainty_context(conflict),
        "ambiguity_composition_passed": validate_virtual_world_situation_uncertainty_context(ambiguity),
        "mixed_boundary_passed": mixed["blocked_reasons"] == [] and mixed["uncertainty_boundary_classification"] == "mixed_virtual_external_uncertainty_boundary",
        "evidence_balance_passed": conflict["evidence_balance_summary"]["conflict_present"] is True,
        "resolution_remains_false_passed": valid["resolution_summary"]["resolved"] is False and valid["automatic_resolution_performed"] is False,
        "malformed_boundary_validation_passed": build_virtual_world_situation_uncertainty_context("missing_information_candidate", "sit-민석", [factor()], {"uncertainty_boundary_classification":""})["blocked_reasons"] == ["malformed_uncertainty_boundary_class"],
        "malformed_confidence_validation_passed": build_virtual_world_situation_uncertainty_context("missing_information_candidate", "sit-민석", [factor()], {"uncertainty_confidence_state":""})["blocked_reasons"] == ["malformed_uncertainty_confidence_state"],
        "recursive_forbidden_metadata_passed": build_virtual_world_situation_uncertainty_context("missing_information_candidate", "sit-민석", [factor()], {"x":[{"memory_write_requested": True}]})["blocked_reasons"] == ["memory_write_requested"],
        "recursive_forbidden_factor_passed": build_virtual_world_situation_uncertainty_context("missing_information_candidate", "sit-민석", [factor(x={"memory_write_requested": True})], {})["blocked_reasons"] == ["memory_write_requested"],
        "scheduling_alias_forbidden_passed": build_virtual_world_situation_uncertainty_context("missing_information_candidate", "sit-민석", [factor()], {"cron_expression":"* * * * *"})["blocked_reasons"] == ["malformed_forbidden_request_field"],
        "non_json_fail_closed_passed": build_virtual_world_situation_uncertainty_context("missing_information_candidate", "sit-민석", [factor()], {1:"bad"})["blocked_reasons"] == ["non_json_serializable_semantic_input"],
        "no_side_effects_passed": all(valid[k] is False for k in ("memory_write_performed", "random_sampling_performed", "persistence_write_performed", "vector_load_performed", "network_action_performed", "agp_bypass_allowed", "fallback_bypass_allowed")),
    }
    checks["all_checks_passed"] = all(checks.values())
    checks["schema_summary"] = build_virtual_world_situation_uncertainty_context_schema_summary()
    print(json.dumps(checks, sort_keys=True, ensure_ascii=False, separators=(",", ":")))
    return 0 if checks["all_checks_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
