import copy
import inspect
import json
import sys

import adapters.virtual_world_situation_causal_context_schema as schema
from adapters.virtual_world_situation_causal_context_schema import (
    build_virtual_world_situation_causal_context,
    build_virtual_world_situation_causal_context_schema_summary,
    validate_virtual_world_situation_causal_context,
)

LINK = {"link_id":"l1","source_situation_id":"s1","target_situation_id":"s2","link_kind":"cause_candidate"}


def main():
    valid = build_virtual_world_situation_causal_context("direct_cause_candidate", "s1", "s2", [LINK])
    invalid = build_virtual_world_situation_causal_context("direct_cause_candidate", "s1", "s1", [LINK])
    stable_a = build_virtual_world_situation_causal_context("direct_cause_candidate", "s1", "s2", [LINK], {"z":1,"a":{"민석":True}})
    stable_b = build_virtual_world_situation_causal_context("direct_cause_candidate", "s1", "s2", [LINK], {"a":{"민석":True},"z":1})
    tampered = copy.deepcopy(valid); tampered["causal_context_id"] += "x"
    chain = build_virtual_world_situation_causal_context("causal_chain_candidate", "s1", "s3", [{**LINK, "sequence_index":0}, {"link_id":"l2","source_situation_id":"s2","target_situation_id":"s3","link_kind":"consequence_candidate","sequence_index":1}])
    bad_chain = build_virtual_world_situation_causal_context("causal_chain_candidate", "s1", "s3", [LINK])
    corr = build_virtual_world_situation_causal_context("correlation_only_candidate", "s1", "s2", [{**LINK, "link_kind":"correlation_candidate"}])
    cf = build_virtual_world_situation_causal_context("counterfactual_cause_candidate", "s1", "s2", [{**LINK, "link_kind":"counterfactual_candidate"}])
    malformed_boundary = build_virtual_world_situation_causal_context("direct_cause_candidate", "s1", "s2", [LINK], {"causal_boundary_classification":""})
    malformed_confidence = build_virtual_world_situation_causal_context("direct_cause_candidate", "s1", "s2", [LINK], {"causal_confidence_state":[]})
    forbidden_metadata = build_virtual_world_situation_causal_context("direct_cause_candidate", "s1", "s2", [LINK], {"nested":[{"memory_write_requested":True}]})
    forbidden_link = build_virtual_world_situation_causal_context("direct_cause_candidate", "s1", "s2", [{**LINK, "unknown":{"tool_execution_requested":True}}])
    circular = {}; circular["self"] = circular
    non_json = build_virtual_world_situation_causal_context("direct_cause_candidate", "s1", "s2", [LINK], {"bad": circular})
    source = inspect.getsource(schema)
    forbidden_tokens = ["import time", "from time", "datetime", "uuid", "random", "requests", "urllib"]
    report = {
        "valid_case_passed": validate_virtual_world_situation_causal_context(valid),
        "invalid_case_passed": validate_virtual_world_situation_causal_context(invalid) is False and invalid["blocked_reasons"] == ["identical_directional_situation_ids"],
        "deterministic_id_passed": stable_a["causal_context_id"] == stable_b["causal_context_id"],
        "tamper_detection_passed": validate_virtual_world_situation_causal_context(tampered) is False,
        "causal_chain_validation_passed": validate_virtual_world_situation_causal_context(chain) is True and bad_chain["blocked_reasons"] == ["causal_chain_sequence_missing"],
        "compatibility_validation_passed": build_virtual_world_situation_causal_context("correlation_only_candidate", "s1", "s2", [LINK])["blocked_reasons"] == ["incompatible_causal_type_link_kind"],
        "endpoint_coherence_passed": build_virtual_world_situation_causal_context("direct_cause_candidate", "s1", "s2", [{"link_id":"l1","source_situation_id":"s2","target_situation_id":"s1","link_kind":"cause_candidate"}])["blocked_reasons"] == ["causal_link_context_endpoint_mismatch"],
        "indirect_path_validation_passed": validate_virtual_world_situation_causal_context(build_virtual_world_situation_causal_context("indirect_cause_candidate", "s1", "s3", [{**LINK, "sequence_index":0}, {"link_id":"l2","source_situation_id":"s2","target_situation_id":"s3","link_kind":"cause_candidate","sequence_index":1}])) is True,
        "common_cause_validation_passed": validate_virtual_world_situation_causal_context(build_virtual_world_situation_causal_context("common_cause_candidate", "s1", "s2", [{"link_id":"l1","source_situation_id":"common","target_situation_id":"s1","link_kind":"common_cause_candidate"}, {"link_id":"l2","source_situation_id":"common","target_situation_id":"s2","link_kind":"common_cause_candidate"}])) is True,
        "correlation_not_causation_passed": corr["causal_direction_summary"]["direction_known"] is False and "correlation_is_not_causation" in corr["causal_integrity_flags"],
        "counterfactual_no_intervention_passed": cf["intervention_performed"] is False and "counterfactual_no_intervention" in cf["boundary_flags"],
        "malformed_boundary_validation_passed": malformed_boundary["blocked_reasons"] == ["malformed_causal_boundary_class"],
        "malformed_confidence_validation_passed": malformed_confidence["blocked_reasons"] == ["malformed_causal_confidence_state"],
        "recursive_forbidden_metadata_passed": forbidden_metadata["blocked_reasons"] == ["memory_write_requested"],
        "recursive_forbidden_link_passed": forbidden_link["blocked_reasons"] == ["tool_execution_requested"],
        "non_json_fail_closed_passed": non_json["blocked_reasons"] == ["non_json_serializable_semantic_input"] and json.dumps(non_json, allow_nan=False) is not None,
        "no_side_effects_passed": all(token not in source for token in forbidden_tokens),
        "schema_summary": build_virtual_world_situation_causal_context_schema_summary(),
    }
    report["all_checks_passed"] = all(v for k, v in report.items() if k != "schema_summary")
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0 if report["all_checks_passed"] else 1

if __name__ == "__main__":
    sys.exit(main())
