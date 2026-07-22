from __future__ import annotations

from scripts.audit import m2_b_decision_groups as groups
from scripts.audit import m2_b_read_capability_manifest as manifest


def _report() -> dict:
    return {
        "schema_version": manifest.SCHEMA_VERSION,
        "report_digest": "f" * 64,
        "candidate_edges": [{"edge_id": "edge-1"}],
        "unresolved_boundary_calls": [{"finding_id": "finding-1"}],
        "parse_errors": [],
    }


def _base() -> dict:
    return {
        "schema_version": groups.SCHEMA_VERSION,
        "candidate_report_digest": "e" * 64,
        "candidate_surface_digest": "a" * 64,
        "edge_decision_groups": [
            {
                "edge_ids": ["edge-1"],
                "decision": "NOT_RAW_TEXT_FALSE_POSITIVE",
                "capability": "No raw-text capability is granted.",
                "provenance": "Synthetic focused-test provenance.",
                "quarantine": "No quarantined source is exposed.",
                "quotation": "No quotation capability is exposed.",
                "denial_semantics": "The candidate remains denied.",
                "rationale": "Focused exact-coverage fixture.",
                "owner": "M2-B audit test",
            }
        ],
        "unresolved_call_decision_groups": [],
        "parse_error_decision_groups": [],
    }


def _addition(report: dict) -> dict:
    return {
        "schema_version": groups.ADDITIONS_SCHEMA_VERSION,
        "introduced_by_pr": 166,
        "base_candidate_surface_digest": "a" * 64,
        "candidate_surface_digest": groups.candidate_surface_digest(report),
        "edge_decision_groups": [],
        "unresolved_call_decision_groups": [
            {
                "finding_ids": ["finding-1"],
                "decision": "NOT_CAPABILITY_BOUNDARY",
                "rationale": "The call is canonical evidence processing only.",
                "owner": "M2-E bounded cutover evidence contract",
                "denial_semantics": "No source-store or expression read capability is granted.",
            }
        ],
        "parse_error_decision_groups": [],
    }


def test_append_only_fragment_extends_exact_coverage() -> None:
    report = _report()
    result = groups.validate_compact_decisions(report, _base(), [_addition(report)])
    assert result["valid"] is True
    assert result["eligible_for_human_review"] is True
    assert result["addition_fragment_count"] == 1
    assert result["edge_decision_count"] == 1
    assert result["unresolved_call_decision_count"] == 1
    assert result["parse_error_decision_count"] == 0
    assert result["compact_group_counts"] == {
        "edge": 1,
        "unresolved_call": 1,
        "parse_error": 0,
    }


def test_fragment_chain_digest_mismatch_fails_closed() -> None:
    report = _report()
    addition = _addition(report)
    addition["base_candidate_surface_digest"] = "b" * 64
    result = groups.validate_compact_decisions(report, _base(), [addition])
    assert result["valid"] is False
    assert result["eligible_for_human_review"] is False
    assert any("base candidate-surface digest mismatch" in error for error in result["errors"])
