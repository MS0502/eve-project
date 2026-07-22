from __future__ import annotations

from copy import deepcopy

from scripts.audit.m2_b_decision_groups import (
    SCHEMA_VERSION,
    candidate_surface_digest,
    validate_compact_decisions,
)
from scripts.audit.m2_b_read_capability_manifest import extract_candidates


def report(tmp_path):
    (tmp_path / "app.py").write_text(
        "def respond(input_text):\n    return generate(input_text)\n",
        encoding="utf-8",
    )
    return extract_candidates(tmp_path)


def compact_for(value):
    edge_id = value["candidate_edges"][0]["edge_id"]
    return {
        "schema_version": SCHEMA_VERSION,
        "candidate_surface_digest": candidate_surface_digest(value),
        "edge_decision_groups": [{
            "edge_ids": [edge_id],
            "decision": "LEGACY_REWRITE",
            "capability": "mapped legacy edge",
            "provenance": "exact candidate evidence",
            "quarantine": "not structurally proven",
            "quotation": "not structurally proven",
            "denial_semantics": "no new capability",
            "rationale": "rewrite required",
            "owner": "tests.m2b",
        }],
        "unresolved_call_decision_groups": [],
        "parse_error_decision_groups": [],
    }


def test_exact_group_expands_and_validates(tmp_path):
    value = report(tmp_path)
    result = validate_compact_decisions(value, compact_for(value))
    assert result["valid"] is True
    assert result["edge_decision_count"] == 1
    assert result["human_accepted"] is False


def test_missing_edge_is_rejected(tmp_path):
    value = report(tmp_path)
    compact = compact_for(value)
    compact["edge_decision_groups"] = []
    result = validate_compact_decisions(value, compact)
    assert result["valid"] is False
    assert "missing edge decisions: 1" in result["errors"]


def test_duplicate_edge_across_groups_is_rejected(tmp_path):
    value = report(tmp_path)
    compact = compact_for(value)
    compact["edge_decision_groups"].append(dict(compact["edge_decision_groups"][0]))
    result = validate_compact_decisions(value, compact)
    assert result["valid"] is False
    assert any(error.startswith("duplicate edge decision:") for error in result["errors"])


def test_unreviewed_shared_field_is_rejected(tmp_path):
    value = report(tmp_path)
    compact = compact_for(value)
    compact["edge_decision_groups"][0]["quarantine"] = "REVIEW_REQUIRED"
    result = validate_compact_decisions(value, compact)
    assert result["valid"] is False
    assert any("fields require review: quarantine" in error for error in result["errors"])


def test_surface_digest_mismatch_is_rejected(tmp_path):
    value = report(tmp_path)
    compact = compact_for(value)
    compact["candidate_surface_digest"] = "0" * 64
    result = validate_compact_decisions(value, compact)
    assert result["valid"] is False
    assert "compact candidate_surface_digest mismatch" in result["errors"]


def test_non_surface_metadata_does_not_invalidate_decisions(tmp_path):
    value = report(tmp_path)
    compact = compact_for(value)
    changed = deepcopy(value)
    changed["python_files_scanned"] += 100
    changed["function_count"] += 100
    changed["summary"]["analyzed_state_count"] += 100
    changed["report_digest"] = "f" * 64
    result = validate_compact_decisions(changed, compact)
    assert result["valid"] is True
    assert result["candidate_surface_digest"] == compact["candidate_surface_digest"]
