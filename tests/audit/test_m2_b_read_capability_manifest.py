from __future__ import annotations

import json
from pathlib import Path

from scripts.audit.m2_b_read_capability_manifest import (
    AUTHORITY,
    DECISION_SCHEMA_VERSION,
    build_output,
    extract_candidates,
    validate_decisions,
)


def write(root: Path, path: str, content: str) -> None:
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")


def approved_decisions(report: dict) -> dict:
    return {
        "schema_version": DECISION_SCHEMA_VERSION,
        "candidate_report_digest": report["report_digest"],
        "edge_decisions": [
            {
                "edge_id": edge["edge_id"],
                "decision": "LEGACY_REWRITE",
                "capability": "legacy raw-input capability retained only as mapped evidence",
                "provenance": "caller-supplied raw input; source identity not yet authoritative",
                "quarantine": "not structurally enforced in legacy path",
                "quotation": "no quotation capability is granted by this decision",
                "denial_semantics": "new raw-text edges remain denied pending replacement",
                "rationale": "test decision covers the exact extracted edge without activation",
                "owner": "tests.m2b",
            }
            for edge in report["candidate_edges"]
        ],
        "unresolved_call_decisions": [
            {
                "finding_id": item["finding_id"],
                "decision": "LEGACY_REWRITE",
                "rationale": "unresolved external boundary remains denied and scheduled for rewrite",
                "owner": "tests.m2b",
                "denial_semantics": "no new capability is approved",
            }
            for item in report["unresolved_boundary_calls"]
        ],
    }


def test_direct_raw_parameter_to_expression_sink_is_extracted(tmp_path: Path):
    write(tmp_path, "app.py", "def respond(input_text):\n    return generate(input_text)\n")
    report = extract_candidates(tmp_path)
    assert report["authority"] == AUTHORITY
    assert report["human_accepted"] is False
    assert report["runtime_integrated"] is False
    assert report["summary"]["candidate_edge_count"] >= 1
    edge = next(
        edge
        for edge in report["candidate_edges"]
        if edge["source"]["symbol"] == "respond"
        and edge["sink"]["symbol"] == "respond"
    )
    assert edge["call_path"] == ["app:respond"]
    assert edge["mechanical_confidence"] == "high"
    assert edge["review_status"] == "REVIEW_REQUIRED"


def test_interprocedural_shortest_path_records_cognition_hop(tmp_path: Path):
    write(
        tmp_path,
        "app.py",
        "def accept(input_text):\n"
        "    return think(input_text)\n\n"
        "def think(value):\n"
        "    return respond(value)\n\n"
        "def respond(value):\n"
        "    return value\n",
    )
    report = extract_candidates(tmp_path)
    edge = next(
        edge
        for edge in report["candidate_edges"]
        if edge["source"]["symbol"] == "accept"
        and edge["sink"]["symbol"] == "respond"
    )
    assert edge["call_path"] == ["app:accept", "app:think", "app:respond"]
    assert edge["cognition_hops"] == [{"path": "app.py", "symbol": "think"}]
    assert edge["mechanical_confidence"] == "medium"


def test_disconnected_source_and_sink_do_not_form_an_edge(tmp_path: Path):
    write(
        tmp_path,
        "app.py",
        "def accept(input_text):\n    return 1\n\n"
        "def emit():\n    return 'fixed'\n",
    )
    report = extract_candidates(tmp_path)
    assert not any(
        edge["source"]["symbol"] == "accept"
        and edge["sink"]["symbol"] == "emit"
        for edge in report["candidate_edges"]
    )


def test_unresolved_source_or_sink_boundary_is_visible(tmp_path: Path):
    write(
        tmp_path,
        "app.py",
        "def respond(input_text):\n    return external.send(input_text)\n",
    )
    report = extract_candidates(tmp_path)
    assert report["summary"]["unresolved_boundary_call_count"] == 1
    finding = report["unresolved_boundary_calls"][0]
    assert finding["target"] == "external.send"
    assert finding["review_status"] == "REVIEW_REQUIRED"
    assert len(finding["finding_id"]) == 64


def test_decisions_require_exact_non_stale_coverage(tmp_path: Path):
    write(
        tmp_path,
        "app.py",
        "def respond(input_text):\n    return external.send(input_text)\n",
    )
    report = extract_candidates(tmp_path)
    decisions = approved_decisions(report)
    result = validate_decisions(report, decisions)
    assert result["valid"] is True
    assert result["eligible_for_human_review"] is True
    assert result["human_accepted"] is False

    decisions["edge_decisions"] = []
    rejected = validate_decisions(report, decisions)
    assert rejected["valid"] is False
    assert any(
        error.startswith("missing edge decisions:") for error in rejected["errors"]
    )


def test_build_output_pins_decision_report_digest(tmp_path: Path):
    write(tmp_path, "app.py", "def respond(input_text):\n    return generate(input_text)\n")
    report = extract_candidates(tmp_path)
    decisions = approved_decisions(report)
    path = tmp_path / "decisions.json"
    path.write_text(json.dumps(decisions), encoding="utf-8")
    output = build_output(tmp_path, path)
    assert output["decision_validation"]["valid"] is True
    decisions["candidate_report_digest"] = "0" * 64
    path.write_text(json.dumps(decisions), encoding="utf-8")
    rejected = build_output(tmp_path, path)
    assert rejected["decision_validation"]["valid"] is False


def test_extraction_is_deterministic_and_excludes_non_runtime_roots(tmp_path: Path):
    write(tmp_path, "app.py", "def respond(input_text):\n    return generate(input_text)\n")
    write(
        tmp_path,
        "tests/test_fake.py",
        "def respond(input_text):\n    return generate(input_text)\n",
    )
    left = extract_candidates(tmp_path)
    right = extract_candidates(tmp_path)
    assert left == right
    assert left["report_digest"] == right["report_digest"]
    assert all(
        edge["source"]["path"] == "app.py" for edge in left["candidate_edges"]
    )


def test_parse_errors_block_decision_eligibility(tmp_path: Path):
    write(tmp_path, "broken.py", "def broken(:\n")
    report = extract_candidates(tmp_path)
    decisions = approved_decisions(report)
    result = validate_decisions(report, decisions)
    assert result["valid"] is False
    assert "candidate report contains parse errors" in result["errors"]
