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


def reviewed_decisions(report: dict) -> dict:
    return {
        "schema_version": DECISION_SCHEMA_VERSION,
        "candidate_report_digest": report["report_digest"],
        "edge_decisions": [
            {
                "edge_id": edge["edge_id"],
                "decision": "LEGACY_REWRITE",
                "capability": "legacy raw-input dependency retained only as mapped evidence",
                "provenance": "caller-supplied raw input; no new authority is inferred",
                "quarantine": "not structurally guaranteed by this legacy edge",
                "quotation": "no quotation capability is granted by this decision",
                "denial_semantics": "new raw-text access remains denied pending replacement",
                "rationale": "test review covers this exact extracted edge without activation",
                "owner": "tests.m2b",
            }
            for edge in report["candidate_edges"]
        ],
        "unresolved_call_decisions": [
            {
                "finding_id": item["finding_id"],
                "decision": "DENIED_NO_CAPABILITY",
                "rationale": "unresolved boundary remains denied",
                "owner": "tests.m2b",
                "denial_semantics": "no capability is approved",
            }
            for item in report["unresolved_boundary_calls"]
        ],
        "parse_error_decisions": [
            {
                "finding_id": item["finding_id"],
                "decision": "LEGACY_REWRITE",
                "rationale": "legacy parse failure is explicit and cannot hide new approval",
                "owner": "tests.m2b",
                "denial_semantics": "the unparsed file grants no M2-B capability",
            }
            for item in report["parse_errors"]
        ],
    }


def test_direct_tainted_argument_to_expression_sink_is_extracted(tmp_path: Path):
    write(tmp_path, "app.py", "def respond(input_text):\n    return generate(input_text)\n")
    report = extract_candidates(tmp_path)
    assert report["authority"] == AUTHORITY
    assert report["human_accepted"] is False
    assert report["runtime_integrated"] is False
    assert report["summary"]["candidate_edge_count"] == 1
    edge = report["candidate_edges"][0]
    assert edge["source"]["symbol"] == "respond"
    assert edge["sink"]["evidence"]["target"] == "generate"
    assert edge["call_path"] == ["app:respond"]
    assert edge["mechanical_confidence"] == "high"
    assert edge["review_status"] == "REVIEW_REQUIRED"


def test_interprocedural_taint_records_only_actual_call_path(tmp_path: Path):
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
    edge = report["candidate_edges"][0]
    assert edge["call_path"] == ["app:accept", "app:think", "app:respond"]
    assert edge["cognition_hops"] == [{"path": "app.py", "symbol": "think"}]
    assert [item["target"] for item in edge["call_evidence"]] == ["think", "respond"]
    assert edge["mechanical_confidence"] == "medium"


def test_disconnected_source_and_sink_do_not_form_an_edge(tmp_path: Path):
    write(
        tmp_path,
        "app.py",
        "def accept(input_text):\n    return 1\n\n"
        "def emit():\n    return 'fixed'\n",
    )
    report = extract_candidates(tmp_path)
    assert report["candidate_edges"] == []


def test_non_tainted_sink_argument_does_not_form_an_edge(tmp_path: Path):
    write(
        tmp_path,
        "app.py",
        "def respond(input_text):\n"
        "    ignored = len(input_text)\n"
        "    return generate('fixed')\n",
    )
    report = extract_candidates(tmp_path)
    assert report["candidate_edges"] == []


def test_cognition_candidate_return_is_not_expression_sink(tmp_path: Path):
    write(
        tmp_path,
        "app.py",
        "def generate_candidates(input_text):\n    return len(input_text)\n",
    )
    report = extract_candidates(tmp_path)
    assert report["candidate_edges"] == []


def test_raw_source_call_matching_is_exact_not_token_overlap(tmp_path: Path):
    write(tmp_path, "app.py", "def status():\n    return sensory.stt_available()\n")
    report = extract_candidates(tmp_path)
    assert report["summary"]["source_seed_count"] == 0


def test_external_sink_is_terminal_evidence_not_hidden_boundary(tmp_path: Path):
    write(
        tmp_path,
        "app.py",
        "def respond(input_text):\n    return external.send(input_text)\n",
    )
    report = extract_candidates(tmp_path)
    assert report["summary"]["candidate_edge_count"] == 1
    assert report["unresolved_boundary_calls"] == []
    assert report["candidate_edges"][0]["sink"]["evidence"]["target"] == "external.send"


def test_decisions_require_exact_non_stale_coverage(tmp_path: Path):
    write(tmp_path, "app.py", "def respond(input_text):\n    return generate(input_text)\n")
    report = extract_candidates(tmp_path)
    decisions = reviewed_decisions(report)
    result = validate_decisions(report, decisions)
    assert result["valid"] is True
    assert result["eligible_for_human_review"] is True
    assert result["human_accepted"] is False

    decisions["edge_decisions"] = []
    rejected = validate_decisions(report, decisions)
    assert rejected["valid"] is False
    assert "missing edge decisions: 1" in rejected["errors"]


def test_parse_errors_require_explicit_exact_decisions(tmp_path: Path):
    write(tmp_path, "broken.py", "def broken(:\n")
    report = extract_candidates(tmp_path)
    assert len(report["parse_errors"]) == 1
    decisions = reviewed_decisions(report)
    accepted = validate_decisions(report, decisions)
    assert accepted["valid"] is True
    decisions["parse_error_decisions"] = []
    rejected = validate_decisions(report, decisions)
    assert rejected["valid"] is False
    assert "missing parse_error decisions: 1" in rejected["errors"]


def test_build_output_pins_decision_report_digest(tmp_path: Path):
    write(tmp_path, "app.py", "def respond(input_text):\n    return generate(input_text)\n")
    report = extract_candidates(tmp_path)
    decisions = reviewed_decisions(report)
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
    assert all(edge["source"]["path"] == "app.py" for edge in left["candidate_edges"])


def test_semantic_or_governance_parameters_are_not_raw_sources(tmp_path: Path):
    write(
        tmp_path,
        "app.py",
        "def write_decision(source_decision, input_understanding):\n"
        "    return path.write_text(str(source_decision) + str(input_understanding))\n",
    )
    report = extract_candidates(tmp_path)
    assert report["summary"]["source_seed_count"] == 0
    assert report["candidate_edges"] == []
