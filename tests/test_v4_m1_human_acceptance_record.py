from __future__ import annotations

import hashlib
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RECORD_PATH = REPO_ROOT / "docs/audit/M1_HUMAN_ACCEPTANCE_RECORD.json"
MARKDOWN_PATH = REPO_ROOT / "docs/audit/M1_HUMAN_ACCEPTANCE_RECORD.md"
RAW_PATH = REPO_ROOT / "docs/audit/M1_EXTENDED_CONTROLLED_OBSERVATION_RAW.json"
STATUS_PATH = REPO_ROOT / "docs/EVE_IMPLEMENTATION_STATUS_v4.md"
CORE_PATH = REPO_ROOT / "core/shadow_acceptance.py"
EXPECTED_RECORD_SHA256 = "aff557da810b7faa0c9dc57bde214a9760a0d3099c8031cb6eb7a24398cf8522"
EXPECTED_RAW_SHA256 = "3618b948cb2e864741412713b5c724632ae9fd72a214479b970d8c4aeeafcaac"
EXPECTED_SOURCE_SHA256 = "06984c653ed2a655f45c7cb27d0777b1c93c6aee872f2cb9c7d1f5a898d9af86"


def _canonical(value: dict) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ) + "\n"


def _transition_sha(before, after) -> str:
    return hashlib.sha256(
        json.dumps(
            {"before": before, "after": after},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _load():
    record_text = RECORD_PATH.read_text(encoding="utf-8")
    raw_text = RAW_PATH.read_text(encoding="utf-8")
    return record_text, json.loads(record_text), raw_text, json.loads(raw_text)


def test_acceptance_record_is_canonical_pinned_and_explicit():
    record_text, record, _, _ = _load()

    assert record_text == _canonical(record)
    assert hashlib.sha256(record_text.encode("utf-8")).hexdigest() == EXPECTED_RECORD_SHA256
    assert record["schema_version"] == "eve.m1-human-acceptance-record.v1"
    assert record["approval_authority"]["automatic"] is False
    assert record["approval_authority"]["project_authority"] == "김민석"
    assert record["decision"] == {
        "human_accepted": True,
        "human_review_status": "accepted",
        "m1_closed": True,
        "m2_started": False,
        "status": "accepted",
        "v4_2_eligible": True,
        "v4_2_review_opened": False,
    }


def test_evidence_pins_match_exact_committed_artifacts():
    _, record, raw_text, raw = _load()
    pins = record["evidence_pins"]

    assert hashlib.sha256(raw_text.encode("utf-8")).hexdigest() == EXPECTED_RAW_SHA256
    assert raw["source_evidence_sha256"] == EXPECTED_SOURCE_SHA256
    assert pins["raw_artifact_sha256"] == EXPECTED_RAW_SHA256
    assert pins["source_evidence_sha256"] == EXPECTED_SOURCE_SHA256
    assert pins["validated_evidence_head"] == "560b9b54f3237d63762b81da38e7c25c36922214"
    assert pins["evidence_merge_sha"] == "7c4573e628e5ac51d0d64ad1040078741f3630e0"
    assert pins["exact_head_run_id"] == 29826184624
    assert pins["exact_head_artifact_sha256"] == "5482da68f38e5d66400d6a32b948d559ce1dd6ce7ec80fe77de08659b8f9d0b9"
    assert pins["focused_tests_passed"] == 12
    assert pins["full_suite_passed"] == 2712


def test_every_acceptance_metric_is_independently_recalculable_from_raw():
    _, record, _, raw = _load()
    criteria = {item["criterion_id"]: item for item in record["reviewed_criteria"]}
    rows = raw["mutation_classification"]["rows"]

    assert {row["form"] for row in rows} == {
        "attribute_assignment",
        "subscript_assignment",
        "augmented_assignment",
        "mutating_method_call",
        "direct_write",
    }
    assert len(rows) == 5
    for row in rows:
        assert row["observed"] is True
        assert row["state_changed"] is True
        assert row["before_value"] != row["after_value"]
        assert row["replay_matches"] is True
        assert row["transition_sha256"] == _transition_sha(
            row["before_value"], row["after_value"]
        )
    assert criteria["mutation_form_state_fidelity"]["passed"] is True

    targets = raw["observation_window"]["adapter_call_paths"]
    assert len(targets) == 3
    assert {row["disposition"] for row in targets} == {"WRAP", "REWRITE"}
    assert criteria["multiple_adapter_dispositions"]["passed"] is True

    activation = raw["raw_observations"]["activation"]
    assert activation["thread_alive_before_mutation"] is True
    assert activation["thread_alive_after_mutation"] is True
    assert activation["mutation_event_delta_while_thread_alive"] == 1
    assert activation["live_tick_event_delta"] == 0
    assert activation["thread_stopped"] is True
    assert criteria["live_tick_thread_concurrency"]["passed"] is True

    replay = raw["replay_equivalence"]
    assert replay["match_rate"] == {"denominator": 4, "numerator": 4, "value": 1.0}
    assert replay["divergence_count"] == 0
    assert replay["divergences"] == []
    assert all(row["matches"] for row in replay["rows"])
    assert all(row["matches"] for row in replay["final_equivalence"])
    assert criteria["complete_replay_equivalence"]["passed"] is True

    visibility = raw["failure_visibility"]
    assert visibility["legacy_failure_event_count"] == 1
    assert visibility["observer_failure"]["event_count"] == 0
    assert visibility["observer_failure"]["legacy_state_preserved"] is True
    assert visibility["observer_failure"]["return_value_preserved"] is True
    assert activation["exception_identity_preserved"] is True
    assert criteria["failure_visibility"]["passed"] is True

    assert raw["granularity"] == {
        "candidate_events": 4,
        "discrete_observed_calls": 4,
        "events_during_live_tick_before_mutation": 0,
        "events_during_standalone_tick_steps": 0,
        "max_events_per_observed_call": 1,
        "standalone_tick_steps": 4,
    }
    assert criteria["discrete_transition_granularity"]["passed"] is True

    persistence = raw["raw_observations"]["persistence"]
    assert persistence["state_matches_unobserved"] is True
    assert persistence["temporary_roots_removed"] is True
    assert persistence["final_snapshot"]["files"][0]["relative_path"] == "state.v41sidecar"
    assert criteria["bounded_direct_write"]["passed"] is True
    assert criteria["raw_observation_recalculability"]["passed"] is True
    assert criteria["exact_head_validation"]["passed"] is True
    assert criteria["zero_unauthorized_effects"]["passed"] is True


def test_external_acceptance_does_not_rewrite_machine_packet_or_runtime_authority():
    _, record, _, raw = _load()
    core_source = CORE_PATH.read_text(encoding="utf-8")

    assert raw["human_gate"] == {
        "eligible_for_human_review": True,
        "human_accepted": False,
        "human_review_status": "required_not_performed",
        "v4_2_eligible": False,
    }
    assert "M1_HUMAN_ACCEPTANCE_RECORD" not in core_source
    assert record["authority_boundary"] == {
        "defaults_changed": False,
        "legacy_runtime_authoritative": True,
        "production_observer_installed": False,
        "production_persistence_enabled": False,
        "runtime_integrated": False,
        "shadow_authority_only": True,
    }
    assert record["decision"]["v4_2_review_opened"] is False
    assert record["decision"]["m2_started"] is False


def test_markdown_and_status_pin_the_decision_without_overclaiming_coverage():
    record_text, record, _, _ = _load()
    markdown = MARKDOWN_PATH.read_text(encoding="utf-8")
    status = STATUS_PATH.read_text(encoding="utf-8")
    record_sha = hashlib.sha256(record_text.encode("utf-8")).hexdigest()

    assert f"Canonical JSON SHA-256: `{record_sha}`" in markdown
    assert "human_accepted: true" in markdown
    assert "v4_2_eligible: true" in markdown
    assert "v4_2_review_opened: false" in markdown
    assert "미관찰 527곳" in markdown
    assert "M1 status: **closed for mechanism verification" in status
    assert "open REWRITE PRs #109, #86, #84, and #82" in status
    assert "absorbed PRs #11, #7, and #4 are closed" in status
    assert record_sha in status
    assert "M2-A remains blocked until v4.2 approval" in status
    assert record["scope_ruling"]["historical_fraction_is_m1_gate"] is False


def test_v4_2_candidate_triangle_is_exact_and_non_activating():
    _, record, _, _ = _load()
    clauses = record["v4_2_candidate_clauses"]

    assert [item["clause_id"] for item in clauses] == [
        "discrete_transition_granularity",
        "raw_observation_recalculability",
        "mutation_state_fidelity",
    ]
    assert record["decision"]["v4_2_eligible"] is True
    assert record["decision"]["v4_2_review_opened"] is False
    assert record["decision"]["m2_started"] is False
