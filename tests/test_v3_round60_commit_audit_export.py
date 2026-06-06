from __future__ import annotations

import json

from main import build_full_engine


def test_round60_commit_audit_snapshot_persists_allowed_and_rejected_records() -> None:
    engine = build_full_engine()
    learner = engine.eve_self_learning

    learner.commit_eve_specific_vectors(words=["민석"], context_words=["오늘", "군대"])
    learner.observe_text("민석 오늘", source="unit_a")
    learner.observe_text("민석 군대", source="unit_b")
    learner.commit_eve_specific_vectors(words=["민석"], context_words=["오늘", "군대"])

    snapshot = learner.export_commit_audit_snapshot()

    assert snapshot["export_version"] == "v3_round60_commit_audit_export"
    assert snapshot["round"] == 60
    assert snapshot["read_only"] is True
    assert snapshot["policy"]["auto_promotion_enabled"] is False
    assert snapshot["policy"]["memory_quarantine_unchanged"] is True
    assert snapshot["record_count"] >= 2
    assert snapshot["rejected_total"] >= 1
    assert snapshot["eligible_total"] == 0
    assert any("민석" in r["rejected_words"] for r in snapshot["records"])
    assert all("민석" not in r["eligible_words"] for r in snapshot["records"])
    assert engine.eve_specific_vector_store.is_eve_specific("민석") is False


def test_round60_write_commit_audit_export_writes_json_without_vector_mutation(tmp_path) -> None:
    engine = build_full_engine()
    learner = engine.eve_self_learning
    learner.observe_text("민석", source="unit")
    before = engine.eve_specific_vector_store.stats()["stored_count"]
    learner.audit_commit_gate(words=["민석"], context_words=["오늘", "군대"])

    export_path = tmp_path / "commit_audit.json"
    result = learner.write_commit_audit_export(export_path)
    after = engine.eve_specific_vector_store.stats()["stored_count"]
    payload = json.loads(export_path.read_text(encoding="utf-8"))

    assert result["export_version"] == "v3_round60_commit_audit_export"
    assert result["vector_mutation"] is False
    assert result["memory_quarantine_unchanged"] is True
    assert before == after
    assert payload["record_count"] >= 1
    assert payload["records"][0]["read_only"] is True
    assert payload["records"][0]["candidate_reports"][0]["word"] == "민석"


def test_round60_state_debug_and_drift_report_surface_audit_export() -> None:
    from adapters.external_seed_manifest import measure_eve_self_learning_drift_accumulation

    engine = build_full_engine()
    learner = engine.eve_self_learning
    learner.observe_text("민석", source="unit")
    learner.audit_commit_gate(words=["민석"], context_words=["오늘", "군대"])

    state = engine.state_debug.snapshot_state()
    report = measure_eve_self_learning_drift_accumulation(engine)

    assert state["eve_self_learning"]["commit_audit_record_count"] >= 1
    assert state["eve_self_learning"]["commit_audit_export_version"] == "v3_round60_commit_audit_export"
    assert report["drift_accumulation"]["commit_audit_record_count"] >= 1
    assert report["commit_gate"]["audit_export_version"] == "v3_round60_commit_audit_export"
    assert report["commit_gate"]["audit_persistence_enabled"] is True
