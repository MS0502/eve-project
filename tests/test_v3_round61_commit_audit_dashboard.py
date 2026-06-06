from __future__ import annotations

from main import build_full_engine


def test_round61_commit_audit_dashboard_counts_reasons_and_recent_trend() -> None:
    engine = build_full_engine()
    learner = engine.eve_self_learning

    learner.commit_eve_specific_vectors(words=["민석"], context_words=["오늘", "군대"])
    learner.observe_text("민석 오늘", source="unit_a")
    learner.observe_text("민석 군대", source="unit_b")
    learner.audit_commit_gate(words=["민석"], context_words=["없는단어999"])
    learner.commit_eve_specific_vectors(words=["민석"], context_words=["오늘", "군대"])

    dashboard = learner.commit_audit_dashboard_snapshot(recent_limit=2)

    assert dashboard["dashboard_version"] == "v3_round61_commit_audit_dashboard"
    assert dashboard["round"] == 61
    assert dashboard["read_only"] is True
    assert dashboard["policy"]["auto_promotion_enabled"] is False
    assert dashboard["record_count"] >= 3
    assert dashboard["eligible_total"] == 0
    assert dashboard["rejected_total"] >= 1
    assert dashboard["gate_pass_records"] == 0
    assert dashboard["gate_block_records"] >= 1
    assert dashboard["event_type_counts"]["audit"] >= 1
    assert dashboard["event_type_counts"]["commit"] >= 1
    assert dashboard["rejection_reason_counts"]["insufficient_observations"] >= 1
    assert dashboard["rejection_reason_counts"]["insufficient_known_context"] >= 1
    assert dashboard["most_attempted_words"][0]["word"] == "민석"
    assert len(dashboard["recent_trend"]) == 2
    assert all("gate_pass" in item for item in dashboard["recent_trend"])


def test_round61_dashboard_is_read_only_and_does_not_mutate_vectors() -> None:
    engine = build_full_engine()
    learner = engine.eve_self_learning
    learner.observe_text("민석", source="unit")
    learner.audit_commit_gate(words=["민석"], context_words=["오늘", "군대"])

    before = engine.eve_specific_vector_store.stats()["stored_count"]
    dashboard = learner.commit_audit_dashboard_snapshot()
    after = engine.eve_specific_vector_store.stats()["stored_count"]

    assert before == after
    assert dashboard["read_only"] is True
    assert dashboard["policy"]["memory_quarantine_unchanged"] is True
    assert dashboard["policy"]["agp_bypass"] is False


def test_round61_state_debug_and_drift_report_surface_dashboard() -> None:
    from adapters.external_seed_manifest import measure_eve_self_learning_drift_accumulation

    engine = build_full_engine()
    learner = engine.eve_self_learning
    learner.observe_text("민석", source="unit")
    learner.audit_commit_gate(words=["민석"], context_words=["오늘", "군대"])

    state = engine.state_debug.snapshot_state()
    report = measure_eve_self_learning_drift_accumulation(engine)

    assert state["eve_self_learning"]["commit_audit_dashboard_version"] == "v3_round61_commit_audit_dashboard"
    assert state["eve_self_learning"]["commit_audit_dashboard"]["record_count"] >= 1
    assert report["commit_gate"]["dashboard_version"] == "v3_round61_commit_audit_dashboard"
    assert report["commit_audit_dashboard"]["dashboard_version"] == "v3_round61_commit_audit_dashboard"
