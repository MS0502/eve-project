"""EVE v3 round75 — commit audit replay/export consolidation.

Round75 consolidates the Round73/74 explicit commit smoke evidence into a
read-only replay/export artifact. The only allowed mutation remains the
explicit gate-approved commit inside the smoke engine; replay/export itself must
not append audit records, mutate vectors, change telemetry, alter policy, or
bypass AGP.
"""

from __future__ import annotations

import json

from main import build_full_engine
from adapters.runtime_smoke_runner import (
    build_round75_commit_audit_replay_export,
    run_round74_explicit_commit_delta_report,
    run_round75_commit_audit_replay_export,
    write_round75_commit_audit_replay_export,
)


def test_round75_replay_export_consolidates_round73_74_evidence() -> None:
    engine = build_full_engine()

    report = run_round75_commit_audit_replay_export(engine)

    assert report["replay_export_version"] == "v3_round75_commit_audit_replay_export"
    assert report["baseline_round"] == 75
    assert report["source_delta_report_version"] == "v3_round74_explicit_commit_drift_telemetry_delta"
    assert report["source_commit_smoke_version"] == "v3_round73_explicit_eve_specific_commit_smoke"
    assert report["target_word"] == "민석"
    assert report["commit_created_target"] is True
    assert report["wrapper_vector_found_after_commit"] is True
    assert report["store_delta"] == 1
    assert report["audit_record_count"] >= 2
    assert report["audit_event_type_counts"]["audit"] >= 1
    assert report["audit_event_type_counts"]["commit"] >= 1

    verification = report["replay_verification"]
    assert verification["has_audit_and_commit_records"] is True
    assert verification["target_created"] is True
    assert verification["target_lookup_shifted_to_eve_specific"] is True
    assert verification["pre_commit_lookup_not_eve_specific"] is True
    assert verification["store_delta_is_one"] is True
    assert verification["audit_record_delta_at_least_two"] is True


def test_round75_replay_layer_is_read_only_after_delta_commit() -> None:
    engine = build_full_engine()

    report = run_round75_commit_audit_replay_export(engine)
    checks = report["replay_read_only_checks"]

    assert checks["audit_records_unchanged_during_replay"] is True
    assert checks["vector_store_unchanged_during_replay"] is True
    assert checks["telemetry_unchanged_during_replay"] is True
    assert checks["policy_changed_during_replay"] is False
    assert report["read_only"] is True
    assert report["decision_policy_changed"] is False

    policy = report["policy"]
    assert policy["explicit_self_learning_commit_path_used"] is True
    assert policy["round72_probe_path_used"] is False
    assert policy["auto_promotion_enabled"] is False
    assert policy["commit_gate_enabled"] is True
    assert policy["min_observations_for_commit"] == 2
    assert policy["context_diversity_gate_enabled"] is True
    assert policy["replay_export_read_only"] is True
    assert policy["replay_does_not_call_commit"] is True
    assert policy["replay_does_not_mutate_vectors"] is True
    assert policy["automatic_rollback_enabled"] is False
    assert policy["fasttext_seed_mutation"] is False
    assert policy["agp_bypass"] is False


def test_round75_build_replay_from_existing_delta_does_not_rerun_commit() -> None:
    engine = build_full_engine()
    delta = run_round74_explicit_commit_delta_report(engine)
    audit_records = engine.eve_self_learning.commit_audit_records()
    store_before = engine.eve_specific_vector_store.stats().copy()
    audit_count_before = len(audit_records)
    telemetry_before = engine.self_embedding.telemetry().copy()

    replay = build_round75_commit_audit_replay_export(delta, audit_records)

    assert replay["replay_export_version"] == "v3_round75_commit_audit_replay_export"
    assert replay["store_delta"] == 1
    assert len(engine.eve_self_learning.commit_audit_records()) == audit_count_before
    assert engine.eve_specific_vector_store.stats() == store_before
    assert engine.self_embedding.telemetry() == telemetry_before
    assert replay["policy"]["replay_does_not_call_commit"] is True
    assert replay["policy"]["replay_does_not_call_lookup"] is True
    assert replay["policy"]["replay_does_not_append_audit"] is True


def test_round75_write_export_writes_existing_artifact_without_recompute(tmp_path) -> None:
    engine = build_full_engine()
    report = run_round75_commit_audit_replay_export(engine)
    audit_count_before = len(engine.eve_self_learning.commit_audit_records())
    store_before = engine.eve_specific_vector_store.stats().copy()

    path = tmp_path / "round75_replay_export.json"
    result = write_round75_commit_audit_replay_export(report, path)
    loaded = json.loads(path.read_text(encoding="utf-8"))

    assert result["export_version"] == "v3_round75_commit_audit_replay_export"
    assert result["recomputed"] is False
    assert result["commit_called"] is False
    assert result["vector_mutation"] is False
    assert result["audit_record_mutation"] is False
    assert loaded["replay_export_version"] == "v3_round75_commit_audit_replay_export"
    assert loaded["target_word"] == "민석"
    assert len(engine.eve_self_learning.commit_audit_records()) == audit_count_before
    assert engine.eve_specific_vector_store.stats() == store_before
