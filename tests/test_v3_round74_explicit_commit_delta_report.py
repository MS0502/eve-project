"""EVE v3 round74 — explicit commit drift/telemetry delta report.

Round73 proved the first real explicit EveSpecific commit path. Round74 wraps
that smoke with before/after telemetry and drift baseline snapshots so the route
shift is auditable without changing self-learning policy.
"""

from __future__ import annotations

from main import build_full_engine
from adapters.runtime_smoke_runner import run_round74_explicit_commit_delta_report


def test_round74_delta_report_records_fail_closed_no_load_commit_attempt() -> None:
    engine = build_full_engine()

    report = run_round74_explicit_commit_delta_report(engine)

    assert report["delta_report_version"] == "v3_round74_explicit_commit_drift_telemetry_delta"
    assert report["baseline_round"] == 74
    assert report["commit_created_target"] is False
    assert report["wrapper_vector_found_after_commit"] is False
    assert report["store_delta"] == 0
    assert report["audit_record_delta"] >= 2  # audit + commit records
    assert report["commit_smoke"]["smoke_version"] == "v3_round73_explicit_eve_specific_commit_smoke"

    route_shift = report["route_shift_summary"]
    assert route_shift["eve_specific_hits_increased"] is False
    assert route_shift["post_commit_lookup_used_eve_specific"] is False
    assert route_shift["pre_commit_lookup_used_eve_specific"] is False


def test_round74_target_lookup_before_after_commit_remains_fallback_without_vector() -> None:
    engine = build_full_engine()

    report = run_round74_explicit_commit_delta_report(engine)

    before_lookup = report["target_lookup_before_commit"]
    after_lookup = report["target_lookup_after_commit"]

    assert before_lookup["eve_specific_hit_delta"] == 0
    assert before_lookup["fallback_delta"] >= 1
    assert after_lookup["found"] is False
    assert after_lookup["eve_specific_hit_delta"] == 0
    assert after_lookup["fallback_delta"] >= 1

    telemetry_delta = report["telemetry_delta"]
    assert telemetry_delta["total_calls"] >= 2
    assert telemetry_delta["eve_specific_hits"] == 0
    assert telemetry_delta["fallback_uses"] >= 1


def test_round74_preserves_self_learning_policy() -> None:
    engine = build_full_engine()

    report = run_round74_explicit_commit_delta_report(engine)
    policy = report["policy"]

    assert policy["explicit_self_learning_commit_path_used"] is True
    assert policy["round72_probe_path_used"] is False
    assert policy["auto_promotion_enabled"] is False
    assert policy["commit_gate_enabled"] is True
    assert policy["min_observations_for_commit"] == 2
    assert policy["context_diversity_gate_enabled"] is True
    assert policy["thresholds_changed"] is False
    assert policy["context_diversity_gate_changed"] is False
    assert policy["automatic_rollback_enabled"] is False
    assert policy["drift_based_runtime_change"] is False
    assert policy["memory_quarantine_unchanged"] is True
    assert policy["fasttext_seed_mutation"] is False
    assert policy["agp_bypass"] is False
    assert report["decision_policy_changed"] is False
    assert report["read_only_decision_surface"] is True


def test_round74_route_distribution_delta_links_to_round72_baselines() -> None:
    engine = build_full_engine()

    report = run_round74_explicit_commit_delta_report(engine)

    assert report["baseline_before"]["measurement_version"] == "v3_round72_eve_specific_smoke_drift_baseline"
    assert report["baseline_after"]["measurement_version"] == "v3_round72_eve_specific_smoke_drift_baseline"
    delta = report["route_distribution_delta"]["count_delta"]
    assert delta["total_calls"] >= 2
    assert delta["eve_specific_hits"] == 0
    assert delta["pmi_svd_fallback_uses"] >= 1
    assert delta["errors"] == 0
