"""EVE v3 round73 — first real explicit EveSpecific commit smoke.

Round72 measured routing with an isolated direct vector-store probe. Round73 uses
actual self-learning path: observe_text -> commit_eve_specific_vectors ->
EmbeddingWrapper lookup. Policy must stay explicit-commit-only.
"""

from __future__ import annotations

from main import build_full_engine
from adapters.runtime_smoke_runner import run_round73_explicit_eve_specific_commit_smoke


def _candidate(report: dict, word: str) -> dict:
    for item in report["candidate_reports"]:
        if item["word"] == word:
            return item
    raise AssertionError(f"missing candidate report for {word}")


def test_round73_explicit_commit_smoke_fails_closed_without_loaded_fasttext_context() -> None:
    engine = build_full_engine()

    result = run_round73_explicit_eve_specific_commit_smoke(
        engine,
        target_word="민석",
        observation_texts=("민석 오늘", "민석 군대"),
        commit_context_words=("오늘", "군대"),
    )

    assert result["smoke_version"] == "v3_round73_explicit_eve_specific_commit_smoke"
    assert result["baseline_round"] == 73
    assert result["self_learning_commit_path_called"] is True
    assert result["policy"]["round72_probe_path_used"] is False
    assert engine.fasttext_embedding.is_loaded() is False
    assert result["commit_created_target"] is False
    assert result["commit_rejected_target"] is True
    assert result["gate_pass"] is False
    assert result["wrapper_vector_found_after_commit"] is False
    assert result["store_delta"] == 0
    assert result["target_update_count"] == 0
    assert engine.eve_specific_vector_store.is_eve_specific("민석") is False

    audit = result["commit_report"]["audit_report"]
    item = _candidate(audit, "민석")
    assert item["observed_count"] == 2
    assert item["is_eve_specific_candidate"] is True
    assert item["context_diverse"] is True
    assert item["evidence_status"] == "threshold_met_context_diverse"
    assert "insufficient_known_context" in item["reasons"]
    assert audit["known_context_count"] == 0


def test_round73_commit_smoke_reports_no_eve_specific_route_without_loaded_context() -> None:
    engine = build_full_engine()

    result = run_round73_explicit_eve_specific_commit_smoke(engine)

    before = result["before_telemetry"]
    after = result["after_telemetry"]
    assert int(after.get("eve_specific_hits", 0)) == int(before.get("eve_specific_hits", 0))
    assert int(after.get("fallback_uses", 0)) > int(before.get("fallback_uses", 0))
    assert result["baseline_after_commit"]["measurement_version"] == "v3_round72_eve_specific_smoke_drift_baseline"
    assert result["baseline_after_commit"]["route_distribution"]["eve_specific_hits"] == 0
    assert result["baseline_after_commit"]["self_learning_policy"]["min_observations_for_commit"] == 2
    assert result["baseline_after_commit"]["self_learning_policy"]["context_diversity_gate_enabled"] is True


def test_round73_policy_remains_explicit_commit_only_after_smoke() -> None:
    engine = build_full_engine()

    result = run_round73_explicit_eve_specific_commit_smoke(engine)
    learner_stats = engine.eve_self_learning.stats()

    assert result["policy"]["auto_promotion_enabled"] is False
    assert result["policy"]["commit_gate_enabled"] is True
    assert result["policy"]["min_observations_for_commit"] == 2
    assert result["policy"]["context_diversity_gate_enabled"] is True
    assert result["policy"]["thresholds_changed"] is False
    assert result["policy"]["context_diversity_gate_changed"] is False
    assert result["policy"]["automatic_rollback_enabled"] is False
    assert result["policy"]["memory_quarantine_unchanged"] is True
    assert result["policy"]["fasttext_seed_mutation"] is False
    assert result["policy"]["agp_bypass"] is False
    assert learner_stats["auto_promotion_enabled"] is False
    assert learner_stats["commit_gate_enabled"] is True
    assert learner_stats["context_diversity_gate_enabled"] is True
