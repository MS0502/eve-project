"""EVE v3 round72 — smoke rerun and EveSpecific drift baseline.

Round72 does not change the active self-learning policy. It clarifies the
EmbeddingWrapper lookup route and re-measures the EveSpecific smoke/drift surface
after the Round71 consolidation baseline.
"""

from __future__ import annotations

import numpy as np

from adapters.embedding_wrapper import EmbeddingWrapper
from adapters.external_seed_manifest import measure_eve_specific_round72_baseline
from adapters.runtime_smoke_runner import run_round72_eve_specific_smoke_baseline
from main import build_full_engine


class _PrimaryMissing:
    def is_loaded(self):
        return True

    def get_embedding(self, word):
        return None

    def get_vector(self, word):
        return None

    def get_dimension(self):
        return 300


class _FallbackCounting:
    def __init__(self):
        self.dim = 50
        self.embeddings = {"local": np.ones(2, dtype=np.float32)}
        self.calls = 0

    def get_embedding(self, word):
        self.calls += 1
        return self.embeddings.get(word)

    def stats(self):
        return {"fallback_test": True}


def test_round72_get_embedding_route_policy_is_explicit_and_no_duplicate_fallback_api_call() -> None:
    fallback = _FallbackCounting()
    wrapper = EmbeddingWrapper(primary=_PrimaryMissing(), fallback=fallback, eve_specific=None)

    local = wrapper.get_embedding("local")
    assert np.allclose(local, np.ones(2, dtype=np.float32))
    assert fallback.calls == 0  # direct local map precheck, not fallback API

    missing = wrapper.get_embedding("missing")
    assert missing is None
    assert fallback.calls == 1  # final fallback API boundary is called once

    policy = wrapper.lookup_route_policy_snapshot()
    assert policy["policy_version"] == "v3_round72_embedding_lookup_route_snapshot"
    assert policy["pmi_svd_local_precheck_enabled"] is True
    assert policy["pmi_svd_local_precheck_calls_fallback_api"] is False
    assert policy["fallback_api_called_at_most_once_per_get_embedding_call"] is True
    assert policy["routing_changed_in_round72"] is False


def test_round72_baseline_reports_route_distribution_and_preserves_policy() -> None:
    engine = build_full_engine()
    wrapper = engine.self_embedding
    assert engine.fasttext_embedding.is_loaded() is False
    wrapper.get_vector("안녕")
    added = engine.eve_specific_vector_store.add_or_update_vector("EVE", ["안녕"], engine=engine)
    assert added is False
    wrapper.get_vector("EVE")
    wrapper.get_vector("없는없는단어")

    report = measure_eve_specific_round72_baseline(engine)
    assert report["measurement_version"] == "v3_round72_eve_specific_smoke_drift_baseline"
    assert report["baseline_round"] == 72
    route = report["route_distribution"]
    assert route["fasttext_primary_hits"] == 0
    assert route["eve_specific_hits"] == 0
    assert route["pmi_svd_fallback_uses"] >= 1
    policy = report["self_learning_policy"]
    assert policy["auto_observe_enabled"] is True
    assert policy["auto_promotion_enabled"] is False
    assert policy["commit_gate_enabled"] is True
    assert policy["min_observations_for_commit"] == 2
    assert policy["context_diversity_gate_enabled"] is True
    assert report["policy"]["routing_changed_in_round72"] is False


def test_round72_smoke_rerun_records_eve_specific_probe_without_commit_path() -> None:
    engine = build_full_engine()
    result = run_round72_eve_specific_smoke_baseline(
        engine,
        ["안녕", "EVE 프로젝트", "없는없는단어"],
        probe_words=("EVE",),
        probe_context_words=("안녕",),
    )
    assert result["smoke_version"] == "v3_round72_eve_specific_smoke_drift_baseline"
    assert result["baseline_round"] == 72
    assert result["self_learning_commit_path_called"] is False
    probe = result["probe_vectors"][0]
    assert probe["added"] is False
    assert probe["reason"] == "no_known_fasttext_context"
    telem = result["smoke"].get("wrapper_telemetry", {})
    assert telem.get("primary_loaded") is False
    assert telem.get("eve_specific_hits", 0) == 0
    assert result["round72_baseline"]["route_distribution"]["eve_specific_hits"] == 0
    assert result["policy"]["auto_promotion_enabled"] is False
    assert result["policy"]["thresholds_changed"] is False
