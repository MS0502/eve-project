"""EVE v3 round20 — AGP operational snapshot and Task 2 readiness.

Round20 is a checkpoint between the AGP integration block and Task 2 planning.
It adds read-only snapshot/readiness helpers and documentation tests only.
"""

from copy import deepcopy
from pathlib import Path

from adapters.agp_adapter import (
    AGP_FALLBACK_SURFACE_POOL,
    AGP_MODE_OBSERVATION,
    AGP_MODE_VETO,
)
from adapters.agp_operational_snapshot import (
    AGP_INVARIANTS_ROUND_1_TO_19,
    AGP_OPERATIONAL_SNAPSHOT_VERSION,
    assess_task2_readiness,
    build_agp_snapshot,
    build_round20_operational_report_data,
)
from main import build_full_engine


def test_snapshot_reflects_current_agp_state():
    """The operational snapshot should reflect the current engine state."""
    engine = build_full_engine()
    snapshot = build_agp_snapshot(engine)

    assert snapshot["version"] == AGP_OPERATIONAL_SNAPSHOT_VERSION
    assert snapshot["mode_state"]["default"] == AGP_MODE_OBSERVATION
    assert snapshot["mode_state"]["agp_adapter_mode"] == engine.agp_adapter.mode
    assert snapshot["mode_state"]["compositor_mode"] == engine.compositor.agp_mode
    assert snapshot["mode_state"]["speech_hub_mode"] == engine.speech_hub.agp_mode
    assert snapshot["mode_state"]["available_modes"] == [AGP_MODE_OBSERVATION, AGP_MODE_VETO]
    assert snapshot["thresholds"] == {
        "category_threshold": engine.agp_adapter.category_threshold,
        "hormone_threshold": engine.agp_adapter.hormone_threshold,
        "explicit_set_required": True,
        "auto_apply_recommendations": False,
    }
    assert snapshot["fallback"]["pool_size"] == len(AGP_FALLBACK_SURFACE_POOL)
    assert snapshot["fallback"]["raw_text_branching_allowed"] is False


def test_snapshot_is_read_only():
    """Snapshot generation must not change modes, thresholds, traces, or pool."""
    engine = build_full_engine()
    engine.compositor.composition_phrase(
        type("Composition", (), {
            "inputs": ("a", "b"),
            "concepts": ["c"],
            "method": "rule",
            "confidence": 0.5,
        })()
    )
    engine.speech_hub.generate("greeting_simple", {"emotional_state": "neutral"}, seed=0)
    before = {
        "modes": (engine.agp_adapter.mode, engine.compositor.agp_mode, engine.speech_hub.agp_mode),
        "thresholds": (engine.agp_adapter.category_threshold, engine.agp_adapter.hormone_threshold),
        "pool": tuple(AGP_FALLBACK_SURFACE_POOL),
        "compositor_trace": deepcopy(engine.compositor.agp_trace),
        "speech_trace": deepcopy(engine.speech_hub.agp_trace),
    }

    snapshot = build_agp_snapshot(engine)
    snapshot["mode_state"]["agp_adapter_mode"] = "mutated copy"

    assert {
        "modes": (engine.agp_adapter.mode, engine.compositor.agp_mode, engine.speech_hub.agp_mode),
        "thresholds": (engine.agp_adapter.category_threshold, engine.agp_adapter.hormone_threshold),
        "pool": tuple(AGP_FALLBACK_SURFACE_POOL),
        "compositor_trace": engine.compositor.agp_trace,
        "speech_trace": engine.speech_hub.agp_trace,
    } == before


def test_snapshot_is_data_dict_not_natural_language_report():
    """Round20 snapshot should be structured data, not a prose recommendation."""
    snapshot = build_agp_snapshot(build_full_engine())

    assert isinstance(snapshot, dict)
    assert "recommendation_text" not in snapshot
    assert "natural_language_summary" not in snapshot
    assert "agp_snapshot" not in snapshot  # this function returns the snapshot itself
    assert isinstance(snapshot["invariants_round_1_to_19"], list)
    assert all(isinstance(item, str) for item in snapshot["invariants_round_1_to_19"])


def test_task2_readiness_check_runs_and_preserves_agp_boundary():
    """Task 2 readiness should identify AGP as indirectly, not directly, affected."""
    engine = build_full_engine()
    readiness = assess_task2_readiness(engine)

    assert readiness["task"] == "fastText_300d_lexical_migration"
    assert readiness["agp_dependency_on_lexical_embedding"]["direct"] is False
    assert readiness["agp_dependency_on_lexical_embedding"]["indirect"] is True
    assert readiness["readiness"] in {"ready_for_task2_planning", "needs_more_audit"}
    assert readiness["external_seed_policy_required"] is True
    assert "self_embedding_adapter" in readiness["affected_modules"]
    assert "concept_memory_adapter" in readiness["affected_modules"]
    assert "agp_anchor_uses_internal_sa_not_seed_space" in readiness["external_seed_policy_checks"]


def test_round1_to_19_invariant_summary_is_complete_enough():
    """The snapshot must preserve the AGP block's main invariants."""
    snapshot = build_agp_snapshot(build_full_engine())
    invariants = set(snapshot["invariants_round_1_to_19"])

    required = {
        "v3_constitution_committed",
        "default_mode_observation",
        "semantic_guard_frozen",
        "trace_analyzer_read_only",
        "recommendation_no_auto_apply",
        "threshold_changes_explicit_only",
        "decision_data_runtime_leak_forbidden",
        "compositor_double_lock",
        "speech_hub_double_lock",
        "fallback_pool_minimal",
        "no_fasttext_hyperbolic_changes_yet",
        "round19_decision_stabilization_audited",
    }
    assert required.issubset(invariants)
    assert len(AGP_INVARIANTS_ROUND_1_TO_19) >= 19


def test_round20_report_data_and_documentation_contract():
    """Round20 should have a combined data package and documented checkpoint."""
    engine = build_full_engine()
    data = build_round20_operational_report_data(engine)

    assert data["version"] == "v3_round20_operational_report_data"
    assert data["agp_snapshot"]["version"] == AGP_OPERATIONAL_SNAPSHOT_VERSION
    assert data["task2_readiness"]["task"] == "fastText_300d_lexical_migration"
    assert "natural_language_summary" not in data

    report = Path("ROUND_V3_R20_REPORT.md")
    assert report.exists()
    text = report.read_text(encoding="utf-8")
    assert "AGP Operational Snapshot" in text
    assert "Task 2 Readiness" in text
    assert "production logic changed: none" in text
    assert "default observation" in text
    assert "External Seed Policy" in text
