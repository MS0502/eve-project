"""EVE v3 round20 — AGP operational snapshot and Task 2 readiness.

This module closes the first AGP integration block with read-only, structured
status reports. It is intentionally data-only:

- no threshold updates
- no mode changes
- no fallback pool changes
- no semantic guard changes
- no trace mutation

The output is meant to be stable enough for later round reports and for the
Task 2 fastText/lexical migration readiness check.
"""

from __future__ import annotations

from typing import Any

from adapters.agp_adapter import (
    AGP_FALLBACK_SURFACE_POOL,
    AGP_MODE_OBSERVATION,
    AGP_MODE_VETO,
    AGP_REASON_HORMONE_MISMATCH,
    AGP_REASON_UNKNOWN_CATEGORY,
)
from adapters.agp_trace_analyzer import AGPTraceAnalyzer


AGP_OPERATIONAL_SNAPSHOT_VERSION = "v3_round20_agp_stability_snapshot"
TASK2_READINESS_VERSION = "v3_round20_task2_readiness"

AGP_INVARIANTS_ROUND_1_TO_19 = (
    "v3_constitution_committed",
    "reproducible_determinism_governance",
    "default_mode_observation",
    "semantic_guard_frozen",
    "agp_input_output_separated",
    "agp_observation_before_veto",
    "fallback_data_before_surface",
    "raw_text_fallback_branching_forbidden",
    "trace_analyzer_read_only",
    "recommendation_data_not_text",
    "recommendation_no_auto_apply",
    "threshold_changes_explicit_only",
    "decision_data_runtime_leak_forbidden",
    "compositor_double_lock",
    "speech_hub_double_lock",
    "fallback_pool_minimal",
    "no_memory_quarantine_changes",
    "no_fasttext_hyperbolic_changes_yet",
    "round19_decision_stabilization_audited",
)

TASK2_AFFECTED_MODULES = (
    "self_embedding_adapter",
    "thought_chain_adapter",
    "concept_memory_adapter",
    "attention_analyzer",
    "semantic_distance_adapter",
)


def build_agp_snapshot(engine: Any) -> dict[str, Any]:
    """Return a read-only structured snapshot of the current AGP state.

    The function only reads engine/adapters. It does not append traces, change
    modes, tune thresholds, or activate veto/fallback. All values are converted
    to simple JSON-like primitives so the result can be copied into a round
    report or future machine-readable artifact.
    """
    agp = getattr(engine, "agp_adapter", None)
    compositor = getattr(engine, "compositor", None)
    speech_hub = getattr(engine, "speech_hub", None)
    analyzer = AGPTraceAnalyzer.from_engine(engine)

    category_threshold = float(getattr(agp, "category_threshold", 0.3))
    hormone_threshold = float(getattr(agp, "hormone_threshold", 0.5))
    agp_mode = getattr(agp, "mode", AGP_MODE_OBSERVATION)
    compositor_mode = getattr(compositor, "agp_mode", AGP_MODE_OBSERVATION)
    speech_hub_mode = getattr(speech_hub, "agp_mode", AGP_MODE_OBSERVATION)

    trace_sources = dict(analyzer.summarize_by_layer())
    reason_counts = dict(analyzer.summarize_by_reason())

    return {
        "version": AGP_OPERATIONAL_SNAPSHOT_VERSION,
        "mode_state": {
            "default": AGP_MODE_OBSERVATION,
            "agp_adapter_mode": agp_mode,
            "compositor_mode": compositor_mode,
            "speech_hub_mode": speech_hub_mode,
            "available_modes": [AGP_MODE_OBSERVATION, AGP_MODE_VETO],
            "compositor_double_lock_required": True,
            "speech_hub_double_lock_required": True,
        },
        "thresholds": {
            "category_threshold": category_threshold,
            "hormone_threshold": hormone_threshold,
            "explicit_set_required": True,
            "auto_apply_recommendations": False,
        },
        "fallback": {
            "pool_size": len(AGP_FALLBACK_SURFACE_POOL),
            "pool": list(AGP_FALLBACK_SURFACE_POOL),
            "reasons": [AGP_REASON_UNKNOWN_CATEGORY, AGP_REASON_HORMONE_MISMATCH],
            "raw_text_branching_allowed": False,
            "construction_grammar_future_owner": True,
        },
        "decision_pipeline": {
            "trace": "active",
            "analyzer": "available",
            "threshold_report": "available",
            "decision_module": "available",
            "explicit_set_required": True,
            "automatic_threshold_application": False,
        },
        "trace_state": {
            "sources": sorted(analyzer.sources.keys()),
            "total_observations": sum(bucket.get("total", 0) for bucket in trace_sources.values()),
            "pass_rate": analyzer.pass_rate(),
            "reason_counts": reason_counts,
            "by_layer": trace_sources,
        },
        "invariants_round_1_to_19": list(AGP_INVARIANTS_ROUND_1_TO_19),
    }


def assess_task2_readiness(engine: Any) -> dict[str, Any]:
    """Return a read-only Task 2 fastText migration readiness assessment.

    Task 2 will rewrite lexical embedding toward fastText 300d. AGP should not
    depend directly on lexical vectors; it should depend on EVE-internal meaning
    and category activation. This helper records that boundary before Task 2
    begins, without starting the migration.
    """
    agp = getattr(engine, "agp_adapter", None)
    components = {
        "agp_adapter": agp is not None,
        "self_embedding_adapter": hasattr(engine, "self_embedding"),
        "concept_memory_adapter": hasattr(engine, "concept_memory"),
        "thought_chain_adapter": hasattr(engine, "thought_chain"),
        "activation_adapter": hasattr(engine, "activation_adapter"),
        "hormone_adapter": hasattr(engine, "hormone_adapter"),
        "appraisal_classifier": True,
    }
    core_ready = all(
        components[name]
        for name in (
            "agp_adapter",
            "activation_adapter",
            "hormone_adapter",
            "self_embedding_adapter",
            "concept_memory_adapter",
        )
    )

    return {
        "version": TASK2_READINESS_VERSION,
        "task": "fastText_300d_lexical_migration",
        "agp_dependency_on_lexical_embedding": {
            "direct": False,
            "indirect": True,
            "boundary": "AGP verifies EVE-internal category activation, not raw lexical vectors.",
            "migration_risk": "indirect_via_lex_concept_mapping_and_category_activation",
        },
        "affected_modules": list(TASK2_AFFECTED_MODULES),
        "component_presence": components,
        "external_seed_policy_required": True,
        "external_seed_policy_checks": [
            "provenance_license_version_manifest",
            "deterministic_incremental_update_only",
            "seed_drift_metric_required",
            "agp_anchor_uses_internal_sa_not_seed_space",
        ],
        "readiness": "ready_for_task2_planning" if core_ready else "needs_more_audit",
        "recommended_next": "task2_patch_1_readiness_plan" if core_ready else "more_agp_audit",
    }


def build_round20_operational_report_data(engine: Any) -> dict[str, Any]:
    """Return the round20 combined snapshot/readiness data package.

    This is a convenience wrapper for reports and tests. It remains read-only.
    """
    return {
        "version": "v3_round20_operational_report_data",
        "agp_snapshot": build_agp_snapshot(engine),
        "task2_readiness": assess_task2_readiness(engine),
    }
