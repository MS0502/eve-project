"""Rounds167-171 concept/runtime mapping diagnosis helpers.

These helpers are read-only operator/debug surfaces for the post-PR #24
validation loop. They classify the currently observed concept/runtime mapping
cluster without touching seed/vector artifacts, runtime mapping defaults,
enforcement, AGP, semantic memory, or production persistence.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

ROUND167_VERSION = "v3_round167_concept_runtime_mapping_failure_taxonomy"
ROUND168_VERSION = "v3_round168_non_artifact_subcluster_selection"
ROUND169_VERSION = "v3_round169_state_debug_baseline_round_gate"
ROUND170_VERSION = "v3_round170_focused_state_debug_verification"
ROUND171_VERSION = "v3_round171_broader_validation_delta_recommendation"


_CONCEPT_RUNTIME_MAPPING_TAXONOMY: dict[str, Any] = {
    "version": ROUND167_VERSION,
    "round": 167,
    "cluster": "concept_runtime_mapping_cascade",
    "total_observed_failures": 43,
    "subclusters": [
        {
            "id": "artifact_dependent_eve_specific_commit_prerequisite",
            "count": 38,
            "artifact_dependent": True,
            "selected_for_fix": False,
            "blocker": "missing registered vectors.npy artifacts keep fastText context unavailable, so 민석 EveSpecific vector commit fixtures fail closed with insufficient_known_context",
            "policy": "hard_block_until_operator_restores_real_artifacts; do_not_create_dummy_vectors_or_fake_checksums",
        },
        {
            "id": "state_debug_baseline_round_metadata",
            "count": 5,
            "artifact_dependent": False,
            "selected_for_fix": True,
            "blocker": "fresh LexConceptMappingAdapter state-debug reported Round96 before any Round95/96 runtime-mapping operator/precheck surface was invoked",
            "policy": "restore the inert baseline to Round94 while preserving explicit Round95/96 transitions and disabled flags",
        },
    ],
    "global_policy_flags": {
        "production_persistence_enabled": False,
        "runtime_mapping_enabled_default": False,
        "enforcement_enabled": False,
        "agp_bypass_used": False,
    },
    "vectors_written": False,
    "read_only": True,
}


def concept_runtime_mapping_failure_taxonomy() -> dict[str, Any]:
    """Return the Round167 read-only failure taxonomy."""
    return deepcopy(_CONCEPT_RUNTIME_MAPPING_TAXONOMY)


def selected_non_artifact_subcluster() -> dict[str, Any]:
    """Return the Round168 selected non-artifact subcluster and rationale."""
    taxonomy = concept_runtime_mapping_failure_taxonomy()
    selected = next(row for row in taxonomy["subclusters"] if row["selected_for_fix"])
    return {
        "version": ROUND168_VERSION,
        "round": 168,
        "selected_subcluster": deepcopy(selected),
        "rationale": [
            "It is deterministic metadata only and does not require fastText or EveSpecific vector artifacts.",
            "It does not enable production persistence, runtime mapping by default, enforcement, or AGP bypass.",
            "It removes a stale state-debug mismatch while leaving artifact-dependent behavior honestly blocked.",
        ],
        "rejected_subclusters": [
            deepcopy(row)
            for row in taxonomy["subclusters"]
            if not row["selected_for_fix"]
        ],
        "read_only": True,
    }


def round169_state_debug_baseline_gate(engine: Any) -> dict[str, Any]:
    """Inspect the inert runtime-mapping state-debug baseline without mutation."""
    lcm = getattr(engine, "lex_concept_mapping", None)
    stats = lcm.stats() if lcm is not None and hasattr(lcm, "stats") else {}
    return {
        "version": ROUND169_VERSION,
        "round": 169,
        "selected_subcluster": "state_debug_baseline_round_metadata",
        "baseline_round": int(stats.get("round", -1) or -1),
        "baseline_round_expected": 94,
        "baseline_round_ok": int(stats.get("round", -1) or -1) == 94,
        "runtime_mapping_enabled": bool(stats.get("runtime_mapping_enabled", False)),
        "enforcement_enabled": bool(stats.get("enforcement_enabled", False)),
        "production_persistence_enabled": False,
        "vectors_written": False,
        "agp_bypass_used": False,
        "read_only": True,
    }


def round170_focused_verification_summary(commands: list[dict[str, str]]) -> dict[str, Any]:
    """Package focused Round170 verification command results."""
    return {
        "version": ROUND170_VERSION,
        "round": 170,
        "selected_subcluster": "state_debug_baseline_round_metadata",
        "commands": deepcopy(commands),
        "focused_verification_status": "passed" if all(row.get("status") == "passed" for row in commands) else "failed",
        "vectors_written": False,
        "runtime_mapping_enabled_default": False,
        "enforcement_enabled": False,
        "production_persistence_enabled": False,
        "read_only": True,
    }


def round171_broader_validation_recommendation() -> dict[str, Any]:
    """Return the Round171 broader validation delta recommendation."""
    return {
        "version": ROUND171_VERSION,
        "round": 171,
        "broader_validation_expected_status": "still_red_until_operator_artifacts_are_restored",
        "expected_delta_from_round166": {
            "failed_tests": -5,
            "passed_tests": 8,
            "collected_tests": 3,
        },
        "remaining_taxonomy_after_selected_fix": {
            "seed_vector_artifact_cascade": 127,
            "eve_specific_vector_self_learning_cascade": 40,
            "concept_runtime_mapping_cascade": 38,
        },
        "next_recommendation": "restore real registered vector artifacts outside the PR, then rerun Round164 load-dependent repair preflight; if artifacts remain unavailable, choose another non-artifact metadata/diagnostic subcluster only",
        "production_persistence_enabled": False,
        "runtime_mapping_enabled_default": False,
        "enforcement_enabled": False,
        "vectors_written": False,
        "read_only": True,
    }
