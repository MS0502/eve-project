"""Rounds182-186 guarded explicit-load repair surfaces.

These helpers advance the load-dependent vector/self-learning cluster only when
operator-side medium 30k readiness is green in the local environment. They do
not download, copy, synthesize, commit, or persist vector artifacts. The actual
load path is opt-in through ``attempt_load=True`` and remains fail-closed unless
the existing Round164 load-dependent preflight is green.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

from adapters.external_seed_manifest import FASTTEXT_KOREAN_SUBSET_MEDIUM_30K_NAME
from adapters.fasttext_embedding_adapter import FasttextEmbeddingAdapter
from adapters.operator_verified_artifact_evidence import SELECTED_CLUSTER_ID
from adapters.seed_vector_restore_contract import build_round164_load_dependent_repair_preflight

ROUND182_VECTOR_SELF_LEARNING_CLUSTER_VERSION = "v3_round182_vector_self_learning_cluster_selection"
ROUND183_GUARDED_EXPLICIT_LOAD_VERSION = "v3_round183_guarded_explicit_load_repair_path"
ROUND184_EXPLICIT_LOAD_GUARD_TEST_PLAN_VERSION = "v3_round184_explicit_load_guard_test_plan"
ROUND185_OPERATOR_LOCAL_VALIDATION_COMMANDS_VERSION = "v3_round185_operator_local_validation_commands"
ROUND186_VALIDATION_DELTA_RECOMMENDATION_VERSION = "v3_round186_validation_delta_recommendation"

SELECTED_VECTOR_SELF_LEARNING_CLUSTER_ID = "eve_specific_commit_known_context_medium30k_explicit_load_guard"

FORBIDDEN_ARTIFACT_PATTERNS: tuple[str, ...] = (
    "_operator_artifacts/**",
    "seeds/subsets/**",
    "**/vectors.npy",
    "**/subset_manifest.json",
    "**/*.zip",
    "**/*.part",
)


def _policy_flags() -> dict[str, bool]:
    return {
        "production_persistence_enabled": False,
        "runtime_mapping_enabled_default": False,
        "enforcement_enabled": False,
        "agp_bypass_used": False,
        "artifacts_included": False,
        "vectors_committed": False,
        "dummy_vectors_created": False,
    }


def select_round182_vector_self_learning_cluster(
    *,
    operator_preflight: dict[str, Any] | None = None,
    previous_cluster_id: str = SELECTED_CLUSTER_ID,
) -> dict[str, Any]:
    """Select one narrow vector/self-learning cluster for load-dependent repair.

    The selected cluster is the EVE-specific vector commit known-context path:
    ``EveSpecificVectorStore`` can create deterministic vectors only when the
    medium 30k fastText adapter is explicitly loaded. This is narrower than
    repairing every vector cascade because it touches only the explicit-load
    gate needed by self-learning commit audits and vector creation.
    """

    preflight = deepcopy(operator_preflight or {})
    preflight_ready = bool(preflight.get("ready") is True and preflight.get("hard_block") is False)
    return {
        "version": ROUND182_VECTOR_SELF_LEARNING_CLUSTER_VERSION,
        "round": 182,
        "selection_layer": "load_dependent_vector_self_learning_cluster",
        "status": "selected_guarded_load_dependent_cluster",
        "previous_cluster_id": str(previous_cluster_id),
        "selected_cluster": {
            "id": SELECTED_VECTOR_SELF_LEARNING_CLUSTER_ID,
            "name": "EVE-specific commit known-context path gated by explicit medium 30k load",
            "taxonomy": "EVE-specific vector/self-learning cascade",
            "selected": True,
            "rationale": [
                "operator-side preflight is green locally, so the next useful repair is a load-dependent self-learning vector path",
                "EveSpecificVectorStore known-context derivation depends on a loaded fastText medium 30k adapter",
                "the cluster is narrow and does not require runtime mapping, enforcement, persistence, or AGP changes",
            ],
            "repair_boundaries": [
                "add an explicit load guard before any FasttextEmbeddingAdapter.load() attempt",
                "do not auto-load from runtime startup or debug/reporting paths",
                "do not commit or synthesize vector artifacts",
                "do not mutate EveSpecificVectorStore during readiness/report calls",
            ],
        },
        "operator_preflight_status": preflight.get("status"),
        "operator_preflight_ready_reported": preflight_ready,
        "actual_load_allowed_by_selection": False,
        "requires_round183_guard": True,
        "read_only": True,
        **_policy_flags(),
    }


def guarded_explicit_medium30k_load(
    *,
    engine: Any | None = None,
    readiness_gate: dict[str, Any] | None = None,
    artifact_dir: str | Path | None = None,
    attempt_load: bool = False,
    attach_to_engine: bool = False,
    adapter_factory: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Run the Round183 explicit-load guard and optionally load the adapter.

    ``attempt_load`` is deliberately false by default. When false, a green
    preflight returns a ready report only. When true, the function requires the
    Round164 preflight to be green before constructing and loading an adapter.
    Test code may pass a deterministic ``adapter_factory``; production use
    defaults to ``FasttextEmbeddingAdapter`` and still relies on its manifest,
    checksum, shape, and dtype checks.
    """

    preflight = build_round164_load_dependent_repair_preflight(readiness_gate=readiness_gate)
    guard_green = bool(preflight.get("ready") is True and preflight.get("hard_block") is False)
    blockers = list(preflight.get("blockers", []) or [])
    base_dir = str(artifact_dir) if artifact_dir is not None else None
    status = "guard_ready_explicit_load_not_attempted"
    vectors_loaded = False
    load_error: str | None = None
    adapter_class: str | None = None

    if not guard_green:
        status = "blocked_preflight_not_green"
    elif not attempt_load:
        status = "guard_ready_explicit_load_not_attempted"
    else:
        factory = adapter_factory or FasttextEmbeddingAdapter
        try:
            kwargs = {"engine": engine}
            if artifact_dir is not None:
                kwargs["subset_dir"] = artifact_dir
            adapter = factory(**kwargs)
            adapter_class = adapter.__class__.__name__
            loaded = adapter.load()
            vectors_loaded = bool(loaded)
            status = "explicit_load_succeeded" if vectors_loaded else "explicit_load_failed_closed"
            if vectors_loaded and attach_to_engine and engine is not None:
                setattr(engine, "fasttext_embedding", adapter)
        except Exception as exc:  # fail closed and report exact class/message
            status = "explicit_load_failed_closed"
            load_error = f"{exc.__class__.__name__}:{exc}"
            blockers.append("explicit_load_exception")

    return {
        "version": ROUND183_GUARDED_EXPLICIT_LOAD_VERSION,
        "round": 183,
        "guard_layer": "round164_preflight_before_fasttext_load",
        "selected_cluster_id": SELECTED_VECTOR_SELF_LEARNING_CLUSTER_ID,
        "subset_name": FASTTEXT_KOREAN_SUBSET_MEDIUM_30K_NAME,
        "status": status,
        "artifact_dir": base_dir,
        "attempt_load_requested": bool(attempt_load),
        "attach_to_engine_requested": bool(attach_to_engine),
        "guard_green": guard_green,
        "preflight_status": preflight.get("status"),
        "preflight": preflight,
        "vectors_loaded": vectors_loaded,
        "adapter_class": adapter_class,
        "load_error": load_error,
        "blockers": sorted(set(str(item) for item in blockers)),
        "read_only_until_attempt_load": not bool(attempt_load),
        "forbidden_artifact_patterns": list(FORBIDDEN_ARTIFACT_PATTERNS),
        **_policy_flags(),
    }


def build_round184_explicit_load_guard_test_plan(*, focused_test_command: str | None = None) -> dict[str, Any]:
    """Return the focused Round184 guard-test contract."""

    command = focused_test_command or "python -m pytest -q tests/test_v3_round182_186_explicit_load_guard.py"
    return {
        "version": ROUND184_EXPLICIT_LOAD_GUARD_TEST_PLAN_VERSION,
        "round": 184,
        "test_layer": "focused_explicit_load_guard_behavior_without_committed_artifacts",
        "focused_test_command": command,
        "required_cases": [
            "red Round164 preflight blocks adapter construction and load",
            "green Round164 preflight with attempt_load=False reports ready without loading",
            "green Round164 preflight with attempt_load=True calls the explicit adapter load path",
            "load exceptions fail closed and do not attach an adapter to the engine",
            "round reports preserve no-persistence/no-runtime-mapping/no-enforcement/no-AGP-bypass flags",
        ],
        "real_vectors_required": False,
        "artifacts_included": False,
        "read_only": True,
        **_policy_flags(),
    }


def build_round185_operator_local_validation_commands() -> dict[str, Any]:
    """Return exact Codespaces-local commands for validation with real artifacts."""

    commands = [
        "git status --short",
        "python - <<'PY'\nfrom adapters.operator_artifact_verification import verify_operator_subset_artifact\nprint(verify_operator_subset_artifact())\nPY",
        "python - <<'PY'\nfrom adapters.seed_vector_restore_contract import build_round164_load_dependent_repair_preflight\nprint(build_round164_load_dependent_repair_preflight())\nPY",
        "python - <<'PY'\nfrom adapters.explicit_load_guard import guarded_explicit_medium30k_load\nprint(guarded_explicit_medium30k_load(attempt_load=True))\nPY",
        "python -m compileall -q adapters tests main.py",
        "python -m pytest --collect-only -q",
        "python -m pytest -q tests/test_v3_round182_186_explicit_load_guard.py tests/test_v3_round172_176_operator_artifact_loop.py tests/test_v3_round162_164_restore_contract_preflight.py",
        "python -m pytest -q",
    ]
    return {
        "version": ROUND185_OPERATOR_LOCAL_VALIDATION_COMMANDS_VERSION,
        "round": 185,
        "validation_layer": "operator_codespaces_local_real_artifact_commands",
        "commands": commands,
        "notes": [
            "Run these only where _operator_artifacts/subset_medium_30k or the registered subset files are locally present and verified green.",
            "Do not commit vectors.npy, vocab.txt, subset_manifest.json, seeds/subsets, zip files, part files, or _operator_artifacts after running validation.",
            "Production persistence, runtime_mapping_enabled default, and enforcement must remain disabled.",
        ],
        "artifacts_included": False,
        "read_only": True,
        **_policy_flags(),
    }


def build_round186_validation_delta_recommendation(
    *,
    focused_tests_status: str = "pending",
    collect_only_status: str = "pending",
    broader_pytest_status: str = "not_run",
    broader_failure_count: int | None = None,
) -> dict[str, Any]:
    """Summarize Round186 validation delta and next recommendation."""

    remaining_taxonomy = {
        "seed_vector_artifact_cascade": 127,
        "eve_specific_vector_self_learning_cascade": 40,
        "concept_runtime_mapping_cascade": 38,
    }
    return {
        "version": ROUND186_VALIDATION_DELTA_RECOMMENDATION_VERSION,
        "round": 186,
        "validation_layer": "broader_validation_delta_and_next_recommendation",
        "selected_cluster_id": SELECTED_VECTOR_SELF_LEARNING_CLUSTER_ID,
        "focused_tests_status": focused_tests_status,
        "collect_only_status": collect_only_status,
        "broader_pytest_status": broader_pytest_status,
        "broader_failure_count": broader_failure_count,
        "remaining_failure_taxonomy_baseline": remaining_taxonomy,
        "recommendation": "operator_run_real_artifact_explicit_load_validation_then_remeasure_self_learning_cascade",
        "hard_stop": False,
        "artifacts_included": False,
        "read_only": True,
        **_policy_flags(),
    }


__all__ = [
    "ROUND182_VECTOR_SELF_LEARNING_CLUSTER_VERSION",
    "ROUND183_GUARDED_EXPLICIT_LOAD_VERSION",
    "ROUND184_EXPLICIT_LOAD_GUARD_TEST_PLAN_VERSION",
    "ROUND185_OPERATOR_LOCAL_VALIDATION_COMMANDS_VERSION",
    "ROUND186_VALIDATION_DELTA_RECOMMENDATION_VERSION",
    "SELECTED_VECTOR_SELF_LEARNING_CLUSTER_ID",
    "build_round184_explicit_load_guard_test_plan",
    "build_round185_operator_local_validation_commands",
    "build_round186_validation_delta_recommendation",
    "guarded_explicit_medium30k_load",
    "select_round182_vector_self_learning_cluster",
]
