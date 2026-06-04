"""Rounds192-197 guarded medium30k runtime-load integration.

This module consumes operator-local validation evidence for the registered
``cc.ko.300.subset.medium.30k`` artifact and provides one narrow integration
surface for load-dependent self-learning tests/runtime checks. It never enables
production persistence, runtime lexical->concept mapping, enforcement, AGP
bypass, artifact download/copy, or default startup loading.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

from adapters.explicit_load_guard import (
    FORBIDDEN_ARTIFACT_PATTERNS,
    SELECTED_VECTOR_SELF_LEARNING_CLUSTER_ID,
    guarded_explicit_medium30k_load,
)
from adapters.external_seed_manifest import FASTTEXT_KOREAN_SUBSET_MEDIUM_30K_NAME
from adapters.operator_artifact_verification import DEFAULT_OPERATOR_MEDIUM_30K_DIR

ROUND192_SEED_VECTOR_CASCADE_DIAGNOSIS_VERSION = "v3_round192_seed_vector_cascade_entrypoint_diagnosis"
ROUND193_GUARDED_RUNTIME_PATH_SELECTION_VERSION = "v3_round193_guarded_runtime_path_selection"
ROUND194_GUARDED_RUNTIME_LOAD_INTEGRATION_VERSION = "v3_round194_guarded_runtime_load_integration"
ROUND195_FOCUSED_INTEGRATION_TEST_PLAN_VERSION = "v3_round195_guarded_integration_focused_tests"
ROUND196_VALIDATION_DELTA_RECOMMENDATION_VERSION = "v3_round196_broader_validation_delta_recommendation"
ROUND197_OPERATOR_GUARDED_INTEGRATION_RESULT_VERSION = "v3_round197_operator_guarded_integration_result"

SELECTED_SEED_VECTOR_CASCADE_ENTRYPOINT = "build_full_engine_fasttext_embedding_for_eve_specific_known_context"


def _policy_flags() -> dict[str, bool]:
    return {
        "production_persistence_enabled": False,
        "runtime_mapping_enabled_default": False,
        "enforcement_enabled": False,
        "agp_bypass_used": False,
        "artifacts_written": False,
        "artifacts_included": False,
        "vectors_committed": False,
        "dummy_vectors_created": False,
        "production_persistence_no_go_respected": True,
    }


def _operator_validation_green(validation: dict[str, Any] | None) -> tuple[bool, list[str]]:
    """Return whether the operator-local validation evidence permits a runtime load.

    The gate intentionally requires the operator command to have been run with
    ``--attempt-load`` and to have succeeded. A no-load readiness report is
    useful evidence, but it is not enough to attach a real adapter to an engine.
    """

    blockers: list[str] = []
    if not isinstance(validation, dict):
        return False, ["operator_validation_report_missing"]
    if validation.get("status") != "operator_medium30k_validation_green":
        blockers.append("operator_validation_status_not_green")
    if validation.get("success") is not True:
        blockers.append("operator_validation_success_not_true")
    if validation.get("exit_code") not in (0, None):
        blockers.append("operator_validation_exit_code_not_zero")
    if validation.get("attempt_load_requested") is not True:
        blockers.append("operator_validation_attempt_load_required")

    verification = validation.get("verification", {}) if isinstance(validation.get("verification"), dict) else {}
    readiness = validation.get("readiness", {}) if isinstance(validation.get("readiness"), dict) else {}
    preflight = validation.get("preflight", {}) if isinstance(validation.get("preflight"), dict) else {}
    explicit_load = validation.get("explicit_load", {}) if isinstance(validation.get("explicit_load"), dict) else {}
    git_safety = validation.get("git_status_safety", {}) if isinstance(validation.get("git_status_safety"), dict) else {}

    if verification.get("status") != "operator_artifact_verified_green" or verification.get("ready") is not True:
        blockers.append("operator_artifact_verification_not_green")
    if readiness.get("ready") is not True or readiness.get("load_should_be_attempted") is not True:
        blockers.append("seed_vector_readiness_not_green")
    if preflight.get("ready") is not True or preflight.get("hard_block") is True:
        blockers.append("round164_preflight_not_green")
    if explicit_load.get("vectors_loaded") is not True or explicit_load.get("status") != "explicit_load_succeeded":
        blockers.append("operator_explicit_load_probe_not_green")
    if git_safety.get("safe") is not True:
        blockers.append("git_status_safety_not_green")

    for key, expected in _policy_flags().items():
        if key in validation and validation.get(key) is not expected:
            blockers.append(f"policy_flag_{key}_mismatch")
    if validation.get("production_persistence_enabled") is not False:
        blockers.append("production_persistence_enabled_forbidden")
    if validation.get("runtime_mapping_enabled_default") is not False:
        blockers.append("runtime_mapping_default_enabled_forbidden")
    if validation.get("enforcement_enabled") is not False:
        blockers.append("enforcement_enabled_forbidden")

    return not blockers, sorted(set(blockers))


def diagnose_round192_seed_vector_cascade_entrypoints(
    *,
    operator_validation: dict[str, Any] | None = None,
    baseline_failure_taxonomy: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Use operator-local green evidence to diagnose seed/vector entrypoints.

    The selected entrypoint is the build-full-engine fastText adapter that backs
    EveSpecificVectorStore known-context derivation. It explains the observed
    seed/vector artifact cascade without touching runtime mapping/enforcement.
    """

    green, blockers = _operator_validation_green(operator_validation)
    taxonomy = baseline_failure_taxonomy or {
        "seed_vector_artifact_cascade": 127,
        "eve_specific_vector_self_learning_cascade": 40,
        "concept_runtime_mapping_cascade": 38,
    }
    return {
        "version": ROUND192_SEED_VECTOR_CASCADE_DIAGNOSIS_VERSION,
        "round": 192,
        "diagnosis_layer": "seed_vector_artifact_cascade_entrypoint",
        "operator_validation_evidence_green": green,
        "operator_validation_blockers": blockers,
        "baseline_failure_taxonomy": dict(taxonomy),
        "selected_entrypoint": SELECTED_SEED_VECTOR_CASCADE_ENTRYPOINT,
        "selected_entrypoint_rationale": [
            "operator-local validation proves the medium30k artifact can pass checksum/shape/dtype/preflight/load guards outside the PR diff",
            "EveSpecificVectorStore known-context derivation fails when engine.fasttext_embedding is not loaded",
            "build_full_engine constructs the engine surface used by the failing EVE-specific vector/self-learning tests",
            "the path is narrower than runtime lexical-to-concept mapping and does not create AGP anchors",
        ],
        "non_selected_entrypoints": [
            {"id": "runtime_lexical_concept_mapping", "reason": "out_of_scope_runtime_mapping_enabled_default_must_remain_false"},
            {"id": "production_persistence", "reason": "explicit_no_go"},
            {"id": "artifact_restoration_in_pr", "reason": "vectors_and_seed_subsets_must_not_be_committed"},
        ],
        "read_only": True,
        **_policy_flags(),
    }


def select_round193_guarded_runtime_path(*, selected_entrypoint: str = SELECTED_SEED_VECTOR_CASCADE_ENTRYPOINT) -> dict[str, Any]:
    """Select the one narrow runtime/test path that may call the load guard."""

    selected = selected_entrypoint == SELECTED_SEED_VECTOR_CASCADE_ENTRYPOINT
    return {
        "version": ROUND193_GUARDED_RUNTIME_PATH_SELECTION_VERSION,
        "round": 193,
        "selection_layer": "narrow_guarded_runtime_load_path",
        "selected": selected,
        "selected_entrypoint": selected_entrypoint,
        "selected_cluster_id": SELECTED_VECTOR_SELF_LEARNING_CLUSTER_ID,
        "path": "main.build_full_engine(..., operator_medium30k_load_authorized=True, operator_medium30k_validation=green_report)",
        "should_call_guard_when": [
            "operator validation report is green",
            "operator validation was run with --attempt-load and loaded vectors successfully",
            "caller explicitly passes operator_medium30k_load_authorized=True",
        ],
        "must_not_call_guard_when": [
            "default build_full_engine() call",
            "operator validation report missing/red/no-load-only",
            "runtime mapping or enforcement request is the only reason",
        ],
        "integration_target": "engine.fasttext_embedding attachment before EveSpecificVectorStore and EmbeddingWrapper wiring",
        "real_vectors_required_for_operator_path": True,
        "focused_tests_use_fakes_only_for_guard_behavior": True,
        "read_only": True,
        **_policy_flags(),
    }


def apply_round194_guarded_medium30k_runtime_load(
    engine: Any,
    *,
    operator_validation: dict[str, Any] | None = None,
    load_authorized: bool = False,
    artifact_dir: str | Path | None = None,
    guard_func: Callable[..., dict[str, Any]] = guarded_explicit_medium30k_load,
    adapter_factory: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Optionally attach a loaded medium30k adapter to ``engine.fasttext_embedding``.

    Default behavior is no-load. Even with authorization, a green operator report
    that includes a successful guarded load probe is required before this helper
    calls the existing explicit load guard for the engine instance.
    """

    green, validation_blockers = _operator_validation_green(operator_validation)
    artifact = Path(artifact_dir or DEFAULT_OPERATOR_MEDIUM_30K_DIR)
    status = "default_no_load_explicit_authorization_required"
    guard_report: dict[str, Any] | None = None
    guard_called = False
    attached = False
    blockers = list(validation_blockers)

    if not load_authorized:
        blockers.append("operator_load_authorization_missing")
    elif not green:
        status = "blocked_operator_validation_not_green"
    else:
        guard_called = True
        kwargs: dict[str, Any] = {
            "engine": engine,
            "readiness_gate": deepcopy(operator_validation.get("readiness", {})) if isinstance(operator_validation, dict) else None,
            "artifact_dir": artifact,
            "attempt_load": True,
            "attach_to_engine": True,
        }
        if adapter_factory is not None:
            kwargs["adapter_factory"] = adapter_factory
        guard_report = guard_func(**kwargs)
        if guard_report.get("vectors_loaded") is True and guard_report.get("status") == "explicit_load_succeeded":
            status = "guarded_runtime_load_attached"
            attached = bool(getattr(engine, "fasttext_embedding", None) is not None)
        else:
            status = "guarded_runtime_load_failed_closed"
            blockers.extend(str(item) for item in guard_report.get("blockers", []) or [])
            blockers.append("guarded_runtime_load_not_successful")

    return {
        "version": ROUND194_GUARDED_RUNTIME_LOAD_INTEGRATION_VERSION,
        "round": 194,
        "integration_layer": "explicit_operator_authorized_medium30k_runtime_load",
        "selected_entrypoint": SELECTED_SEED_VECTOR_CASCADE_ENTRYPOINT,
        "selected_cluster_id": SELECTED_VECTOR_SELF_LEARNING_CLUSTER_ID,
        "subset_name": FASTTEXT_KOREAN_SUBSET_MEDIUM_30K_NAME,
        "status": status,
        "operator_validation_green": green,
        "operator_validation_blockers": validation_blockers,
        "load_authorized": bool(load_authorized),
        "guard_called": guard_called,
        "guard_report": guard_report,
        "attached_to_engine": attached,
        "artifact_dir": str(artifact),
        "default_runtime_no_load": not bool(load_authorized),
        "forbidden_artifact_patterns": list(FORBIDDEN_ARTIFACT_PATTERNS),
        "blockers": sorted(set(blockers)),
        **_policy_flags(),
    }


def build_round195_focused_integration_test_plan() -> dict[str, Any]:
    """Return focused tests for the guarded integration behavior."""

    command = "python -m pytest -q tests/test_v3_round192_196_guarded_medium30k_integration.py"
    return {
        "version": ROUND195_FOCUSED_INTEGRATION_TEST_PLAN_VERSION,
        "round": 195,
        "test_layer": "focused_guarded_runtime_integration_without_committed_artifacts",
        "focused_test_command": command,
        "required_cases": [
            "Round192 selects the build_full_engine fastText known-context entrypoint",
            "Round193 documents when the runtime path may call the load guard",
            "Round194 default path does not call the guard or load vectors",
            "Round194 red/no-load-only validation blocks guard calls even when authorized",
            "Round194 green operator validation plus explicit authorization calls the guard and attaches the adapter",
            "main.build_full_engine forwards only explicit operator authorization to the integration helper",
        ],
        "real_vectors_required": False,
        "fake_vectors_allowed": False,
        "fakes_scope": "guard call/adapter attachment behavior only",
        **_policy_flags(),
    }


def build_round196_validation_delta_recommendation(
    *,
    focused_tests_status: str,
    compileall_status: str,
    collect_only_status: str,
    broader_pytest_status: str,
    broader_failure_count: int | None = None,
    broader_taxonomy: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Return the Round196 broader validation delta and next recommendation."""

    return {
        "version": ROUND196_VALIDATION_DELTA_RECOMMENDATION_VERSION,
        "round": 196,
        "validation_layer": "broader_validation_delta_and_next_recommendation",
        "rounds_completed": [192, 193, 194, 195, 196],
        "selected_entrypoint": SELECTED_SEED_VECTOR_CASCADE_ENTRYPOINT,
        "focused_tests_status": focused_tests_status,
        "compileall_status": compileall_status,
        "collect_only_status": collect_only_status,
        "broader_pytest_status": broader_pytest_status,
        "broader_failure_count": broader_failure_count,
        "broader_failure_taxonomy": broader_taxonomy or {
            "seed_vector_artifact_cascade": 127,
            "eve_specific_vector_self_learning_cascade": 40,
            "concept_runtime_mapping_cascade": 38,
        },
        "operator_local_validation_command": "python scripts/operator_validate_medium30k.py --attempt-load",
        "operator_local_validation_evidence": {
            "status": "operator_artifact_verified_green",
            "ready": True,
            "shape": [30000, 300],
            "dtype": "float32",
            "sha256_match": True,
            "git_status_safety_clean": True,
            "production_persistence_enabled": False,
            "runtime_mapping_enabled_default": False,
            "enforcement_enabled": False,
        },
        "recommendation": "operator_run_build_full_engine_with_green_validation_and_explicit_load_authorization_then_remeasure_seed_vector_and_eve_specific_cascades",
        **_policy_flags(),
    }


def build_round197_operator_guarded_integration_result(
    *,
    operator_validation: dict[str, Any],
    runtime_load_report: dict[str, Any],
    engine_surface: dict[str, bool] | None = None,
) -> dict[str, Any]:
    """Record the operator-local guarded integration result after Codespaces validation.

    This is an evidence consolidation surface only. It accepts the operator's
    already-produced validation JSON and runtime load report, verifies the
    expected safety invariants, and points the next step at remeasuring the
    load-dependent EVE-specific vector/self-learning cascade.
    """

    green, validation_blockers = _operator_validation_green(operator_validation)
    report = runtime_load_report if isinstance(runtime_load_report, dict) else {}
    surface = engine_surface or {}

    blockers: list[str] = list(validation_blockers)
    if not green:
        blockers.append("operator_validation_not_green")
    if report.get("status") != "guarded_runtime_load_attached":
        blockers.append("runtime_load_status_not_attached")
    if report.get("attached_to_engine") is not True:
        blockers.append("runtime_load_not_attached_to_engine")
    if report.get("guard_called") is not True:
        blockers.append("runtime_load_guard_not_called")
    if report.get("blockers") not in ([], None):
        blockers.append("runtime_load_report_has_blockers")
    if surface and surface.get("self_embedding_present") is not True:
        blockers.append("self_embedding_missing")
    if surface and surface.get("self_embedding_backup_present") is not True:
        blockers.append("self_embedding_backup_missing")

    policy = _policy_flags()
    for key, expected in policy.items():
        validation_value = operator_validation.get(key) if isinstance(operator_validation, dict) else None
        report_value = report.get(key)
        if validation_value is not None and validation_value is not expected:
            blockers.append(f"operator_validation_policy_flag_{key}_mismatch")
        if report_value is not None and report_value is not expected:
            blockers.append(f"runtime_load_policy_flag_{key}_mismatch")

    unique_blockers = sorted(set(blockers))
    accepted = not unique_blockers

    return {
        "version": ROUND197_OPERATOR_GUARDED_INTEGRATION_RESULT_VERSION,
        "round": 197,
        "result_layer": "operator_local_guarded_medium30k_integration_result",
        "operator_validation_green": green,
        "operator_validation_command": "python scripts/operator_validate_medium30k.py --attempt-load",
        "build_full_engine_operator_authorized_path_succeeded": accepted,
        "medium30k_runtime_attached_to_engine": report.get("attached_to_engine") is True,
        "medium30k_runtime_load_status": report.get("status"),
        "medium30k_runtime_guard_called": report.get("guard_called") is True,
        "engine_surface": {
            "self_embedding_present": bool(surface.get("self_embedding_present")),
            "self_embedding_backup_present": bool(surface.get("self_embedding_backup_present")),
        },
        "blockers": unique_blockers,
        "accepted_as_guarded_integration_evidence": accepted,
        "next_recommendation": "remeasure_focused_eve_specific_vector_self_learning_cascade_before_broader_repairs",
        "must_not_change_next": [
            "production_persistence_enabled",
            "runtime_mapping_enabled_default",
            "enforcement_enabled",
            "agp_bypass",
            "fasttext_seed_vectors",
            "semantic_memory_or_quarantine",
        ],
        **policy,
    }


__all__ = [
    "ROUND192_SEED_VECTOR_CASCADE_DIAGNOSIS_VERSION",
    "ROUND193_GUARDED_RUNTIME_PATH_SELECTION_VERSION",
    "ROUND194_GUARDED_RUNTIME_LOAD_INTEGRATION_VERSION",
    "ROUND195_FOCUSED_INTEGRATION_TEST_PLAN_VERSION",
    "ROUND196_VALIDATION_DELTA_RECOMMENDATION_VERSION",
    "ROUND197_OPERATOR_GUARDED_INTEGRATION_RESULT_VERSION",
    "SELECTED_SEED_VECTOR_CASCADE_ENTRYPOINT",
    "apply_round194_guarded_medium30k_runtime_load",
    "build_round195_focused_integration_test_plan",
    "build_round196_validation_delta_recommendation",
    "build_round197_operator_guarded_integration_result",
    "diagnose_round192_seed_vector_cascade_entrypoints",
    "select_round193_guarded_runtime_path",
]
