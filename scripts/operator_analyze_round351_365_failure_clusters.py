#!/usr/bin/env python3
"""Rounds351-365 captured pytest failure evidence, cluster taxonomy, and repair delta.

This module is read-only over operator-local pytest evidence.  It may classify
captured nodeids and summarize a guarded repair selection, but it must not load
vectors, fabricate artifacts, enable runtime mapping by default, enable
enforcement, bypass AGP, or write production persistence.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.operator_remeasure_eve_self_learning import _compact_json  # noqa: E402
from scripts.operator_run_local_validation_suite import POLICY_FLAGS  # noqa: E402

ROUND351_365_VERSION = "v3_round351_365_captured_failure_cluster_repair_delta"
PYTEST_NODEID_RE = re.compile(r"^FAILED\s+(?P<nodeid>tests/[^\s]+?::[^\s]+)(?:\s+-\s+.*)?$")
PYTEST_SHORT_COUNT_RE = re.compile(r"(?P<failed>\d+)\s+failed(?:,\s+(?P<passed>\d+)\s+passed)?")

REQUESTED_CLUSTER_ORDER: tuple[str, ...] = (
    "round40_41_wrapper_swap_audit",
    "round94_98_runtime_mapping_enable_smoke",
    "round306_320_artifact_guard_regression",
    "unclassified_pytest_failure",
)

OPERATOR_CAPTURED_135_FAILURE_EVIDENCE: dict[str, Any] = {
    "failed_count": 135,
    "passed_count": 1279,
    "nodeids_captured": True,
    "sample_nodeids": [
        "tests/test_v3_round40_pre_swap_audit.py::test_round40_all_trace_modules_have_parallel_observation_option_unloaded",
        "tests/test_v3_round41_final_swap.py::test_round41_main_py_contains_wrapper_swap_and_backup",
        "tests/test_v3_round41_final_swap.py::test_round41_post_swap_assessment_ready",
        "tests/test_v3_round41_final_swap.py::test_round41_wrapper_primary_used_for_known_word",
        "tests/test_v3_round306_320_split_validation_execution.py::test_round306_320_cli_artifact_dependent_missing_artifacts_fails_closed",
    ],
    "source": "operator-local_round336_350_artifact_dependent_execution_after_PR_46_merge",
}


def extract_pytest_failure_nodeids(pytest_text: str) -> list[str]:
    """Extract unique pytest failure nodeids from full or short-summary text."""

    nodeids: list[str] = []
    seen: set[str] = set()
    for line in pytest_text.splitlines():
        match = PYTEST_NODEID_RE.match(line.strip())
        if not match:
            continue
        nodeid = match.group("nodeid")
        if nodeid not in seen:
            seen.add(nodeid)
            nodeids.append(nodeid)
    return nodeids


def extract_pytest_summary_counts(pytest_text: str) -> dict[str, int | None]:
    """Extract failed/passed counts when pytest printed a terminal summary."""

    failed: int | None = None
    passed: int | None = None
    for line in pytest_text.splitlines():
        match = PYTEST_SHORT_COUNT_RE.search(line)
        if not match:
            continue
        failed = int(match.group("failed"))
        if match.group("passed") is not None:
            passed = int(match.group("passed"))
    return {"failed": failed, "passed": passed}


def _is_round_inclusive(nodeid: str, start: int, end: int) -> bool:
    lowered = nodeid.lower()
    for round_number in range(start, end + 1):
        if f"round{round_number}" in lowered or f"round_{round_number}" in lowered:
            return True
    return False


def cluster_nodeid(nodeid: str) -> str:
    """Map a pytest nodeid into the Round351-355 concrete taxonomy."""

    lowered = nodeid.lower()
    if _is_round_inclusive(lowered, 40, 41) or "wrapper_swap" in lowered or "pre_swap" in lowered or "final_swap" in lowered:
        return "round40_41_wrapper_swap_audit"
    if _is_round_inclusive(lowered, 94, 98) or "runtime_mapping_enable" in lowered or "controlled_smoke" in lowered:
        return "round94_98_runtime_mapping_enable_smoke"
    if _is_round_inclusive(lowered, 306, 320) or "artifact_guard" in lowered or "split_validation" in lowered:
        return "round306_320_artifact_guard_regression"
    return "unclassified_pytest_failure"


def classify_round351_355_failures(nodeids: Iterable[str], *, expected_failed_count: int | None = None) -> dict[str, Any]:
    """Classify captured nodeids into the operator-requested concrete clusters."""

    unique_nodeids = sorted(dict.fromkeys(nodeids))
    clusters: dict[str, list[str]] = {cluster: [] for cluster in REQUESTED_CLUSTER_ORDER}
    for nodeid in unique_nodeids:
        clusters.setdefault(cluster_nodeid(nodeid), []).append(nodeid)

    cluster_counts = {cluster: len(clusters.get(cluster, [])) for cluster in REQUESTED_CLUSTER_ORDER}
    classified = sum(cluster_counts.values())
    unexpanded_count = max(int(expected_failed_count) - classified, 0) if isinstance(expected_failed_count, int) else None
    return {
        "version": "v3_round352_355_concrete_failure_cluster_taxonomy",
        "rounds": [352, 353, 354, 355],
        "expected_failed_count": expected_failed_count,
        "captured_nodeid_count": len(unique_nodeids),
        "classified_captured_nodeid_count": classified,
        "unexpanded_failure_count": unexpanded_count,
        "nodeids_captured": bool(unique_nodeids),
        "cluster_counts": cluster_counts,
        "clusters": {
            cluster: {"count": len(clusters.get(cluster, [])), "sample_nodeids": clusters.get(cluster, [])[:8]}
            for cluster in REQUESTED_CLUSTER_ORDER
        },
    }


def consolidate_round351_captured_evidence(
    *,
    operator_evidence: Mapping[str, Any] | None = None,
    pytest_text: str = "",
) -> dict[str, Any]:
    """Round351: consolidate the 135-failure captured-nodeid evidence."""

    source = dict(operator_evidence or OPERATOR_CAPTURED_135_FAILURE_EVIDENCE)
    parsed_nodeids = extract_pytest_failure_nodeids(pytest_text) if pytest_text else []
    parsed_counts = extract_pytest_summary_counts(pytest_text) if pytest_text else {"failed": None, "passed": None}
    nodeids = parsed_nodeids or list(source.get("sample_nodeids", []))
    failed_count = parsed_counts.get("failed") if parsed_counts.get("failed") is not None else source.get("failed_count")
    passed_count = parsed_counts.get("passed") if parsed_counts.get("passed") is not None else source.get("passed_count")
    checks = {
        "captured_failure_count_recorded": failed_count == 135,
        "captured_passed_count_recorded": isinstance(passed_count, int),
        "nodeid_evidence_present": bool(nodeids),
        "production_persistence_no_go": POLICY_FLAGS.get("production_persistence_enabled") is False,
        "runtime_mapping_default_false": POLICY_FLAGS.get("runtime_mapping_enabled_default") is False,
        "enforcement_disabled": POLICY_FLAGS.get("enforcement_enabled") is False,
    }
    blockers = sorted(key for key, passed in checks.items() if not passed)
    return {
        "version": "v3_round351_captured_135_failure_evidence_consolidation",
        "round": 351,
        "success": not blockers,
        "status": "round351_captured_nodeid_evidence_consolidated" if not blockers else "blocked_round351_captured_evidence_incomplete",
        "failed_count": failed_count,
        "passed_count": passed_count,
        "nodeids": nodeids,
        "nodeid_count_in_payload": len(nodeids),
        "full_135_nodeid_list_available_in_repo": len(nodeids) == failed_count,
        "source": source.get("source", "pytest_text" if pytest_text else "operator_provided_evidence"),
        "checks": checks,
        "blockers": blockers,
        "vectors_created": False,
        "vectors_loaded": False,
        **POLICY_FLAGS,
    }


def select_round356_360_repair_cluster(taxonomy: Mapping[str, Any]) -> dict[str, Any]:
    """Select one narrow recent cluster for guarded repair."""

    counts = taxonomy.get("cluster_counts", {}) if isinstance(taxonomy, Mapping) else {}
    artifact_count = int(counts.get("round306_320_artifact_guard_regression", 0)) if isinstance(counts, Mapping) else 0
    selected = "round306_320_artifact_guard_regression" if artifact_count > 0 else "unclassified_pytest_failure"
    action = (
        "guarded_artifact_dependent_pytest_summary_taxonomy_delta_and_fail_closed_metadata"
        if selected == "round306_320_artifact_guard_regression"
        else "blocked_until_more_captured_nodeids_are_available"
    )
    return {
        "version": "v3_round356_360_selected_guarded_repair_cluster",
        "rounds": [356, 357, 358, 359, 360],
        "success": selected == "round306_320_artifact_guard_regression",
        "selected_cluster": selected,
        "selected_cluster_count": artifact_count if selected == "round306_320_artifact_guard_regression" else int(counts.get("unclassified_pytest_failure", 0)) if isinstance(counts, Mapping) else 0,
        "implementation_action": action,
        "repair_scope": [
            "classify Round306-320 artifact-guard failures in captured pytest summaries",
            "preserve fail-closed behavior when artifacts are missing or unsafe",
            "record bounded nodeid samples before stdout/stderr tail truncation",
        ],
        "production_persistence_go": False,
        "runtime_mapping_default_go": False,
        "enforcement_go": False,
        "agp_bypass_go": False,
        "vectors_created": False,
        **POLICY_FLAGS,
    }


def build_round361_365_validation_delta(
    round351: Mapping[str, Any],
    taxonomy: Mapping[str, Any],
    selected: Mapping[str, Any],
    *,
    focused_test_status: str = "pending_operator_run",
    artifact_free_status: str = "pending_operator_run",
    artifact_dependent_status: str = "pending_operator_run",
    broader_pytest_status: str = "pending_operator_run",
) -> dict[str, Any]:
    """Rounds361-365: focused validation delta and next recommendation."""

    return {
        "version": "v3_round361_365_validation_delta_next_recommendation",
        "rounds": [361, 362, 363, 364, 365],
        "success": round351.get("success") is True and selected.get("success") is True,
        "rounds_completed": list(range(351, 366)),
        "captured_failure_evidence": {
            "failed_count": round351.get("failed_count"),
            "passed_count": round351.get("passed_count"),
            "nodeid_count_in_payload": round351.get("nodeid_count_in_payload"),
            "full_135_nodeid_list_available_in_repo": round351.get("full_135_nodeid_list_available_in_repo"),
        },
        "failure_cluster_taxonomy": taxonomy,
        "selected_repair_cluster": selected,
        "focused_test_status": focused_test_status,
        "artifact_free_validation_status": artifact_free_status,
        "artifact_dependent_validation_status": artifact_dependent_status,
        "broader_pytest_status": broader_pytest_status,
        "remaining_taxonomy": taxonomy.get("cluster_counts", {}),
        "next_recommendation": (
            "Keep production persistence, runtime_mapping_enabled default, enforcement, and AGP bypass disabled. "
            "Re-run the guarded artifact-dependent command operator-locally; use the Round351-365 concrete taxonomy and bounded nodeid samples to determine whether Round40/41 wrapper-swap audit or Round94-98 runtime-mapping enable-smoke should be repaired next."
        ),
        "production_persistence_go": False,
        "runtime_mapping_default_go": False,
        "enforcement_go": False,
        **POLICY_FLAGS,
    }


def build_round351_365_report(
    *,
    operator_evidence: Mapping[str, Any] | None = None,
    pytest_text: str = "",
    focused_test_status: str = "pending_operator_run",
    artifact_free_status: str = "pending_operator_run",
    artifact_dependent_status: str = "pending_operator_run",
    broader_pytest_status: str = "pending_operator_run",
) -> dict[str, Any]:
    round351 = consolidate_round351_captured_evidence(operator_evidence=operator_evidence, pytest_text=pytest_text)
    taxonomy = classify_round351_355_failures(round351.get("nodeids", []), expected_failed_count=round351.get("failed_count"))
    selected = select_round356_360_repair_cluster(taxonomy)
    delta = build_round361_365_validation_delta(
        round351,
        taxonomy,
        selected,
        focused_test_status=focused_test_status,
        artifact_free_status=artifact_free_status,
        artifact_dependent_status=artifact_dependent_status,
        broader_pytest_status=broader_pytest_status,
    )
    return {
        "version": ROUND351_365_VERSION,
        "success": round351.get("success") is True and selected.get("success") is True and delta.get("success") is True,
        "rounds_completed": list(range(351, 366)),
        "round351_captured_evidence": round351,
        "round352_355_failure_taxonomy": taxonomy,
        "round356_360_selected_repair": selected,
        "round361_365_validation_delta": delta,
        "production_persistence_enabled": False,
        "runtime_mapping_enabled_default": False,
        "enforcement_enabled": False,
        "vectors_created": False,
        "vectors_loaded": False,
        **POLICY_FLAGS,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pytest-log", help="Optional pytest output/log path to parse instead of the embedded operator sample.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    pytest_text = Path(args.pytest_log).read_text(encoding="utf-8") if args.pytest_log else ""
    report = build_round351_365_report(pytest_text=pytest_text)
    print(_compact_json(report))
    return 0 if report.get("success") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
