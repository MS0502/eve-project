#!/usr/bin/env python3
"""Rounds336-350 artifact-dependent failure taxonomy and guarded repair selection.

This module is intentionally read-only over operator-local execution artifacts. It
may parse pytest text/log evidence and summarize failure clusters, but it must not
load vectors, create artifacts, enable persistence, enable runtime mapping by
default, or enable enforcement.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.operator_remeasure_eve_self_learning import _compact_json  # noqa: E402
from scripts.operator_run_local_validation_suite import POLICY_FLAGS  # noqa: E402

ROUND336_350_VERSION = "v3_round336_350_artifact_dependent_failure_taxonomy"
PYTEST_NODEID_RE = re.compile(r"^FAILED\s+(?P<nodeid>tests/[^\s]+?::[^\s]+)(?:\s+-\s+.*)?$")
SHORT_COUNT_RE = re.compile(r"(?P<failed>\d+)\s+failed(?:,\s+(?P<passed>\d+)\s+passed)?")


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def extract_pytest_failure_nodeids(pytest_text: str) -> list[str]:
    """Extract failed pytest node ids from full or short-summary pytest text."""

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
    """Return pytest failed/passed counts when a summary line is present."""

    failed: int | None = None
    passed: int | None = None
    for line in pytest_text.splitlines():
        match = SHORT_COUNT_RE.search(line)
        if not match:
            continue
        failed = int(match.group("failed"))
        passed_group = match.group("passed")
        passed = int(passed_group) if passed_group is not None else passed
    return {"failed": failed, "passed": passed}


def _cluster_for_nodeid(nodeid: str) -> str:
    lowered = nodeid.lower()
    if "round29" in lowered or "subset" in lowered or "seed" in lowered or "fasttext" in lowered or "medium30k" in lowered:
        return "seed_subset_or_fasttext_artifact_boundary"
    if "runtime_mapping" in lowered or "concept_mapping" in lowered:
        return "runtime_mapping_or_concept_mapping_readiness"
    if "eve_self_learning" in lowered or "eve_specific" in lowered or "vector_store" in lowered or "embedding_wrapper" in lowered:
        return "eve_specific_vector_learning_boundary"
    if "smoke" in lowered or "drift" in lowered or "telemetry" in lowered or "dashboard" in lowered:
        return "telemetry_smoke_or_drift_measurement"
    if "agp" in lowered or "category" in lowered or "meaning_bridge" in lowered:
        return "agp_category_bridge_boundary"
    if "state_debug" in lowered or "export" in lowered or "replay" in lowered:
        return "debug_export_or_replay_surface"
    return "unclassified_pytest_failure"


def classify_pytest_failures(nodeids: Iterable[str]) -> dict[str, Any]:
    """Classify pytest failures into deterministic high-level clusters."""

    sorted_nodeids = sorted(dict.fromkeys(nodeids))
    by_cluster: dict[str, list[str]] = {}
    for nodeid in sorted_nodeids:
        by_cluster.setdefault(_cluster_for_nodeid(nodeid), []).append(nodeid)
    counts = {cluster: len(items) for cluster, items in sorted(by_cluster.items())}
    return {
        "version": "v3_round337_340_pytest_failure_cluster_taxonomy",
        "rounds": [337, 338, 339, 340],
        "total_classified_failures": len(sorted_nodeids),
        "cluster_counts": counts,
        "clusters": {cluster: {"count": len(items), "sample_nodeids": items[:5]} for cluster, items in sorted(by_cluster.items())},
        "nodeids_available": bool(sorted_nodeids),
    }


def consolidate_round336_execution_evidence(
    execution_report: Mapping[str, Any] | None = None,
    *,
    operator_failed_count: int | None = None,
    operator_passed_count: int | None = None,
) -> dict[str, Any]:
    """Round336: consolidate readiness-green and executed-red evidence."""

    readiness = execution_report.get("artifact_dependent_readiness", {}) if execution_report else {}
    ready = readiness.get("ready") is True if isinstance(readiness, Mapping) else False
    missing = readiness.get("missing_artifact_ids", []) if isinstance(readiness, Mapping) else []
    unsafe = readiness.get("unsafe_artifact_ids", []) if isinstance(readiness, Mapping) else []
    vector_contents_read = readiness.get("vector_contents_read") is True if isinstance(readiness, Mapping) else False
    steps = execution_report.get("steps", []) if execution_report else []
    pytest_step = next((step for step in steps if isinstance(step, Mapping) and step.get("command") == "python -m pytest -q"), None)
    parsed_counts = None
    if isinstance(pytest_step, Mapping):
        summary = pytest_step.get("pytest_failure_summary", {})
        if isinstance(summary, Mapping):
            parsed_counts = {"failed": summary.get("failed_count"), "passed": summary.get("passed_count")}
    failed_count = operator_failed_count if operator_failed_count is not None else (parsed_counts or {}).get("failed")
    passed_count = operator_passed_count if operator_passed_count is not None else (parsed_counts or {}).get("passed")
    executed = execution_report.get("execute_requested") is True if execution_report else operator_failed_count is not None
    checks = {
        "artifact_dependent_readiness_green": ready,
        "missing_artifacts_empty": missing == [],
        "unsafe_artifacts_empty": unsafe == [],
        "vector_contents_not_read_by_readiness": vector_contents_read is False,
        "artifact_dependent_execution_observed": executed,
        "executed_failure_count_recorded": isinstance(failed_count, int),
        "production_persistence_no_go": POLICY_FLAGS.get("production_persistence_enabled") is False,
        "runtime_mapping_default_false": POLICY_FLAGS.get("runtime_mapping_enabled_default") is False,
        "enforcement_disabled": POLICY_FLAGS.get("enforcement_enabled") is False,
    }
    blockers = sorted(key for key, passed in checks.items() if not passed)
    return {
        "version": "v3_round336_artifact_dependent_execution_evidence_consolidation",
        "round": 336,
        "success": not blockers,
        "status": "round336_artifact_dependent_executed_failure_evidence_consolidated" if not blockers else "blocked_round336_artifact_dependent_execution_evidence_incomplete",
        "operator_failed_count": failed_count,
        "operator_passed_count": passed_count,
        "readiness_green_evidence": {
            "ready": ready,
            "missing": list(missing) if isinstance(missing, list) else missing,
            "unsafe": list(unsafe) if isinstance(unsafe, list) else unsafe,
            "vector_contents_read": vector_contents_read,
        },
        "checks": checks,
        "blockers": blockers,
        **POLICY_FLAGS,
    }


def select_round341_345_repair_cluster(taxonomy: Mapping[str, Any]) -> dict[str, Any]:
    """Select a narrow non-persistence cluster for repair or measurement improvement."""

    counts = taxonomy.get("cluster_counts", {}) if isinstance(taxonomy, Mapping) else {}
    nodeids_available = taxonomy.get("nodeids_available") is True if isinstance(taxonomy, Mapping) else False
    if nodeids_available and isinstance(counts, Mapping) and counts:
        eligible = [(cluster, int(count)) for cluster, count in counts.items() if cluster != "runtime_mapping_or_concept_mapping_readiness"]
        selected, count = sorted(eligible or [(str(k), int(v)) for k, v in counts.items()], key=lambda item: (-item[1], item[0]))[0]
        action = "classify_and_target_cluster_with_focused_tests"
    else:
        selected = "pytest_failure_taxonomy_capture_gap"
        count = 0
        action = "measurement_improvement_capture_full_pytest_failure_summary_before_tail_truncation"
    return {
        "version": "v3_round341_345_selected_non_persistence_repair_cluster",
        "rounds": [341, 342, 343, 344, 345],
        "success": True,
        "selected_cluster": selected,
        "selected_cluster_count": count,
        "implementation_action": action,
        "production_persistence_go": False,
        "runtime_mapping_default_go": False,
        "enforcement_go": False,
        "agp_bypass_go": False,
        "vectors_created": False,
        **POLICY_FLAGS,
    }


def build_round346_350_validation_delta(
    round336: Mapping[str, Any],
    taxonomy: Mapping[str, Any],
    selected: Mapping[str, Any],
    *,
    focused_test_status: str = "pending_operator_run",
    artifact_free_status: str = "pending_operator_run",
    artifact_dependent_status: str = "pending_operator_run",
    broader_pytest_status: str = "pending_operator_run",
) -> dict[str, Any]:
    """Rounds346-350: validation delta and next recommendation."""

    failure_count = round336.get("operator_failed_count")
    remaining = dict(taxonomy.get("cluster_counts", {})) if isinstance(taxonomy.get("cluster_counts", {}), Mapping) else {}
    if not taxonomy.get("nodeids_available") and isinstance(failure_count, int):
        remaining = {"unparsed_operator_report_failures_needing_full_pytest_summary": failure_count}
    return {
        "version": "v3_round346_350_validation_delta_next_recommendation",
        "rounds": [346, 347, 348, 349, 350],
        "success": round336.get("success") is True and selected.get("success") is True,
        "rounds_completed": list(range(336, 351)),
        "artifact_dependent_readiness_green_evidence": round336.get("readiness_green_evidence"),
        "operator_executed_failure_count": failure_count,
        "operator_executed_passed_count": round336.get("operator_passed_count"),
        "failure_taxonomy": taxonomy,
        "selected_repair_cluster": selected,
        "focused_test_status": focused_test_status,
        "artifact_free_validation_status": artifact_free_status,
        "artifact_dependent_validation_status": artifact_dependent_status,
        "broader_pytest_status": broader_pytest_status,
        "remaining_taxonomy": remaining,
        "next_recommendation": (
            "Keep persistence, runtime mapping defaults, enforcement, and AGP bypass disabled. If prior artifact-dependent JSON lacks full failure nodeids, rerun the guarded artifact-dependent command with the Round336-350 failure-summary capture and use the largest non-persistence cluster for the next narrow repair."
        ),
        "production_persistence_go": False,
        "runtime_mapping_default_go": False,
        "enforcement_go": False,
        **POLICY_FLAGS,
    }


def build_round336_350_report(
    *,
    execution_report: Mapping[str, Any] | None = None,
    pytest_text: str = "",
    operator_failed_count: int | None = None,
    operator_passed_count: int | None = None,
) -> dict[str, Any]:
    nodeids = extract_pytest_failure_nodeids(pytest_text)
    counts = extract_pytest_summary_counts(pytest_text)
    failed_count = operator_failed_count if operator_failed_count is not None else counts.get("failed")
    passed_count = operator_passed_count if operator_passed_count is not None else counts.get("passed")
    round336 = consolidate_round336_execution_evidence(
        execution_report,
        operator_failed_count=failed_count,
        operator_passed_count=passed_count,
    )
    taxonomy = classify_pytest_failures(nodeids)
    selected = select_round341_345_repair_cluster(taxonomy)
    delta = build_round346_350_validation_delta(round336, taxonomy, selected)
    return {
        "version": ROUND336_350_VERSION,
        "status": "round336_350_artifact_dependent_failure_taxonomy_ready" if delta.get("success") else "blocked_round336_350_artifact_dependent_failure_taxonomy",
        "success": delta.get("success") is True,
        "exit_code": 0 if delta.get("success") else 1,
        "round336_execution_evidence": round336,
        "round337_340_failure_taxonomy": taxonomy,
        "round341_345_selected_repair_cluster": selected,
        "round346_350_validation_delta": delta,
        **POLICY_FLAGS,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze Round336-350 artifact-dependent pytest failures.")
    parser.add_argument("--execution-json", help="Optional ignored operator-local artifact-dependent execution JSON.")
    parser.add_argument("--pytest-output", help="Optional pytest output/log text path used for nodeid taxonomy.")
    parser.add_argument("--operator-failed-count", type=int, help="Operator-reported executed failure count when nodeids are unavailable.")
    parser.add_argument("--operator-passed-count", type=int, help="Operator-reported executed passed count when nodeids are unavailable.")
    parser.add_argument("--output", help="Optional JSON output path, normally under ignored _operator_artifacts/.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    execution = _read_json(args.execution_json) if args.execution_json else None
    pytest_text = Path(args.pytest_output).read_text(encoding="utf-8") if args.pytest_output else ""
    report = build_round336_350_report(
        execution_report=execution,
        pytest_text=pytest_text,
        operator_failed_count=args.operator_failed_count,
        operator_passed_count=args.operator_passed_count,
    )
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(_compact_json(report))
    return int(report.get("exit_code", 1))


if __name__ == "__main__":  # pragma: no cover - operator command entrypoint
    raise SystemExit(main())
