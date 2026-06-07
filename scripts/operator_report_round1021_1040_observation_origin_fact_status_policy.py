#!/usr/bin/env python3
"""Compact operator report for Round1021-1040 observation origin and fact status policy."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adapters.observation_origin_fact_status_policy import (  # noqa: E402
    FEATURE_TRACK,
    NEXT_IMPLEMENTATION_RECOMMENDATION,
    SUPPORTED_FACT_STATUSES,
    SUPPORTED_ORIGINS,
    build_observation_origin_fact_status_policy_summary,
    build_origin_fact_status_record,
    validate_origin_fact_status_record,
)
from scripts.operator_rehearse_runtime_mapping_no_persistence import FORBIDDEN_GIT_PATTERNS  # noqa: E402

OPERATOR_COMMAND = "python scripts/operator_report_round1021_1040_observation_origin_fact_status_policy.py"
OPERATOR_REPORT_PATH = "docs/round1021_1040_observation_origin_fact_status_policy.md"
FOCUSED_TEST_COMMAND = "python -m pytest -q tests/test_v3_round1021_1040_observation_origin_fact_status_policy.py"


def _compact_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _git_status_short(repo_root: Path) -> tuple[bool, list[str]]:
    proc = subprocess.run(
        ["git", "status", "--short"],
        cwd=repo_root,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return proc.returncode == 0, [line for line in proc.stdout.splitlines() if line.strip()]


def _forbidden_entries(lines: Sequence[str]) -> list[str]:
    return [line for line in lines if any(pattern in line for pattern in FORBIDDEN_GIT_PATTERNS)]


def _proof(value: bool, **extra: Any) -> dict[str, Any]:
    payload = {"passed": bool(value)}
    payload.update(extra)
    return payload


def build_operator_report(*, repo_root: str | Path = REPO_ROOT) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    summary = build_observation_origin_fact_status_policy_summary()

    # Create various combination records for validation
    valid_external = build_origin_fact_status_record("external_visual", "observed_external")
    valid_virtual = build_origin_fact_status_record("virtual_world_visual", "observed_internal_virtual")
    valid_memory = build_origin_fact_status_record("memory_replay", "reconstructed_memory")

    invalid_imagination = build_origin_fact_status_record("imagination", "observed_external")
    invalid_virtual = build_origin_fact_status_record("virtual_world_visual", "observed_external")
    unknown_origin = build_origin_fact_status_record("unknown_origin", "observed_external")

    validation_rows = {
        "valid_external": validate_origin_fact_status_record(valid_external),
        "valid_virtual": validate_origin_fact_status_record(valid_virtual),
        "valid_memory": validate_origin_fact_status_record(valid_memory),
        "invalid_imagination": validate_origin_fact_status_record(invalid_imagination),
        "invalid_virtual": validate_origin_fact_status_record(invalid_virtual),
        "unknown_origin": validate_origin_fact_status_record(unknown_origin),
    }

    git_ok, git_lines = _git_status_short(root)
    forbidden = _forbidden_entries(git_lines)
    artifact_proof = {
        "git_status_ok": git_ok,
        "git_status_short_lines": git_lines,
        "forbidden_patterns": list(FORBIDDEN_GIT_PATTERNS),
        "forbidden_entries": forbidden,
        "no_forbidden_artifact_creation_or_staging": git_ok and not forbidden,
    }

    invariant_rows = (valid_external, valid_virtual, valid_memory, invalid_imagination, invalid_virtual, unknown_origin)

    proofs = {
        "external_fact_assertion_guard_proof": _proof(
            valid_external["external_fact_asserted"] is True
            and invalid_imagination["external_fact_asserted"] is False
            and not invalid_imagination["origin_fact_status_passed"]
        ),
        "no_memory_write_proof": _proof(all(row["memory_write_performed"] is False for row in invariant_rows)),
        "no_self_model_update_proof": _proof(all(row["self_model_update_allowed"] is False for row in invariant_rows)),
        "no_affect_hormone_transition_proof": _proof(all(row["affect_transition_allowed"] is False and row["hormone_transition_allowed"] is False for row in invariant_rows)),
        "no_persistence_runtime_vector_artifact_proof": _proof(
            all(row["persistence_write_performed"] is False and row["runtime_mutation_performed"] is False and row["vector_read_performed"] is False and row["vector_load_performed"] is False and row["artifact_created_or_staged"] is False for row in invariant_rows)
            and artifact_proof["no_forbidden_artifact_creation_or_staging"]
        ),
        "agp_fallback_non_bypass_proof": _proof(all(row["agp_bypass_allowed"] is False and row["fallback_bypass_allowed"] is False for row in invariant_rows)),
    }

    checks = {
        "valid_records_passed": all(row["validation_passed"] for row in (validation_rows["valid_external"], validation_rows["valid_virtual"], validation_rows["valid_memory"])),
        "invalid_records_failed": all(row["validation_passed"] is False for row in (validation_rows["invalid_imagination"], validation_rows["invalid_virtual"], validation_rows["unknown_origin"])),
        "all_proofs_passed": all(row["passed"] for row in proofs.values()),
        "artifact_safety_passed": artifact_proof["no_forbidden_artifact_creation_or_staging"],
        "exactly_one_next_implementation_recommendation": len(summary["next_implementation_recommendations"]) == 1,
    }

    success = all(checks.values())

    return {
        "version": "v3_round1021_1040_operator_observation_origin_fact_status_policy_report",
        "feature_track": FEATURE_TRACK,
        "success": success,
        "status": "round1021_1040_read_only_observation_origin_fact_status_policy_green" if success else "blocked_round1021_1040_observation_origin_fact_status_policy",
        "operator_command": OPERATOR_COMMAND,
        "operator_report_path": OPERATOR_REPORT_PATH,
        "policy_summary": summary,
        "schema_validation_results": validation_rows,
        "proofs": proofs,
        "no_artifact_creation_staging_proof": artifact_proof,
        "checks": checks,
        "exactly_one_next_implementation_recommendation": NEXT_IMPLEMENTATION_RECOMMENDATION,
    }


def main() -> None:
    print(_compact_json(build_operator_report()))


if __name__ == "__main__":
    main()
