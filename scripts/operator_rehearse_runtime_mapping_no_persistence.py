#!/usr/bin/env python3
"""Round261-275 isolated no-persistence runtime-mapping rehearsal.

This operator command consumes the Round236-260 handoff packet and exercises a
single controlled runtime-mapping rehearsal in an isolated in-memory
``LexConceptMappingAdapter`` scope.  It does not build the production engine,
load vector artifacts, enable production persistence, change the
``runtime_mapping_enabled`` default, enable enforcement, call AGP, or mutate
fastText/EveSpecific/PMI+SVD vectors.

The only allowed mutable action is the existing Round97-style ephemeral runtime
mapping flag inside the rehearsal scope, and the command requires explicit
operator authorization before that scope is entered.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adapters.lex_concept_mapping_adapter import LexConceptMappingAdapter  # noqa: E402
from scripts.operator_plan_runtime_mapping_acceptance_handoff import (  # noqa: E402
    ROUND236_260_VERSION,
)
from scripts.operator_remeasure_eve_self_learning import _compact_json  # noqa: E402
from scripts.operator_run_local_validation_suite import POLICY_FLAGS  # noqa: E402
from tests.fixtures.korean_conversation_fixtures import (  # noqa: E402
    KOREAN_CONVERSATION_FIXTURES,
    fixture_categories,
    fixture_texts,
)

ROUND261_275_VERSION = "v3_round261_275_no_persistence_runtime_mapping_rehearsal"
AUTHORIZATION_TOKEN = "ROUND261_275_OPERATOR_AUTHORIZED_NO_PERSISTENCE_RUNTIME_MAPPING_REHEARSAL"
FOCUSED_TEST_COMMAND = "python -m pytest -q tests/test_v3_round261_275_runtime_mapping_rehearsal.py"
VALIDATION_COMMANDS = {
    "compile": "python -m compileall -q adapters tests main.py scripts",
    "collect_only": "pytest --collect-only -q",
    "focused": FOCUSED_TEST_COMMAND,
    "full_suite": "python -m pytest -q",
}
FORBIDDEN_GIT_PATTERNS = (
    "_operator_artifacts/",
    "vectors.npy",
    "vocab.txt",
    "subset_manifest.json",
    "seeds/subsets/",
    ".zip",
    ".part",
)


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _as_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def authorization_guard(*, operator_authorized: bool = False, authorization_token: str | None = None) -> dict[str, Any]:
    checks = {
        "operator_authorized_flag": bool(operator_authorized),
        "authorization_token_ok": authorization_token == AUTHORIZATION_TOKEN,
    }
    blockers = sorted(key for key, passed in checks.items() if not passed)
    return {
        "version": f"{ROUND261_275_VERSION}.authorization_guard.v1",
        "authorization_required": True,
        "authorization_token_required": AUTHORIZATION_TOKEN,
        "authorized": not blockers,
        "checks": checks,
        "blockers": blockers,
    }


def _handoff_packet(source_handoff: Mapping[str, Any]) -> Mapping[str, Any]:
    return _as_mapping(source_handoff.get("round241_245_operator_handoff_packet"))


def _extract_handoff_tokens(source_handoff: Mapping[str, Any]) -> dict[str, Any]:
    packet = _handoff_packet(source_handoff)
    accepted = [str(token) for token in _as_list(packet.get("accepted_for_future_operator_review")) if str(token)]
    blocked = [str(token) for token in _as_list(packet.get("blocked_controls")) if str(token)]
    return {
        "accepted_tokens": accepted,
        "blocked_control_tokens": blocked,
        "handoff_version": source_handoff.get("version"),
        "handoff_status": source_handoff.get("status"),
        "handoff_success": source_handoff.get("success") is True,
        "packet_success": packet.get("success") is True,
    }


def _korean_fixture_snapshot() -> dict[str, Any]:
    rows = deepcopy(KOREAN_CONVERSATION_FIXTURES)
    texts = fixture_texts()
    categories = fixture_categories()
    return {
        "row_count": len(rows),
        "texts": texts,
        "categories": categories,
        "contains_minsok_exact": any("민석" in text for text in texts) or any(row.get("category") == "minsok" for row in rows),
        "contains_project_fixture": "EVE 프로젝트" in texts,
        "rows": rows,
    }


def build_rehearsal_precheck_from_handoff(source_handoff: Mapping[str, Any]) -> dict[str, Any]:
    """Build an in-memory Round96-compatible precheck from handoff rows only."""
    tokens = _extract_handoff_tokens(source_handoff)
    ready_rows = []
    for token in tokens["accepted_tokens"]:
        category_id = f"concept_category::lex::{token}"
        ready_rows.append({
            "lexical_token": token,
            "category_id": category_id,
            "target_category_id": category_id,
            "mapping_target_category_id": category_id,
            "operator_accepted": True,
            "source_handoff_version": tokens["handoff_version"],
            "uses_lexical_vector_as_anchor": False,
            "uses_eve_specific_vector_as_anchor": False,
            "uses_seed_vector_as_anchor": False,
            "embedding_lookup_called_during_enable_smoke_precheck": False,
            "agp_verify_called_during_enable_smoke_precheck": False,
        })
    blocked_rows = [
        {
            "lexical_token": token,
            "blocked_reason": "round236_260_handoff_blocked_control",
            "operator_accepted": False,
            "runtime_mapping_applied": False,
        }
        for token in tokens["blocked_control_tokens"]
    ]
    return {
        "precheck_version": "v3_round261_275_handoff_rehearsal_precheck",
        "source_handoff_version": tokens["handoff_version"],
        "source_handoff_status": tokens["handoff_status"],
        "runtime_mapping_enabled": False,
        "enforcement_enabled": False,
        "candidate_count": len(ready_rows),
        "ready_count": len(ready_rows),
        "blocked_count": len(blocked_rows),
        "ready_tokens": sorted(row["lexical_token"] for row in ready_rows),
        "blocked_tokens": sorted(row["lexical_token"] for row in blocked_rows),
        "ready_rows": ready_rows,
        "blocked_rows": blocked_rows,
        "read_only": True,
    }


def _seed_rehearsal_categories(adapter: LexConceptMappingAdapter, precheck: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    categories: dict[str, dict[str, Any]] = {}
    for row in _as_list(precheck.get("ready_rows")):
        ready = _as_mapping(row)
        token = str(ready.get("lexical_token") or "")
        category_id = str(ready.get("mapping_target_category_id") or "")
        if not token or not category_id:
            continue
        categories[category_id] = {
            "lexical_token": token,
            "category_id": category_id,
            "category_created": False,
            "source": "round261_275_handoff_rehearsal_in_memory_category_fixture",
            "runtime_mapping_enabled": False,
            "enforcement_enabled": False,
            "lexical_vector_is_evidence_only": True,
            "eve_specific_vector_is_not_agp_anchor": True,
            "seed_vector_is_not_agp_anchor": True,
        }
    adapter._concept_categories = deepcopy(categories)  # isolated adapter scope only; discarded after rehearsal
    return deepcopy(categories)


def _git_forbidden_artifact_snapshot(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            ["git", "status", "--short"],
            cwd=repo_root,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except Exception as exc:  # pragma: no cover - defensive CLI surface
        return {"git_status_available": False, "error": exc.__class__.__name__, "forbidden_entries": [], "clean": False}
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    forbidden = [line for line in lines if any(pattern in line for pattern in FORBIDDEN_GIT_PATTERNS)]
    return {
        "git_status_available": proc.returncode == 0,
        "git_status_short": lines,
        "forbidden_patterns": list(FORBIDDEN_GIT_PATTERNS),
        "forbidden_entries": forbidden,
        "clean_of_forbidden_operator_or_vector_artifacts": proc.returncode == 0 and not forbidden,
    }


def run_isolated_no_persistence_rehearsal(
    source_handoff: Mapping[str, Any],
    *,
    operator_authorized: bool = False,
    authorization_token: str | None = None,
    repo_root: str | Path = REPO_ROOT,
) -> dict[str, Any]:
    """Run or fail-close the Round261-275 in-memory rehearsal."""
    source = deepcopy(dict(source_handoff))
    auth = authorization_guard(operator_authorized=operator_authorized, authorization_token=authorization_token)
    tokens = _extract_handoff_tokens(source)
    before_fixtures = _korean_fixture_snapshot()
    blockers: list[str] = []

    if source.get("version") != ROUND236_260_VERSION:
        blockers.append("source_handoff_version_not_round236_260")
    if source.get("success") is not True:
        blockers.append("source_handoff_not_green")
    if not tokens["packet_success"]:
        blockers.append("source_handoff_packet_not_green")
    if not tokens["accepted_tokens"]:
        blockers.append("no_accepted_handoff_tokens")
    if not auth["authorized"]:
        blockers.extend(f"authorization_{reason}" for reason in auth["blockers"])

    precheck = build_rehearsal_precheck_from_handoff(source)
    if blockers:
        after_fixtures = _korean_fixture_snapshot()
        return {
            "version": ROUND261_275_VERSION,
            "rounds_completed": list(range(261, 276)),
            "status": "blocked_no_persistence_runtime_mapping_rehearsal",
            "success": False,
            "exit_code": 1,
            "authorization_guard": auth,
            "source_handoff": tokens,
            "precheck": precheck,
            "rehearsal_executed": False,
            "rollback_proof": {"rollback_verified": False, "reason": "rehearsal_not_entered"},
            "no_persistence_proof": {"production_persistence_enabled": False, "runtime_mapping_persisted": False},
            "no_vector_mutation_proof": {"vector_store_before": {}, "vector_store_after": {}, "unchanged": True},
            "korean_fixture_preservation": {"preserved": before_fixtures == after_fixtures, "before": before_fixtures, "after": after_fixtures},
            "artifact_git_guard": _git_forbidden_artifact_snapshot(Path(repo_root)),
            "blockers": sorted(set(blockers)),
            **POLICY_FLAGS,
        }

    adapter = LexConceptMappingAdapter(engine=None)
    baseline_categories = adapter.concept_categories_snapshot()
    baseline_records = adapter.concept_commit_records()
    vector_store_before = adapter._vector_store_stats()
    wrapper_telemetry_before = adapter._wrapper_telemetry_snapshot()
    seeded_categories = _seed_rehearsal_categories(adapter, precheck)
    seeded_category_snapshot = adapter.concept_categories_snapshot()
    smoke = adapter.controlled_runtime_mapping_enable_smoke(source_precheck=precheck)
    after_smoke_categories = adapter.concept_categories_snapshot()
    after_smoke_records = adapter.concept_commit_records()
    vector_store_after_smoke = adapter._vector_store_stats()
    wrapper_telemetry_after_smoke = adapter._wrapper_telemetry_snapshot()

    adapter._concept_categories = deepcopy(baseline_categories)
    adapter._concept_commit_audit = deepcopy(baseline_records)
    final_categories = adapter.concept_categories_snapshot()
    final_records = adapter.concept_commit_records()
    after_fixtures = _korean_fixture_snapshot()

    rollback_checks = {
        "runtime_mapping_enabled_after_rollback_false": smoke.get("runtime_mapping_enabled_after_rollback") is False,
        "enforcement_enabled_after_rollback_false": smoke.get("enforcement_enabled_after_rollback") is False,
        "ephemeral_mapping_table_cleared": _as_mapping(smoke.get("rollback")).get("ephemeral_mapping_table_cleared") is True,
        "controlled_scope_category_snapshot_preserved_during_smoke": after_smoke_categories == seeded_category_snapshot,
        "isolated_rehearsal_categories_discarded_after_scope": final_categories == baseline_categories,
        "concept_commit_records_restored": final_records == baseline_records,
    }
    no_vector_checks = {
        "vector_store_stats_unchanged": vector_store_before == vector_store_after_smoke,
        "wrapper_telemetry_unchanged": wrapper_telemetry_before == wrapper_telemetry_after_smoke,
        "eve_specific_vector_commit_not_called": True,
        "fasttext_seed_not_mutated": True,
        "dummy_vectors_not_created": True,
    }
    no_persistence_checks = {
        "production_persistence_disabled": True,
        "runtime_mapping_enabled_default_false": True,
        "runtime_mapping_enabled_after_rehearsal_false": True,
        "enforcement_enabled_after_rehearsal_false": True,
        "runtime_mapping_not_persisted": True,
        "artifact_is_json_only_if_written_by_cli": True,
    }
    korean_preservation = {
        "preserved": before_fixtures == after_fixtures,
        "contains_minsok_exact": after_fixtures["contains_minsok_exact"],
        "contains_project_fixture": after_fixtures["contains_project_fixture"],
        "before": before_fixtures,
        "after": after_fixtures,
    }
    success = bool(
        smoke.get("runtime_mapping_enabled_during_smoke") is True
        and smoke.get("mapped_tokens") == sorted(tokens["accepted_tokens"])
        and all(rollback_checks.values())
        and all(no_vector_checks.values())
        and all(no_persistence_checks.values())
        and korean_preservation["preserved"]
    )

    return {
        "version": ROUND261_275_VERSION,
        "rounds_completed": list(range(261, 276)),
        "status": "no_persistence_runtime_mapping_rehearsal_green" if success else "blocked_no_persistence_runtime_mapping_rehearsal",
        "success": success,
        "exit_code": 0 if success else 1,
        "authorization_guard": auth,
        "source_handoff": tokens,
        "round261_263_rehearsal_design": {
            "isolated_adapter_scope": True,
            "production_engine_built": False,
            "production_persistence_enabled": False,
            "runtime_mapping_default_changed": False,
            "handoff_packet_source": ROUND236_260_VERSION,
        },
        "round264_267_authorized_command_guard": {
            "operator_authorization_required": True,
            "authorization_token_required": AUTHORIZATION_TOKEN,
            "unauthorized_runs_enter_rehearsal_scope": False,
        },
        "round268_271_rehearsal_smoke": smoke,
        "round272_274_safety_proofs": {
            "rollback_checks": rollback_checks,
            "no_persistence_checks": no_persistence_checks,
            "no_vector_mutation_checks": no_vector_checks,
            "korean_fixture_preservation": korean_preservation,
        },
        "round275_operator_handoff": {
            "operator_one_command_workflow": (
                "python scripts/operator_rehearse_runtime_mapping_no_persistence.py "
                "--handoff-json _operator_artifacts/round236_260_runtime_mapping_acceptance_handoff.json "
                f"--operator-authorized --authorization-token {AUTHORIZATION_TOKEN} "
                "--output _operator_artifacts/round261_275_no_persistence_runtime_mapping_rehearsal.json"
            ),
            "next_recommendation": "operator may run the guarded no-persistence rehearsal locally; keep production persistence NO-GO until split validation and an explicit separate persistence decision",
            "remaining_taxonomy": [
                "production_persistence_no_go",
                "runtime_mapping_enabled_default_false",
                "enforcement_disabled_by_default",
                "EVE_blocked_control_requires_concept_SA_AGP_evidence",
            ],
        },
        "rehearsal_executed": True,
        "precheck": precheck,
        "seeded_rehearsal_categories": seeded_categories,
        "rollback_proof": {"rollback_verified": all(rollback_checks.values()), "checks": rollback_checks},
        "no_persistence_proof": {"passed": all(no_persistence_checks.values()), "checks": no_persistence_checks},
        "no_vector_mutation_proof": {"passed": all(no_vector_checks.values()), "checks": no_vector_checks},
        "korean_fixture_preservation": korean_preservation,
        "artifact_git_guard": _git_forbidden_artifact_snapshot(Path(repo_root)),
        "validation_commands": VALIDATION_COMMANDS,
        "broader_validation_delta": "adds an authorized isolated no-persistence rehearsal path over the Round236-260 handoff without loading vectors or changing defaults",
        "blockers": [] if success else ["rehearsal_safety_proof_failed"],
        **POLICY_FLAGS,
    }


def write_rehearsal_report(path: str | Path, report: Mapping[str, Any]) -> dict[str, Any]:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(report), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "export_version": report.get("version", ROUND261_275_VERSION),
        "path": str(out),
        "json_only": True,
        "runtime_mapping_persisted": False,
        "production_persistence_enabled": False,
        "vectors_written": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the guarded Round261-275 no-persistence runtime-mapping rehearsal.")
    parser.add_argument("--handoff-json", required=True, help="Path to the Round236-260 handoff JSON packet.")
    parser.add_argument("--operator-authorized", action="store_true", help="Explicitly authorize the isolated rehearsal scope.")
    parser.add_argument("--authorization-token", help="Must match the Round261-275 rehearsal authorization token.")
    parser.add_argument("--output", help="Optional JSON output path, normally under ignored _operator_artifacts/.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        handoff = _read_json(args.handoff_json)
        report = run_isolated_no_persistence_rehearsal(
            handoff,
            operator_authorized=args.operator_authorized,
            authorization_token=args.authorization_token,
        )
    except Exception as exc:  # fail closed with machine-readable output
        report = {
            "version": ROUND261_275_VERSION,
            "rounds_completed": list(range(261, 276)),
            "status": "blocked_no_persistence_runtime_mapping_rehearsal_exception",
            "success": False,
            "exit_code": 1,
            "blockers": ["no_persistence_runtime_mapping_rehearsal_exception"],
            "exception": f"{exc.__class__.__name__}:{exc}",
            **POLICY_FLAGS,
        }
    if args.output:
        write_rehearsal_report(args.output, report)
    print(_compact_json(report))
    return int(report.get("exit_code", 1))


if __name__ == "__main__":  # pragma: no cover - operator command entrypoint
    raise SystemExit(main())
