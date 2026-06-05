#!/usr/bin/env python3
"""Operator-local runtime-mapping measurement after green EVE self-learning.

Rounds203-207 consume the operator-local Round198-202 green evidence without
making production persistence available. The command validates the real
medium30k artifact through the existing guard, builds the engine only under
explicit operator authorization, recreates the in-memory EVE-specific vector
smoke for a Korean-first target such as ``민석``, then runs the existing
lexical→concept runtime-mapping dry-run/precheck/controlled-smoke chain.

The path is local-only and guarded: default runtime remains no-load, production
persistence remains NO-GO, enforcement remains disabled, runtime mapping is
rolled back after the controlled smoke, and no seed/vector artifacts are
created, copied, downloaded, or written into tracked repository paths.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adapters.operator_artifact_verification import DEFAULT_OPERATOR_MEDIUM_30K_DIR
from scripts.operator_remeasure_eve_self_learning import (
    DEFAULT_CONTEXT_WORDS,
    DEFAULT_OBSERVATION_TEXTS,
    DEFAULT_TARGET_WORD,
    POLICY_FLAGS,
    _compact_json,
    _dedupe_preserve_order,
    _summarize_validation,
    run_measurement_on_engine,
)
from scripts.operator_validate_medium30k import run_validation

ROUND203_207_VERSION = "v3_round203_207_runtime_mapping_after_self_learning"
SELECTED_CLUSTER_ID = "concept_runtime_mapping_after_eve_self_learning_guarded_medium30k"
DEFAULT_NEGATIVE_TOKEN = "EVE"


def build_round203_operator_green_evidence_record() -> dict[str, Any]:
    """Record the operator-local green result supplied after PR #32 merged."""

    return {
        "version": "v3_round203_operator_local_green_evidence_record",
        "round": 203,
        "source": "operator_local_codespaces_remeasurement_after_pr32_merge",
        "source_command": (
            "python scripts/operator_remeasure_eve_self_learning.py "
            "--artifact-dir _operator_artifacts/subset_medium_30k "
            "--target-word 민석 "
            "--context-word 한국어 --context-word 감정 --context-word 기억 --context-word 대화"
        ),
        "exit_code": 0,
        "status": "operator_eve_self_learning_remeasurement_green",
        "success": True,
        "target_word": DEFAULT_TARGET_WORD,
        "selected_cluster_id": "eve_specific_vector_self_learning_cascade",
        "runtime_load_summary": {
            "attached_to_engine": True,
            "guard_called": True,
            "blockers": [],
        },
        "real_vector_load": "explicit_guarded_operator_authorized_medium30k_only",
        "operator_green_delta": {
            "commit_gate_pass": True,
            "created_vectors_contains_target_word": True,
            "wrapper_eve_specific_hit_delta_green": True,
            "production_persistence_required": False,
            "runtime_mapping_enabled_default": False,
        },
        "seed_vectors_mutated": False,
        "semantic_memory_or_quarantine_mutated": False,
        "marker_required": False,
        **POLICY_FLAGS,
    }


def build_round204_next_cluster_selection() -> dict[str, Any]:
    """Select the next narrow cluster that benefits from guarded medium30k."""

    return {
        "version": "v3_round204_next_narrow_failure_cluster_selection",
        "round": 204,
        "selected_cluster_id": SELECTED_CLUSTER_ID,
        "previous_green_evidence": build_round203_operator_green_evidence_record(),
        "rationale": [
            "The self-learning cascade is now operator-local green for 민석 with known Korean context words.",
            "Concept/runtime mapping fixtures historically depend on having the EVE-specific lexical vector committed before the mapping gate can be exercised.",
            "A local-only guarded chain can reuse the verified medium30k load and in-memory vector commit without production persistence or default runtime loading.",
        ],
        "measurement_goal": "prove the 민석 vector can flow into the existing concept commit and runtime-mapping dry-run/precheck/controlled-smoke chain while rollback restores runtime_mapping_enabled=False",
        "out_of_scope": [
            "production_persistence_enablement",
            "runtime_mapping_enabled_default_true",
            "enforcement_enablement",
            "AGP_bypass",
            "seed_or_subset_artifact_creation",
            "dummy_vectors_or_fake_checksums",
        ],
        "operator_local_required": True,
        "cloud_expected_blocker_without_artifacts": "operator_artifact_files_missing",
        "next_round": "round205_guarded_local_only_measurement_repair_path",
        **POLICY_FLAGS,
    }


def build_round205_operator_command_contract() -> dict[str, Any]:
    """Return the stable one-command operator-local workflow for Round205."""

    command = (
        "python scripts/operator_measure_runtime_mapping_after_self_learning.py "
        "--artifact-dir _operator_artifacts/subset_medium_30k "
        "--target-word 민석 "
        "--context-word 한국어 --context-word 감정 --context-word 기억 --context-word 대화 "
        "--negative-token EVE "
        "--output eve_v3_autonomous_handoff/validation/operator_local_round203_207_runtime_mapping_after_self_learning.json"
    )
    return {
        "version": "v3_round205_guarded_local_only_runtime_mapping_measurement_contract",
        "round": 205,
        "selected_cluster_id": SELECTED_CLUSTER_ID,
        "stable_operator_local_command": command,
        "command_sequence": [
            "git status --short -- _operator_artifacts seeds/subsets",
            command,
            "git status --short -- _operator_artifacts seeds/subsets",
        ],
        "manual_python_snippet_required": False,
        "requires_real_operator_medium30k_artifact": True,
        "default_runtime_load": False,
        "controlled_smoke_rolls_back_runtime_mapping": True,
        **POLICY_FLAGS,
    }


def _runtime_load_summary_from_engine(engine: Any) -> dict[str, Any]:
    report = getattr(engine, "medium30k_runtime_load_report", {})
    if not isinstance(report, dict):
        report = {}
    return {
        "status": report.get("status"),
        "attached_to_engine": report.get("attached_to_engine") is True,
        "guard_called": report.get("guard_called") is True,
        "blockers": list(report.get("blockers", []) or []),
    }


def run_mapping_pipeline_on_engine(
    engine: Any,
    *,
    target_word: str = DEFAULT_TARGET_WORD,
    negative_token: str = DEFAULT_NEGATIVE_TOKEN,
) -> dict[str, Any]:
    """Run the existing mapping chain after a successful in-memory vector commit."""

    from adapters.runtime_smoke_runner import (
        run_round89_explicit_concept_commit_smoke,
        run_round90_concept_commit_delta_replay_report,
        run_round91_concept_commit_replay_export_checkpoint,
        run_round92_runtime_mapping_gate_dry_run,
        run_round93_runtime_mapping_proposal_report,
        run_round94_runtime_mapping_enforcement_dry_run,
        run_round95_runtime_mapping_operator_acceptance_fixture,
        run_round96_runtime_mapping_enable_smoke_precheck,
        run_round97_controlled_runtime_mapping_enable_smoke,
    )

    target = str(target_word or "").strip() or DEFAULT_TARGET_WORD
    negative = str(negative_token or "").strip() or DEFAULT_NEGATIVE_TOKEN
    tokens = _dedupe_preserve_order([target, negative])
    lcm = getattr(engine, "lex_concept_mapping", None)
    if lcm is None:
        raise RuntimeError("lex_concept_mapping_missing")
    before_runtime = bool(getattr(lcm, "runtime_mapping_enabled", False))
    before_enforcement = bool(getattr(lcm, "enforcement_enabled", False))

    concept_commit = run_round89_explicit_concept_commit_smoke(engine, planning_tokens=tokens, accepted_tokens=[target])
    replay = run_round90_concept_commit_delta_replay_report(engine, source_commit_report=concept_commit)
    checkpoint = run_round91_concept_commit_replay_export_checkpoint(
        engine,
        source_commit_report=concept_commit,
        source_replay_report=replay,
    )
    gate = run_round92_runtime_mapping_gate_dry_run(engine, tokens=tokens, source_checkpoint=checkpoint)
    proposal = run_round93_runtime_mapping_proposal_report(engine, source_dry_run=gate)
    enforcement = run_round94_runtime_mapping_enforcement_dry_run(engine, tokens=tokens, source_proposal=proposal)
    fixture = run_round95_runtime_mapping_operator_acceptance_fixture(
        engine,
        source_enforcement=enforcement,
        accepted_tokens=[target],
    )
    precheck = run_round96_runtime_mapping_enable_smoke_precheck(engine, source_fixture=fixture)
    smoke = run_round97_controlled_runtime_mapping_enable_smoke(engine, source_precheck=precheck)

    after_runtime = bool(getattr(lcm, "runtime_mapping_enabled", False))
    after_enforcement = bool(getattr(lcm, "enforcement_enabled", False))
    return {
        "mapping_pipeline_version": ROUND203_207_VERSION,
        "selected_cluster_id": SELECTED_CLUSTER_ID,
        "target_word": target,
        "negative_token": negative,
        "tokens": tokens,
        "concept_commit": concept_commit,
        "replay": replay,
        "checkpoint": checkpoint,
        "gate": gate,
        "proposal": proposal,
        "enforcement_dry_run": enforcement,
        "operator_acceptance_fixture": fixture,
        "precheck": precheck,
        "controlled_smoke": smoke,
        "target_would_map": target in list(gate.get("would_map_tokens", []) or []),
        "target_precheck_ready": target in list(precheck.get("ready_tokens", []) or []),
        "target_mapped_in_controlled_smoke": target in list(smoke.get("mapped_tokens", []) or []),
        "negative_token_blocked": negative in list(gate.get("blocked_tokens", []) or []),
        "runtime_mapping_enabled_before": before_runtime,
        "runtime_mapping_enabled_after": after_runtime,
        "enforcement_enabled_before": before_enforcement,
        "enforcement_enabled_after": after_enforcement,
        "rollback_complete": bool((smoke.get("rollback") or {}).get("rollback_complete", False)),
        "runtime_mapping_persisted": False,
        "production_persistence_enabled": False,
        "enforcement_enabled": False,
        "agp_bypass_used": False,
    }


def run_guarded_runtime_mapping_measurement(
    *,
    artifact_dir: str | Path = DEFAULT_OPERATOR_MEDIUM_30K_DIR,
    target_word: str = DEFAULT_TARGET_WORD,
    observation_texts: Sequence[str] | None = None,
    context_words: Sequence[str] | None = None,
    negative_token: str = DEFAULT_NEGATIVE_TOKEN,
    repo_root: str | Path = REPO_ROOT,
    validation_func: Callable[..., dict[str, Any]] = run_validation,
    engine_builder: Callable[..., Any] | None = None,
    self_learning_measurement_func: Callable[..., dict[str, Any]] = run_measurement_on_engine,
    mapping_pipeline_func: Callable[..., dict[str, Any]] = run_mapping_pipeline_on_engine,
) -> dict[str, Any]:
    """Run Round203-207 guarded local-only measurement and return JSON data."""

    validation = validation_func(artifact_dir=artifact_dir, attempt_load=True, repo_root=repo_root)
    validation_summary = _summarize_validation(validation)
    blockers = list(validation_summary.get("blockers", []) or [])
    runtime_load_summary = {
        "status": "not_attempted_operator_validation_not_green",
        "attached_to_engine": False,
        "guard_called": False,
        "blockers": [],
    }
    self_learning_measurement: dict[str, Any] = {
        "measurement_executed": False,
        "reason": "operator_validation_not_green",
        "persistent_vector_artifacts_written": False,
        "in_memory_vector_mutation_attempted": False,
    }
    mapping_measurement: dict[str, Any] = {
        "measurement_executed": False,
        "reason": "self_learning_or_runtime_load_not_green",
        "runtime_mapping_persisted": False,
        "production_persistence_enabled": False,
        "enforcement_enabled": False,
    }

    if validation.get("success") is True:
        if engine_builder is None:
            from main import build_full_engine

            engine_builder = build_full_engine
        engine = engine_builder(
            operator_medium30k_validation=validation,
            operator_medium30k_load_authorized=True,
            operator_medium30k_artifact_dir=str(artifact_dir),
        )
        runtime_load_summary = _runtime_load_summary_from_engine(engine)
        blockers.extend(runtime_load_summary["blockers"])
        if runtime_load_summary["attached_to_engine"] is True:
            self_learning_measurement = self_learning_measurement_func(
                engine,
                target_word=target_word,
                observation_texts=observation_texts or DEFAULT_OBSERVATION_TEXTS,
                context_words=context_words or DEFAULT_CONTEXT_WORDS,
            )
            self_learning_measurement["measurement_executed"] = True
            self_learning_green = bool(
                self_learning_measurement.get("wrapper_primary_loaded") is True
                and self_learning_measurement.get("commit_gate_pass") is True
                and target_word in list(self_learning_measurement.get("created_vectors", []) or [])
                and int(self_learning_measurement.get("wrapper_eve_specific_hits_delta", 0) or 0) >= 1
            )
            if self_learning_green:
                mapping_measurement = mapping_pipeline_func(
                    engine,
                    target_word=target_word,
                    negative_token=negative_token,
                )
                mapping_measurement["measurement_executed"] = True
            else:
                blockers.append("self_learning_measurement_not_green")
        else:
            blockers.append("guarded_runtime_load_not_attached")

    mapping_green = bool(
        mapping_measurement.get("measurement_executed") is True
        and mapping_measurement.get("target_would_map") is True
        and mapping_measurement.get("target_precheck_ready") is True
        and mapping_measurement.get("target_mapped_in_controlled_smoke") is True
        and mapping_measurement.get("negative_token_blocked") is True
        and mapping_measurement.get("rollback_complete") is True
        and mapping_measurement.get("runtime_mapping_enabled_after") is False
        and mapping_measurement.get("enforcement_enabled_after") is False
        and mapping_measurement.get("runtime_mapping_persisted") is False
    )
    if not mapping_green and not blockers:
        blockers.append("runtime_mapping_measurement_not_green")

    return {
        "version": ROUND203_207_VERSION,
        "rounds_completed": [203, 204, 205, 206, 207],
        "selected_cluster_id": SELECTED_CLUSTER_ID,
        "status": "operator_runtime_mapping_after_self_learning_green" if mapping_green else "blocked_operator_runtime_mapping_after_self_learning",
        "success": mapping_green,
        "exit_code": 0 if mapping_green else 1,
        "operator_green_evidence": build_round203_operator_green_evidence_record(),
        "next_cluster_selection": build_round204_next_cluster_selection(),
        "operator_command_contract": build_round205_operator_command_contract(),
        "operator_validation_summary": validation_summary,
        "runtime_load_summary": runtime_load_summary,
        "self_learning_measurement": self_learning_measurement,
        "mapping_measurement": mapping_measurement,
        "focused_tests": {
            "round": 206,
            "command": "python -m pytest -q tests/test_v3_round203_207_runtime_mapping_after_self_learning.py",
            "status": "defined_marker_free_fake_guard_only",
        },
        "broader_validation_delta": {
            "round": 207,
            "cloud_expected_blocker_without_artifacts": "operator_artifact_files_missing",
            "production_persistence_remains_no_go": True,
            "next_recommendation": "operator_run_one_command_runtime_mapping_after_self_learning_then compare mapped/precheck/rollback deltas before any persistence discussion",
        },
        "blockers": sorted(set(str(item) for item in blockers)),
        **POLICY_FLAGS,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Measure guarded runtime mapping after EVE self-learning with operator-local medium30k.")
    parser.add_argument("--artifact-dir", default=str(DEFAULT_OPERATOR_MEDIUM_30K_DIR), help="Operator-local medium30k artifact directory.")
    parser.add_argument("--target-word", default=DEFAULT_TARGET_WORD, help="Korean-first EVE-specific target word to measure.")
    parser.add_argument("--observation-text", action="append", dest="observation_texts", help="Korean-first observation text; may be repeated.")
    parser.add_argument("--context-word", action="append", dest="context_words", help="Known medium30k context word candidate; may be repeated.")
    parser.add_argument("--negative-token", default=DEFAULT_NEGATIVE_TOKEN, help="Control token expected to remain blocked by the mapping gate.")
    parser.add_argument("--output", help="Optional JSON output path for operator-local handoff evidence.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run_guarded_runtime_mapping_measurement(
            artifact_dir=args.artifact_dir,
            target_word=args.target_word,
            observation_texts=args.observation_texts or DEFAULT_OBSERVATION_TEXTS,
            context_words=args.context_words or DEFAULT_CONTEXT_WORDS,
            negative_token=args.negative_token,
        )
    except Exception as exc:  # fail closed with machine-readable output
        report = {
            "version": ROUND203_207_VERSION,
            "rounds_completed": [203, 204, 205, 206, 207],
            "selected_cluster_id": SELECTED_CLUSTER_ID,
            "status": "blocked_operator_runtime_mapping_after_self_learning_exception",
            "success": False,
            "exit_code": 1,
            "blockers": ["operator_runtime_mapping_after_self_learning_exception"],
            "exception": f"{exc.__class__.__name__}:{exc}",
            **POLICY_FLAGS,
        }
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(_compact_json(report))
    return int(report.get("exit_code", 1))


if __name__ == "__main__":  # pragma: no cover - operator command entrypoint
    raise SystemExit(main())
