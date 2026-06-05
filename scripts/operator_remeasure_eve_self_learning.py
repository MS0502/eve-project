#!/usr/bin/env python3
"""Operator-local EVE-specific self-learning remeasurement for rounds198-202.

The command is intentionally artifact-neutral: it validates the operator-local
medium30k artifact through the existing guarded workflow, builds the engine only
when that validation is green and explicitly authorized, then runs one narrow
in-memory EVE-specific vector/self-learning smoke. It does not enable production
persistence, runtime mapping defaults, enforcement, AGP bypass, artifact
creation/copy/download, or repository seed/vector writes.
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
from scripts.operator_validate_medium30k import run_validation

ROUND198_202_REMEASURE_VERSION = "v3_round198_202_eve_self_learning_remeasurement"
SELECTED_CLUSTER_ID = "eve_specific_vector_self_learning_cascade"
DEFAULT_TARGET_WORD = "민석"
DEFAULT_OBSERVATION_TEXTS = [
    "민석 감정 기억",
    "민석 한국어 대화",
]
DEFAULT_CONTEXT_WORDS = ["한국어", "감정", "기억", "대화"]

POLICY_FLAGS: dict[str, bool] = {
    "production_persistence_enabled": False,
    "runtime_mapping_enabled_default": False,
    "enforcement_enabled": False,
    "agp_bypass_used": False,
    "artifacts_written": False,
    "artifacts_included": False,
    "artifact_download_attempted": False,
    "seed_vectors_mutated": False,
    "semantic_memory_or_quarantine_mutated": False,
    "dummy_vectors_created": False,
    "production_persistence_no_go_respected": True,
}


class RemeasurementError(RuntimeError):
    """Raised when the guarded operator-local path cannot continue."""


def _compact_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _dedupe_preserve_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for raw in values:
        item = str(raw or "").strip()
        if not item or item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def build_round198_operator_remeasurement_command_set() -> dict[str, Any]:
    """Return the exact operator-local command set for Round198."""

    stable_command = (
        "python scripts/operator_remeasure_eve_self_learning.py "
        "--artifact-dir _operator_artifacts/subset_medium_30k "
        "--target-word 민석 "
        "--context-word 한국어 --context-word 감정 --context-word 기억 --context-word 대화 "
        "--output eve_v3_autonomous_handoff/validation/operator_local_round198_202_eve_self_learning_remeasurement.json"
    )
    return {
        "version": "v3_round198_operator_remeasurement_command_set",
        "round": 198,
        "selected_cluster_id": SELECTED_CLUSTER_ID,
        "operator_local_prerequisite_command": "python scripts/operator_validate_medium30k.py --attempt-load",
        "stable_remeasurement_command": stable_command,
        "command_sequence": [
            "git status --short -- _operator_artifacts seeds/subsets",
            "python scripts/operator_validate_medium30k.py --attempt-load",
            stable_command,
            "git status --short -- _operator_artifacts seeds/subsets",
        ],
        "requires_real_operator_medium30k_artifact": True,
        "manual_python_snippet_required": False,
        "read_only_default_runtime": True,
        **POLICY_FLAGS,
    }


def build_round199_local_smoke_contract() -> dict[str, Any]:
    """Return the Round199 stable command contract."""

    return {
        "version": "v3_round199_eve_self_learning_local_smoke_contract",
        "round": 199,
        "selected_cluster_id": SELECTED_CLUSTER_ID,
        "script": "scripts/operator_remeasure_eve_self_learning.py",
        "pytest_marker_required": False,
        "production_persistence_required": False,
        "real_vector_load": "explicit_guarded_operator_authorized_medium30k_only",
        "default_runtime_load": False,
        "measurement_steps": [
            "run operator_validate_medium30k with --attempt-load",
            "build_full_engine with operator_medium30k_validation and operator_medium30k_load_authorized=True",
            "observe Korean-first EVE-specific target text twice with distinct contexts",
            "audit commit gate for the selected target word",
            "attempt explicit in-memory EveSpecificVectorStore vector creation from known medium30k context words",
            "query wrapper get_vector for the target word and report telemetry deltas",
        ],
        "target_word": DEFAULT_TARGET_WORD,
        "observation_texts": list(DEFAULT_OBSERVATION_TEXTS),
        "context_words": list(DEFAULT_CONTEXT_WORDS),
        **POLICY_FLAGS,
    }


def build_round201_expected_delta_report_schema() -> dict[str, Any]:
    """Return the expected operator-local delta schema for Round201."""

    return {
        "version": "v3_round201_expected_operator_local_delta_report_schema",
        "round": 201,
        "selected_cluster_id": SELECTED_CLUSTER_ID,
        "required_top_level_keys": [
            "version",
            "rounds_completed",
            "selected_cluster_id",
            "status",
            "success",
            "exit_code",
            "operator_validation_summary",
            "runtime_load_summary",
            "measurement",
            "expected_delta_report_schema",
            "broader_validation_delta",
            "blockers",
        ],
        "measurement_required_keys": [
            "target_word",
            "observation_texts",
            "context_words",
            "known_context_words",
            "observed_count",
            "is_eve_specific_candidate",
            "commit_gate_pass",
            "created_vectors",
            "rejected_words",
            "vector_store_count_before",
            "vector_store_count_after",
            "wrapper_eve_specific_hits_before",
            "wrapper_eve_specific_hits_after",
            "wrapper_primary_loaded",
            "in_memory_vector_mutation_attempted",
            "persistent_vector_artifacts_written",
        ],
        "expected_operator_local_green_delta": {
            "operator_validation_success": True,
            "runtime_load_attached_to_engine": True,
            "wrapper_primary_loaded": True,
            "target_word": DEFAULT_TARGET_WORD,
            "observed_count_minimum": 2,
            "commit_gate_pass": True,
            "created_vectors_contains_target_word": True,
            "vector_store_count_delta_minimum": 1,
            "wrapper_eve_specific_hits_delta_minimum": 1,
            "persistent_vector_artifacts_written": False,
            "production_persistence_enabled": False,
            "runtime_mapping_enabled_default": False,
            "enforcement_enabled": False,
            "agp_bypass_used": False,
        },
        "blocked_cloud_delta": {
            "operator_validation_success": False,
            "expected_blocker": "operator_artifact_files_missing",
            "runtime_load_attached_to_engine": False,
            "measurement_executed": False,
        },
        **POLICY_FLAGS,
    }


def build_round202_broader_validation_delta(
    *,
    focused_command_status: str = "pending_operator_local_real_artifact_run",
    focused_tests_status: str = "pending",
    compileall_status: str = "pending",
    collect_only_status: str = "pending",
    broader_pytest_status: str = "not_run",
    broader_failure_count: int | None = None,
) -> dict[str, Any]:
    """Return the Round202 broader validation delta and recommendation."""

    return {
        "version": "v3_round202_broader_validation_delta_next_recommendation",
        "round": 202,
        "selected_cluster_id": SELECTED_CLUSTER_ID,
        "focused_command_status": focused_command_status,
        "focused_tests_status": focused_tests_status,
        "compileall_status": compileall_status,
        "collect_only_status": collect_only_status,
        "broader_pytest_status": broader_pytest_status,
        "broader_failure_count": broader_failure_count,
        "baseline_cloud_failure_taxonomy": {
            "seed_vector_artifact_cascade": "major_remaining_cloud_baseline_without_operator_artifacts",
            "eve_specific_vector_self_learning_cascade": "selected_for_operator_local_remeasurement",
            "concept_runtime_mapping_cascade": "defer_until_vector_self_learning_delta_is_known",
        },
        "next_recommendation": "operator_run_stable_remeasurement_command_then_compare_created_vector_and_wrapper_hit_deltas_before_concept_runtime_mapping_repairs",
        "production_persistence_remains_no_go": True,
        **POLICY_FLAGS,
    }


def _summarize_validation(validation: dict[str, Any]) -> dict[str, Any]:
    explicit_load = validation.get("explicit_load", {}) if isinstance(validation.get("explicit_load"), dict) else {}
    return {
        "status": validation.get("status"),
        "success": validation.get("success") is True,
        "exit_code": validation.get("exit_code"),
        "attempt_load_requested": validation.get("attempt_load_requested") is True,
        "explicit_vectors_loaded": explicit_load.get("vectors_loaded") is True,
        "blockers": list(validation.get("blockers", []) or []),
    }


def _engine_surface(engine: Any) -> dict[str, Any]:
    wrapper = getattr(engine, "self_embedding", None)
    fasttext = getattr(engine, "fasttext_embedding", None)
    store = getattr(engine, "eve_specific_vector_store", None)
    tracker = getattr(engine, "eve_vocab_tracker", None)
    self_learning = getattr(engine, "eve_self_learning", None)
    try:
        primary_loaded = bool(fasttext.is_loaded()) if hasattr(fasttext, "is_loaded") else False
    except Exception:
        primary_loaded = False
    return {
        "self_embedding_class": wrapper.__class__.__name__ if wrapper is not None else None,
        "self_embedding_backup_present": getattr(engine, "self_embedding_backup", None) is not None,
        "fasttext_class": fasttext.__class__.__name__ if fasttext is not None else None,
        "fasttext_loaded": primary_loaded,
        "eve_vocab_tracker_class": tracker.__class__.__name__ if tracker is not None else None,
        "eve_specific_vector_store_class": store.__class__.__name__ if store is not None else None,
        "eve_self_learning_class": self_learning.__class__.__name__ if self_learning is not None else None,
    }


def run_measurement_on_engine(
    engine: Any,
    *,
    target_word: str = DEFAULT_TARGET_WORD,
    observation_texts: Sequence[str] | None = None,
    context_words: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Run the focused in-memory self-learning smoke on an already loaded engine."""

    target = str(target_word or "").strip() or DEFAULT_TARGET_WORD
    texts = _dedupe_preserve_order(observation_texts or DEFAULT_OBSERVATION_TEXTS)
    contexts = _dedupe_preserve_order(context_words or DEFAULT_CONTEXT_WORDS)
    learner = getattr(engine, "eve_self_learning", None)
    store = getattr(engine, "eve_specific_vector_store", None)
    wrapper = getattr(engine, "self_embedding", None)
    if learner is None or store is None or wrapper is None:
        raise RemeasurementError("eve_self_learning_surface_missing")

    before_store_stats = store.stats() if hasattr(store, "stats") else {}
    before_wrapper_telemetry = wrapper.telemetry() if hasattr(wrapper, "telemetry") else {}
    observations: list[dict[str, Any]] = []
    for index, text in enumerate(texts, start=1):
        observations.append(
            learner.observe_text(
                text,
                source=f"round199_operator_local_smoke_{index}",
                context={"round": 199, "selected_cluster_id": SELECTED_CLUSTER_ID},
            )
        )
    quality = learner.observation_evidence_quality_summary(words=[target])
    audit = learner.audit_commit_gate(words=[target], context_words=contexts)
    commit = learner.commit_eve_specific_vectors(words=[target], context_words=contexts)
    after_store_stats = store.stats() if hasattr(store, "stats") else {}
    vector_lookup_result = wrapper.get_vector(target) if hasattr(wrapper, "get_vector") else None
    after_wrapper_telemetry = wrapper.telemetry() if hasattr(wrapper, "telemetry") else {}
    candidate = (quality.get("candidate_summaries") or [{}])[0] if quality.get("candidate_summaries") else {}
    before_count = int(before_store_stats.get("stored_count", 0) or 0)
    after_count = int(after_store_stats.get("stored_count", 0) or 0)
    before_hits = int(before_wrapper_telemetry.get("eve_specific_hits", 0) or 0)
    after_hits = int(after_wrapper_telemetry.get("eve_specific_hits", 0) or 0)
    return {
        "measurement_version": ROUND198_202_REMEASURE_VERSION,
        "selected_cluster_id": SELECTED_CLUSTER_ID,
        "target_word": target,
        "observation_texts": list(texts),
        "context_words": list(contexts),
        "observations": observations,
        "quality_summary": quality,
        "audit_report": audit,
        "commit_report": commit,
        "known_context_words": list(audit.get("known_context_words", []) or []),
        "observed_count": int(candidate.get("observed_count", 0) or 0),
        "is_eve_specific_candidate": bool(candidate.get("is_eve_specific_candidate", False)),
        "commit_gate_pass": bool(audit.get("gate_pass", False)),
        "created_vectors": list(commit.get("created", []) or []),
        "rejected_words": list(commit.get("rejected", []) or []),
        "vector_lookup_after_commit": vector_lookup_result is not None,
        "vector_store_count_before": before_count,
        "vector_store_count_after": after_count,
        "vector_store_count_delta": after_count - before_count,
        "wrapper_eve_specific_hits_before": before_hits,
        "wrapper_eve_specific_hits_after": after_hits,
        "wrapper_eve_specific_hits_delta": after_hits - before_hits,
        "wrapper_primary_loaded": bool(after_wrapper_telemetry.get("primary_loaded", False)),
        "wrapper_telemetry_after": after_wrapper_telemetry,
        "engine_surface": _engine_surface(engine),
        "in_memory_vector_mutation_attempted": True,
        "persistent_vector_artifacts_written": False,
        **POLICY_FLAGS,
    }


def run_remeasurement(
    *,
    artifact_dir: str | Path = DEFAULT_OPERATOR_MEDIUM_30K_DIR,
    target_word: str = DEFAULT_TARGET_WORD,
    observation_texts: Sequence[str] | None = None,
    context_words: Sequence[str] | None = None,
    repo_root: str | Path = REPO_ROOT,
    validation_func: Callable[..., dict[str, Any]] = run_validation,
    engine_builder: Callable[..., Any] | None = None,
    measurement_func: Callable[..., dict[str, Any]] = run_measurement_on_engine,
) -> dict[str, Any]:
    """Run the guarded operator-local remeasurement and return JSON data."""

    validation = validation_func(artifact_dir=artifact_dir, attempt_load=True, repo_root=repo_root)
    validation_summary = _summarize_validation(validation)
    blockers = list(validation_summary["blockers"])
    runtime_load_summary: dict[str, Any] = {
        "status": "not_attempted_operator_validation_not_green",
        "attached_to_engine": False,
        "guard_called": False,
        "blockers": [],
    }
    measurement: dict[str, Any] = {
        "measurement_executed": False,
        "reason": "operator_validation_not_green",
        "persistent_vector_artifacts_written": False,
        "in_memory_vector_mutation_attempted": False,
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
        runtime_report = getattr(engine, "medium30k_runtime_load_report", {})
        runtime_load_summary = {
            "status": runtime_report.get("status"),
            "attached_to_engine": runtime_report.get("attached_to_engine") is True,
            "guard_called": runtime_report.get("guard_called") is True,
            "blockers": list(runtime_report.get("blockers", []) or []),
        }
        blockers.extend(runtime_load_summary["blockers"])
        if runtime_load_summary["attached_to_engine"] is True:
            measurement = measurement_func(
                engine,
                target_word=target_word,
                observation_texts=observation_texts or DEFAULT_OBSERVATION_TEXTS,
                context_words=context_words or DEFAULT_CONTEXT_WORDS,
            )
            measurement["measurement_executed"] = True
        else:
            measurement = {
                "measurement_executed": False,
                "reason": "guarded_runtime_load_not_attached",
                "persistent_vector_artifacts_written": False,
                "in_memory_vector_mutation_attempted": False,
            }
            blockers.append("guarded_runtime_load_not_attached")

    success = bool(
        validation.get("success") is True
        and runtime_load_summary.get("attached_to_engine") is True
        and measurement.get("measurement_executed") is True
        and measurement.get("wrapper_primary_loaded") is True
        and measurement.get("commit_gate_pass") is True
        and target_word in list(measurement.get("created_vectors", []) or [])
        and int(measurement.get("wrapper_eve_specific_hits_delta", 0) or 0) >= 1
    )
    if not success and not blockers:
        blockers.append("measurement_delta_not_green")

    return {
        "version": ROUND198_202_REMEASURE_VERSION,
        "rounds_completed": [198, 199, 200, 201, 202],
        "selected_cluster_id": SELECTED_CLUSTER_ID,
        "status": "operator_eve_self_learning_remeasurement_green" if success else "blocked_operator_eve_self_learning_remeasurement",
        "success": success,
        "exit_code": 0 if success else 1,
        "operator_validation_summary": validation_summary,
        "runtime_load_summary": runtime_load_summary,
        "measurement": measurement,
        "round198_command_set": build_round198_operator_remeasurement_command_set(),
        "round199_smoke_contract": build_round199_local_smoke_contract(),
        "expected_delta_report_schema": build_round201_expected_delta_report_schema(),
        "broader_validation_delta": build_round202_broader_validation_delta(
            focused_command_status="green" if success else "blocked_or_pending_operator_local_real_artifact_run"
        ),
        "blockers": sorted(set(str(item) for item in blockers)),
        **POLICY_FLAGS,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Remeasure the guarded EVE-specific vector/self-learning cluster with operator-local medium30k.")
    parser.add_argument("--artifact-dir", default=str(DEFAULT_OPERATOR_MEDIUM_30K_DIR), help="Operator-local medium30k artifact directory.")
    parser.add_argument("--target-word", default=DEFAULT_TARGET_WORD, help="EVE-specific target word to observe and commit in memory.")
    parser.add_argument("--observation-text", action="append", dest="observation_texts", help="Korean-first observation text; may be repeated.")
    parser.add_argument("--context-word", action="append", dest="context_words", help="Known medium30k context word candidate; may be repeated.")
    parser.add_argument("--output", help="Optional JSON output path for operator-local handoff evidence.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run_remeasurement(
            artifact_dir=args.artifact_dir,
            target_word=args.target_word,
            observation_texts=args.observation_texts or DEFAULT_OBSERVATION_TEXTS,
            context_words=args.context_words or DEFAULT_CONTEXT_WORDS,
        )
    except Exception as exc:  # fail closed with machine-readable output
        report = {
            "version": ROUND198_202_REMEASURE_VERSION,
            "rounds_completed": [198, 199, 200, 201, 202],
            "selected_cluster_id": SELECTED_CLUSTER_ID,
            "status": "blocked_operator_eve_self_learning_remeasurement_exception",
            "success": False,
            "exit_code": 1,
            "blockers": ["operator_remeasurement_exception"],
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
