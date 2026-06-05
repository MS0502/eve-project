from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import scripts.operator_measure_runtime_mapping_after_self_learning as runtime_measure


def _green_validation() -> dict:
    return {
        "status": "operator_medium30k_validation_green",
        "success": True,
        "exit_code": 0,
        "attempt_load_requested": True,
        "explicit_load": {"status": "explicit_load_succeeded", "vectors_loaded": True, "blockers": []},
        "blockers": [],
    }


def test_round203_records_operator_local_green_evidence_without_persistence() -> None:
    evidence = runtime_measure.build_round203_operator_green_evidence_record()

    assert evidence["version"] == "v3_round203_operator_local_green_evidence_record"
    assert evidence["round"] == 203
    assert evidence["status"] == "operator_eve_self_learning_remeasurement_green"
    assert evidence["success"] is True
    assert evidence["target_word"] == "민석"
    assert evidence["runtime_load_summary"]["attached_to_engine"] is True
    assert evidence["runtime_load_summary"]["guard_called"] is True
    assert evidence["production_persistence_enabled"] is False
    assert evidence["runtime_mapping_enabled_default"] is False
    assert evidence["enforcement_enabled"] is False
    assert evidence["agp_bypass_used"] is False


def test_round204_selects_concept_runtime_mapping_cluster_from_green_self_learning() -> None:
    selection = runtime_measure.build_round204_next_cluster_selection()

    assert selection["version"] == "v3_round204_next_narrow_failure_cluster_selection"
    assert selection["round"] == 204
    assert selection["selected_cluster_id"] == runtime_measure.SELECTED_CLUSTER_ID
    assert selection["previous_green_evidence"]["target_word"] == "민석"
    assert selection["operator_local_required"] is True
    assert selection["cloud_expected_blocker_without_artifacts"] == "operator_artifact_files_missing"
    assert "production_persistence_enablement" in selection["out_of_scope"]
    assert selection["production_persistence_enabled"] is False


def test_round205_operator_command_is_one_stable_korean_first_workflow() -> None:
    contract = runtime_measure.build_round205_operator_command_contract()

    assert contract["version"] == "v3_round205_guarded_local_only_runtime_mapping_measurement_contract"
    assert contract["round"] == 205
    assert contract["manual_python_snippet_required"] is False
    assert "python scripts/operator_measure_runtime_mapping_after_self_learning.py" in contract["stable_operator_local_command"]
    assert "--target-word 민석" in contract["stable_operator_local_command"]
    assert "--context-word 한국어" in contract["stable_operator_local_command"]
    assert "--negative-token EVE" in contract["stable_operator_local_command"]
    assert contract["default_runtime_load"] is False
    assert contract["controlled_smoke_rolls_back_runtime_mapping"] is True


def test_round205_blocked_validation_does_not_build_engine_or_measure(tmp_path: Path) -> None:
    calls = {"engine": 0, "self_learning": 0, "mapping": 0}

    def red_validation(**kwargs):
        return {
            "status": "blocked_operator_medium30k_validation_failed",
            "success": False,
            "exit_code": 1,
            "attempt_load_requested": True,
            "explicit_load": {"vectors_loaded": False},
            "blockers": ["operator_artifact_files_missing"],
        }

    def engine_builder(**kwargs):
        calls["engine"] += 1
        return SimpleNamespace()

    def self_learning_func(*args, **kwargs):
        calls["self_learning"] += 1
        return {}

    def mapping_func(*args, **kwargs):
        calls["mapping"] += 1
        return {}

    report = runtime_measure.run_guarded_runtime_mapping_measurement(
        artifact_dir=tmp_path / "missing",
        repo_root=tmp_path,
        validation_func=red_validation,
        engine_builder=engine_builder,
        self_learning_measurement_func=self_learning_func,
        mapping_pipeline_func=mapping_func,
    )

    assert report["success"] is False
    assert report["exit_code"] == 1
    assert report["status"] == "blocked_operator_runtime_mapping_after_self_learning"
    assert "operator_artifact_files_missing" in report["blockers"]
    assert report["self_learning_measurement"]["measurement_executed"] is False
    assert report["mapping_measurement"]["measurement_executed"] is False
    assert calls == {"engine": 0, "self_learning": 0, "mapping": 0}
    assert report["production_persistence_enabled"] is False


def test_round206_green_guard_runs_mapping_pipeline_with_fakes_not_vector_contents(tmp_path: Path) -> None:
    calls: dict[str, object] = {}

    def validation_func(**kwargs):
        calls["validation_kwargs"] = dict(kwargs)
        return _green_validation()

    def engine_builder(**kwargs):
        calls["engine_kwargs"] = dict(kwargs)
        return SimpleNamespace(
            medium30k_runtime_load_report={
                "status": "guarded_runtime_load_attached",
                "attached_to_engine": True,
                "guard_called": True,
                "blockers": [],
            }
        )

    def self_learning_func(engine, **kwargs):
        calls["self_learning_kwargs"] = dict(kwargs)
        return {
            "measurement_executed": True,
            "target_word": kwargs["target_word"],
            "created_vectors": [kwargs["target_word"]],
            "commit_gate_pass": True,
            "wrapper_primary_loaded": True,
            "wrapper_eve_specific_hits_delta": 1,
            "persistent_vector_artifacts_written": False,
            "in_memory_vector_mutation_attempted": True,
        }

    def mapping_func(engine, **kwargs):
        calls["mapping_kwargs"] = dict(kwargs)
        return {
            "measurement_executed": True,
            "target_word": kwargs["target_word"],
            "negative_token": kwargs["negative_token"],
            "target_would_map": True,
            "target_precheck_ready": True,
            "target_mapped_in_controlled_smoke": True,
            "negative_token_blocked": True,
            "runtime_mapping_enabled_before": False,
            "runtime_mapping_enabled_after": False,
            "enforcement_enabled_before": False,
            "enforcement_enabled_after": False,
            "rollback_complete": True,
            "runtime_mapping_persisted": False,
            "production_persistence_enabled": False,
            "enforcement_enabled": False,
            "agp_bypass_used": False,
        }

    report = runtime_measure.run_guarded_runtime_mapping_measurement(
        artifact_dir=tmp_path / "subset_medium_30k",
        repo_root=tmp_path,
        validation_func=validation_func,
        engine_builder=engine_builder,
        self_learning_measurement_func=self_learning_func,
        mapping_pipeline_func=mapping_func,
    )

    assert report["success"] is True
    assert report["exit_code"] == 0
    assert report["status"] == "operator_runtime_mapping_after_self_learning_green"
    assert report["rounds_completed"] == [203, 204, 205, 206, 207]
    assert calls["validation_kwargs"]["attempt_load"] is True
    assert calls["engine_kwargs"]["operator_medium30k_load_authorized"] is True
    assert calls["self_learning_kwargs"]["target_word"] == "민석"
    assert calls["mapping_kwargs"]["target_word"] == "민석"
    assert report["mapping_measurement"]["runtime_mapping_enabled_after"] is False
    assert report["mapping_measurement"]["runtime_mapping_persisted"] is False
    assert report["production_persistence_enabled"] is False
    assert report["runtime_mapping_enabled_default"] is False
    assert report["enforcement_enabled"] is False
    assert report["dummy_vectors_created"] is False


def test_main_writes_optional_json_and_prints_compact_report(monkeypatch, tmp_path: Path, capsys) -> None:
    payload = {
        "version": runtime_measure.ROUND203_207_VERSION,
        "rounds_completed": [203, 204, 205, 206, 207],
        "selected_cluster_id": runtime_measure.SELECTED_CLUSTER_ID,
        "status": "blocked_operator_runtime_mapping_after_self_learning",
        "success": False,
        "exit_code": 1,
        "blockers": ["operator_artifact_files_missing"],
        **runtime_measure.POLICY_FLAGS,
    }
    monkeypatch.setattr(runtime_measure, "run_guarded_runtime_mapping_measurement", lambda **kwargs: dict(payload))
    out = tmp_path / "report.json"

    exit_code = runtime_measure.main(["--output", str(out)])
    printed = capsys.readouterr().out.strip()

    assert exit_code == 1
    assert "\n" not in printed
    assert json.loads(printed)["status"] == "blocked_operator_runtime_mapping_after_self_learning"
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["rounds_completed"] == [203, 204, 205, 206, 207]
