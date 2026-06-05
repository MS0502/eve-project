from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import scripts.operator_remeasure_eve_self_learning as remeasure


def _green_validation() -> dict:
    return {
        "status": "operator_medium30k_validation_green",
        "success": True,
        "exit_code": 0,
        "attempt_load_requested": True,
        "explicit_load": {"status": "explicit_load_succeeded", "vectors_loaded": True, "blockers": []},
        "blockers": [],
    }


def test_round198_command_set_is_one_stable_operator_command() -> None:
    report = remeasure.build_round198_operator_remeasurement_command_set()

    assert report["version"] == "v3_round198_operator_remeasurement_command_set"
    assert report["round"] == 198
    assert report["selected_cluster_id"] == remeasure.SELECTED_CLUSTER_ID
    assert report["manual_python_snippet_required"] is False
    assert "python scripts/operator_remeasure_eve_self_learning.py" in report["stable_remeasurement_command"]
    assert "--target-word 민석" in report["stable_remeasurement_command"]
    assert "--context-word 한국어" in report["stable_remeasurement_command"]
    assert report["production_persistence_enabled"] is False
    assert report["runtime_mapping_enabled_default"] is False
    assert report["enforcement_enabled"] is False
    assert report["agp_bypass_used"] is False


def test_round199_local_smoke_contract_preserves_korean_first_fixture() -> None:
    contract = remeasure.build_round199_local_smoke_contract()

    assert contract["version"] == "v3_round199_eve_self_learning_local_smoke_contract"
    assert contract["pytest_marker_required"] is False
    assert contract["production_persistence_required"] is False
    assert contract["real_vector_load"] == "explicit_guarded_operator_authorized_medium30k_only"
    assert contract["target_word"] == "민석"
    assert contract["observation_texts"] == ["민석 감정 기억", "민석 한국어 대화"]
    assert contract["context_words"] == ["한국어", "감정", "기억", "대화"]


def test_round200_blocked_validation_does_not_build_engine_or_measure(tmp_path: Path) -> None:
    calls = {"engine": 0, "measurement": 0}

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

    def measurement_func(*args, **kwargs):
        calls["measurement"] += 1
        return {}

    report = remeasure.run_remeasurement(
        artifact_dir=tmp_path / "missing",
        repo_root=tmp_path,
        validation_func=red_validation,
        engine_builder=engine_builder,
        measurement_func=measurement_func,
    )

    assert report["success"] is False
    assert report["exit_code"] == 1
    assert report["status"] == "blocked_operator_eve_self_learning_remeasurement"
    assert "operator_artifact_files_missing" in report["blockers"]
    assert report["measurement"]["measurement_executed"] is False
    assert report["measurement"]["in_memory_vector_mutation_attempted"] is False
    assert calls == {"engine": 0, "measurement": 0}
    assert report["production_persistence_enabled"] is False
    assert report["artifacts_written"] is False


def test_round200_green_guard_invokes_engine_and_measurement_without_fake_vector_contents(tmp_path: Path) -> None:
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

    def measurement_func(engine, **kwargs):
        calls["measurement_kwargs"] = dict(kwargs)
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

    report = remeasure.run_remeasurement(
        artifact_dir=tmp_path / "subset_medium_30k",
        repo_root=tmp_path,
        validation_func=validation_func,
        engine_builder=engine_builder,
        measurement_func=measurement_func,
    )

    assert report["success"] is True
    assert report["exit_code"] == 0
    assert calls["validation_kwargs"]["attempt_load"] is True
    assert calls["engine_kwargs"]["operator_medium30k_load_authorized"] is True
    assert calls["engine_kwargs"]["operator_medium30k_validation"]["status"] == "operator_medium30k_validation_green"
    assert calls["measurement_kwargs"]["target_word"] == "민석"
    assert report["measurement"]["persistent_vector_artifacts_written"] is False
    assert report["dummy_vectors_created"] is False
    assert report["seed_vectors_mutated"] is False


def test_round201_expected_delta_schema_documents_green_and_blocked_shapes() -> None:
    schema = remeasure.build_round201_expected_delta_report_schema()

    assert schema["version"] == "v3_round201_expected_operator_local_delta_report_schema"
    assert "measurement" in schema["required_top_level_keys"]
    assert "known_context_words" in schema["measurement_required_keys"]
    assert schema["expected_operator_local_green_delta"]["target_word"] == "민석"
    assert schema["expected_operator_local_green_delta"]["persistent_vector_artifacts_written"] is False
    assert schema["blocked_cloud_delta"]["expected_blocker"] == "operator_artifact_files_missing"


def test_main_writes_optional_json_and_prints_compact_report(monkeypatch, tmp_path: Path, capsys) -> None:
    payload = {
        "version": remeasure.ROUND198_202_REMEASURE_VERSION,
        "rounds_completed": [198, 199, 200, 201, 202],
        "selected_cluster_id": remeasure.SELECTED_CLUSTER_ID,
        "status": "blocked_operator_eve_self_learning_remeasurement",
        "success": False,
        "exit_code": 1,
        "blockers": ["operator_artifact_files_missing"],
        **remeasure.POLICY_FLAGS,
    }
    monkeypatch.setattr(remeasure, "run_remeasurement", lambda **kwargs: dict(payload))
    out = tmp_path / "report.json"

    exit_code = remeasure.main(["--output", str(out)])
    printed = capsys.readouterr().out.strip()

    assert exit_code == 1
    assert "\n" not in printed
    assert json.loads(printed)["status"] == "blocked_operator_eve_self_learning_remeasurement"
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["rounds_completed"] == [198, 199, 200, 201, 202]
