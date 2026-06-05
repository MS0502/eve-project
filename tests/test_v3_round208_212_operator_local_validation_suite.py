from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Sequence

import scripts.operator_run_local_validation_suite as suite


def _completed(payload: dict, returncode: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n", stderr="")


def test_round208_suite_contract_is_one_stable_korean_first_command() -> None:
    contract = suite.build_round208_suite_contract()

    assert contract["version"] == "v3_round208_operator_one_command_suite_contract"
    assert contract["round"] == 208
    assert contract["stable_operator_command"] == "python scripts/operator_run_local_validation_suite.py"
    assert contract["manual_python_snippet_required"] is False
    assert contract["default_target_word"] == "민석"
    assert contract["default_context_words"] == ["한국어", "감정", "기억", "대화"]
    assert contract["production_persistence_enabled"] is False
    assert contract["runtime_mapping_enabled_default"] is False
    assert contract["enforcement_enabled"] is False
    assert contract["agp_bypass_used"] is False


def test_command_sequence_runs_required_operator_checks_in_order() -> None:
    commands = suite.build_operator_command_sequence()
    joined = [" ".join(command) for command in commands]

    assert "scripts/operator_validate_medium30k.py --attempt-load" in joined[0]
    assert "scripts/operator_remeasure_eve_self_learning.py" in joined[1]
    assert "--target-word 민석" in joined[1]
    assert "--context-word 한국어" in joined[1]
    assert "scripts/operator_measure_runtime_mapping_after_self_learning.py" in joined[2]
    assert "--negative-token EVE" in joined[2]


def test_round209_schema_documents_compact_json_and_copy_paste_summary() -> None:
    schema = suite.build_round209_report_schema()

    assert schema["version"] == "v3_round209_operator_suite_report_schema"
    assert schema["compact_json_stdout"] is True
    assert schema["copy_paste_summary_stdout"] is True
    assert schema["local_report_path"] == "_operator_artifacts/operator_local_validation_latest.json"
    assert "operator_prompt_summary" in schema["required_top_level_keys"]
    assert "target_word" in schema["operator_prompt_summary_required_fields"]


def test_round210_suite_stops_on_first_failure_without_running_real_vectors(tmp_path: Path) -> None:
    calls: list[str] = []

    def runner(command: Sequence[str], **kwargs):
        calls.append(" ".join(command))
        return _completed(
            {
                "status": "blocked_operator_medium30k_validation_failed",
                "success": False,
                "exit_code": 1,
                "blockers": ["operator_artifact_files_missing"],
            },
            returncode=1,
        )

    report = suite.run_suite(repo_root=tmp_path, report_path=tmp_path / "latest.json", runner=runner)

    assert report["success"] is False
    assert report["exit_code"] == 1
    assert report["status"] == "blocked_operator_local_validation_suite"
    assert report["blockers"] == ["operator_artifact_files_missing"]
    assert len(report["steps"]) == 1
    assert "operator_validate_medium30k.py" in calls[0]
    assert all("operator_remeasure_eve_self_learning.py" not in call for call in calls)
    assert report["git_status_safety"]["status"] == "not_run_previous_step_failed"
    assert report["dummy_vectors_created"] is False


def test_round210_green_suite_writes_report_and_extracts_prompt_summary(tmp_path: Path) -> None:
    calls: list[str] = []

    def runner(command: Sequence[str], **kwargs):
        joined = " ".join(command)
        calls.append(joined)
        if "git status" in joined:
            return subprocess.CompletedProcess(args=list(command), returncode=0, stdout="", stderr="")
        if "git ls-files" in joined:
            return subprocess.CompletedProcess(args=list(command), returncode=0, stdout="", stderr="")
        if "operator_remeasure_eve_self_learning.py" in joined:
            return _completed(
                {
                    "selected_cluster_id": "eve_specific_vector_self_learning_cascade",
                    "status": "operator_eve_self_learning_remeasurement_green",
                    "success": True,
                    "exit_code": 0,
                    "blockers": [],
                    "measurement": {
                        "target_word": "민석",
                        "vector_lookup_after_commit": True,
                        "vector_store_count_before": 0,
                        "vector_store_count_after": 1,
                        "wrapper_eve_specific_hits_before": 0,
                        "wrapper_eve_specific_hits_after": 1,
                        "wrapper_primary_loaded": True,
                    },
                }
            )
        if "operator_measure_runtime_mapping_after_self_learning.py" in joined:
            return _completed(
                {
                    "selected_cluster_id": "concept_runtime_mapping_after_eve_self_learning_guarded_medium30k",
                    "status": "operator_runtime_mapping_after_self_learning_green",
                    "success": True,
                    "exit_code": 0,
                    "blockers": [],
                    "self_learning_measurement": {
                        "target_word": "민석",
                        "vector_lookup_after_commit": True,
                        "vector_store_count_before": 0,
                        "vector_store_count_after": 1,
                        "wrapper_eve_specific_hits_before": 0,
                        "wrapper_eve_specific_hits_after": 1,
                        "wrapper_primary_loaded": True,
                    },
                    "mapping_measurement": {
                        "runtime_mapping_enabled_after": False,
                        "enforcement_enabled_after": False,
                        "runtime_mapping_persisted": False,
                    },
                }
            )
        return _completed({"status": "operator_medium30k_validation_green", "success": True, "exit_code": 0, "blockers": []})

    report_path = tmp_path / "operator_local_validation_latest.json"
    report = suite.run_suite(repo_root=tmp_path, report_path=report_path, runner=runner)

    assert report["success"] is True
    assert report["exit_code"] == 0
    assert len(report["steps"]) == 3
    assert report["git_status_safety"]["safe"] is True
    assert report["operator_prompt_summary"]["target_word"] == "민석"
    assert report["operator_prompt_summary"]["selected_cluster_id"] == "concept_runtime_mapping_after_eve_self_learning_guarded_medium30k"
    assert report["operator_prompt_summary"]["vector_lookup_after_commit"] is True
    assert report["operator_prompt_summary"]["production_persistence_enabled"] is False
    assert report["operator_prompt_summary"]["runtime_mapping_enabled_default"] is False
    assert report_path.exists()
    written = json.loads(report_path.read_text(encoding="utf-8"))
    assert written["rounds_completed"] == [208, 209, 210, 211, 212]
    assert calls[0].endswith("scripts/operator_validate_medium30k.py --attempt-load")


def test_git_status_safety_blocks_guarded_path_changes(tmp_path: Path) -> None:
    def runner(command: Sequence[str], **kwargs):
        joined = " ".join(command)
        if "git status" in joined:
            return subprocess.CompletedProcess(args=list(command), returncode=0, stdout=" M seeds/subsets/cc.ko.300.subset.medium.30k/vocab.txt\n", stderr="")
        return subprocess.CompletedProcess(args=list(command), returncode=0, stdout="", stderr="")

    safety = suite.run_git_status_safety_check(repo_root=tmp_path, runner=runner)

    assert safety["success"] is False
    assert safety["safe"] is False
    assert "guarded_paths_present_in_git_status" in safety["blockers"]


def test_main_prints_compact_json_then_copy_paste_summary(monkeypatch, tmp_path: Path, capsys) -> None:
    payload = {
        "version": suite.ROUND208_212_VERSION,
        "rounds_completed": [208, 209, 210, 211, 212],
        "status": "blocked_operator_local_validation_suite",
        "success": False,
        "exit_code": 1,
        "steps": [],
        "git_status_safety": {"safe": False},
        "blockers": ["operator_artifact_files_missing"],
        **suite.POLICY_FLAGS,
    }
    payload["operator_prompt_summary"] = suite.build_operator_prompt_summary(payload)
    monkeypatch.setattr(suite, "run_suite", lambda: dict(payload))

    exit_code = suite.main([])
    output = capsys.readouterr().out.splitlines()

    assert exit_code == 1
    assert json.loads(output[0])["status"] == "blocked_operator_local_validation_suite"
    assert output[1] == "OPERATOR_LOCAL_VALIDATION_SUMMARY:"
    assert any("target_word: 민석" in line for line in output)


def test_round211_and_round212_reports_preserve_no_go_policy() -> None:
    workflow = suite.build_round211_operator_workflow()
    delta = suite.build_round212_broader_validation_delta(focused_suite_tests_status="green", compileall_status="green")

    assert workflow["version"] == "v3_round211_one_command_operator_workflow"
    assert workflow["report_path"] == "_operator_artifacts/operator_local_validation_latest.json"
    assert workflow["no_manual_python_snippets"] is True
    assert delta["version"] == "v3_round212_operator_suite_broader_validation_delta"
    assert delta["focused_suite_tests_status"] == "green"
    assert delta["production_persistence_enabled"] is False
    assert delta["next_recommendation"].startswith("operator_run_one_command_suite_after_merge")
