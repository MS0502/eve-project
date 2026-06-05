from __future__ import annotations

import subprocess

import scripts.operator_analyze_round351_365_failure_clusters as round351_365
import scripts.operator_run_round306_320_split_validation as round306_320


def test_round351_consolidates_captured_135_failure_evidence_without_vectors() -> None:
    evidence = round351_365.consolidate_round351_captured_evidence()

    assert evidence["version"] == "v3_round351_captured_135_failure_evidence_consolidation"
    assert evidence["round"] == 351
    assert evidence["success"] is True
    assert evidence["failed_count"] == 135
    assert evidence["passed_count"] == 1279
    assert evidence["nodeid_count_in_payload"] >= 5
    assert evidence["full_135_nodeid_list_available_in_repo"] is False
    assert evidence["checks"]["runtime_mapping_default_false"] is True
    assert evidence["vectors_created"] is False
    assert evidence["vectors_loaded"] is False


def test_round352_355_classifies_requested_failure_clusters() -> None:
    nodeids = [
        "tests/test_v3_round40_pre_swap_audit.py::test_round40_all_trace_modules_have_parallel_observation_option_unloaded",
        "tests/test_v3_round41_final_swap.py::test_round41_wrapper_primary_used_for_known_word",
        "tests/test_v3_round96_runtime_mapping_enable_smoke_precheck.py::test_round96_precheck_keeps_runtime_mapping_default_false",
        "tests/test_v3_round306_320_split_validation_execution.py::test_round306_320_cli_artifact_dependent_missing_artifacts_fails_closed",
        "tests/test_unrelated.py::test_other_failure",
    ]

    taxonomy = round351_365.classify_round351_355_failures(nodeids, expected_failed_count=135)

    assert taxonomy["version"] == "v3_round352_355_concrete_failure_cluster_taxonomy"
    assert taxonomy["rounds"] == [352, 353, 354, 355]
    assert taxonomy["cluster_counts"] == {
        "round40_41_wrapper_swap_audit": 2,
        "round94_98_runtime_mapping_enable_smoke": 1,
        "round306_320_artifact_guard_regression": 1,
        "unclassified_pytest_failure": 1,
    }
    assert taxonomy["unexpanded_failure_count"] == 130


def test_round356_360_selects_recent_artifact_guard_cluster_for_guarded_repair() -> None:
    taxonomy = round351_365.classify_round351_355_failures(
        [
            "tests/test_v3_round40_pre_swap_audit.py::test_round40_all_trace_modules_have_parallel_observation_option_unloaded",
            "tests/test_v3_round306_320_split_validation_execution.py::test_round306_320_cli_artifact_dependent_missing_artifacts_fails_closed",
        ],
        expected_failed_count=135,
    )

    selected = round351_365.select_round356_360_repair_cluster(taxonomy)

    assert selected["version"] == "v3_round356_360_selected_guarded_repair_cluster"
    assert selected["rounds"] == [356, 357, 358, 359, 360]
    assert selected["success"] is True
    assert selected["selected_cluster"] == "round306_320_artifact_guard_regression"
    assert selected["implementation_action"] == "guarded_artifact_dependent_pytest_summary_taxonomy_delta_and_fail_closed_metadata"
    assert selected["runtime_mapping_default_go"] is False
    assert selected["enforcement_go"] is False
    assert selected["vectors_created"] is False


def test_round361_365_report_records_delta_and_remaining_taxonomy() -> None:
    report = round351_365.build_round351_365_report(
        focused_test_status="green",
        artifact_free_status="green",
        artifact_dependent_status="red_135_failed_1279_passed",
        broader_pytest_status="red_135_failed_1279_passed",
    )

    assert report["version"] == "v3_round351_365_captured_failure_cluster_repair_delta"
    assert report["success"] is True
    assert report["rounds_completed"] == list(range(351, 366))
    assert report["round361_365_validation_delta"]["captured_failure_evidence"]["failed_count"] == 135
    assert report["round356_360_selected_repair"]["selected_cluster"] == "round306_320_artifact_guard_regression"
    remaining = report["round361_365_validation_delta"]["remaining_taxonomy"]
    assert remaining["round40_41_wrapper_swap_audit"] >= 4
    assert remaining["round306_320_artifact_guard_regression"] == 1
    assert report["production_persistence_enabled"] is False
    assert report["runtime_mapping_enabled_default"] is False
    assert report["enforcement_enabled"] is False


def test_artifact_dependent_runner_exports_round351_365_taxonomy_before_tail_truncation() -> None:
    stdout = "\n".join(
        [
            "FAILED tests/test_v3_round40_pre_swap_audit.py::test_round40_all_trace_modules_have_parallel_observation_option_unloaded - AssertionError",
            "FAILED tests/test_v3_round96_runtime_mapping_enable_smoke_precheck.py::test_round96_precheck_keeps_runtime_mapping_default_false - AssertionError",
            "FAILED tests/test_v3_round306_320_split_validation_execution.py::test_round306_320_cli_artifact_dependent_missing_artifacts_fails_closed - AssertionError",
            "3 failed, 1407 passed in 3.00s",
        ]
    )

    def runner(command, **kwargs):
        return subprocess.CompletedProcess(args=list(command), returncode=1, stdout=stdout, stderr="")

    result = round306_320._run_command("python -m pytest -q", runner=runner)
    summary = result["pytest_failure_summary"]

    assert result["success"] is False
    assert summary["nodeids_captured_before_tail_truncation"] is True
    assert summary["sample_nodeids"] == [
        "tests/test_v3_round40_pre_swap_audit.py::test_round40_all_trace_modules_have_parallel_observation_option_unloaded",
        "tests/test_v3_round96_runtime_mapping_enable_smoke_precheck.py::test_round96_precheck_keeps_runtime_mapping_default_false",
        "tests/test_v3_round306_320_split_validation_execution.py::test_round306_320_cli_artifact_dependent_missing_artifacts_fails_closed",
    ]
    assert summary["round351_365_concrete_cluster_counts"] == {
        "round40_41_wrapper_swap_audit": 1,
        "round94_98_runtime_mapping_enable_smoke": 1,
        "round306_320_artifact_guard_regression": 1,
        "unclassified_pytest_failure": 0,
    }
    assert summary["round351_365_unexpanded_failure_count"] == 0
    assert summary["full_stdout_retained"] is False
    assert summary["vectors_created"] is False
