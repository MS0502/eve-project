from __future__ import annotations

import subprocess

import scripts.operator_analyze_round336_350_artifact_failures as round336_350
import scripts.operator_run_round306_320_split_validation as round306_320


def _execution_report() -> dict:
    return {
        "execute_requested": True,
        "artifact_dependent_readiness": {
            "ready": True,
            "missing_artifact_ids": [],
            "unsafe_artifact_ids": [],
            "vector_contents_read": False,
        },
        "steps": [],
    }


def test_round336_consolidates_operator_executed_failure_evidence_without_loading_vectors() -> None:
    evidence = round336_350.consolidate_round336_execution_evidence(
        _execution_report(), operator_failed_count=135, operator_passed_count=1274
    )

    assert evidence["version"] == "v3_round336_artifact_dependent_execution_evidence_consolidation"
    assert evidence["round"] == 336
    assert evidence["success"] is True
    assert evidence["operator_failed_count"] == 135
    assert evidence["operator_passed_count"] == 1274
    assert evidence["readiness_green_evidence"] == {
        "ready": True,
        "missing": [],
        "unsafe": [],
        "vector_contents_read": False,
    }
    assert evidence["checks"]["runtime_mapping_default_false"] is True
    assert evidence["checks"]["enforcement_disabled"] is True


def test_round337_340_parses_and_classifies_pytest_failure_nodeids() -> None:
    pytest_text = """
FAILED tests/test_v3_round29_subset_extraction_mini_1k.py::test_vectors_shape_dtype_and_checksum - FileNotFoundError
FAILED tests/test_v3_round92_runtime_mapping_gate_dry_run.py::test_round92_runtime_mapping_gate_dry_run_maps_only_committed_category_without_enabling_runtime_mapping - AssertionError
FAILED tests/test_v3_round73_explicit_eve_specific_commit_smoke.py::test_round73_real_explicit_commit_smoke_creates_eve_specific_vector - AssertionError
135 failed, 1274 passed in 20.00s
"""

    nodeids = round336_350.extract_pytest_failure_nodeids(pytest_text)
    counts = round336_350.extract_pytest_summary_counts(pytest_text)
    taxonomy = round336_350.classify_pytest_failures(nodeids)

    assert counts == {"failed": 135, "passed": 1274}
    assert len(nodeids) == 3
    assert taxonomy["version"] == "v3_round337_340_pytest_failure_cluster_taxonomy"
    assert taxonomy["cluster_counts"] == {
        "eve_specific_vector_learning_boundary": 1,
        "runtime_mapping_or_concept_mapping_readiness": 1,
        "seed_subset_or_fasttext_artifact_boundary": 1,
    }
    assert taxonomy["nodeids_available"] is True


def test_round341_345_selects_measurement_improvement_when_prior_report_lacks_nodeids() -> None:
    taxonomy = round336_350.classify_pytest_failures([])
    selected = round336_350.select_round341_345_repair_cluster(taxonomy)

    assert selected["version"] == "v3_round341_345_selected_non_persistence_repair_cluster"
    assert selected["rounds"] == [341, 342, 343, 344, 345]
    assert selected["selected_cluster"] == "pytest_failure_taxonomy_capture_gap"
    assert selected["implementation_action"] == "measurement_improvement_capture_full_pytest_failure_summary_before_tail_truncation"
    assert selected["production_persistence_go"] is False
    assert selected["runtime_mapping_default_go"] is False
    assert selected["enforcement_go"] is False
    assert selected["vectors_created"] is False


def test_round346_350_report_preserves_135_unparsed_taxonomy_when_no_full_log_available() -> None:
    report = round336_350.build_round336_350_report(
        execution_report=_execution_report(), operator_failed_count=135, operator_passed_count=1274
    )

    assert report["version"] == "v3_round336_350_artifact_dependent_failure_taxonomy"
    assert report["success"] is True
    assert report["round346_350_validation_delta"]["rounds_completed"] == list(range(336, 351))
    assert report["round346_350_validation_delta"]["remaining_taxonomy"] == {
        "unparsed_operator_report_failures_needing_full_pytest_summary": 135
    }
    assert report["production_persistence_enabled"] is False
    assert report["runtime_mapping_enabled_default"] is False
    assert report["enforcement_enabled"] is False


def test_artifact_dependent_runner_captures_pytest_summary_before_tail_truncation() -> None:
    stdout = "\n".join(
        [
            "FAILED tests/test_v3_round29_subset_extraction_mini_1k.py::test_subset_directory_and_files_exist - AssertionError",
            "FAILED tests/test_v3_round73_explicit_eve_specific_commit_smoke.py::test_round73_real_explicit_commit_smoke_creates_eve_specific_vector - AssertionError",
            "2 failed, 1407 passed in 3.00s",
        ]
    )

    def runner(command, **kwargs):
        return subprocess.CompletedProcess(args=list(command), returncode=1, stdout=stdout, stderr="")

    result = round306_320._run_command("python -m pytest -q", runner=runner)

    summary = result["pytest_failure_summary"]
    assert result["success"] is False
    assert summary["version"] == "v3_round336_350_full_pytest_failure_summary_capture"
    assert summary["failed_count"] == 2
    assert summary["passed_count"] == 1407
    assert summary["classified_failure_count"] == 2
    assert summary["cluster_counts"] == {
        "eve_specific_vector_learning_boundary": 1,
        "seed_subset_or_fasttext_artifact_boundary": 1,
    }
    assert summary["nodeids_captured_before_tail_truncation"] is True
    assert summary["full_stdout_retained"] is False
    assert summary["vectors_created"] is False
