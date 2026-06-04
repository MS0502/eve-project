from types import SimpleNamespace

from adapters.explicit_load_guard import (
    ROUND182_VECTOR_SELF_LEARNING_CLUSTER_VERSION,
    ROUND183_GUARDED_EXPLICIT_LOAD_VERSION,
    ROUND184_EXPLICIT_LOAD_GUARD_TEST_PLAN_VERSION,
    ROUND185_OPERATOR_LOCAL_VALIDATION_COMMANDS_VERSION,
    ROUND186_VALIDATION_DELTA_RECOMMENDATION_VERSION,
    SELECTED_VECTOR_SELF_LEARNING_CLUSTER_ID,
    build_round184_explicit_load_guard_test_plan,
    build_round185_operator_local_validation_commands,
    build_round186_validation_delta_recommendation,
    guarded_explicit_medium30k_load,
    select_round182_vector_self_learning_cluster,
)


class RecordingAdapter:
    constructed = 0
    loaded = 0

    def __init__(self, **kwargs):
        type(self).constructed += 1
        self.kwargs = kwargs

    def load(self):
        type(self).loaded += 1
        return True


class FailingAdapter:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def load(self):
        raise ValueError("operator_artifact_checksum_mismatch")


def _green_gate() -> dict:
    return {
        "status": "ready_for_explicit_operator_artifact_load",
        "ready": True,
        "load_should_be_attempted": True,
        "readiness_blockers": [],
    }


def _red_gate() -> dict:
    return {
        "status": "blocked_operator_artifact_required",
        "ready": False,
        "load_should_be_attempted": False,
        "readiness_blockers": ["operator_artifact_files_missing"],
    }


def test_round182_selects_narrow_vector_self_learning_cluster_without_load() -> None:
    selection = select_round182_vector_self_learning_cluster(
        operator_preflight={"status": "load_dependent_repair_preflight_green", "ready": True, "hard_block": False}
    )

    assert selection["version"] == ROUND182_VECTOR_SELF_LEARNING_CLUSTER_VERSION
    assert selection["selected_cluster"]["id"] == SELECTED_VECTOR_SELF_LEARNING_CLUSTER_ID
    assert selection["selected_cluster"]["taxonomy"] == "EVE-specific vector/self-learning cascade"
    assert selection["actual_load_allowed_by_selection"] is False
    assert selection["production_persistence_enabled"] is False
    assert selection["runtime_mapping_enabled_default"] is False
    assert selection["enforcement_enabled"] is False
    assert selection["agp_bypass_used"] is False
    assert selection["artifacts_included"] is False


def test_round183_red_preflight_blocks_adapter_construction_and_load() -> None:
    RecordingAdapter.constructed = 0
    RecordingAdapter.loaded = 0

    report = guarded_explicit_medium30k_load(
        readiness_gate=_red_gate(),
        attempt_load=True,
        adapter_factory=RecordingAdapter,
    )

    assert report["version"] == ROUND183_GUARDED_EXPLICIT_LOAD_VERSION
    assert report["status"] == "blocked_preflight_not_green"
    assert report["guard_green"] is False
    assert report["vectors_loaded"] is False
    assert "operator_artifact_files_missing" in report["blockers"]
    assert RecordingAdapter.constructed == 0
    assert RecordingAdapter.loaded == 0


def test_round183_green_preflight_can_report_ready_without_loading() -> None:
    RecordingAdapter.constructed = 0
    RecordingAdapter.loaded = 0

    report = guarded_explicit_medium30k_load(
        readiness_gate=_green_gate(),
        attempt_load=False,
        adapter_factory=RecordingAdapter,
    )

    assert report["status"] == "guard_ready_explicit_load_not_attempted"
    assert report["guard_green"] is True
    assert report["attempt_load_requested"] is False
    assert report["vectors_loaded"] is False
    assert report["read_only_until_attempt_load"] is True
    assert RecordingAdapter.constructed == 0
    assert RecordingAdapter.loaded == 0


def test_round183_green_preflight_attempts_explicit_load_only_when_requested() -> None:
    RecordingAdapter.constructed = 0
    RecordingAdapter.loaded = 0
    engine = SimpleNamespace()

    report = guarded_explicit_medium30k_load(
        engine=engine,
        readiness_gate=_green_gate(),
        artifact_dir="/operator/local/subset_medium_30k",
        attempt_load=True,
        attach_to_engine=True,
        adapter_factory=RecordingAdapter,
    )

    assert report["status"] == "explicit_load_succeeded"
    assert report["guard_green"] is True
    assert report["vectors_loaded"] is True
    assert report["artifact_dir"] == "/operator/local/subset_medium_30k"
    assert report["adapter_class"] == "RecordingAdapter"
    assert RecordingAdapter.constructed == 1
    assert RecordingAdapter.loaded == 1
    assert engine.fasttext_embedding.__class__.__name__ == "RecordingAdapter"


def test_round183_load_exception_fails_closed_without_engine_attachment() -> None:
    engine = SimpleNamespace()

    report = guarded_explicit_medium30k_load(
        engine=engine,
        readiness_gate=_green_gate(),
        attempt_load=True,
        attach_to_engine=True,
        adapter_factory=FailingAdapter,
    )

    assert report["status"] == "explicit_load_failed_closed"
    assert report["vectors_loaded"] is False
    assert "explicit_load_exception" in report["blockers"]
    assert report["load_error"] == "ValueError:operator_artifact_checksum_mismatch"
    assert not hasattr(engine, "fasttext_embedding")
    assert report["production_persistence_enabled"] is False
    assert report["runtime_mapping_enabled_default"] is False
    assert report["enforcement_enabled"] is False
    assert report["agp_bypass_used"] is False


def test_round184_185_186_reports_are_artifact_free_and_operator_oriented() -> None:
    test_plan = build_round184_explicit_load_guard_test_plan()
    commands = build_round185_operator_local_validation_commands()
    delta = build_round186_validation_delta_recommendation(
        focused_tests_status="passed",
        collect_only_status="passed",
        broader_pytest_status="red_baseline_expected",
        broader_failure_count=205,
    )

    assert test_plan["version"] == ROUND184_EXPLICIT_LOAD_GUARD_TEST_PLAN_VERSION
    assert test_plan["real_vectors_required"] is False
    assert "tests/test_v3_round182_186_explicit_load_guard.py" in test_plan["focused_test_command"]
    assert commands["version"] == ROUND185_OPERATOR_LOCAL_VALIDATION_COMMANDS_VERSION
    assert any("guarded_explicit_medium30k_load(attempt_load=True)" in command for command in commands["commands"])
    assert any(command == "python -m pytest -q" for command in commands["commands"])
    assert delta["version"] == ROUND186_VALIDATION_DELTA_RECOMMENDATION_VERSION
    assert delta["focused_tests_status"] == "passed"
    assert delta["broader_failure_count"] == 205
    assert delta["recommendation"] == "operator_run_real_artifact_explicit_load_validation_then_remeasure_self_learning_cascade"
    for report in (test_plan, commands, delta):
        assert report["artifacts_included"] is False
        assert report["production_persistence_enabled"] is False
        assert report["runtime_mapping_enabled_default"] is False
        assert report["enforcement_enabled"] is False
        assert report["agp_bypass_used"] is False
