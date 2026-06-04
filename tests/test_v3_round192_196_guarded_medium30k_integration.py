from __future__ import annotations

from types import SimpleNamespace

import adapters.medium30k_runtime_load_integration as integration


class RecordingAdapter:
    constructed = 0
    loaded = 0

    def __init__(self, **kwargs):
        type(self).constructed += 1
        self.kwargs = dict(kwargs)
        self._loaded = False

    def load(self) -> bool:
        type(self).loaded += 1
        self._loaded = True
        return True

    def is_loaded(self) -> bool:
        return self._loaded


def _green_validation() -> dict:
    return {
        "status": "operator_medium30k_validation_green",
        "success": True,
        "exit_code": 0,
        "attempt_load_requested": True,
        "verification": {
            "status": "operator_artifact_verified_green",
            "ready": True,
            "blockers": [],
            "vectors": {"shape": [30000, 300], "dtype": "float32", "shape_match": True, "dtype_match": True},
            "manifest_consistency": {"consistent": True},
        },
        "readiness": {
            "status": "ready_for_explicit_operator_artifact_load",
            "ready": True,
            "load_should_be_attempted": True,
            "readiness_blockers": [],
        },
        "preflight": {
            "status": "load_dependent_repair_preflight_green",
            "ready": True,
            "hard_block": False,
            "blockers": [],
        },
        "explicit_load": {
            "status": "explicit_load_succeeded",
            "attempt_load_requested": True,
            "vectors_loaded": True,
            "blockers": [],
        },
        "git_status_safety": {"safe": True, "lines": []},
        "production_persistence_enabled": False,
        "runtime_mapping_enabled_default": False,
        "enforcement_enabled": False,
        "agp_bypass_used": False,
        "artifacts_written": False,
        "vectors_committed": False,
        "dummy_vectors_created": False,
    }


def _no_load_validation() -> dict:
    payload = _green_validation()
    payload["attempt_load_requested"] = False
    payload["explicit_load"] = {
        "status": "guard_ready_explicit_load_not_attempted",
        "attempt_load_requested": False,
        "vectors_loaded": False,
        "blockers": [],
    }
    return payload


def test_round192_diagnoses_build_full_engine_known_context_entrypoint() -> None:
    report = integration.diagnose_round192_seed_vector_cascade_entrypoints(operator_validation=_green_validation())

    assert report["version"] == integration.ROUND192_SEED_VECTOR_CASCADE_DIAGNOSIS_VERSION
    assert report["round"] == 192
    assert report["operator_validation_evidence_green"] is True
    assert report["selected_entrypoint"] == integration.SELECTED_SEED_VECTOR_CASCADE_ENTRYPOINT
    assert report["baseline_failure_taxonomy"]["seed_vector_artifact_cascade"] == 127
    assert report["production_persistence_enabled"] is False
    assert report["runtime_mapping_enabled_default"] is False
    assert report["enforcement_enabled"] is False
    assert report["agp_bypass_used"] is False


def test_round193_selects_only_explicit_authorized_runtime_path() -> None:
    report = integration.select_round193_guarded_runtime_path()

    assert report["version"] == integration.ROUND193_GUARDED_RUNTIME_PATH_SELECTION_VERSION
    assert report["selected"] is True
    assert report["selected_entrypoint"] == integration.SELECTED_SEED_VECTOR_CASCADE_ENTRYPOINT
    assert "operator_medium30k_load_authorized=True" in report["path"]
    assert "default build_full_engine() call" in report["must_not_call_guard_when"]
    assert report["real_vectors_required_for_operator_path"] is True
    assert report["focused_tests_use_fakes_only_for_guard_behavior"] is True


def test_round194_default_does_not_call_guard_or_attach(monkeypatch) -> None:
    calls = {"count": 0}

    def guard(**kwargs):
        calls["count"] += 1
        return {"status": "explicit_load_succeeded", "vectors_loaded": True, "blockers": []}

    engine = SimpleNamespace()
    report = integration.apply_round194_guarded_medium30k_runtime_load(
        engine,
        operator_validation=_green_validation(),
        load_authorized=False,
        guard_func=guard,
    )

    assert report["version"] == integration.ROUND194_GUARDED_RUNTIME_LOAD_INTEGRATION_VERSION
    assert report["status"] == "default_no_load_explicit_authorization_required"
    assert report["guard_called"] is False
    assert report["attached_to_engine"] is False
    assert calls["count"] == 0
    assert not hasattr(engine, "fasttext_embedding")
    assert "operator_load_authorization_missing" in report["blockers"]


def test_round194_no_load_only_validation_blocks_even_when_authorized() -> None:
    calls = {"count": 0}

    def guard(**kwargs):
        calls["count"] += 1
        return {"status": "explicit_load_succeeded", "vectors_loaded": True, "blockers": []}

    engine = SimpleNamespace()
    report = integration.apply_round194_guarded_medium30k_runtime_load(
        engine,
        operator_validation=_no_load_validation(),
        load_authorized=True,
        guard_func=guard,
    )

    assert report["status"] == "blocked_operator_validation_not_green"
    assert report["operator_validation_green"] is False
    assert "operator_validation_attempt_load_required" in report["operator_validation_blockers"]
    assert "operator_explicit_load_probe_not_green" in report["operator_validation_blockers"]
    assert report["guard_called"] is False
    assert calls["count"] == 0
    assert not hasattr(engine, "fasttext_embedding")


def test_round194_green_validation_and_authorization_calls_guard_and_attaches() -> None:
    RecordingAdapter.constructed = 0
    RecordingAdapter.loaded = 0
    engine = SimpleNamespace()

    report = integration.apply_round194_guarded_medium30k_runtime_load(
        engine,
        operator_validation=_green_validation(),
        load_authorized=True,
        artifact_dir="/operator/local/subset_medium_30k",
        adapter_factory=RecordingAdapter,
    )

    assert report["status"] == "guarded_runtime_load_attached"
    assert report["operator_validation_green"] is True
    assert report["guard_called"] is True
    assert report["attached_to_engine"] is True
    assert report["artifact_dir"] == "/operator/local/subset_medium_30k"
    assert report["guard_report"]["attempt_load_requested"] is True
    assert report["guard_report"]["attach_to_engine_requested"] is True
    assert RecordingAdapter.constructed == 1
    assert RecordingAdapter.loaded == 1
    assert engine.fasttext_embedding.__class__.__name__ == "RecordingAdapter"
    assert report["production_persistence_enabled"] is False
    assert report["runtime_mapping_enabled_default"] is False
    assert report["enforcement_enabled"] is False


def test_round195_196_reports_preserve_operator_and_validation_contracts() -> None:
    plan = integration.build_round195_focused_integration_test_plan()
    delta = integration.build_round196_validation_delta_recommendation(
        focused_tests_status="passed",
        compileall_status="passed",
        collect_only_status="passed",
        broader_pytest_status="red_baseline_expected",
        broader_failure_count=205,
    )

    assert plan["version"] == integration.ROUND195_FOCUSED_INTEGRATION_TEST_PLAN_VERSION
    assert "tests/test_v3_round192_196_guarded_medium30k_integration.py" in plan["focused_test_command"]
    assert plan["real_vectors_required"] is False
    assert plan["fake_vectors_allowed"] is False
    assert delta["version"] == integration.ROUND196_VALIDATION_DELTA_RECOMMENDATION_VERSION
    assert delta["rounds_completed"] == [192, 193, 194, 195, 196]
    assert delta["operator_local_validation_command"] == "python scripts/operator_validate_medium30k.py --attempt-load"
    assert delta["broader_failure_count"] == 205
    assert delta["recommendation"] == "operator_run_build_full_engine_with_green_validation_and_explicit_load_authorization_then_remeasure_seed_vector_and_eve_specific_cascades"


def test_main_build_full_engine_forwards_only_explicit_operator_authorization(monkeypatch) -> None:
    import adapters.medium30k_runtime_load_integration as runtime_integration
    import main

    calls: list[dict] = []

    def fake_apply(engine, **kwargs):
        calls.append(dict(kwargs))
        return {
            "status": "guarded_runtime_load_attached" if kwargs.get("load_authorized") else "default_no_load_explicit_authorization_required",
            "guard_called": bool(kwargs.get("load_authorized")),
            "production_persistence_enabled": False,
            "runtime_mapping_enabled_default": False,
            "enforcement_enabled": False,
        }

    monkeypatch.setattr(runtime_integration, "apply_round194_guarded_medium30k_runtime_load", fake_apply)

    default_engine = main.build_full_engine()
    assert calls[-1]["operator_validation"] is None
    assert calls[-1]["load_authorized"] is False
    assert default_engine.medium30k_runtime_load_report["guard_called"] is False

    authorized_engine = main.build_full_engine(
        operator_medium30k_validation=_green_validation(),
        operator_medium30k_load_authorized=True,
        operator_medium30k_artifact_dir="/operator/local/subset_medium_30k",
    )
    assert calls[-1]["operator_validation"]["status"] == "operator_medium30k_validation_green"
    assert calls[-1]["load_authorized"] is True
    assert calls[-1]["artifact_dir"] == "/operator/local/subset_medium_30k"
    assert authorized_engine.medium30k_runtime_load_report["guard_called"] is True


def test_round197_accepts_operator_codespaces_guarded_integration_result() -> None:
    runtime_load_report = {
        "status": "guarded_runtime_load_attached",
        "attached_to_engine": True,
        "guard_called": True,
        "blockers": [],
        "production_persistence_enabled": False,
        "runtime_mapping_enabled_default": False,
        "enforcement_enabled": False,
        "agp_bypass_used": False,
        "vectors_committed": False,
        "dummy_vectors_created": False,
    }

    result = integration.build_round197_operator_guarded_integration_result(
        operator_validation=_green_validation(),
        runtime_load_report=runtime_load_report,
        engine_surface={"self_embedding_present": True, "self_embedding_backup_present": True},
    )

    assert result["version"] == integration.ROUND197_OPERATOR_GUARDED_INTEGRATION_RESULT_VERSION
    assert result["round"] == 197
    assert result["operator_validation_green"] is True
    assert result["build_full_engine_operator_authorized_path_succeeded"] is True
    assert result["medium30k_runtime_attached_to_engine"] is True
    assert result["blockers"] == []
    assert result["vectors_committed"] is False
    assert result["dummy_vectors_created"] is False
    assert result["production_persistence_enabled"] is False
    assert result["runtime_mapping_enabled_default"] is False
    assert result["enforcement_enabled"] is False
    assert result["agp_bypass_used"] is False
    assert result["next_recommendation"] == "remeasure_focused_eve_specific_vector_self_learning_cascade_before_broader_repairs"


def test_round197_blocks_if_runtime_load_report_is_not_attached() -> None:
    result = integration.build_round197_operator_guarded_integration_result(
        operator_validation=_green_validation(),
        runtime_load_report={"status": "default_no_load_explicit_authorization_required", "attached_to_engine": False, "guard_called": False},
        engine_surface={"self_embedding_present": True, "self_embedding_backup_present": True},
    )

    assert result["accepted_as_guarded_integration_evidence"] is False
    assert "runtime_load_status_not_attached" in result["blockers"]
    assert "runtime_load_not_attached_to_engine" in result["blockers"]
    assert "runtime_load_guard_not_called" in result["blockers"]
