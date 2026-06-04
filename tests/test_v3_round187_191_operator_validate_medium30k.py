from __future__ import annotations

import json
from pathlib import Path

import scripts.operator_validate_medium30k as validator


def _green_verification() -> dict:
    return {
        "ready": True,
        "status": "operator_artifact_verified_green",
        "blockers": [],
        "files": {
            "vectors": {
                "exists": True,
                "sha256": validator.EXPECTED_MEDIUM30K_VECTORS_SHA256,
                "expected_sha256": validator.EXPECTED_MEDIUM30K_VECTORS_SHA256,
                "sha256_match": True,
            }
        },
        "vectors": {"shape": [30000, 300], "dtype": "float32", "shape_match": True, "dtype_match": True},
        "manifest_consistency": {"consistent": True},
    }


def _green_readiness() -> dict:
    return {
        "status": "ready_for_explicit_operator_artifact_load",
        "ready": True,
        "load_should_be_attempted": True,
        "readiness_blockers": [],
    }


def _green_preflight() -> dict:
    return {
        "status": "load_dependent_repair_preflight_green",
        "ready": True,
        "hard_block": False,
        "blockers": [],
    }


def _load_report(*, attempt_load: bool) -> dict:
    return {
        "status": "explicit_load_succeeded" if attempt_load else "guard_ready_explicit_load_not_attempted",
        "attempt_load_requested": attempt_load,
        "vectors_loaded": attempt_load,
        "blockers": [],
        "production_persistence_enabled": False,
        "runtime_mapping_enabled_default": False,
        "enforcement_enabled": False,
        "agp_bypass_used": False,
    }


def test_round187_188_default_command_is_compact_json_no_load(monkeypatch, capsys) -> None:
    calls: dict[str, object] = {}

    monkeypatch.setattr(validator, "verify_operator_subset_artifact", lambda **kwargs: _green_verification())
    monkeypatch.setattr(validator, "_git_status_safety", lambda repo, artifact_dir: {"safe": True, "lines": []})
    monkeypatch.setattr(validator, "build_seed_vector_artifact_readiness_gate", lambda **kwargs: _green_readiness())
    monkeypatch.setattr(validator, "build_round164_load_dependent_repair_preflight", lambda **kwargs: _green_preflight())

    def guarded(**kwargs):
        calls["attempt_load"] = kwargs["attempt_load"]
        calls["artifact_dir"] = kwargs["artifact_dir"]
        return _load_report(attempt_load=kwargs["attempt_load"])

    monkeypatch.setattr(validator, "guarded_explicit_medium30k_load", guarded)

    exit_code = validator.main([])
    captured = capsys.readouterr().out.strip()
    payload = json.loads(captured)

    assert exit_code == 0
    assert "\n" not in captured
    assert payload["status"] == "operator_medium30k_validation_green"
    assert payload["one_command_workflow"] == "python scripts/operator_validate_medium30k.py"
    assert payload["default_read_only_no_load"] is True
    assert payload["attempt_load_requested"] is False
    assert payload["explicit_load"]["vectors_loaded"] is False
    assert calls["attempt_load"] is False
    assert payload["production_persistence_enabled"] is False
    assert payload["runtime_mapping_enabled_default"] is False
    assert payload["enforcement_enabled"] is False
    assert payload["agp_bypass_used"] is False


def test_round188_missing_artifact_fails_closed_with_nonzero(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        validator,
        "verify_operator_subset_artifact",
        lambda **kwargs: {"ready": False, "status": "blocked_operator_artifact_required", "blockers": ["operator_artifact_files_missing"]},
    )
    monkeypatch.setattr(validator, "_git_status_safety", lambda repo, artifact_dir: {"safe": True, "lines": []})
    monkeypatch.setattr(
        validator,
        "build_seed_vector_artifact_readiness_gate",
        lambda **kwargs: {"status": "blocked_operator_artifact_required", "ready": False, "load_should_be_attempted": False, "readiness_blockers": ["operator_artifact_files_missing"]},
    )
    monkeypatch.setattr(
        validator,
        "build_round164_load_dependent_repair_preflight",
        lambda **kwargs: {"status": "hard_block_load_dependent_repair_until_artifacts_ready", "ready": False, "hard_block": True, "blockers": ["operator_artifact_files_missing"]},
    )
    monkeypatch.setattr(
        validator,
        "guarded_explicit_medium30k_load",
        lambda **kwargs: {"status": "blocked_preflight_not_green", "vectors_loaded": False, "blockers": ["operator_artifact_files_missing"]},
    )

    report = validator.run_validation(artifact_dir=tmp_path / "missing", attempt_load=False, repo_root=tmp_path)

    assert report["success"] is False
    assert report["exit_code"] == 1
    assert report["status"] == "blocked_operator_medium30k_validation_failed"
    assert "operator_artifact_files_missing" in report["blockers"]
    assert report["explicit_load"]["vectors_loaded"] is False


def test_round189_attempt_load_is_opt_in_and_guarded(monkeypatch, tmp_path: Path) -> None:
    calls: dict[str, object] = {}
    monkeypatch.setattr(validator, "verify_operator_subset_artifact", lambda **kwargs: _green_verification())
    monkeypatch.setattr(validator, "_git_status_safety", lambda repo, artifact_dir: {"safe": True, "lines": []})
    monkeypatch.setattr(validator, "build_seed_vector_artifact_readiness_gate", lambda **kwargs: _green_readiness())
    monkeypatch.setattr(validator, "build_round164_load_dependent_repair_preflight", lambda **kwargs: _green_preflight())

    def guarded(**kwargs):
        calls.update(kwargs)
        return _load_report(attempt_load=kwargs["attempt_load"])

    monkeypatch.setattr(validator, "guarded_explicit_medium30k_load", guarded)

    report = validator.run_validation(artifact_dir=tmp_path / "subset_medium_30k", attempt_load=True, repo_root=tmp_path)

    assert report["success"] is True
    assert report["exit_code"] == 0
    assert report["attempt_load_requested"] is True
    assert report["default_read_only_no_load"] is False
    assert calls["attempt_load"] is True
    assert calls["attach_to_engine"] is False
    assert report["explicit_load"]["vectors_loaded"] is True


def test_round189_attempt_load_failure_returns_validation_failure(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(validator, "verify_operator_subset_artifact", lambda **kwargs: _green_verification())
    monkeypatch.setattr(validator, "_git_status_safety", lambda repo, artifact_dir: {"safe": True, "lines": []})
    monkeypatch.setattr(validator, "build_seed_vector_artifact_readiness_gate", lambda **kwargs: _green_readiness())
    monkeypatch.setattr(validator, "build_round164_load_dependent_repair_preflight", lambda **kwargs: _green_preflight())
    monkeypatch.setattr(
        validator,
        "guarded_explicit_medium30k_load",
        lambda **kwargs: {"status": "explicit_load_failed_closed", "vectors_loaded": False, "blockers": ["explicit_load_exception"]},
    )

    report = validator.run_validation(artifact_dir=tmp_path / "subset_medium_30k", attempt_load=True, repo_root=tmp_path)

    assert report["success"] is False
    assert report["exit_code"] == 1
    assert "explicit_load_not_successful" in report["blockers"]
    assert "explicit_load_exception" in report["blockers"]


def test_round190_191_expected_contract_and_git_status_paths(monkeypatch, tmp_path: Path) -> None:
    captured_paths: dict[str, object] = {}

    monkeypatch.setattr(validator, "verify_operator_subset_artifact", lambda **kwargs: _green_verification())
    monkeypatch.setattr(validator, "build_seed_vector_artifact_readiness_gate", lambda **kwargs: _green_readiness())
    monkeypatch.setattr(validator, "build_round164_load_dependent_repair_preflight", lambda **kwargs: _green_preflight())
    monkeypatch.setattr(validator, "guarded_explicit_medium30k_load", lambda **kwargs: _load_report(attempt_load=False))

    def fake_git(repo, artifact_dir):
        captured_paths["artifact_dir"] = artifact_dir
        return {
            "safe": True,
            "lines": [],
            "guarded_paths": [
                "_operator_artifacts",
                str(artifact_dir / "vectors.npy"),
                str(artifact_dir / "vocab.txt"),
                str(artifact_dir / "subset_manifest.json"),
                "seeds/subsets",
            ],
        }

    monkeypatch.setattr(validator, "_git_status_safety", fake_git)

    report = validator.run_validation(artifact_dir="_operator_artifacts/subset_medium_30k", attempt_load=False, repo_root=tmp_path)

    assert report["rounds_completed"] == [187, 188, 189, 190, 191]
    assert report["expected"]["required_files"] == ["vocab.txt", "vectors.npy", "subset_manifest.json"]
    assert report["expected"]["vectors_shape"] == [30000, 300]
    assert report["expected"]["vectors_dtype"] == "float32"
    assert report["expected"]["vectors_sha256_bare"] == "f228cbca9816d539ce9532e63fbb1ea95e4c66a7c3df286c788f817e2055bd05"
    assert "seeds/subsets" in report["git_status_safety"]["guarded_paths"]
    assert "_operator_artifacts/subset_medium_30k/vectors.npy" in report["git_status_safety"]["guarded_paths"]
