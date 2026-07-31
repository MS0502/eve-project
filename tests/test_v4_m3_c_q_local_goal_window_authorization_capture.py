from __future__ import annotations

import ast
import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path

import pytest

from core.m3_c_p_private_device_goal_window_authorization_pin import (
    ACCEPTED_M3_C_O_IMPLEMENTATION_EVIDENCE,
    M3CPOperatorAuthorizationBinding,
)
from core.m3_c_q_local_goal_window_authorization_capture import (
    M3CQLocalAuthorizationCaptureError,
    authorization_capture_manifest,
    build_local_reviewed_authorization_pin,
    capture_local_reviewed_authorization,
    local_pin_from_mapping,
)

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "core/m3_c_q_local_goal_window_authorization_capture.py"
SCRIPT = ROOT / "scripts/operator/m3_c_q_capture_local_goal_window_authorization.py"
DESIGN = ROOT / "docs/audit/M3_C_Q_LOCAL_GOAL_WINDOW_AUTHORIZATION_CAPTURE.md"
REUSE = ROOT / "docs/audit/M3_C_P_PR240_VALIDATION_REUSE_PIN.json"


def _d(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _binding() -> M3CPOperatorAuthorizationBinding:
    return M3CPOperatorAuthorizationBinding(
        implementation_head=ACCEPTED_M3_C_O_IMPLEMENTATION_EVIDENCE.exact_head,
        authorization_digest=_d("authorization"),
        package_digest=_d("package"),
        review_artifact_digest=_d("review"),
        path_binding_digest=_d("paths"),
        rollback_digest=_d("rollback"),
        mapping_digest=_d("mapping"),
        evaluator_digest=_d("evaluator"),
        reviewer_id="kim-minseok",
    )


def test_build_pin_binds_exact_implementation_and_grants_nothing():
    binding = _binding()
    pin = build_local_reviewed_authorization_pin(binding)
    assert pin.implementation_evidence_digest == (
        ACCEPTED_M3_C_O_IMPLEMENTATION_EVIDENCE.evidence_digest
    )
    assert pin.binding_digest == binding.binding_digest
    assert pin.package_digest == binding.package_digest
    assert pin.reviewer_id == binding.reviewer_id
    assert pin.private_device_single_use_only is True
    assert pin.existing_m3_c_j_path_reuse_authorized is False
    assert pin.raw_private_text_or_path_publication_authorized is False
    assert pin.legacy_goal_authority_transfer_authorized is False
    assert pin.legacy_migration_authorized is False
    assert pin.action_authorized is False
    assert pin.scheduler_authorized is False
    assert pin.speech_authorized is False
    assert pin.m3_e_authority_open is False


def test_capture_writes_one_canonical_private_pin_and_public_safe_receipt(
    tmp_path: Path,
):
    binding = _binding()
    output = (tmp_path / "private-pin.json").resolve()
    pin, receipt = capture_local_reviewed_authorization(
        binding,
        private_output_path=output,
        human_review_confirmed=True,
    )
    assert output.is_file()
    text = output.read_text(encoding="utf-8")
    mapping = json.loads(text)
    assert text == json.dumps(
        mapping,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ) + "\n"
    assert local_pin_from_mapping(mapping) == pin
    assert receipt.pin_digest == pin.pin_digest
    assert receipt.binding_digest == binding.binding_digest
    assert receipt.operator_executed is False
    assert receipt.active_local_authorization_installed is False
    assert receipt.operator_pin_opened is False
    assert receipt.existing_m3_c_j_database_accessed is False
    assert receipt.raw_private_text_or_path_public is False
    assert str(output) not in json.dumps(receipt.to_mapping())
    assert binding.reviewer_id not in json.dumps(receipt.to_mapping())
    if os.name != "nt":
        assert output.stat().st_mode & 0o077 == 0


def test_missing_review_and_existing_output_fail_before_overwrite(tmp_path: Path):
    binding = _binding()
    output = (tmp_path / "private-pin.json").resolve()
    with pytest.raises(M3CQLocalAuthorizationCaptureError, match="review confirmation"):
        capture_local_reviewed_authorization(
            binding,
            private_output_path=output,
            human_review_confirmed=False,
        )
    assert not output.exists()

    output.write_text("immutable-first-attempt", encoding="utf-8")
    with pytest.raises(M3CQLocalAuthorizationCaptureError, match="already exists"):
        capture_local_reviewed_authorization(
            binding,
            private_output_path=output,
            human_review_confirmed=True,
        )
    assert output.read_text(encoding="utf-8") == "immutable-first-attempt"


def test_relative_output_and_missing_parent_fail_closed(tmp_path: Path):
    with pytest.raises(M3CQLocalAuthorizationCaptureError, match="absolute"):
        capture_local_reviewed_authorization(
            _binding(),
            private_output_path=Path("relative-pin.json"),
            human_review_confirmed=True,
        )
    missing = (tmp_path / "missing" / "private-pin.json").resolve()
    with pytest.raises(M3CQLocalAuthorizationCaptureError, match="parent"):
        capture_local_reviewed_authorization(
            _binding(),
            private_output_path=missing,
            human_review_confirmed=True,
        )


def test_pin_readback_rejects_extra_keys_and_scope_escape():
    pin = build_local_reviewed_authorization_pin(_binding())
    mapping = pin.to_mapping()
    with pytest.raises(M3CQLocalAuthorizationCaptureError, match="keys"):
        local_pin_from_mapping({**mapping, "unexpected": True})
    escaped = replace(pin, action_authorized=True).to_mapping() if False else dict(mapping)
    escaped["action_authorized"] = True
    with pytest.raises(M3CQLocalAuthorizationCaptureError, match="invalid"):
        local_pin_from_mapping(escaped)


def test_capture_manifest_keeps_runtime_and_authority_closed():
    manifest = authorization_capture_manifest()
    assert manifest["capture_requires_canonical_private_package"] is True
    assert manifest["private_pin_output_single_use"] is True
    assert manifest["active_local_authorization_installed"] is False
    assert manifest["operator_pin_opened"] is False
    assert manifest["actual_private_device_execution"] is False
    assert manifest["default_runtime_integration"] is False
    assert manifest["existing_m3_c_j_database_access"] is False
    assert manifest["legacy_goal_authority_transfer"] is False
    assert manifest["legacy_migration_authorization"] is False


def test_explicit_script_never_installs_pin_or_executes_operator_and_reuse_is_durable():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "--expected-head" in source
    assert "--package-file" in source
    assert "--expected-package-digest" in source
    assert "--pin-output" in source
    assert "--reviewed" in source
    assert "read_canonical_private_package" in source
    assert "binding_from_private_package" in source
    assert "capture_local_reviewed_authorization" in source
    assert "build_full_engine" not in source
    assert "execute_private_device_goal_dual_read_window(" not in source
    assert "reviewed_operator_pin_session(" not in source

    module_tree = ast.parse(MODULE.read_text(encoding="utf-8"), filename=str(MODULE))
    imports = {
        node.names[0].name
        for node in ast.walk(module_tree)
        if isinstance(node, ast.Import)
    }
    assert not imports.intersection({"sqlite3", "socket", "requests", "subprocess"})

    design = DESIGN.read_text(encoding="utf-8")
    reuse = REUSE.read_text(encoding="utf-8")
    for token in (
        "active local authorization installed: false",
        "operator execution: false",
        ACCEPTED_M3_C_O_IMPLEMENTATION_EVIDENCE.exact_head,
        ACCEPTED_M3_C_O_IMPLEMENTATION_EVIDENCE.artifact_sha256,
    ):
        assert token in design or token in reuse
    assert '"chat_change":true' in reuse
    assert '"rerun_full_suite_pr_240":true' in reuse
    assert '"rerun_m2_e_pr_240":true' in reuse
