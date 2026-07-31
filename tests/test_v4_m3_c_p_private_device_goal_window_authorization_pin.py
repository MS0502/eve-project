from __future__ import annotations

import ast
import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

import core.m3_c_o_private_device_goal_dual_read_operator as m3_c_o_operator
import core.m3_c_p_private_device_goal_window_authorization_pin as pin_module
from core.m3_c_p_private_device_goal_window_authorization_pin import (
    ACCEPTED_M3_C_O_IMPLEMENTATION_EVIDENCE,
    M3CPAuthorizationPinError,
    M3CPLocalReviewedAuthorizationPin,
    M3CPOperatorAuthorizationBinding,
    active_local_reviewed_authorization_pin,
    authorization_pin_preflight_manifest,
    reviewed_operator_pin_session,
    verify_active_local_pin,
)

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "core/m3_c_p_private_device_goal_window_authorization_pin.py"
DESIGN = ROOT / "docs/audit/M3_C_P_PRIVATE_DEVICE_GOAL_WINDOW_AUTHORIZATION_PIN.md"
REUSE = ROOT / "docs/audit/M3_C_O_PR239_VALIDATION_REUSE_PIN.json"


def _d(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _binding() -> M3CPOperatorAuthorizationBinding:
    return M3CPOperatorAuthorizationBinding(
        implementation_head=pin_module.PR239_EXACT_HEAD,
        authorization_digest=_d("authorization"),
        package_digest=_d("package"),
        review_artifact_digest=_d("review"),
        path_binding_digest=_d("paths"),
        rollback_digest=_d("rollback"),
        mapping_digest=_d("mapping"),
        evaluator_digest=_d("evaluator"),
        reviewer_id="kim-minseok",
    )


def _pin(binding: M3CPOperatorAuthorizationBinding) -> M3CPLocalReviewedAuthorizationPin:
    return M3CPLocalReviewedAuthorizationPin(
        implementation_evidence_digest=(
            ACCEPTED_M3_C_O_IMPLEMENTATION_EVIDENCE.evidence_digest
        ),
        binding_digest=binding.binding_digest,
        authorization_digest=binding.authorization_digest,
        package_digest=binding.package_digest,
        review_artifact_digest=binding.review_artifact_digest,
        path_binding_digest=binding.path_binding_digest,
        rollback_digest=binding.rollback_digest,
        mapping_digest=binding.mapping_digest,
        evaluator_digest=binding.evaluator_digest,
        reviewer_id=binding.reviewer_id,
    )


def test_accepted_pr239_implementation_evidence_is_exact_and_complete():
    evidence = ACCEPTED_M3_C_O_IMPLEMENTATION_EVIDENCE
    assert evidence.pr_number == 239
    assert evidence.base_sha == "9a26f6040679013066425887c3bcee5a2846a025"
    assert evidence.exact_head == "57da278ce01e04257efc8a84933092715b371dec"
    assert evidence.exact_run == 30643858724
    assert evidence.focused_passed == 8
    assert evidence.full_passed == 3377
    assert evidence.m0_byte_identical is True
    assert evidence.m2_b_valid is True
    assert evidence.m2_b_errors == 0
    assert evidence.m2_e_run == 30643857677
    assert evidence.m2_e_passed == evidence.m2_e_required == 6
    assert evidence.artifact_sha256 == (
        "097f8025b587bd77156eb966fb4cbf584f0a437b2fdbbbfaff1d7c4200a88068"
    )
    assert evidence.merge_sha == "f0a01b8e138dd1111c323dd54bf92c8527eb5b30"


def test_checked_in_local_authorization_and_operator_pins_remain_absent():
    with pytest.raises(M3CPAuthorizationPinError, match="no active local reviewed"):
        active_local_reviewed_authorization_pin()
    assert pin_module._ACTIVE_LOCAL_REVIEWED_AUTHORIZATION_PIN is None
    assert m3_c_o_operator._ACTIVE_REVIEWED_OPERATOR_IMPLEMENTATION_HEAD is None
    assert m3_c_o_operator._ACTIVE_REVIEWED_OPERATOR_AUTHORIZATION_DIGEST is None
    manifest = authorization_pin_preflight_manifest()
    assert manifest["active_local_authorization_present"] is False
    assert manifest["concrete_private_binding_present"] is False
    assert manifest["actual_private_device_execution"] is False
    assert manifest["default_runtime_integration"] is False


def test_local_pin_rejects_every_downstream_authority_escape():
    binding = _binding()
    valid = _pin(binding)
    for field in (
        "existing_m3_c_j_path_reuse_authorized",
        "raw_private_text_or_path_publication_authorized",
        "legacy_goal_authority_transfer_authorized",
        "legacy_migration_authorized",
        "action_authorized",
        "scheduler_authorized",
        "speech_authorized",
        "m3_e_authority_open",
    ):
        with pytest.raises(M3CPAuthorizationPinError, match="escaped"):
            replace(valid, **{field: True})


def test_active_pin_must_match_every_private_binding_digest(
    monkeypatch: pytest.MonkeyPatch,
):
    binding = _binding()
    pin = _pin(binding)
    monkeypatch.setattr(
        pin_module,
        "_ACTIVE_LOCAL_REVIEWED_AUTHORIZATION_PIN",
        pin,
    )
    assert verify_active_local_pin(binding) is pin
    with pytest.raises(M3CPAuthorizationPinError, match="differs"):
        verify_active_local_pin(
            replace(binding, package_digest=_d("different-package"))
        )


def test_reviewed_session_opens_exact_two_operator_pins_and_restores(
    monkeypatch: pytest.MonkeyPatch,
):
    binding = _binding()
    pin = _pin(binding)
    monkeypatch.setattr(
        pin_module,
        "_ACTIVE_LOCAL_REVIEWED_AUTHORIZATION_PIN",
        pin,
    )
    with reviewed_operator_pin_session(binding) as active:
        assert active is pin
        assert (
            m3_c_o_operator._ACTIVE_REVIEWED_OPERATOR_IMPLEMENTATION_HEAD
            == pin_module.PR239_EXACT_HEAD
        )
        assert (
            m3_c_o_operator._ACTIVE_REVIEWED_OPERATOR_AUTHORIZATION_DIGEST
            == binding.authorization_digest
        )
    assert m3_c_o_operator._ACTIVE_REVIEWED_OPERATOR_IMPLEMENTATION_HEAD is None
    assert m3_c_o_operator._ACTIVE_REVIEWED_OPERATOR_AUTHORIZATION_DIGEST is None


def test_reviewed_session_restores_after_exception_and_refuses_reentry(
    monkeypatch: pytest.MonkeyPatch,
):
    binding = _binding()
    pin = _pin(binding)
    monkeypatch.setattr(
        pin_module,
        "_ACTIVE_LOCAL_REVIEWED_AUTHORIZATION_PIN",
        pin,
    )
    with pytest.raises(RuntimeError, match="synthetic stop"):
        with reviewed_operator_pin_session(binding):
            raise RuntimeError("synthetic stop")
    assert m3_c_o_operator._ACTIVE_REVIEWED_OPERATOR_IMPLEMENTATION_HEAD is None
    assert m3_c_o_operator._ACTIVE_REVIEWED_OPERATOR_AUTHORIZATION_DIGEST is None

    monkeypatch.setattr(
        m3_c_o_operator,
        "_ACTIVE_REVIEWED_OPERATOR_IMPLEMENTATION_HEAD",
        pin_module.PR239_EXACT_HEAD,
    )
    with pytest.raises(M3CPAuthorizationPinError, match="already open"):
        with reviewed_operator_pin_session(binding):
            pytest.fail("session must not open")


def test_module_has_no_io_command_or_ambient_runtime_integration_surface():
    source = MODULE.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(MODULE))
    imports = {
        node.names[0].name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
    }
    imported_from = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert not imports.intersection(
        {"os", "pathlib", "sqlite3", "subprocess", "socket", "requests"}
    )
    assert not imported_from.intersection(
        {"os", "pathlib", "sqlite3", "subprocess", "socket", "requests"}
    )
    for token in (
        "open(",
        "read_text(",
        "write_text(",
        "build_full_engine",
        "execute_private_device_goal_dual_read_window(",
    ):
        assert token not in source

    design = DESIGN.read_text(encoding="utf-8")
    reuse = REUSE.read_text(encoding="utf-8")
    for token in (
        pin_module.PR239_EXACT_HEAD,
        pin_module.PR239_ARTIFACT_SHA256,
        pin_module.PR239_MERGE_SHA,
        "active local authorization: false",
        "actual private-device execution: false",
    ):
        assert token in design or token in reuse
    assert '"chat_change":true' in reuse
    assert '"rerun_full_suite_pr_239":true' in reuse
    assert '"rerun_m2_e_pr_239":true' in reuse
