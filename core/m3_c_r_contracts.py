"""Immutable input, review, and pin contracts for M3-C-R."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from core.m3_c_o_private_device_goal_dual_read_operator import (
    PrivateDeviceGoalDualReadPackage,
    read_canonical_private_package,
)
from core.m3_c_p_private_device_goal_window_authorization_pin import (
    M3CPOperatorAuthorizationBinding,
    binding_from_private_package,
)
from core.m3_c_q_local_goal_window_authorization_capture import (
    LocalAuthorizationCaptureReceipt,
    build_local_reviewed_authorization_pin,
    capture_local_reviewed_authorization,
    local_pin_from_mapping,
    private_output_path_digest,
)

PRIVATE_ROOT_BASENAME = "eve-m3c-private-goal-window-4d22013a-20260801"
PACKAGE_FILENAME = "canonical_goal_window_package.json"
SUMMARY_FILENAME = "package_review_summary.json"
FORBIDDEN_FILENAME = "forbidden_prior_path_digests.json"
REVIEW_CONFIRMATION_FILENAME = "operator_review_confirmation.json"
PIN_FILENAME = "local_authorization_pin.json"
CONSUMED_PIN_FILENAME = "local_authorization_pin.consumed.json"
PIN_RECEIPT_FILENAME = "authorization_capture_receipt.json"
PREFLIGHT_FILENAME = "execution_preflight.json"
WORKING_STORE_FILENAME = "goal_dual_read_window.jsonl"
BASELINE_BACKUP_FILENAME = "legacy_goal_baseline.backup"
SEPARATE_RESTORE_FILENAME = "legacy_goal_baseline.restore"
OPERATOR_RECEIPT_FILENAME = "operator_execution_receipt.json"
PUBLIC_REVIEW_FILENAME = "m3_c_private_device_goal_window_public_review.json"
REVIEW_CONFIRMATION_SCHEMA = "eve.m3-c-r.operator-review-confirmation.v1"
PREFLIGHT_SCHEMA = "eve.m3-c-r.execution-preflight.v1"
PUBLIC_REVIEW_SCHEMA = "eve.m3-c-r.private-device-public-review.v1"
MINIMUM_AVAILABLE_MIB = 3072
EXPECTED_FORBIDDEN_DIGEST_COUNT = 34
EXPECTED_PACKAGE_DIGEST = "bdc250fce7c746d527c378e240ec1fd3b307c3c1763306f43b8f4fafc3bd6c88"
EXPECTED_LAUNCH_HEAD = "4d22013a34760f974b363615133972de47f02bb9"
EXPECTED_OPERATOR_IMPLEMENTATION_HEAD = "57da278ce01e04257efc8a84933092715b371dec"
EXPECTED_REVIEW_SUMMARY: dict[str, Any] = {
    "authorization_digest": "232c8bbfb23dee4cfcfea505a1c731baa02994ba14f876bc381ae3d08853b8c0",
    "baseline_state_digest": "363539371b711689eb322f11ae6e317fc7f6ddbb2d47953c302da8fa3ec8a398",
    "evaluator_digest": "8009866d1f3155307894483217badda1503cfc2cf67cbb742ac27a1f7f4982ab",
    "forbidden_prior_path_digest_count": 34,
    "launch_repository_head": EXPECTED_LAUNCH_HEAD,
    "legacy_goal_authority_transfer_authorized": False,
    "mapping_digest": "a28142ffed814a55f3bc50a1ab5bba16ab3eee2fd5fd1d25f6885e4c92adde3c",
    "operator_implementation_head": EXPECTED_OPERATOR_IMPLEMENTATION_HEAD,
    "package_digest": EXPECTED_PACKAGE_DIGEST,
    "path_binding_digest": "230aa8c0ad91fe98f223b89bda3094cd5e8020ae4a66692b94d9e3a7fcc26a14",
    "probe_count": 4,
    "probe_operation_summary": ["goal_set", "tick", "goal_set", "tick"],
    "raw_private_text_or_path_output": False,
    "review_artifact_digest": "a934841c69cec67e4d2c20f0ca8b8b402db9e95e219d10f860c8497de2ed95fd",
    "rollback_digest": "319308e4ed8ab8dae2b667f44e4b34762cfcb3b375c83979aaa0e8063e6290b2",
    "schema_version": "eve.m3-c.private-package-review-summary.v1",
}


class M3CRResumableOperatorError(RuntimeError):
    """Fail-closed staged private-phone workflow error."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def mapping_digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def text_digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def private_paths(private_root: str | Path) -> dict[str, Path]:
    root = Path(private_root).expanduser()
    if not root.is_absolute():
        raise M3CRResumableOperatorError("private root must be absolute")
    root = root.resolve()
    if root.name != PRIVATE_ROOT_BASENAME:
        raise M3CRResumableOperatorError("private root basename mismatch")
    names = {
        "package": PACKAGE_FILENAME,
        "summary": SUMMARY_FILENAME,
        "forbidden": FORBIDDEN_FILENAME,
        "review_confirmation": REVIEW_CONFIRMATION_FILENAME,
        "pin": PIN_FILENAME,
        "consumed_pin": CONSUMED_PIN_FILENAME,
        "pin_receipt": PIN_RECEIPT_FILENAME,
        "preflight": PREFLIGHT_FILENAME,
        "working_store": WORKING_STORE_FILENAME,
        "baseline_backup": BASELINE_BACKUP_FILENAME,
        "separate_restore": SEPARATE_RESTORE_FILENAME,
        "operator_receipt": OPERATOR_RECEIPT_FILENAME,
        "public_review": PUBLIC_REVIEW_FILENAME,
    }
    return {"root": root, **{key: root / name for key, name in names.items()}}


def _require_private_file(path: Path, field: str) -> Path:
    if not path.is_file() or path.is_symlink():
        raise M3CRResumableOperatorError(f"{field} must be an existing regular file")
    if os.name != "nt" and path.stat().st_mode & 0o077:
        raise M3CRResumableOperatorError(f"{field} must not grant group/other permissions")
    return path


def load_canonical_mapping(path: Path, *, field: str) -> dict[str, Any]:
    _require_private_file(path, field)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise M3CRResumableOperatorError(f"{field} is not valid JSON") from exc
    if not isinstance(value, Mapping):
        raise M3CRResumableOperatorError(f"{field} must be a JSON object")
    return dict(value)


def write_idempotent_canonical(path: Path, value: Mapping[str, Any]) -> bool:
    payload = canonical_json(value) + "\n"
    if path.exists():
        if not path.is_file() or path.is_symlink():
            raise M3CRResumableOperatorError(f"existing staged output is not a regular file: {path.name}")
        if path.read_text(encoding="utf-8") != payload:
            raise M3CRResumableOperatorError(f"conflicting or partial staged output preserved: {path.name}")
        if os.name != "nt" and path.stat().st_mode & 0o077:
            raise M3CRResumableOperatorError(f"existing staged output permissions are not private: {path.name}")
        return False
    if not path.parent.is_dir():
        raise M3CRResumableOperatorError("private root must already exist")
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if os.name != "nt":
            os.chmod(path, 0o600)
    except OSError as exc:
        raise M3CRResumableOperatorError(f"failed to create staged output: {path.name}") from exc
    if path.read_text(encoding="utf-8") != payload:
        raise M3CRResumableOperatorError(f"staged output readback mismatch: {path.name}")
    return True


def _collect_sha256(value: Any, output: set[str]) -> None:
    if isinstance(value, str):
        if len(value) == 64 and all(ch in "0123456789abcdef" for ch in value):
            output.add(value)
    elif isinstance(value, Mapping):
        for item in value.values():
            _collect_sha256(item, output)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            _collect_sha256(item, output)


def read_forbidden_digests(path: Path) -> tuple[str, ...]:
    value = load_canonical_mapping(path, field="forbidden prior-path digests")
    found: set[str] = set()
    _collect_sha256(value, found)
    if len(found) != EXPECTED_FORBIDDEN_DIGEST_COUNT:
        raise M3CRResumableOperatorError("forbidden prior-path digest set must contain exactly 34 unique digests")
    return tuple(sorted(found))


def validate_immutable_inputs(private_root: str | Path) -> tuple[PrivateDeviceGoalDualReadPackage, dict[str, Any], tuple[str, ...]]:
    paths = private_paths(private_root)
    summary = load_canonical_mapping(paths["summary"], field="package review summary")
    if summary != EXPECTED_REVIEW_SUMMARY:
        raise M3CRResumableOperatorError("package review summary differs from the accepted operator review")
    forbidden = read_forbidden_digests(paths["forbidden"])
    _require_private_file(paths["package"], "canonical private package")
    package = read_canonical_private_package(paths["package"])
    if package.package_digest != EXPECTED_PACKAGE_DIGEST or len(package.probes) != 4:
        raise M3CRResumableOperatorError("canonical package differs from accepted review")
    return package, summary, forbidden


def expected_review_confirmation(private_root: str | Path) -> dict[str, Any]:
    paths = private_paths(private_root)
    validate_immutable_inputs(private_root)
    return {
        "authorization_digest": EXPECTED_REVIEW_SUMMARY["authorization_digest"],
        "forbidden_prior_path_digest_count": 34,
        "forbidden_prior_path_file_sha256": file_sha256(paths["forbidden"]),
        "human_reviewed": True,
        "legacy_goal_authority_transfer_authorized": False,
        "m3_e_authority_open": False,
        "package_digest": EXPECTED_PACKAGE_DIGEST,
        "package_file_sha256": file_sha256(paths["package"]),
        "probe_count": 4,
        "raw_private_text_or_path_output": False,
        "review_summary_file_sha256": file_sha256(paths["summary"]),
        "schema_version": REVIEW_CONFIRMATION_SCHEMA,
        "single_use_review_scope": True,
    }


def record_review_confirmation(private_root: str | Path) -> dict[str, Any]:
    paths = private_paths(private_root)
    confirmation = expected_review_confirmation(private_root)
    write_idempotent_canonical(paths["review_confirmation"], confirmation)
    return confirmation


def require_review_confirmation(private_root: str | Path) -> dict[str, Any]:
    paths = private_paths(private_root)
    expected = expected_review_confirmation(private_root)
    actual = load_canonical_mapping(paths["review_confirmation"], field="operator review confirmation")
    if actual != expected:
        raise M3CRResumableOperatorError("operator review confirmation differs from immutable inputs")
    return actual


def _existing_pin_receipt(
    binding: M3CPOperatorAuthorizationBinding,
    pin_path: Path,
    *,
    original_pin_path: Path,
):
    pin = local_pin_from_mapping(load_canonical_mapping(pin_path, field="local authorization pin"))
    expected = build_local_reviewed_authorization_pin(binding)
    if pin != expected:
        raise M3CRResumableOperatorError("existing local authorization pin conflicts with reviewed package")
    receipt = LocalAuthorizationCaptureReceipt(
        implementation_head=binding.implementation_head,
        implementation_evidence_digest=pin.implementation_evidence_digest,
        binding_digest=binding.binding_digest,
        authorization_digest=binding.authorization_digest,
        package_digest=binding.package_digest,
        review_artifact_digest=binding.review_artifact_digest,
        path_binding_digest=binding.path_binding_digest,
        rollback_digest=binding.rollback_digest,
        mapping_digest=binding.mapping_digest,
        evaluator_digest=binding.evaluator_digest,
        reviewer_id_digest=text_digest(binding.reviewer_id),
        pin_digest=pin.pin_digest,
        private_output_path_digest=private_output_path_digest(original_pin_path),
        private_pin_file_sha256=file_sha256(pin_path),
    )
    return pin, receipt


def capture_or_reuse_local_pin(private_root: str | Path) -> dict[str, Any]:
    paths = private_paths(private_root)
    require_review_confirmation(private_root)
    package, _, _ = validate_immutable_inputs(private_root)
    binding = binding_from_private_package(package)
    if paths["consumed_pin"].exists() and paths["pin"].exists():
        raise M3CRResumableOperatorError("both active and consumed local pin paths exist")
    pin_path = paths["consumed_pin"] if paths["consumed_pin"].exists() else paths["pin"]
    if pin_path.exists():
        pin, receipt = _existing_pin_receipt(
            binding, pin_path, original_pin_path=paths["pin"]
        )
    else:
        pin, receipt = capture_local_reviewed_authorization(
            binding, private_output_path=paths["pin"], human_review_confirmed=True
        )
    public = {
        "capture_receipt": receipt.to_mapping(),
        "capture_receipt_digest": receipt.receipt_digest,
        "launch_repository_head": EXPECTED_LAUNCH_HEAD,
        "local_pin_digest": pin.pin_digest,
        "operator_executed": paths["operator_receipt"].exists(),
        "private_output_path_plaintext_public": False,
        "raw_private_text_public": False,
    }
    write_idempotent_canonical(paths["pin_receipt"], public)
    return public
