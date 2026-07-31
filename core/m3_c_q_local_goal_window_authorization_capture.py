"""M3-C-Q local goal-window authorization capture.

This module turns one already canonical and locally reviewed M3-C-O private
package binding into one single-use private M3-C-P authorization-pin artifact.
It does not activate the pin, open the M3-C-O operator seam, build an engine, or
execute a private-device observation.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Mapping

from core.m3_c_p_private_device_goal_window_authorization_pin import (
    ACCEPTED_M3_C_O_IMPLEMENTATION_EVIDENCE,
    M3CPAuthorizationPinError,
    M3CPLocalReviewedAuthorizationPin,
    M3CPOperatorAuthorizationBinding,
)

CAPTURE_RECEIPT_SCHEMA = "eve.m3-c-q.local-authorization-capture-receipt.v1"
CAPTURE_MANIFEST_SCHEMA = "eve.m3-c-q.local-authorization-capture-manifest.v1"


class M3CQLocalAuthorizationCaptureError(RuntimeError):
    """Fail-closed local authorization capture error."""


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _text_digest(value: str) -> str:
    if not isinstance(value, str):
        raise M3CQLocalAuthorizationCaptureError("text digest input must be str")
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _require_sha256(value: str, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise M3CQLocalAuthorizationCaptureError(
            f"{field} must be a lowercase 64-character SHA-256"
        )
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], *, field: str) -> None:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise M3CQLocalAuthorizationCaptureError(f"{field} keys do not match schema")


def private_output_path_digest(path: str | Path) -> str:
    candidate = Path(path)
    if not candidate.is_absolute():
        raise M3CQLocalAuthorizationCaptureError("private output path must be absolute")
    return _text_digest(str(candidate.resolve()))


def build_local_reviewed_authorization_pin(
    binding: M3CPOperatorAuthorizationBinding,
) -> M3CPLocalReviewedAuthorizationPin:
    if not isinstance(binding, M3CPOperatorAuthorizationBinding):
        raise M3CQLocalAuthorizationCaptureError("authorization binding type mismatch")
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


def local_pin_from_mapping(
    value: Mapping[str, Any],
) -> M3CPLocalReviewedAuthorizationPin:
    expected = {item.name for item in fields(M3CPLocalReviewedAuthorizationPin)}
    _exact_keys(value, expected, field="local reviewed authorization pin")
    try:
        return M3CPLocalReviewedAuthorizationPin(**dict(value))
    except M3CPAuthorizationPinError as exc:
        raise M3CQLocalAuthorizationCaptureError(
            "local reviewed authorization pin is invalid"
        ) from exc


@dataclass(frozen=True, slots=True)
class LocalAuthorizationCaptureReceipt:
    implementation_head: str
    implementation_evidence_digest: str
    binding_digest: str
    authorization_digest: str
    package_digest: str
    review_artifact_digest: str
    path_binding_digest: str
    rollback_digest: str
    mapping_digest: str
    evaluator_digest: str
    reviewer_id_digest: str
    pin_digest: str
    private_output_path_digest: str
    private_pin_file_sha256: str
    local_human_reviewed: bool = True
    single_use_output_created: bool = True
    active_local_authorization_installed: bool = False
    operator_pin_opened: bool = False
    operator_executed: bool = False
    existing_m3_c_j_database_accessed: bool = False
    raw_private_text_or_path_public: bool = False
    legacy_goal_authority_transferred: bool = False
    legacy_migration_authorized: bool = False
    action_authorized: bool = False
    scheduler_authorized: bool = False
    speech_authorized: bool = False
    m3_e_authority_open: bool = False
    schema_version: str = CAPTURE_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        if self.implementation_head != (
            ACCEPTED_M3_C_O_IMPLEMENTATION_EVIDENCE.exact_head
        ):
            raise M3CQLocalAuthorizationCaptureError(
                "capture receipt implementation head mismatch"
            )
        for name in (
            "implementation_evidence_digest",
            "binding_digest",
            "authorization_digest",
            "package_digest",
            "review_artifact_digest",
            "path_binding_digest",
            "rollback_digest",
            "mapping_digest",
            "evaluator_digest",
            "reviewer_id_digest",
            "pin_digest",
            "private_output_path_digest",
            "private_pin_file_sha256",
        ):
            _require_sha256(getattr(self, name), field=name)
        if (
            self.schema_version != CAPTURE_RECEIPT_SCHEMA
            or not self.local_human_reviewed
            or not self.single_use_output_created
            or self.active_local_authorization_installed
            or self.operator_pin_opened
            or self.operator_executed
            or self.existing_m3_c_j_database_accessed
            or self.raw_private_text_or_path_public
            or self.legacy_goal_authority_transferred
            or self.legacy_migration_authorized
            or self.action_authorized
            or self.scheduler_authorized
            or self.speech_authorized
            or self.m3_e_authority_open
        ):
            raise M3CQLocalAuthorizationCaptureError(
                "capture receipt escaped private pin-artifact scope"
            )

    def to_mapping(self) -> dict[str, Any]:
        return {item.name: getattr(self, item.name) for item in fields(self)}

    @property
    def receipt_digest(self) -> str:
        return _digest(self.to_mapping())


def _file_sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def capture_local_reviewed_authorization(
    binding: M3CPOperatorAuthorizationBinding,
    *,
    private_output_path: str | Path,
    human_review_confirmed: bool,
) -> tuple[M3CPLocalReviewedAuthorizationPin, LocalAuthorizationCaptureReceipt]:
    if human_review_confirmed is not True:
        raise M3CQLocalAuthorizationCaptureError(
            "explicit local human review confirmation is required"
        )
    output = Path(private_output_path)
    output_digest = private_output_path_digest(output)
    if output.exists():
        raise M3CQLocalAuthorizationCaptureError(
            "single-use local authorization output already exists"
        )
    if not output.parent.exists() or not output.parent.is_dir():
        raise M3CQLocalAuthorizationCaptureError(
            "private output parent directory must already exist"
        )
    pin = build_local_reviewed_authorization_pin(binding)
    payload = _canonical(pin.to_mapping()) + "\n"
    try:
        with output.open("x", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if os.name != "nt":
            os.chmod(output, 0o600)
        readback_text = output.read_text(encoding="utf-8")
        readback_value = json.loads(readback_text)
    except (OSError, json.JSONDecodeError) as exc:
        raise M3CQLocalAuthorizationCaptureError(
            "private local authorization artifact write/readback failed"
        ) from exc
    if readback_text != payload or not isinstance(readback_value, Mapping):
        raise M3CQLocalAuthorizationCaptureError(
            "private local authorization artifact is not canonical"
        )
    restored = local_pin_from_mapping(readback_value)
    if restored != pin or restored.pin_digest != pin.pin_digest:
        raise M3CQLocalAuthorizationCaptureError(
            "private local authorization artifact readback mismatch"
        )
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
        reviewer_id_digest=_text_digest(binding.reviewer_id),
        pin_digest=pin.pin_digest,
        private_output_path_digest=output_digest,
        private_pin_file_sha256=_file_sha256(output),
    )
    return pin, receipt


def authorization_capture_manifest() -> dict[str, Any]:
    return {
        "active_local_authorization_installed": False,
        "actual_private_device_execution": False,
        "capture_requires_canonical_private_package": True,
        "default_runtime_integration": False,
        "existing_m3_c_j_database_access": False,
        "legacy_goal_authority_transfer": False,
        "legacy_migration_authorization": False,
        "operator_pin_opened": False,
        "private_pin_output_single_use": True,
        "raw_private_text_or_path_public": False,
        "schema_version": CAPTURE_MANIFEST_SCHEMA,
    }
