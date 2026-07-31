"""Default-absent M3-C-P authorization-pin preflight.

This module immutably records the accepted M3-C-O implementation evidence from
PR #239 and defines the exact local authorization binding required by a later
private-device launch.  No concrete private package, path, rollback, reviewer,
or authorization digest is checked in here, so the operator remains
unreachable.

A future isolated local-review pin may instantiate the exact binding and use the
scoped context manager below.  The context manager opens the two M3-C-O module
pins only for one synchronous caller and restores their prior values in
``finally``.  It performs no I/O and issues no operator command.
"""
from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from dataclasses import dataclass, fields
from typing import Any, Iterator

import core.m3_c_o_private_device_goal_dual_read_operator as m3_c_o_operator

IMPLEMENTATION_EVIDENCE_SCHEMA = "eve.m3-c-p.m3-c-o-implementation-evidence.v1"
AUTHORIZATION_BINDING_SCHEMA = "eve.m3-c-p.operator-authorization-binding.v1"
LOCAL_PIN_SCHEMA = "eve.m3-c-p.local-reviewed-authorization-pin.v1"
MANIFEST_SCHEMA = "eve.m3-c-p.authorization-pin-preflight-manifest.v1"

PR239_NUMBER = 239
PR239_BASE_SHA = "9a26f6040679013066425887c3bcee5a2846a025"
PR239_EXACT_HEAD = "57da278ce01e04257efc8a84933092715b371dec"
PR239_EXACT_RUN = 30643858724
PR239_FOCUSED_PASSED = 8
PR239_FULL_PASSED = 3377
PR239_M2E_RUN = 30643857677
PR239_M2E_PASSED = 6
PR239_ARTIFACT_NAME = (
    "exact-head-validation-57da278ce01e04257efc8a84933092715b371dec"
)
PR239_ARTIFACT_SHA256 = (
    "097f8025b587bd77156eb966fb4cbf584f0a437b2fdbbbfaff1d7c4200a88068"
)
PR239_MERGE_SHA = "f0a01b8e138dd1111c323dd54bf92c8527eb5b30"


class M3CPAuthorizationPinError(RuntimeError):
    """Fail-closed M3-C-P implementation-evidence or local-pin error."""


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


def _require_git_sha(value: str, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise M3CPAuthorizationPinError(
            f"{field} must be a lowercase 40-character Git SHA"
        )
    return value


def _require_sha256(value: str, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise M3CPAuthorizationPinError(
            f"{field} must be a lowercase 64-character SHA-256"
        )
    return value


def _to_mapping(value: Any) -> dict[str, Any]:
    return {item.name: getattr(value, item.name) for item in fields(value)}


@dataclass(frozen=True, slots=True)
class M3COImplementationEvidence:
    pr_number: int = PR239_NUMBER
    base_sha: str = PR239_BASE_SHA
    exact_head: str = PR239_EXACT_HEAD
    exact_run: int = PR239_EXACT_RUN
    focused_passed: int = PR239_FOCUSED_PASSED
    full_passed: int = PR239_FULL_PASSED
    m0_byte_identical: bool = True
    m2_b_valid: bool = True
    m2_b_errors: int = 0
    forward_venv: int = 0
    forward_production: int = 0
    forward_monkeypatch: int = 0
    forward_unregistered: int = 0
    m2_e_run: int = PR239_M2E_RUN
    m2_e_passed: int = PR239_M2E_PASSED
    m2_e_required: int = 6
    artifact_name: str = PR239_ARTIFACT_NAME
    artifact_sha256: str = PR239_ARTIFACT_SHA256
    merge_sha: str = PR239_MERGE_SHA
    schema_version: str = IMPLEMENTATION_EVIDENCE_SCHEMA

    def __post_init__(self) -> None:
        if self.pr_number != PR239_NUMBER:
            raise M3CPAuthorizationPinError("implementation evidence PR mismatch")
        _require_git_sha(self.base_sha, field="base_sha")
        _require_git_sha(self.exact_head, field="exact_head")
        _require_git_sha(self.merge_sha, field="merge_sha")
        _require_sha256(self.artifact_sha256, field="artifact_sha256")
        if (
            self.exact_head != PR239_EXACT_HEAD
            or self.base_sha != PR239_BASE_SHA
            or self.merge_sha != PR239_MERGE_SHA
            or self.exact_run != PR239_EXACT_RUN
            or self.focused_passed != PR239_FOCUSED_PASSED
            or self.full_passed != PR239_FULL_PASSED
            or self.m2_e_run != PR239_M2E_RUN
            or self.m2_e_passed != self.m2_e_required
            or self.artifact_name != PR239_ARTIFACT_NAME
            or self.artifact_sha256 != PR239_ARTIFACT_SHA256
            or self.schema_version != IMPLEMENTATION_EVIDENCE_SCHEMA
            or not self.m0_byte_identical
            or not self.m2_b_valid
            or self.m2_b_errors != 0
            or any(
                (
                    self.forward_venv,
                    self.forward_production,
                    self.forward_monkeypatch,
                    self.forward_unregistered,
                )
            )
        ):
            raise M3CPAuthorizationPinError(
                "implementation evidence differs from accepted PR #239"
            )

    def to_mapping(self) -> dict[str, Any]:
        return _to_mapping(self)

    @property
    def evidence_digest(self) -> str:
        return _digest(self.to_mapping())


ACCEPTED_M3_C_O_IMPLEMENTATION_EVIDENCE = M3COImplementationEvidence()


@dataclass(frozen=True, slots=True)
class M3CPOperatorAuthorizationBinding:
    implementation_head: str
    authorization_digest: str
    package_digest: str
    review_artifact_digest: str
    path_binding_digest: str
    rollback_digest: str
    mapping_digest: str
    evaluator_digest: str
    reviewer_id: str
    schema_version: str = AUTHORIZATION_BINDING_SCHEMA

    def __post_init__(self) -> None:
        _require_git_sha(self.implementation_head, field="implementation_head")
        for name in (
            "authorization_digest",
            "package_digest",
            "review_artifact_digest",
            "path_binding_digest",
            "rollback_digest",
            "mapping_digest",
            "evaluator_digest",
        ):
            _require_sha256(getattr(self, name), field=name)
        if self.implementation_head != PR239_EXACT_HEAD:
            raise M3CPAuthorizationPinError(
                "authorization binding targets an unaccepted implementation head"
            )
        if not isinstance(self.reviewer_id, str) or not self.reviewer_id:
            raise M3CPAuthorizationPinError("reviewer_id must be non-empty")
        if self.schema_version != AUTHORIZATION_BINDING_SCHEMA:
            raise M3CPAuthorizationPinError("authorization binding schema mismatch")

    def to_mapping(self) -> dict[str, Any]:
        return _to_mapping(self)

    @property
    def binding_digest(self) -> str:
        return _digest(self.to_mapping())


@dataclass(frozen=True, slots=True)
class M3CPLocalReviewedAuthorizationPin:
    implementation_evidence_digest: str
    binding_digest: str
    authorization_digest: str
    package_digest: str
    review_artifact_digest: str
    path_binding_digest: str
    rollback_digest: str
    mapping_digest: str
    evaluator_digest: str
    reviewer_id: str
    local_human_reviewed: bool = True
    private_device_single_use_only: bool = True
    existing_m3_c_j_path_reuse_authorized: bool = False
    raw_private_text_or_path_publication_authorized: bool = False
    legacy_goal_authority_transfer_authorized: bool = False
    legacy_migration_authorized: bool = False
    action_authorized: bool = False
    scheduler_authorized: bool = False
    speech_authorized: bool = False
    m3_e_authority_open: bool = False
    schema_version: str = LOCAL_PIN_SCHEMA

    def __post_init__(self) -> None:
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
        ):
            _require_sha256(getattr(self, name), field=name)
        if (
            self.implementation_evidence_digest
            != ACCEPTED_M3_C_O_IMPLEMENTATION_EVIDENCE.evidence_digest
        ):
            raise M3CPAuthorizationPinError(
                "local pin does not bind accepted implementation evidence"
            )
        if not isinstance(self.reviewer_id, str) or not self.reviewer_id:
            raise M3CPAuthorizationPinError("reviewer_id must be non-empty")
        if (
            self.schema_version != LOCAL_PIN_SCHEMA
            or not self.local_human_reviewed
            or not self.private_device_single_use_only
            or self.existing_m3_c_j_path_reuse_authorized
            or self.raw_private_text_or_path_publication_authorized
            or self.legacy_goal_authority_transfer_authorized
            or self.legacy_migration_authorized
            or self.action_authorized
            or self.scheduler_authorized
            or self.speech_authorized
            or self.m3_e_authority_open
        ):
            raise M3CPAuthorizationPinError(
                "local pin escaped bounded private shadow-observation scope"
            )

    def to_mapping(self) -> dict[str, Any]:
        return _to_mapping(self)

    @property
    def pin_digest(self) -> str:
        return _digest(self.to_mapping())


# A concrete instance may only be added by a later isolated local-review pin PR.
# Its absence is the checked-in production boundary for this preflight.
_ACTIVE_LOCAL_REVIEWED_AUTHORIZATION_PIN: M3CPLocalReviewedAuthorizationPin | None = None


def binding_from_private_package(package: Any) -> M3CPOperatorAuthorizationBinding:
    """Derive the public-safe exact binding from one already validated package."""

    required_attributes = (
        "authorization",
        "package_digest",
        "review_artifact",
        "mapping_table",
        "evaluator_digest",
        "rollback",
    )
    if any(not hasattr(package, name) for name in required_attributes):
        raise M3CPAuthorizationPinError("private package lacks required binding fields")
    authorization = package.authorization
    return M3CPOperatorAuthorizationBinding(
        implementation_head=authorization.window_implementation_head,
        authorization_digest=authorization.authorization_digest,
        package_digest=package.package_digest,
        review_artifact_digest=package.review_artifact.review_digest,
        path_binding_digest=authorization.path_binding_digest,
        rollback_digest=package.rollback.rollback_digest,
        mapping_digest=package.mapping_table.table_digest,
        evaluator_digest=package.evaluator_digest,
        reviewer_id=authorization.reviewer_id,
    )


def active_local_reviewed_authorization_pin() -> M3CPLocalReviewedAuthorizationPin:
    pin = _ACTIVE_LOCAL_REVIEWED_AUTHORIZATION_PIN
    if pin is None:
        raise M3CPAuthorizationPinError(
            "no active local reviewed M3-C-P authorization pin"
        )
    return pin


def verify_active_local_pin(
    binding: M3CPOperatorAuthorizationBinding,
) -> M3CPLocalReviewedAuthorizationPin:
    if not isinstance(binding, M3CPOperatorAuthorizationBinding):
        raise M3CPAuthorizationPinError("authorization binding type mismatch")
    pin = active_local_reviewed_authorization_pin()
    expected = {
        "authorization_digest": binding.authorization_digest,
        "binding_digest": binding.binding_digest,
        "evaluator_digest": binding.evaluator_digest,
        "mapping_digest": binding.mapping_digest,
        "package_digest": binding.package_digest,
        "path_binding_digest": binding.path_binding_digest,
        "review_artifact_digest": binding.review_artifact_digest,
        "reviewer_id": binding.reviewer_id,
        "rollback_digest": binding.rollback_digest,
    }
    actual = {name: getattr(pin, name) for name in expected}
    if actual != expected:
        raise M3CPAuthorizationPinError(
            "active local pin differs from exact private package binding"
        )
    return pin


@contextmanager
def reviewed_operator_pin_session(
    binding: M3CPOperatorAuthorizationBinding,
) -> Iterator[M3CPLocalReviewedAuthorizationPin]:
    """Open the accepted M3-C-O pins for one synchronous caller only."""

    pin = verify_active_local_pin(binding)
    if (
        m3_c_o_operator._ACTIVE_REVIEWED_OPERATOR_IMPLEMENTATION_HEAD is not None
        or m3_c_o_operator._ACTIVE_REVIEWED_OPERATOR_AUTHORIZATION_DIGEST is not None
    ):
        raise M3CPAuthorizationPinError("M3-C-O operator pin seam is already open")
    previous_head = m3_c_o_operator._ACTIVE_REVIEWED_OPERATOR_IMPLEMENTATION_HEAD
    previous_authorization = (
        m3_c_o_operator._ACTIVE_REVIEWED_OPERATOR_AUTHORIZATION_DIGEST
    )
    m3_c_o_operator._ACTIVE_REVIEWED_OPERATOR_IMPLEMENTATION_HEAD = (
        binding.implementation_head
    )
    m3_c_o_operator._ACTIVE_REVIEWED_OPERATOR_AUTHORIZATION_DIGEST = (
        binding.authorization_digest
    )
    try:
        yield pin
    finally:
        m3_c_o_operator._ACTIVE_REVIEWED_OPERATOR_IMPLEMENTATION_HEAD = previous_head
        m3_c_o_operator._ACTIVE_REVIEWED_OPERATOR_AUTHORIZATION_DIGEST = (
            previous_authorization
        )


def authorization_pin_preflight_manifest() -> dict[str, Any]:
    return {
        "accepted_implementation_evidence_digest": (
            ACCEPTED_M3_C_O_IMPLEMENTATION_EVIDENCE.evidence_digest
        ),
        "accepted_implementation_head": PR239_EXACT_HEAD,
        "active_local_authorization_present": False,
        "actual_private_device_execution": False,
        "checked_in_operator_pin_open": False,
        "concrete_private_binding_present": False,
        "default_runtime_integration": False,
        "existing_m3_c_j_database_access": False,
        "legacy_goal_authority_transfer": False,
        "legacy_migration_authorization": False,
        "m3_e_authority": False,
        "raw_private_text_or_path_retention": False,
        "schema_version": MANIFEST_SCHEMA,
    }
