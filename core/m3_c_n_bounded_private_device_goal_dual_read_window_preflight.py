"""Pure M3-C-N bounded private-device goal dual-read window preflight.

This module fixes the accepted M3-C-M evidence, defines a correct raw Git-SHA
binding around the v1 compatibility pin, and specifies bounded private-only
retention, chaining, rollback, and human-gate review receipts.  It performs no
I/O and deliberately exposes no active authorization or executable operator.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from core.m3_c_l_goal_dual_read_comparator_preflight import (
    COMPARISON_VERDICTS,
    GoalDualReadComparisonReceipt,
)
from core.m3_c_m_dormant_production_origin_shadow_tap import (
    PRODUCTION_CALLSITE_MANIFEST_DIGEST,
    ShadowTapExecution,
    ShadowTapImplementationPin,
)

WINDOW_POLICY_SCHEMA = "eve.m3-c-n.bounded-dual-read-window-policy.v1"
PATH_BINDING_SCHEMA = "eve.m3-c-n.private-device-path-binding.v1"
ROLLBACK_SCHEMA = "eve.m3-c-n.private-device-window-rollback.v1"
AUTHORIZATION_SCHEMA = "eve.m3-c-n.private-device-window-authorization.v1"
RECORD_SCHEMA = "eve.m3-c-n.private-device-window-record.v1"
RECEIPT_SCHEMA = "eve.m3-c-n.private-device-window-receipt.v1"
EVIDENCE_SCHEMA = "eve.m3-c-n.accepted-m3-c-m-evidence.v1"
OPERATOR_MANIFEST_SCHEMA = "eve.m3-c-n.private-device-operator-manifest.v1"
GENESIS_RECORD_DIGEST = "0" * 64

M3_C_M_EXACT_BASE = "dd524a820a58947f0b589cd0cd521ee35eda73da"
M3_C_M_EXACT_HEAD = "ca9e8a13ae0308060fa0c0505a2b1b4a6558b3a4"
M3_C_M_EXACT_RUN = 30635460387
M3_C_M_FOCUSED_PASSED = 10
M3_C_M_FULL_PASSED = 3358
M3_C_M_ARTIFACT_NAME = (
    "exact-head-validation-ca9e8a13ae0308060fa0c0505a2b1b4a6558b3a4"
)
M3_C_M_ARTIFACT_SHA256 = (
    "9b72851c017af41c6c8d423d3ccd79a94b8a4fe10e4ee432f0b64750e0c0117c"
)
M3_C_M_M2E_RUN = 30635460203
M3_C_M_M2E_PASSED = 6
M3_C_M_MERGE_SHA = "1ebd3e27ad4582c67b9b2f072ebd58c625af2057"

BLOCKING_VERDICTS = frozenset(
    {
        "unexplained_divergence",
        "comparison_unavailable",
        "legacy_only_behavior",
        "v4_only_behavior",
    }
)

_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9._:/-]{0,127}$")

# No checked-in value may activate a private-device window.  A later operator
# pin slice must set both values to one exact reviewed implementation package.
_ACTIVE_REVIEWED_WINDOW_IMPLEMENTATION_HEAD: str | None = None
_ACTIVE_REVIEWED_WINDOW_AUTHORIZATION_DIGEST: str | None = None


class M3CNDualReadWindowError(ValueError):
    """Fail-closed error for M3-C-N window material."""


def _canonical(value: Mapping[str, Any]) -> str:
    return json.dumps(
        dict(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _text_digest(value: str) -> str:
    if not isinstance(value, str):
        raise M3CNDualReadWindowError("text digest input must be str")
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _require_digest(value: str, *, field: str) -> str:
    if not isinstance(value, str) or not _DIGEST.fullmatch(value):
        raise M3CNDualReadWindowError(f"{field} must be lowercase SHA-256")
    return value


def _require_git_sha(value: str, *, field: str) -> str:
    if not isinstance(value, str) or not _GIT_SHA.fullmatch(value):
        raise M3CNDualReadWindowError(f"{field} must be lowercase 40-character Git SHA")
    return value


def _require_identifier(value: str, *, field: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise M3CNDualReadWindowError(f"{field} must be a canonical identifier")
    return value


def _require_positive(value: int, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise M3CNDualReadWindowError(f"{field} must be a positive integer")
    return value


@dataclass(frozen=True, slots=True)
class AcceptedM3CMImplementationEvidence:
    """Raw accepted Git evidence plus a wire-compatible v1 shadow pin."""

    base_sha: str = M3_C_M_EXACT_BASE
    exact_head: str = M3_C_M_EXACT_HEAD
    exact_run: int = M3_C_M_EXACT_RUN
    focused_passed: int = M3_C_M_FOCUSED_PASSED
    full_passed: int = M3_C_M_FULL_PASSED
    artifact_name: str = M3_C_M_ARTIFACT_NAME
    artifact_sha256: str = M3_C_M_ARTIFACT_SHA256
    m2e_run: int = M3_C_M_M2E_RUN
    m2e_passed: int = M3_C_M_M2E_PASSED
    merge_sha: str = M3_C_M_MERGE_SHA
    artifact_digest_independently_recomputed: bool = True
    human_reviewed: bool = True
    schema_version: str = EVIDENCE_SCHEMA

    def __post_init__(self) -> None:
        for field in ("base_sha", "exact_head", "merge_sha"):
            _require_git_sha(getattr(self, field), field=field)
        _require_positive(self.exact_run, field="exact_run")
        _require_positive(self.focused_passed, field="focused_passed")
        _require_positive(self.full_passed, field="full_passed")
        _require_identifier(self.artifact_name, field="artifact_name")
        _require_digest(self.artifact_sha256, field="artifact_sha256")
        _require_positive(self.m2e_run, field="m2e_run")
        if self.m2e_passed != 6:
            raise M3CNDualReadWindowError("accepted M3-C-M M2-E result must be 6/6")
        if self.artifact_digest_independently_recomputed is not True:
            raise M3CNDualReadWindowError("artifact digest must be independently recomputed")
        if self.human_reviewed is not True:
            raise M3CNDualReadWindowError("accepted M3-C-M evidence must be reviewed")
        if self.schema_version != EVIDENCE_SCHEMA:
            raise M3CNDualReadWindowError("unsupported M3-C-M evidence schema")
        exact = (
            self.base_sha,
            self.exact_head,
            self.exact_run,
            self.focused_passed,
            self.full_passed,
            self.artifact_name,
            self.artifact_sha256,
            self.m2e_run,
            self.m2e_passed,
            self.merge_sha,
        )
        expected = (
            M3_C_M_EXACT_BASE,
            M3_C_M_EXACT_HEAD,
            M3_C_M_EXACT_RUN,
            M3_C_M_FOCUSED_PASSED,
            M3_C_M_FULL_PASSED,
            M3_C_M_ARTIFACT_NAME,
            M3_C_M_ARTIFACT_SHA256,
            M3_C_M_M2E_RUN,
            M3_C_M_M2E_PASSED,
            M3_C_M_MERGE_SHA,
        )
        if exact != expected:
            raise M3CNDualReadWindowError("accepted M3-C-M evidence mismatch")

    def to_mapping(self) -> dict[str, Any]:
        return {
            "artifact_digest_independently_recomputed": (
                self.artifact_digest_independently_recomputed
            ),
            "artifact_name": self.artifact_name,
            "artifact_sha256": self.artifact_sha256,
            "base_sha": self.base_sha,
            "exact_head": self.exact_head,
            "exact_run": self.exact_run,
            "focused_passed": self.focused_passed,
            "full_passed": self.full_passed,
            "human_reviewed": self.human_reviewed,
            "m2e_passed": self.m2e_passed,
            "m2e_run": self.m2e_run,
            "merge_sha": self.merge_sha,
            "schema_version": self.schema_version,
        }

    @property
    def evidence_digest(self) -> str:
        return _digest(self.to_mapping())

    @property
    def compatibility_shadow_pin(self) -> ShadowTapImplementationPin:
        """Project raw Git SHAs into the v1 pin's 64-hex compatibility slots.

        M3-C-M v1 named these slots ``exact_head`` and ``merge_sha`` but
        validated 64-hex values.  M3-C-N preserves wire compatibility while the
        raw 40-character values remain separately and exactly bound above.
        """
        return ShadowTapImplementationPin(
            exact_head=_text_digest(self.exact_head),
            exact_run=self.exact_run,
            artifact_name=self.artifact_name,
            artifact_sha256=self.artifact_sha256,
            merge_sha=_text_digest(self.merge_sha),
            reviewed=True,
            callsite_manifest_digest=PRODUCTION_CALLSITE_MANIFEST_DIGEST,
        )


ACCEPTED_M3_C_M_EVIDENCE = AcceptedM3CMImplementationEvidence()


@dataclass(frozen=True, slots=True)
class BoundedDualReadWindowPolicy:
    min_observations: int = 4
    max_observations: int = 16
    allowed_verdicts: tuple[str, ...] = tuple(sorted(COMPARISON_VERDICTS))
    blocking_verdicts: tuple[str, ...] = tuple(sorted(BLOCKING_VERDICTS))
    max_unexplained_divergences: int = 0
    max_unavailable_comparisons: int = 0
    single_use_private_device_only: bool = True
    exact_sequence_required: bool = True
    raw_text_retention_authorized: bool = False
    existing_private_database_access_authorized: bool = False
    schema_version: str = WINDOW_POLICY_SCHEMA

    def __post_init__(self) -> None:
        _require_positive(self.min_observations, field="min_observations")
        _require_positive(self.max_observations, field="max_observations")
        if self.min_observations > self.max_observations or self.max_observations > 64:
            raise M3CNDualReadWindowError("window observation bound is invalid")
        if tuple(sorted(set(self.allowed_verdicts))) != tuple(sorted(COMPARISON_VERDICTS)):
            raise M3CNDualReadWindowError("allowed verdict catalog mismatch")
        if tuple(sorted(set(self.blocking_verdicts))) != tuple(sorted(BLOCKING_VERDICTS)):
            raise M3CNDualReadWindowError("blocking verdict catalog mismatch")
        if self.max_unexplained_divergences != 0 or self.max_unavailable_comparisons != 0:
            raise M3CNDualReadWindowError("migration review requires zero unknown/unavailable")
        if not self.single_use_private_device_only or not self.exact_sequence_required:
            raise M3CNDualReadWindowError("window must be exact single-use private-device only")
        if self.raw_text_retention_authorized or self.existing_private_database_access_authorized:
            raise M3CNDualReadWindowError("policy cannot retain raw text or access existing DB")
        if self.schema_version != WINDOW_POLICY_SCHEMA:
            raise M3CNDualReadWindowError("unsupported window policy schema")

    def to_mapping(self) -> dict[str, Any]:
        return {
            "allowed_verdicts": list(self.allowed_verdicts),
            "blocking_verdicts": list(self.blocking_verdicts),
            "exact_sequence_required": self.exact_sequence_required,
            "existing_private_database_access_authorized": (
                self.existing_private_database_access_authorized
            ),
            "max_observations": self.max_observations,
            "max_unavailable_comparisons": self.max_unavailable_comparisons,
            "max_unexplained_divergences": self.max_unexplained_divergences,
            "min_observations": self.min_observations,
            "raw_text_retention_authorized": self.raw_text_retention_authorized,
            "schema_version": self.schema_version,
            "single_use_private_device_only": self.single_use_private_device_only,
        }

    @property
    def policy_digest(self) -> str:
        return _digest(self.to_mapping())


@dataclass(frozen=True, slots=True)
class PrivateDeviceWindowPathBinding:
    operator_input_path_digest: str
    working_store_path_digest: str
    baseline_backup_path_digest: str
    separate_restore_path_digest: str
    forbidden_existing_path_digests: tuple[str, ...]
    schema_version: str = PATH_BINDING_SCHEMA

    def __post_init__(self) -> None:
        active = (
            self.operator_input_path_digest,
            self.working_store_path_digest,
            self.baseline_backup_path_digest,
            self.separate_restore_path_digest,
        )
        for index, value in enumerate(active):
            _require_digest(value, field=f"active_path_digest_{index}")
        if len(set(active)) != len(active):
            raise M3CNDualReadWindowError("operator, store, backup, and restore paths must differ")
        if not self.forbidden_existing_path_digests:
            raise M3CNDualReadWindowError("prior private path digests must be supplied locally")
        forbidden = tuple(sorted(set(self.forbidden_existing_path_digests)))
        if forbidden != tuple(sorted(self.forbidden_existing_path_digests)):
            raise M3CNDualReadWindowError("forbidden path digests must be unique and sorted")
        for value in forbidden:
            _require_digest(value, field="forbidden_existing_path_digest")
        if set(active) & set(forbidden):
            raise M3CNDualReadWindowError("new M3-C-N paths overlap prior private evidence")
        if self.schema_version != PATH_BINDING_SCHEMA:
            raise M3CNDualReadWindowError("unsupported private path binding schema")

    def to_mapping(self) -> dict[str, Any]:
        return {
            "baseline_backup_path_digest": self.baseline_backup_path_digest,
            "forbidden_existing_path_digests": list(self.forbidden_existing_path_digests),
            "operator_input_path_digest": self.operator_input_path_digest,
            "schema_version": self.schema_version,
            "separate_restore_path_digest": self.separate_restore_path_digest,
            "working_store_path_digest": self.working_store_path_digest,
        }

    @property
    def path_binding_digest(self) -> str:
        return _digest(self.to_mapping())


@dataclass(frozen=True, slots=True)
class PrivateDeviceWindowRollbackPlan:
    path_binding_digest: str
    baseline_state_digest: str
    disable_shadow_tap_before_rollback: bool = True
    preserve_public_review_bundle: bool = True
    delete_working_store_after_review: bool = True
    restore_only_to_separate_path: bool = True
    legacy_state_rewrite_authorized: bool = False
    schema_version: str = ROLLBACK_SCHEMA

    def __post_init__(self) -> None:
        _require_digest(self.path_binding_digest, field="path_binding_digest")
        _require_digest(self.baseline_state_digest, field="baseline_state_digest")
        if not all(
            (
                self.disable_shadow_tap_before_rollback,
                self.preserve_public_review_bundle,
                self.delete_working_store_after_review,
                self.restore_only_to_separate_path,
            )
        ):
            raise M3CNDualReadWindowError("rollback safeguards must all be enabled")
        if self.legacy_state_rewrite_authorized:
            raise M3CNDualReadWindowError("rollback cannot rewrite legacy goal state")
        if self.schema_version != ROLLBACK_SCHEMA:
            raise M3CNDualReadWindowError("unsupported rollback schema")

    def to_mapping(self) -> dict[str, Any]:
        return {
            "baseline_state_digest": self.baseline_state_digest,
            "delete_working_store_after_review": self.delete_working_store_after_review,
            "disable_shadow_tap_before_rollback": self.disable_shadow_tap_before_rollback,
            "legacy_state_rewrite_authorized": self.legacy_state_rewrite_authorized,
            "path_binding_digest": self.path_binding_digest,
            "preserve_public_review_bundle": self.preserve_public_review_bundle,
            "restore_only_to_separate_path": self.restore_only_to_separate_path,
            "schema_version": self.schema_version,
        }

    @property
    def rollback_digest(self) -> str:
        return _digest(self.to_mapping())


@dataclass(frozen=True, slots=True)
class BoundedDualReadWindowAuthorizationPacket:
    window_implementation_head: str
    accepted_m3_c_m_evidence_digest: str
    compatibility_shadow_pin_digest: str
    legacy_mapping_digest: str
    v4_evaluator_digest: str
    policy_digest: str
    path_binding_digest: str
    rollback_digest: str
    authorization_artifact_digest: str
    reviewer_id: str
    human_reviewed: bool = True
    single_use_authorized: bool = True
    private_device_shadow_observation_authorized: bool = True
    bounded_private_retention_authorized: bool = True
    existing_private_database_access_authorized: bool = False
    raw_text_retention_authorized: bool = False
    default_runtime_integration_authorized: bool = False
    action_authorized: bool = False
    scheduler_authorized: bool = False
    speech_authorized: bool = False
    legacy_goal_authority_transferred: bool = False
    legacy_migration_authorized: bool = False
    m3_e_authority_open: bool = False
    prerequisite_exact_head: str = M3_C_M_EXACT_HEAD
    prerequisite_exact_run: int = M3_C_M_EXACT_RUN
    prerequisite_artifact_sha256: str = M3_C_M_ARTIFACT_SHA256
    prerequisite_m2e_run: int = M3_C_M_M2E_RUN
    prerequisite_merge_sha: str = M3_C_M_MERGE_SHA
    schema_version: str = AUTHORIZATION_SCHEMA

    def __post_init__(self) -> None:
        _require_git_sha(self.window_implementation_head, field="window_implementation_head")
        _require_identifier(self.reviewer_id, field="reviewer_id")
        for field in (
            "accepted_m3_c_m_evidence_digest",
            "compatibility_shadow_pin_digest",
            "legacy_mapping_digest",
            "v4_evaluator_digest",
            "policy_digest",
            "path_binding_digest",
            "rollback_digest",
            "authorization_artifact_digest",
            "prerequisite_artifact_sha256",
        ):
            _require_digest(getattr(self, field), field=field)
        _require_git_sha(self.prerequisite_exact_head, field="prerequisite_exact_head")
        _require_git_sha(self.prerequisite_merge_sha, field="prerequisite_merge_sha")
        if (
            self.accepted_m3_c_m_evidence_digest
            != ACCEPTED_M3_C_M_EVIDENCE.evidence_digest
            or self.compatibility_shadow_pin_digest
            != ACCEPTED_M3_C_M_EVIDENCE.compatibility_shadow_pin.pin_digest
            or self.prerequisite_exact_head != M3_C_M_EXACT_HEAD
            or self.prerequisite_exact_run != M3_C_M_EXACT_RUN
            or self.prerequisite_artifact_sha256 != M3_C_M_ARTIFACT_SHA256
            or self.prerequisite_m2e_run != M3_C_M_M2E_RUN
            or self.prerequisite_merge_sha != M3_C_M_MERGE_SHA
        ):
            raise M3CNDualReadWindowError("authorization prerequisite mismatch")
        if not all(
            (
                self.human_reviewed,
                self.single_use_authorized,
                self.private_device_shadow_observation_authorized,
                self.bounded_private_retention_authorized,
            )
        ):
            raise M3CNDualReadWindowError("authorization lacks required private-window scope")
        if any(
            (
                self.existing_private_database_access_authorized,
                self.raw_text_retention_authorized,
                self.default_runtime_integration_authorized,
                self.action_authorized,
                self.scheduler_authorized,
                self.speech_authorized,
                self.legacy_goal_authority_transferred,
                self.legacy_migration_authorized,
                self.m3_e_authority_open,
            )
        ):
            raise M3CNDualReadWindowError("authorization escaped shadow-only boundary")
        if self.schema_version != AUTHORIZATION_SCHEMA:
            raise M3CNDualReadWindowError("unsupported authorization schema")

    def to_mapping(self) -> dict[str, Any]:
        return {
            field: getattr(self, field)
            for field in self.__dataclass_fields__
        }

    @property
    def authorization_digest(self) -> str:
        return _digest(self.to_mapping())


@dataclass(frozen=True, slots=True)
class GoalDualReadWindowRecord:
    sequence: int
    previous_record_digest: str
    operation_digest: str
    execution_digest: str
    comparison_receipt_digest: str
    comparison_input_digest: str
    source_observation_digest: str
    legacy_before_state_digest: str
    legacy_after_state_digest: str
    structural_manifest_digest: str
    verdict: str
    authoritative_call_count: int = 1
    raw_text_retained: bool = False
    event_append_performed: bool = False
    action_authorized: bool = False
    scheduler_authorized: bool = False
    speech_authorized: bool = False
    legacy_goal_authority_transferred: bool = False
    legacy_migration_authorized: bool = False
    m3_e_authority_open: bool = False
    schema_version: str = RECORD_SCHEMA

    def __post_init__(self) -> None:
        _require_positive(self.sequence, field="sequence")
        for field in (
            "previous_record_digest",
            "operation_digest",
            "execution_digest",
            "comparison_receipt_digest",
            "comparison_input_digest",
            "source_observation_digest",
            "legacy_before_state_digest",
            "legacy_after_state_digest",
            "structural_manifest_digest",
        ):
            _require_digest(getattr(self, field), field=field)
        if self.verdict not in COMPARISON_VERDICTS:
            raise M3CNDualReadWindowError("unsupported comparison verdict")
        if self.authoritative_call_count != 1:
            raise M3CNDualReadWindowError("legacy authoritative call count must be exactly one")
        if any(
            (
                self.raw_text_retained,
                self.event_append_performed,
                self.action_authorized,
                self.scheduler_authorized,
                self.speech_authorized,
                self.legacy_goal_authority_transferred,
                self.legacy_migration_authorized,
                self.m3_e_authority_open,
            )
        ):
            raise M3CNDualReadWindowError("window record escaped observation-only scope")
        if self.schema_version != RECORD_SCHEMA:
            raise M3CNDualReadWindowError("unsupported window record schema")

    @classmethod
    def from_shadow_execution(
        cls,
        *,
        sequence: int,
        previous_record_digest: str,
        execution: ShadowTapExecution,
    ) -> "GoalDualReadWindowRecord":
        if not isinstance(execution, ShadowTapExecution):
            raise M3CNDualReadWindowError("execution must be ShadowTapExecution")
        receipt = execution.comparison_receipt
        if (
            execution.status != "comparison_ready_in_memory_only"
            or not isinstance(receipt, GoalDualReadComparisonReceipt)
            or execution.legacy_before is None
            or execution.legacy_after is None
            or execution.authoritative_call_count != 1
            or not execution.comparison_performed
            or execution.event_append_performed
            or execution.persistence_write_performed
        ):
            raise M3CNDualReadWindowError("execution is not a retainable M3-C-M comparison")
        return cls(
            sequence=sequence,
            previous_record_digest=previous_record_digest,
            operation_digest=_digest(execution.operation.to_mapping()),
            execution_digest=execution.execution_digest,
            comparison_receipt_digest=receipt.receipt_digest,
            comparison_input_digest=receipt.comparison_input_digest,
            source_observation_digest=receipt.source_observation_digest,
            legacy_before_state_digest=execution.legacy_before.state_digest,
            legacy_after_state_digest=execution.legacy_after.state_digest,
            structural_manifest_digest=execution.legacy_after.structural_manifest_digest,
            verdict=receipt.verdict,
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            field: getattr(self, field)
            for field in self.__dataclass_fields__
        }

    @property
    def record_digest(self) -> str:
        return _digest(self.to_mapping())


@dataclass(frozen=True, slots=True)
class GoalDualReadWindowReceipt:
    authorization_digest: str
    policy_digest: str
    record_count: int
    final_record_digest: str
    verdict_counts: tuple[tuple[str, int], ...]
    blocking_verdict_count: int
    human_gate_review_eligible: bool
    retention_scope: str = "bounded_private_digest_only"
    legacy_authority: str = "legacy_authoritative"
    v4_authority: str = "shadow_only"
    production_runtime_integration_authorized: bool = False
    action_authorized: bool = False
    scheduler_authorized: bool = False
    speech_authorized: bool = False
    legacy_goal_authority_transferred: bool = False
    legacy_migration_authorized: bool = False
    m3_e_authority_open: bool = False
    schema_version: str = RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        _require_digest(self.authorization_digest, field="authorization_digest")
        _require_digest(self.policy_digest, field="policy_digest")
        _require_positive(self.record_count, field="record_count")
        _require_digest(self.final_record_digest, field="final_record_digest")
        if tuple(sorted(self.verdict_counts)) != self.verdict_counts:
            raise M3CNDualReadWindowError("verdict counts must be sorted")
        if sum(count for _, count in self.verdict_counts) != self.record_count:
            raise M3CNDualReadWindowError("verdict counts do not match record count")
        if self.blocking_verdict_count < 0:
            raise M3CNDualReadWindowError("blocking verdict count cannot be negative")
        if self.human_gate_review_eligible != (self.blocking_verdict_count == 0):
            raise M3CNDualReadWindowError("human gate eligibility must be derived")
        if self.retention_scope != "bounded_private_digest_only":
            raise M3CNDualReadWindowError("receipt retention scope mismatch")
        if self.legacy_authority != "legacy_authoritative" or self.v4_authority != "shadow_only":
            raise M3CNDualReadWindowError("receipt authority boundary mismatch")
        if any(
            (
                self.production_runtime_integration_authorized,
                self.action_authorized,
                self.scheduler_authorized,
                self.speech_authorized,
                self.legacy_goal_authority_transferred,
                self.legacy_migration_authorized,
                self.m3_e_authority_open,
            )
        ):
            raise M3CNDualReadWindowError("window receipt cannot grant downstream authority")
        if self.schema_version != RECEIPT_SCHEMA:
            raise M3CNDualReadWindowError("unsupported window receipt schema")

    def to_mapping(self) -> dict[str, Any]:
        return {
            "action_authorized": self.action_authorized,
            "authorization_digest": self.authorization_digest,
            "blocking_verdict_count": self.blocking_verdict_count,
            "final_record_digest": self.final_record_digest,
            "human_gate_review_eligible": self.human_gate_review_eligible,
            "legacy_goal_authority_transferred": self.legacy_goal_authority_transferred,
            "legacy_migration_authorized": self.legacy_migration_authorized,
            "legacy_authority": self.legacy_authority,
            "m3_e_authority_open": self.m3_e_authority_open,
            "policy_digest": self.policy_digest,
            "production_runtime_integration_authorized": (
                self.production_runtime_integration_authorized
            ),
            "record_count": self.record_count,
            "retention_scope": self.retention_scope,
            "scheduler_authorized": self.scheduler_authorized,
            "schema_version": self.schema_version,
            "speech_authorized": self.speech_authorized,
            "v4_authority": self.v4_authority,
            "verdict_counts": [list(item) for item in self.verdict_counts],
        }

    @property
    def receipt_digest(self) -> str:
        return _digest(self.to_mapping())


def evaluate_bounded_dual_read_window(
    records: Sequence[GoalDualReadWindowRecord],
    *,
    policy: BoundedDualReadWindowPolicy,
    authorization: BoundedDualReadWindowAuthorizationPacket,
) -> GoalDualReadWindowReceipt:
    if not isinstance(policy, BoundedDualReadWindowPolicy):
        raise M3CNDualReadWindowError("policy must be BoundedDualReadWindowPolicy")
    if not isinstance(authorization, BoundedDualReadWindowAuthorizationPacket):
        raise M3CNDualReadWindowError("authorization packet type mismatch")
    if authorization.policy_digest != policy.policy_digest:
        raise M3CNDualReadWindowError("authorization policy digest mismatch")
    material = tuple(records)
    if not policy.min_observations <= len(material) <= policy.max_observations:
        raise M3CNDualReadWindowError("window record count is outside reviewed bounds")
    expected_previous = GENESIS_RECORD_DIGEST
    execution_digests: set[str] = set()
    source_digests: set[str] = set()
    verdict_counts: dict[str, int] = {}
    for sequence, record in enumerate(material, start=1):
        if not isinstance(record, GoalDualReadWindowRecord):
            raise M3CNDualReadWindowError("window contains wrong record type")
        if record.sequence != sequence or record.previous_record_digest != expected_previous:
            raise M3CNDualReadWindowError("window sequence or digest chain mismatch")
        if record.execution_digest in execution_digests:
            raise M3CNDualReadWindowError("duplicate shadow execution retained")
        if record.source_observation_digest in source_digests:
            raise M3CNDualReadWindowError("duplicate production observation retained")
        execution_digests.add(record.execution_digest)
        source_digests.add(record.source_observation_digest)
        verdict_counts[record.verdict] = verdict_counts.get(record.verdict, 0) + 1
        expected_previous = record.record_digest
    blocking = sum(verdict_counts.get(verdict, 0) for verdict in BLOCKING_VERDICTS)
    return GoalDualReadWindowReceipt(
        authorization_digest=authorization.authorization_digest,
        policy_digest=policy.policy_digest,
        record_count=len(material),
        final_record_digest=material[-1].record_digest,
        verdict_counts=tuple(sorted(verdict_counts.items())),
        blocking_verdict_count=blocking,
        human_gate_review_eligible=(blocking == 0),
    )


def active_reviewed_window_authorization() -> tuple[str, str]:
    """Fail closed until a later exact-reviewed private-device pin slice."""
    if (
        _ACTIVE_REVIEWED_WINDOW_IMPLEMENTATION_HEAD is None
        or _ACTIVE_REVIEWED_WINDOW_AUTHORIZATION_DIGEST is None
    ):
        raise M3CNDualReadWindowError("no active reviewed M3-C-N window authorization")
    return (
        _ACTIVE_REVIEWED_WINDOW_IMPLEMENTATION_HEAD,
        _ACTIVE_REVIEWED_WINDOW_AUTHORIZATION_DIGEST,
    )


def private_device_operator_manifest() -> dict[str, Any]:
    """Describe the future explicit operator without making it executable."""
    return {
        "active_authorization_present": False,
        "default_runtime_integration": False,
        "entrypoint": "scripts/run_m3_c_n_private_device_goal_dual_read_window.py",
        "execution_available_in_this_slice": False,
        "existing_m3_c_j_database_access": False,
        "required_local_inputs": [
            "reviewed_authorization.json",
            "reviewed_legacy_mapping.json",
            "reviewed_v4_evaluator.py",
            "new_private_operator_input.json",
            "new_empty_window_store_path",
            "separate_backup_path",
            "separate_restore_path",
            "prior_private_path_digests",
        ],
        "schema_version": OPERATOR_MANIFEST_SCHEMA,
        "single_use": True,
    }
