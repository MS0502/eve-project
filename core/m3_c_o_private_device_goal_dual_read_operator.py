"""Default-absent M3-C-O private-device goal dual-read operator.

The implementation reuses the accepted M3-C-N bounded-window contracts.  It is
unreachable until a later exact-reviewed implementation head and authorization
digest are pinned.  One explicit process may then inject a bounded collector
into the existing GoalAdapter production-origin seam, preserve legacy authority,
retain only digest records in a new private JSONL store, and restore the seam in
``finally``.

No existing M3-C-J database, sidecar, journal, bundle, backup, restore path, raw
private path, or raw private goal text is read or retained by this module.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Mapping, Sequence

from core.m3_c_b_goal_selection_kernel import (
    ALLOWED_DRIVES,
    DriveSample,
    GoalCandidate,
    select_goal_proposal,
)
from core.m3_c_c_goal_lifecycle_kernel import (
    GoalLifecycleState,
    LifecycleEvidence,
    evaluate_lifecycle_transition,
)
from core.m3_c_l_goal_dual_read_comparator_preflight import V4ShadowGoalObservation
from core.m3_c_m_dormant_production_origin_shadow_tap import (
    DormantProductionOriginGoalShadowTap,
    LegacyGoalMappingEntry,
    LegacyGoalMappingTable,
    ProductionGoalOperation,
    ShadowTapAuthorizationPin,
    ShadowTapExecution,
    capture_legacy_goal_state,
)
from core.m3_c_n_bounded_private_device_goal_dual_read_window_preflight import (
    ACCEPTED_M3_C_M_EVIDENCE,
    BoundedDualReadWindowAuthorizationPacket,
    BoundedDualReadWindowPolicy,
    GoalDualReadWindowRecord,
    GoalDualReadWindowReceipt,
    PrivateDeviceWindowPathBinding,
    PrivateDeviceWindowRollbackPlan,
    evaluate_bounded_dual_read_window,
)

PACKAGE_SCHEMA = "eve.m3-c-o.private-device-goal-dual-read-package.v1"
PROBE_SCHEMA = "eve.m3-c-o.reviewed-goal-probe.v1"
REVIEW_SCHEMA = "eve.m3-c-o.local-human-review.v1"
STORE_SCHEMA = "eve.m3-c-o.private-digest-window-store.v1"
RECEIPT_SCHEMA = "eve.m3-c-o.private-device-operator-receipt.v1"
OPERATOR_SCOPE = "m3-c-o.bounded-private-device-goal-dual-read"
OPERATOR_DECISION = "authorize_single_use_private_device_shadow_observation"

M3_C_N_EXACT_BASE = "1ebd3e27ad4582c67b9b2f072ebd58c625af2057"
M3_C_N_EXACT_HEAD = "b3f599883b9101d7c3b0609fe0680ba4511784d8"
M3_C_N_EXACT_RUN = 30637887864
M3_C_N_FOCUSED_PASSED = 11
M3_C_N_FULL_PASSED = 3369
M3_C_N_ARTIFACT_NAME = (
    "exact-head-validation-b3f599883b9101d7c3b0609fe0680ba4511784d8"
)
M3_C_N_ARTIFACT_SHA256 = (
    "8df56600f347e0d73ab07d8251469eb678542de9d4b405ca723d05533229ab0c"
)
M3_C_N_M2E_RUN = 30637887888
M3_C_N_MERGE_SHA = "9a26f6040679013066425887c3bcee5a2846a025"

# A later isolated exact-pin slice may bind these two values.  They stay absent
# in this implementation PR, so import and the checked-in command are dormant.
_ACTIVE_REVIEWED_OPERATOR_IMPLEMENTATION_HEAD: str | None = None
_ACTIVE_REVIEWED_OPERATOR_AUTHORIZATION_DIGEST: str | None = None


class M3COOperatorError(RuntimeError):
    """Base fail-closed M3-C-O error."""


class M3COOperatorAuthorizationError(M3COOperatorError):
    """Exact implementation or local reviewed packet is absent or mismatched."""


class M3COOperatorInputError(M3COOperatorError):
    """Private reviewed package is malformed or escapes the reviewed scope."""


class M3COOperatorExecutionError(M3COOperatorError):
    """Single-use path, production seam, retention, or rollback proof failed."""


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
        raise M3COOperatorInputError("text digest input must be str")
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def private_path_digest(path: str | Path) -> str:
    candidate = Path(path)
    if not candidate.is_absolute():
        raise M3COOperatorExecutionError("private path must be absolute")
    return _text_digest(str(candidate.resolve()))


def _file_sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], *, field: str) -> None:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise M3COOperatorInputError(f"{field} keys do not match schema")


def _require_git_sha(value: str, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise M3COOperatorAuthorizationError(
            f"{field} must be lowercase 40-character Git SHA"
        )
    return value


def _require_digest(value: str, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise M3COOperatorInputError(f"{field} must be lowercase SHA-256")
    return value


def _candidate_from_mapping(value: Mapping[str, Any]) -> GoalCandidate:
    expected = {
        "base_value",
        "candidate_id",
        "continuity",
        "cost",
        "decision_epoch",
        "drive_alignment",
        "drive_confidence",
        "evidence_digest",
        "expected_value",
        "risk",
        "schema_version",
        "scoring_policy_version",
        "semantic_goal_id",
        "urgency",
    }
    _require_exact_keys(value, expected, field="goal candidate")
    candidate = GoalCandidate(
        semantic_goal_id=value["semantic_goal_id"],
        decision_epoch=value["decision_epoch"],
        evidence_digest=value["evidence_digest"],
        base_value=value["base_value"],
        expected_value=value["expected_value"],
        urgency=value["urgency"],
        continuity=value["continuity"],
        cost=value["cost"],
        risk=value["risk"],
        drive_alignment=value["drive_alignment"],
        drive_confidence=value["drive_confidence"],
        schema_version=value["schema_version"],
        scoring_policy_version=value["scoring_policy_version"],
    )
    if candidate.to_mapping() != dict(value):
        raise M3COOperatorInputError("goal candidate canonical mapping mismatch")
    return candidate


def _sample_from_mapping(value: Mapping[str, Any]) -> DriveSample:
    expected = {
        "drive",
        "dynamics_version",
        "lower_bound",
        "normalized",
        "predicate_version",
        "replay_elapsed_seconds",
        "sample_digest",
        "upper_bound",
        "value",
    }
    _require_exact_keys(value, expected, field="drive sample")
    sample = DriveSample(
        drive=value["drive"],
        value=value["value"],
        lower_bound=value["lower_bound"],
        upper_bound=value["upper_bound"],
        sample_digest=value["sample_digest"],
        replay_elapsed_seconds=value["replay_elapsed_seconds"],
        dynamics_version=value["dynamics_version"],
        predicate_version=value["predicate_version"],
    )
    if sample.to_mapping() != dict(value):
        raise M3COOperatorInputError("drive sample canonical mapping mismatch")
    return sample


@dataclass(frozen=True, slots=True)
class LocalHumanReviewArtifact:
    reviewer_id: str
    review_statement_digest: str
    human_reviewed: bool = True
    decision: str = OPERATOR_DECISION
    scope: str = OPERATOR_SCOPE
    existing_m3_c_j_private_path_reuse_authorized: bool = False
    raw_private_text_publication_authorized: bool = False
    legacy_goal_authority_transfer_authorized: bool = False
    legacy_migration_authorized: bool = False
    action_authorized: bool = False
    scheduler_authorized: bool = False
    speech_authorized: bool = False
    m3_e_authority_open: bool = False
    schema_version: str = REVIEW_SCHEMA

    def __post_init__(self) -> None:
        if not isinstance(self.reviewer_id, str) or not self.reviewer_id:
            raise M3COOperatorInputError("reviewer_id must be non-empty")
        _require_digest(self.review_statement_digest, field="review_statement_digest")
        if (
            self.human_reviewed is not True
            or self.decision != OPERATOR_DECISION
            or self.scope != OPERATOR_SCOPE
            or self.schema_version != REVIEW_SCHEMA
        ):
            raise M3COOperatorInputError("local review artifact scope mismatch")
        if any(
            (
                self.existing_m3_c_j_private_path_reuse_authorized,
                self.raw_private_text_publication_authorized,
                self.legacy_goal_authority_transfer_authorized,
                self.legacy_migration_authorized,
                self.action_authorized,
                self.scheduler_authorized,
                self.speech_authorized,
                self.m3_e_authority_open,
            )
        ):
            raise M3COOperatorInputError("local review artifact grants forbidden authority")

    def to_mapping(self) -> dict[str, Any]:
        return {item.name: getattr(self, item.name) for item in fields(self)}

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "LocalHumanReviewArtifact":
        expected = {item.name for item in fields(cls)}
        _require_exact_keys(value, expected, field="local review artifact")
        return cls(**dict(value))

    @property
    def review_digest(self) -> str:
        return _digest(self.to_mapping())


@dataclass(frozen=True, slots=True)
class ReviewedGoalProbe:
    operation_kind: str
    legacy_goal_code: str
    expected_decision_epoch: int
    logical_step: int
    candidate: GoalCandidate
    drive_samples: tuple[DriveSample, ...]
    category: str | None = None
    priority: float = 0.5
    source: str = "m3-c-o-private-operator"
    dt: float | None = None
    new_private_material: bool = True
    phone_witness_replayed: bool = False
    retained_sequences_replayed: bool = False
    raw_text_publication_authorized: bool = False
    legacy_goal_authority_transfer_requested: bool = False
    legacy_migration_requested: bool = False
    m3_e_requested: bool = False
    schema_version: str = PROBE_SCHEMA

    def __post_init__(self) -> None:
        if self.operation_kind not in {"goal_set", "tick"}:
            raise M3COOperatorInputError("unsupported reviewed probe operation")
        if not isinstance(self.legacy_goal_code, str) or not self.legacy_goal_code:
            raise M3COOperatorInputError("legacy_goal_code must be non-empty")
        if (
            isinstance(self.expected_decision_epoch, bool)
            or not isinstance(self.expected_decision_epoch, int)
            or self.expected_decision_epoch < 0
        ):
            raise M3COOperatorInputError("expected_decision_epoch must be non-negative")
        if isinstance(self.logical_step, bool) or not isinstance(self.logical_step, int) or self.logical_step <= 0:
            raise M3COOperatorInputError("logical_step must be positive")
        if not isinstance(self.candidate, GoalCandidate):
            raise M3COOperatorInputError("probe candidate must be GoalCandidate")
        if self.candidate.decision_epoch != self.expected_decision_epoch:
            raise M3COOperatorInputError("candidate and production epoch mismatch")
        if (
            len(self.drive_samples) != len(ALLOWED_DRIVES)
            or tuple(item.drive for item in self.drive_samples) != ALLOWED_DRIVES
        ):
            raise M3COOperatorInputError("probe drive samples must follow canonical order")
        if self.operation_kind == "goal_set":
            if not isinstance(self.category, str) or not self.category:
                raise M3COOperatorInputError("goal_set probe requires private category")
            if self.dt is not None:
                raise M3COOperatorInputError("goal_set probe cannot carry dt")
        else:
            if self.category is not None or self.dt is None:
                raise M3COOperatorInputError("tick probe requires dt and no category")
        if (
            self.schema_version != PROBE_SCHEMA
            or not self.new_private_material
            or self.phone_witness_replayed
            or self.retained_sequences_replayed
            or self.raw_text_publication_authorized
            or self.legacy_goal_authority_transfer_requested
            or self.legacy_migration_requested
            or self.m3_e_requested
        ):
            raise M3COOperatorInputError("reviewed probe escaped new observation-only scope")

    @property
    def source_material(self) -> dict[str, Any]:
        if self.operation_kind == "goal_set":
            return {
                "category": self.category,
                "priority": self.priority,
                "source": self.source,
            }
        return {"dt": self.dt}

    @property
    def operation(self) -> ProductionGoalOperation:
        return ProductionGoalOperation.from_source_material(
            operation_kind=self.operation_kind,
            legacy_goal_code=self.legacy_goal_code,
            decision_epoch=self.expected_decision_epoch,
            source_material=self.source_material,
        )

    def evaluator_mapping(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate.to_mapping(),
            "drive_samples": [item.to_mapping() for item in self.drive_samples],
            "logical_step": self.logical_step,
            "source_observation_digest": self.operation.source_observation_digest,
        }

    def to_mapping(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate.to_mapping(),
            "category": self.category,
            "drive_samples": [item.to_mapping() for item in self.drive_samples],
            "dt": self.dt,
            "expected_decision_epoch": self.expected_decision_epoch,
            "legacy_goal_authority_transfer_requested": (
                self.legacy_goal_authority_transfer_requested
            ),
            "legacy_goal_code": self.legacy_goal_code,
            "legacy_migration_requested": self.legacy_migration_requested,
            "logical_step": self.logical_step,
            "m3_e_requested": self.m3_e_requested,
            "new_private_material": self.new_private_material,
            "operation_kind": self.operation_kind,
            "phone_witness_replayed": self.phone_witness_replayed,
            "priority": self.priority,
            "raw_text_publication_authorized": self.raw_text_publication_authorized,
            "retained_sequences_replayed": self.retained_sequences_replayed,
            "schema_version": self.schema_version,
            "source": self.source,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ReviewedGoalProbe":
        expected = {
            "candidate",
            "category",
            "drive_samples",
            "dt",
            "expected_decision_epoch",
            "legacy_goal_authority_transfer_requested",
            "legacy_goal_code",
            "legacy_migration_requested",
            "logical_step",
            "m3_e_requested",
            "new_private_material",
            "operation_kind",
            "phone_witness_replayed",
            "priority",
            "raw_text_publication_authorized",
            "retained_sequences_replayed",
            "schema_version",
            "source",
        }
        _require_exact_keys(value, expected, field="reviewed goal probe")
        raw_samples = value["drive_samples"]
        if not isinstance(raw_samples, list):
            raise M3COOperatorInputError("drive_samples must be a list")
        return cls(
            operation_kind=value["operation_kind"],
            legacy_goal_code=value["legacy_goal_code"],
            expected_decision_epoch=value["expected_decision_epoch"],
            logical_step=value["logical_step"],
            candidate=_candidate_from_mapping(value["candidate"]),
            drive_samples=tuple(_sample_from_mapping(item) for item in raw_samples),
            category=value["category"],
            priority=value["priority"],
            source=value["source"],
            dt=value["dt"],
            new_private_material=value["new_private_material"],
            phone_witness_replayed=value["phone_witness_replayed"],
            retained_sequences_replayed=value["retained_sequences_replayed"],
            raw_text_publication_authorized=value["raw_text_publication_authorized"],
            legacy_goal_authority_transfer_requested=value[
                "legacy_goal_authority_transfer_requested"
            ],
            legacy_migration_requested=value["legacy_migration_requested"],
            m3_e_requested=value["m3_e_requested"],
            schema_version=value["schema_version"],
        )


def _mapping_entry_from_mapping(value: Mapping[str, Any]) -> LegacyGoalMappingEntry:
    expected = {
        "category_sha256",
        "legacy_goal_code",
        "legacy_status",
        "semantic_goal_id",
        "v4_lifecycle_state",
    }
    _require_exact_keys(value, expected, field="legacy mapping entry")
    return LegacyGoalMappingEntry(**dict(value))


def _dataclass_from_mapping(cls, value: Mapping[str, Any], *, field: str):
    expected = {item.name for item in fields(cls)}
    _require_exact_keys(value, expected, field=field)
    return cls(**dict(value))


@dataclass(frozen=True, slots=True)
class PrivateDeviceGoalDualReadPackage:
    authorization: BoundedDualReadWindowAuthorizationPacket
    policy: BoundedDualReadWindowPolicy
    rollback: PrivateDeviceWindowRollbackPlan
    mapping_table: LegacyGoalMappingTable
    probes: tuple[ReviewedGoalProbe, ...]
    review_artifact: LocalHumanReviewArtifact
    schema_version: str = PACKAGE_SCHEMA
    private_package_only: bool = True
    raw_paths_embedded: bool = False
    existing_m3_c_j_material_embedded: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.authorization, BoundedDualReadWindowAuthorizationPacket):
            raise M3COOperatorInputError("package authorization type mismatch")
        if not isinstance(self.policy, BoundedDualReadWindowPolicy):
            raise M3COOperatorInputError("package policy type mismatch")
        if not isinstance(self.rollback, PrivateDeviceWindowRollbackPlan):
            raise M3COOperatorInputError("package rollback type mismatch")
        if not isinstance(self.mapping_table, LegacyGoalMappingTable):
            raise M3COOperatorInputError("package mapping type mismatch")
        if not isinstance(self.review_artifact, LocalHumanReviewArtifact):
            raise M3COOperatorInputError("package review artifact type mismatch")
        if not all(isinstance(item, ReviewedGoalProbe) for item in self.probes):
            raise M3COOperatorInputError("package probes have wrong type")
        if not self.policy.min_observations <= len(self.probes) <= self.policy.max_observations:
            raise M3COOperatorInputError("package probe count is outside reviewed bounds")
        if self.authorization.policy_digest != self.policy.policy_digest:
            raise M3COOperatorInputError("package policy digest mismatch")
        if self.authorization.legacy_mapping_digest != self.mapping_table.table_digest:
            raise M3COOperatorInputError("package mapping digest mismatch")
        if self.authorization.v4_evaluator_digest != self.evaluator_digest:
            raise M3COOperatorInputError("package evaluator digest mismatch")
        if (
            self.authorization.authorization_artifact_digest
            != self.review_artifact.review_digest
            or self.authorization.reviewer_id != self.review_artifact.reviewer_id
        ):
            raise M3COOperatorInputError("package local review binding mismatch")
        operations = tuple(item.operation.source_observation_digest for item in self.probes)
        if len(set(operations)) != len(operations):
            raise M3COOperatorInputError("package contains duplicate production observations")
        if (
            self.schema_version != PACKAGE_SCHEMA
            or not self.private_package_only
            or self.raw_paths_embedded
            or self.existing_m3_c_j_material_embedded
        ):
            raise M3COOperatorInputError("package escaped private new-material scope")

    @property
    def evaluator_digest(self) -> str:
        return _digest(
            {
                "probes": [item.evaluator_mapping() for item in self.probes],
                "schema_version": "eve.m3-c-o.reviewed-v4-evaluator.v1",
            }
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "authorization": self.authorization.to_mapping(),
            "existing_m3_c_j_material_embedded": self.existing_m3_c_j_material_embedded,
            "mapping_entries": [
                item.to_mapping() for item in self.mapping_table.entries
            ],
            "mapping_version": self.mapping_table.mapping_version,
            "policy": self.policy.to_mapping(),
            "private_package_only": self.private_package_only,
            "probes": [item.to_mapping() for item in self.probes],
            "raw_paths_embedded": self.raw_paths_embedded,
            "review_artifact": self.review_artifact.to_mapping(),
            "rollback": self.rollback.to_mapping(),
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any]
    ) -> "PrivateDeviceGoalDualReadPackage":
        expected = {
            "authorization",
            "existing_m3_c_j_material_embedded",
            "mapping_entries",
            "mapping_version",
            "policy",
            "private_package_only",
            "probes",
            "raw_paths_embedded",
            "review_artifact",
            "rollback",
            "schema_version",
        }
        _require_exact_keys(value, expected, field="private operator package")
        raw_entries = value["mapping_entries"]
        raw_probes = value["probes"]
        if not isinstance(raw_entries, list) or not isinstance(raw_probes, list):
            raise M3COOperatorInputError("mapping_entries and probes must be lists")
        mapping = LegacyGoalMappingTable(
            entries=tuple(_mapping_entry_from_mapping(item) for item in raw_entries),
            mapping_version=value["mapping_version"],
        )
        return cls(
            authorization=_dataclass_from_mapping(
                BoundedDualReadWindowAuthorizationPacket,
                value["authorization"],
                field="bounded authorization",
            ),
            policy=_dataclass_from_mapping(
                BoundedDualReadWindowPolicy,
                value["policy"],
                field="bounded policy",
            ),
            rollback=_dataclass_from_mapping(
                PrivateDeviceWindowRollbackPlan,
                value["rollback"],
                field="rollback plan",
            ),
            mapping_table=mapping,
            probes=tuple(ReviewedGoalProbe.from_mapping(item) for item in raw_probes),
            review_artifact=LocalHumanReviewArtifact.from_mapping(
                value["review_artifact"]
            ),
            schema_version=value["schema_version"],
            private_package_only=value["private_package_only"],
            raw_paths_embedded=value["raw_paths_embedded"],
            existing_m3_c_j_material_embedded=value[
                "existing_m3_c_j_material_embedded"
            ],
        )

    @property
    def package_digest(self) -> str:
        return _digest(self.to_mapping())


class ReviewedV4GoalEvaluator:
    """Deterministic read-only evaluator bound to reviewed private probe digests."""

    def __init__(self, package: PrivateDeviceGoalDualReadPackage) -> None:
        self.evaluator_digest = package.evaluator_digest
        self._probes = {
            item.operation.source_observation_digest: item for item in package.probes
        }

    def __call__(self, comparison_input, legacy_after) -> V4ShadowGoalObservation:
        probe = self._probes.get(
            comparison_input.operation.source_observation_digest
        )
        if probe is None:
            raise M3COOperatorExecutionError(
                "production observation is absent from reviewed evaluator package"
            )
        samples = {item.drive: item for item in probe.drive_samples}
        selection = select_goal_proposal([probe.candidate], samples)
        lifecycle = None
        if selection.selected_candidate_id is not None:
            state = GoalLifecycleState(
                candidate_id=probe.candidate.candidate_id,
                semantic_goal_id=probe.candidate.semantic_goal_id,
                decision_epoch=probe.candidate.decision_epoch,
                evidence_digest=probe.candidate.evidence_digest,
                lifecycle_state="eligible",
            )
            lifecycle = evaluate_lifecycle_transition(
                state,
                LifecycleEvidence(
                    candidate_score=selection.scored_candidates[0],
                    logical_step=probe.logical_step,
                    selection_receipt=selection,
                ),
            )
        return V4ShadowGoalObservation(
            comparison_input_digest=comparison_input.comparison_input_digest,
            source_observation_digest=(
                comparison_input.operation.source_observation_digest
            ),
            projected_before_state_digest=comparison_input.legacy_before.state_digest,
            projected_after_state_digest=legacy_after.state_digest,
            structural_manifest_digest=legacy_after.structural_manifest_digest,
            selection_receipt=selection,
            lifecycle_receipt=lifecycle,
        )


class BoundedDigestRecordCollector:
    """In-memory collector around the exact M3-C-M tap; no I/O occurs here."""

    def __init__(
        self,
        delegate: DormantProductionOriginGoalShadowTap,
        *,
        max_observations: int,
    ) -> None:
        self._delegate = delegate
        self._max_observations = max_observations
        self._records: list[GoalDualReadWindowRecord] = []
        self._execution_digests: set[str] = set()
        self._source_digests: set[str] = set()

    @property
    def records(self) -> tuple[GoalDualReadWindowRecord, ...]:
        return tuple(self._records)

    def execute_authoritative_once(
        self,
        *,
        goal_management: Any,
        operation: ProductionGoalOperation,
        authoritative_call,
    ) -> ShadowTapExecution:
        if len(self._records) >= self._max_observations:
            raise M3COOperatorExecutionError(
                "reviewed observation bound reached before legacy call"
            )
        execution = self._delegate.execute_authoritative_once(
            goal_management=goal_management,
            operation=operation,
            authoritative_call=authoritative_call,
        )
        if execution.status != "comparison_ready_in_memory_only":
            raise M3COOperatorExecutionError(
                f"shadow comparison failed after authoritative call: {execution.status}"
            )
        previous = (
            self._records[-1].record_digest
            if self._records
            else "0" * 64
        )
        record = GoalDualReadWindowRecord.from_shadow_execution(
            sequence=len(self._records) + 1,
            previous_record_digest=previous,
            execution=execution,
        )
        if (
            record.execution_digest in self._execution_digests
            or record.source_observation_digest in self._source_digests
        ):
            raise M3COOperatorExecutionError("duplicate production evidence refused")
        self._execution_digests.add(record.execution_digest)
        self._source_digests.add(record.source_observation_digest)
        self._records.append(record)
        return execution


@dataclass(frozen=True, slots=True)
class PrivateDeviceGoalDualReadOperatorReceipt:
    operator_implementation_head: str
    launch_repository_head: str
    authorization_digest: str
    package_digest: str
    path_binding_digest: str
    rollback_digest: str
    baseline_state_digest: str
    record_count: int
    final_record_digest: str
    window_receipt_digest: str
    private_store_sha256: str
    baseline_backup_sha256: str
    separate_restore_sha256: str
    human_gate_review_eligible: bool
    explicit_operator_injection_performed: bool = True
    default_runtime_integration_performed: bool = False
    existing_m3_c_j_database_accessed: bool = False
    raw_private_text_retained_in_store: bool = False
    event_append_performed: bool = False
    action_authorized: bool = False
    scheduler_authorized: bool = False
    speech_authorized: bool = False
    legacy_goal_authority_transferred: bool = False
    legacy_migration_authorized: bool = False
    m3_e_authority_open: bool = False
    schema_version: str = RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        _require_git_sha(
            self.operator_implementation_head,
            field="operator_implementation_head",
        )
        _require_git_sha(self.launch_repository_head, field="launch_repository_head")
        for name in (
            "authorization_digest",
            "package_digest",
            "path_binding_digest",
            "rollback_digest",
            "baseline_state_digest",
            "final_record_digest",
            "window_receipt_digest",
            "private_store_sha256",
            "baseline_backup_sha256",
            "separate_restore_sha256",
        ):
            _require_digest(getattr(self, name), field=name)
        if self.record_count <= 0:
            raise M3COOperatorExecutionError("operator receipt record count must be positive")
        if (
            self.schema_version != RECEIPT_SCHEMA
            or not self.explicit_operator_injection_performed
            or self.default_runtime_integration_performed
            or self.existing_m3_c_j_database_accessed
            or self.raw_private_text_retained_in_store
            or self.event_append_performed
            or self.action_authorized
            or self.scheduler_authorized
            or self.speech_authorized
            or self.legacy_goal_authority_transferred
            or self.legacy_migration_authorized
            or self.m3_e_authority_open
        ):
            raise M3COOperatorExecutionError(
                "operator receipt escaped bounded shadow-observation scope"
            )

    def to_mapping(self) -> dict[str, Any]:
        return {item.name: getattr(self, item.name) for item in fields(self)}

    @property
    def receipt_digest(self) -> str:
        return _digest(self.to_mapping())


def active_reviewed_operator_pin() -> tuple[str, str]:
    if (
        _ACTIVE_REVIEWED_OPERATOR_IMPLEMENTATION_HEAD is None
        or _ACTIVE_REVIEWED_OPERATOR_AUTHORIZATION_DIGEST is None
    ):
        raise M3COOperatorAuthorizationError(
            "no active reviewed M3-C-O operator authorization"
        )
    return (
        _ACTIVE_REVIEWED_OPERATOR_IMPLEMENTATION_HEAD,
        _ACTIVE_REVIEWED_OPERATOR_AUTHORIZATION_DIGEST,
    )


def verify_active_operator_authorization(
    authorization: BoundedDualReadWindowAuthorizationPacket,
) -> str:
    implementation_head, authorization_digest = active_reviewed_operator_pin()
    if not isinstance(authorization, BoundedDualReadWindowAuthorizationPacket):
        raise M3COOperatorAuthorizationError("authorization packet type mismatch")
    if authorization.window_implementation_head != implementation_head:
        raise M3COOperatorAuthorizationError(
            "authorization implementation head is not exact-reviewed"
        )
    if authorization.authorization_digest != authorization_digest:
        raise M3COOperatorAuthorizationError(
            "local authorization digest is not exact-reviewed"
        )
    return authorization_digest


def build_private_path_binding(
    *,
    package_path: str | Path,
    working_store_path: str | Path,
    baseline_backup_path: str | Path,
    separate_restore_path: str | Path,
    forbidden_existing_path_digests: Sequence[str],
) -> PrivateDeviceWindowPathBinding:
    return PrivateDeviceWindowPathBinding(
        operator_input_path_digest=private_path_digest(package_path),
        working_store_path_digest=private_path_digest(working_store_path),
        baseline_backup_path_digest=private_path_digest(baseline_backup_path),
        separate_restore_path_digest=private_path_digest(separate_restore_path),
        forbidden_existing_path_digests=tuple(
            sorted(forbidden_existing_path_digests)
        ),
    )


def require_single_use_private_paths(
    *,
    package_path: str | Path,
    working_store_path: str | Path,
    baseline_backup_path: str | Path,
    separate_restore_path: str | Path,
    path_binding: PrivateDeviceWindowPathBinding,
) -> tuple[Path, Path, Path, Path]:
    package = Path(package_path)
    store = Path(working_store_path)
    backup = Path(baseline_backup_path)
    restore = Path(separate_restore_path)
    paths = (package, store, backup, restore)
    if not all(path.is_absolute() for path in paths):
        raise M3COOperatorExecutionError("all private operator paths must be absolute")
    actual = build_private_path_binding(
        package_path=package,
        working_store_path=store,
        baseline_backup_path=backup,
        separate_restore_path=restore,
        forbidden_existing_path_digests=path_binding.forbidden_existing_path_digests,
    )
    if actual != path_binding:
        raise M3COOperatorExecutionError("actual private paths differ from reviewed binding")
    if not package.is_file():
        raise M3COOperatorExecutionError("reviewed private package file is absent")
    for target in (store, backup, restore):
        if target.exists():
            raise M3COOperatorExecutionError(
                "single-use operator target already exists"
            )
    return package, store, backup, restore


def read_canonical_private_package(path: str | Path) -> PrivateDeviceGoalDualReadPackage:
    package_path = Path(path)
    try:
        text = package_path.read_text(encoding="utf-8")
        value = json.loads(text)
    except (OSError, json.JSONDecodeError) as exc:
        raise M3COOperatorInputError("private package is unreadable or invalid JSON") from exc
    if not isinstance(value, Mapping) or text.strip() != _canonical(value):
        raise M3COOperatorInputError("private package must be canonical JSON")
    return PrivateDeviceGoalDualReadPackage.from_mapping(value)


def _write_line(path: Path, value: Mapping[str, Any], *, create: bool = False) -> None:
    mode = "x" if create else "a"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open(mode, encoding="utf-8") as handle:
        handle.write(_canonical(value) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    if os.name != "nt":
        os.chmod(path, 0o600)


def _execute_probe(goal_adapter: Any, probe: ReviewedGoalProbe) -> Any:
    current_epoch = int(getattr(goal_adapter.gm, "tick_count", -1))
    if current_epoch != probe.expected_decision_epoch:
        raise M3COOperatorExecutionError(
            "live legacy decision epoch differs from reviewed probe"
        )
    if probe.operation_kind == "goal_set":
        return goal_adapter._goal_call(
            "goal_set",
            probe.legacy_goal_code,
            probe.source_material,
            lambda: goal_adapter.gm.goal_set(
                probe.category,
                priority=probe.priority,
                source=probe.source,
            ),
        )
    return goal_adapter._goal_call(
        "tick",
        probe.legacy_goal_code,
        probe.source_material,
        lambda: goal_adapter.gm.tick(dt=probe.dt),
    )


def execute_private_device_goal_dual_read_window(
    package: PrivateDeviceGoalDualReadPackage,
    *,
    goal_adapter: Any,
    path_binding: PrivateDeviceWindowPathBinding,
    working_store_path: str | Path,
    baseline_backup_path: str | Path,
    separate_restore_path: str | Path,
    launch_repository_head: str,
) -> tuple[
    PrivateDeviceGoalDualReadOperatorReceipt,
    GoalDualReadWindowReceipt,
]:
    authorization_digest = verify_active_operator_authorization(package.authorization)
    implementation_head, _ = active_reviewed_operator_pin()
    _require_git_sha(launch_repository_head, field="launch_repository_head")
    if package.authorization.path_binding_digest != path_binding.path_binding_digest:
        raise M3COOperatorAuthorizationError("authorization path binding mismatch")
    if package.rollback.path_binding_digest != path_binding.path_binding_digest:
        raise M3COOperatorAuthorizationError("rollback path binding mismatch")
    if package.authorization.rollback_digest != package.rollback.rollback_digest:
        raise M3COOperatorAuthorizationError("authorization rollback mismatch")
    if not hasattr(goal_adapter, "_goal_call") or not hasattr(goal_adapter, "gm"):
        raise M3COOperatorExecutionError("goal_adapter lacks reviewed production seam")
    if getattr(goal_adapter, "production_origin_shadow_tap", None) is not None:
        raise M3COOperatorExecutionError("goal shadow seam is already occupied")

    baseline = capture_legacy_goal_state(goal_adapter.gm)
    if baseline.state_digest != package.rollback.baseline_state_digest:
        raise M3COOperatorExecutionError("legacy baseline differs from reviewed rollback")

    store = Path(working_store_path)
    backup = Path(baseline_backup_path)
    restore = Path(separate_restore_path)
    for target in (store, backup, restore):
        if not target.is_absolute() or target.exists():
            raise M3COOperatorExecutionError("single-use output path is invalid or exists")

    evaluator = ReviewedV4GoalEvaluator(package)
    shadow_authorization = ShadowTapAuthorizationPin(
        implementation_pin_digest=(
            ACCEPTED_M3_C_M_EVIDENCE.compatibility_shadow_pin.pin_digest
        ),
        legacy_mapping_digest=package.mapping_table.table_digest,
        v4_evaluator_digest=evaluator.evaluator_digest,
        authorization_artifact_digest=(
            package.authorization.authorization_artifact_digest
        ),
        reviewer_id=package.authorization.reviewer_id,
    )
    delegate = DormantProductionOriginGoalShadowTap(
        implementation_pin=ACCEPTED_M3_C_M_EVIDENCE.compatibility_shadow_pin,
        authorization_pin=shadow_authorization,
        mapping_table=package.mapping_table,
        v4_evaluator=evaluator,
    )
    collector = BoundedDigestRecordCollector(
        delegate,
        max_observations=package.policy.max_observations,
    )

    _write_line(
        store,
        {
            "authorization_digest": authorization_digest,
            "baseline_state_digest": baseline.state_digest,
            "package_digest": package.package_digest,
            "path_binding_digest": path_binding.path_binding_digest,
            "schema_version": STORE_SCHEMA,
            "stage": "empty_baseline",
        },
        create=True,
    )
    backup.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(store, backup)
    baseline_backup_sha256 = _file_sha256(backup)
    if baseline_backup_sha256 != _file_sha256(store):
        raise M3COOperatorExecutionError("baseline backup verification failed")

    previous_tap = goal_adapter.production_origin_shadow_tap
    goal_adapter.production_origin_shadow_tap = collector
    try:
        for probe in package.probes:
            before_count = len(collector.records)
            _execute_probe(goal_adapter, probe)
            if len(collector.records) != before_count + 1:
                raise M3COOperatorExecutionError(
                    "production probe did not retain exactly one digest record"
                )
            _write_line(
                store,
                {
                    "record": collector.records[-1].to_mapping(),
                    "schema_version": STORE_SCHEMA,
                    "stage": "digest_record",
                },
            )
    finally:
        goal_adapter.production_origin_shadow_tap = previous_tap

    if goal_adapter.production_origin_shadow_tap is not None:
        raise M3COOperatorExecutionError("goal shadow seam was not restored")
    window_receipt = evaluate_bounded_dual_read_window(
        collector.records,
        policy=package.policy,
        authorization=package.authorization,
    )
    _write_line(
        store,
        {
            "schema_version": STORE_SCHEMA,
            "stage": "window_receipt",
            "window_receipt": window_receipt.to_mapping(),
        },
    )
    restore.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(backup, restore)
    separate_restore_sha256 = _file_sha256(restore)
    if separate_restore_sha256 != baseline_backup_sha256:
        raise M3COOperatorExecutionError("separate baseline restore verification failed")

    receipt = PrivateDeviceGoalDualReadOperatorReceipt(
        operator_implementation_head=implementation_head,
        launch_repository_head=launch_repository_head,
        authorization_digest=authorization_digest,
        package_digest=package.package_digest,
        path_binding_digest=path_binding.path_binding_digest,
        rollback_digest=package.rollback.rollback_digest,
        baseline_state_digest=baseline.state_digest,
        record_count=len(collector.records),
        final_record_digest=collector.records[-1].record_digest,
        window_receipt_digest=window_receipt.receipt_digest,
        private_store_sha256=_file_sha256(store),
        baseline_backup_sha256=baseline_backup_sha256,
        separate_restore_sha256=separate_restore_sha256,
        human_gate_review_eligible=window_receipt.human_gate_review_eligible,
    )
    return receipt, window_receipt


def operator_manifest() -> dict[str, Any]:
    return {
        "active_authorization_present": False,
        "default_runtime_integration": False,
        "entrypoint": "scripts/operator/m3_c_o_private_device_goal_dual_read_window.py",
        "existing_m3_c_j_database_access": False,
        "execution_available_in_this_slice": False,
        "private_store_format": "canonical-jsonl-digest-only",
        "required_observations": {"maximum": 16, "minimum": 4},
        "schema_version": "eve.m3-c-o.private-device-operator-manifest.v1",
        "single_use": True,
    }
