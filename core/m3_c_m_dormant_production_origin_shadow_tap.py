"""Dormant M3-C-M production-origin legacy/v4 goal shadow tap.

The wrapper is unreachable by default.  It activates only when an exact
implementation pin and a separately reviewed shadow-only authorization pin bind
the call-site manifest, legacy mapping table, comparator, and v4 evaluator.
The authoritative legacy callable always executes exactly once.

No file, SQLite, network, event append, persistence, action, scheduler, speech,
authority-transfer, migration, memory, affect, hormone, or M3-E work exists here.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol, TypeVar

from core.m3_c_l_goal_dual_read_comparator_preflight import (
    COMPARATOR_SCHEMA_VERSION,
    GoalDualReadComparisonReceipt,
    LegacyGoalObservation,
    V4ShadowGoalObservation,
    compare_goal_observations,
)

SHADOW_TAP_SCHEMA_VERSION = "eve.m3-c-m.production-origin-shadow-tap.v1"
IMPLEMENTATION_PIN_SCHEMA_VERSION = "eve.m3-c-m.shadow-tap-implementation-pin.v1"
AUTHORIZATION_PIN_SCHEMA_VERSION = "eve.m3-c-m.shadow-tap-authorization-pin.v1"
LEGACY_SNAPSHOT_SCHEMA_VERSION = "eve.m3-c-m.legacy-goal-snapshot.v1"
LEGACY_MAPPING_SCHEMA_VERSION = "eve.m3-c-m.legacy-goal-mapping.v1"
OPERATION_SCHEMA_VERSION = "eve.m3-c-m.production-goal-operation.v1"
COMPARISON_INPUT_SCHEMA_VERSION = "eve.m3-c-m.production-comparison-input.v1"
EXECUTION_SCHEMA_VERSION = "eve.m3-c-m.shadow-tap-execution.v1"

LEGACY_AUTHORITY = "legacy_authoritative"
V4_AUTHORITY = "shadow_only"
AUTHORIZATION_SCOPE = "m3-c-n.bounded-private-device-goal-dual-read"
AUTHORIZATION_DECISION = "authorize_shadow_observation_only"

PRODUCTION_CALLSITES = (
    {
        "path": "adapters/goal_adapter.py",
        "callable": "GoalAdapter.observe_meaning",
        "legacy_callable": "GoalManagement.goal_set",
        "operation_kind": "goal_set",
    },
    {
        "path": "adapters/goal_adapter.py",
        "callable": "GoalAdapter.tick",
        "legacy_callable": "GoalManagement.tick",
        "operation_kind": "tick",
    },
)

_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9._:/-]{0,127}$")
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_ALLOWED_OPERATIONS = frozenset({"goal_set", "tick"})
_ALLOWED_LEGACY_STATES = frozenset({"active", "completed", "abandoned", "expired"})
_ALLOWED_V4_STATES = frozenset(
    {
        "absent",
        "proposed",
        "validated",
        "eligible",
        "selected",
        "rejected",
        "expired",
        "withdrawn",
        "superseded",
    }
)
T = TypeVar("T")


class M3CShadowTapError(ValueError):
    """Fail-closed contract error for M3-C-M material."""


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
        raise M3CShadowTapError("text digest input must be str")
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256(value: str, *, field: str) -> str:
    if not isinstance(value, str) or not _DIGEST.fullmatch(value):
        raise M3CShadowTapError(f"{field} must be lowercase SHA-256")
    return value


def _identifier(value: str, *, field: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise M3CShadowTapError(f"{field} must be a canonical internal identifier")
    return value


def _non_negative_int(value: int, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise M3CShadowTapError(f"{field} must be a non-negative integer")
    return value


def _finite(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise M3CShadowTapError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise M3CShadowTapError(f"{field} must be finite")
    return result


def _sanitize_scalar(value: Any, *, field: str) -> Any:
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return _finite(value, field=field)
    if isinstance(value, str):
        return {"text_sha256": _text_digest(value)}
    raise M3CShadowTapError(f"{field} contains unsupported scalar type")


def _sanitize_goal(goal_id: str, goal: Any) -> dict[str, Any]:
    category = getattr(goal, "category", None)
    if not isinstance(category, str) or not category:
        raise M3CShadowTapError("legacy goal category must be non-empty str")
    result = {
        "goal_id_sha256": _text_digest(str(goal_id)),
        "category_sha256": _text_digest(category),
    }
    for field in (
        "priority",
        "deadline",
        "status",
        "progress",
        "created",
        "last_evaluated",
        "completed_at",
        "source",
        "abandon_reason",
    ):
        result[field] = _sanitize_scalar(
            getattr(goal, field, None),
            field=f"goal.{field}",
        )
    return result


def _sanitize_history(history: Any) -> list[dict[str, Any]]:
    try:
        records = list(history)
    except TypeError as exc:
        raise M3CShadowTapError("legacy history must be iterable") from exc
    clean: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise M3CShadowTapError("legacy history record must be a mapping")
        clean.append(
            {
                str(key): _sanitize_scalar(value, field=f"history[{index}].{key}")
                for key, value in sorted(record.items(), key=lambda item: str(item[0]))
            }
        )
    return clean


PRODUCTION_CALLSITE_MANIFEST_DIGEST = _digest(
    {
        "callsites": list(PRODUCTION_CALLSITES),
        "schema_version": SHADOW_TAP_SCHEMA_VERSION,
    }
)


@dataclass(frozen=True, slots=True)
class ShadowTapImplementationPin:
    exact_head: str
    exact_run: int
    artifact_name: str
    artifact_sha256: str
    merge_sha: str
    reviewed: bool = True
    comparator_schema_version: str = COMPARATOR_SCHEMA_VERSION
    callsite_manifest_digest: str = PRODUCTION_CALLSITE_MANIFEST_DIGEST
    schema_version: str = IMPLEMENTATION_PIN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _sha256(self.exact_head, field="exact_head")
        _non_negative_int(self.exact_run, field="exact_run")
        _identifier(self.artifact_name, field="artifact_name")
        _sha256(self.artifact_sha256, field="artifact_sha256")
        _sha256(self.merge_sha, field="merge_sha")
        if self.reviewed is not True:
            raise M3CShadowTapError("implementation pin must be explicitly reviewed")
        if self.comparator_schema_version != COMPARATOR_SCHEMA_VERSION:
            raise M3CShadowTapError("implementation pin comparator version mismatch")
        if self.callsite_manifest_digest != PRODUCTION_CALLSITE_MANIFEST_DIGEST:
            raise M3CShadowTapError("implementation pin callsite manifest mismatch")
        if self.schema_version != IMPLEMENTATION_PIN_SCHEMA_VERSION:
            raise M3CShadowTapError("unsupported implementation pin schema")

    def to_mapping(self) -> dict[str, Any]:
        return {
            "artifact_name": self.artifact_name,
            "artifact_sha256": self.artifact_sha256,
            "callsite_manifest_digest": self.callsite_manifest_digest,
            "comparator_schema_version": self.comparator_schema_version,
            "exact_head": self.exact_head,
            "exact_run": self.exact_run,
            "merge_sha": self.merge_sha,
            "reviewed": self.reviewed,
            "schema_version": self.schema_version,
        }

    @property
    def pin_digest(self) -> str:
        return _digest(self.to_mapping())


@dataclass(frozen=True, slots=True)
class LegacyGoalMappingEntry:
    legacy_goal_code: str
    category_sha256: str
    legacy_status: str
    semantic_goal_id: str
    v4_lifecycle_state: str

    def __post_init__(self) -> None:
        _identifier(self.legacy_goal_code, field="legacy_goal_code")
        _sha256(self.category_sha256, field="category_sha256")
        if self.legacy_status not in _ALLOWED_LEGACY_STATES:
            raise M3CShadowTapError("unsupported legacy goal status")
        _identifier(self.semantic_goal_id, field="semantic_goal_id")
        if self.v4_lifecycle_state not in _ALLOWED_V4_STATES:
            raise M3CShadowTapError("unsupported mapped lifecycle state")

    def to_mapping(self) -> dict[str, Any]:
        return {
            "category_sha256": self.category_sha256,
            "legacy_goal_code": self.legacy_goal_code,
            "legacy_status": self.legacy_status,
            "semantic_goal_id": self.semantic_goal_id,
            "v4_lifecycle_state": self.v4_lifecycle_state,
        }


@dataclass(frozen=True, slots=True)
class LegacyGoalMappingTable:
    entries: tuple[LegacyGoalMappingEntry, ...]
    mapping_version: str = "eve.m3-c-k.legacy-goal-mapping.v1"
    schema_version: str = LEGACY_MAPPING_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.entries:
            raise M3CShadowTapError("legacy mapping table cannot be empty")
        keys: set[tuple[str, str, str]] = set()
        for entry in self.entries:
            if not isinstance(entry, LegacyGoalMappingEntry):
                raise M3CShadowTapError("mapping entries must be LegacyGoalMappingEntry")
            key = (entry.legacy_goal_code, entry.category_sha256, entry.legacy_status)
            if key in keys:
                raise M3CShadowTapError("duplicate legacy mapping tuple")
            keys.add(key)
        _identifier(self.mapping_version, field="mapping_version")
        if self.schema_version != LEGACY_MAPPING_SCHEMA_VERSION:
            raise M3CShadowTapError("unsupported legacy mapping schema")

    def to_mapping(self) -> dict[str, Any]:
        return {
            "entries": [entry.to_mapping() for entry in self.entries],
            "mapping_version": self.mapping_version,
            "schema_version": self.schema_version,
        }

    @property
    def table_digest(self) -> str:
        return _digest(self.to_mapping())

    def resolve(
        self,
        *,
        legacy_goal_code: str,
        category_sha256: str,
        legacy_status: str,
    ) -> LegacyGoalMappingEntry:
        matches = tuple(
            entry
            for entry in self.entries
            if entry.legacy_goal_code == legacy_goal_code
            and entry.category_sha256 == category_sha256
            and entry.legacy_status == legacy_status
        )
        if len(matches) != 1:
            raise M3CShadowTapError("exact legacy goal mapping unavailable")
        return matches[0]


@dataclass(frozen=True, slots=True)
class ShadowTapAuthorizationPin:
    implementation_pin_digest: str
    legacy_mapping_digest: str
    v4_evaluator_digest: str
    authorization_artifact_digest: str
    reviewer_id: str
    reviewed: bool = True
    decision: str = AUTHORIZATION_DECISION
    scope: str = AUTHORIZATION_SCOPE
    callsite_manifest_digest: str = PRODUCTION_CALLSITE_MANIFEST_DIGEST
    comparator_schema_version: str = COMPARATOR_SCHEMA_VERSION
    legacy_authority: str = LEGACY_AUTHORITY
    v4_authority: str = V4_AUTHORITY
    persistence_write_authorized: bool = False
    event_append_authorized: bool = False
    action_authorized: bool = False
    scheduler_authorized: bool = False
    speech_authorized: bool = False
    legacy_goal_authority_transfer_authorized: bool = False
    legacy_migration_authorized: bool = False
    m3_e_authority_open: bool = False
    schema_version: str = AUTHORIZATION_PIN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for field in (
            "implementation_pin_digest",
            "legacy_mapping_digest",
            "v4_evaluator_digest",
            "authorization_artifact_digest",
            "callsite_manifest_digest",
        ):
            _sha256(getattr(self, field), field=field)
        _identifier(self.reviewer_id, field="reviewer_id")
        if self.reviewed is not True:
            raise M3CShadowTapError("authorization pin must be explicitly reviewed")
        if self.decision != AUTHORIZATION_DECISION or self.scope != AUTHORIZATION_SCOPE:
            raise M3CShadowTapError("authorization is not exact shadow-only scope")
        if self.callsite_manifest_digest != PRODUCTION_CALLSITE_MANIFEST_DIGEST:
            raise M3CShadowTapError("authorization callsite manifest mismatch")
        if self.comparator_schema_version != COMPARATOR_SCHEMA_VERSION:
            raise M3CShadowTapError("authorization comparator version mismatch")
        if self.legacy_authority != LEGACY_AUTHORITY or self.v4_authority != V4_AUTHORITY:
            raise M3CShadowTapError("authorization authority boundary mismatch")
        if any(
            (
                self.persistence_write_authorized,
                self.event_append_authorized,
                self.action_authorized,
                self.scheduler_authorized,
                self.speech_authorized,
                self.legacy_goal_authority_transfer_authorized,
                self.legacy_migration_authorized,
                self.m3_e_authority_open,
            )
        ):
            raise M3CShadowTapError("authorization pin cannot grant downstream authority")
        if self.schema_version != AUTHORIZATION_PIN_SCHEMA_VERSION:
            raise M3CShadowTapError("unsupported authorization pin schema")

    def binds(
        self,
        implementation_pin: ShadowTapImplementationPin,
        mapping_table: LegacyGoalMappingTable,
        evaluator_digest: str,
    ) -> bool:
        return (
            self.implementation_pin_digest == implementation_pin.pin_digest
            and self.legacy_mapping_digest == mapping_table.table_digest
            and self.v4_evaluator_digest == evaluator_digest
        )


@dataclass(frozen=True, slots=True)
class LegacyGoalStateSnapshot:
    state_digest: str
    structural_manifest_digest: str
    active_count: int
    top_goal_category_sha256: str | None
    top_goal_status: str | None
    schema_version: str = LEGACY_SNAPSHOT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _sha256(self.state_digest, field="state_digest")
        _sha256(self.structural_manifest_digest, field="structural_manifest_digest")
        _non_negative_int(self.active_count, field="active_count")
        if (self.top_goal_category_sha256 is None) != (self.top_goal_status is None):
            raise M3CShadowTapError("top goal category/status must be present together")
        if self.top_goal_category_sha256 is not None:
            _sha256(self.top_goal_category_sha256, field="top_goal_category_sha256")
            if self.top_goal_status not in _ALLOWED_LEGACY_STATES:
                raise M3CShadowTapError("unsupported top goal status")
        if self.schema_version != LEGACY_SNAPSHOT_SCHEMA_VERSION:
            raise M3CShadowTapError("unsupported legacy snapshot schema")

    def to_mapping(self) -> dict[str, Any]:
        return {
            "active_count": self.active_count,
            "schema_version": self.schema_version,
            "state_digest": self.state_digest,
            "structural_manifest_digest": self.structural_manifest_digest,
            "top_goal_category_sha256": self.top_goal_category_sha256,
            "top_goal_status": self.top_goal_status,
        }


def capture_legacy_goal_state(goal_management: Any) -> LegacyGoalStateSnapshot:
    """Capture deterministic text-digest-only in-memory legacy state."""
    goals_obj = getattr(goal_management, "goals", None)
    if not isinstance(goals_obj, Mapping):
        raise M3CShadowTapError("legacy GoalManagement.goals must be a mapping")
    goals = [
        _sanitize_goal(str(goal_id), goal)
        for goal_id, goal in sorted(goals_obj.items(), key=lambda item: str(item[0]))
    ]
    active = [
        item
        for item in goals
        if item["status"]["text_sha256"] == _text_digest("active")
    ]
    active.sort(
        key=lambda item: (
            -float(item["priority"]),
            float(item["created"]),
            item["goal_id_sha256"],
        )
    )
    counters = {
        field: _non_negative_int(int(getattr(goal_management, field, 0)), field=field)
        for field in (
            "tick_count",
            "_next_goal_id",
            "set_count",
            "completed_count",
            "abandoned_count",
            "expired_count",
            "suggested_count",
            "proposed_count",
        )
    }
    state = {
        "counters": counters,
        "goals": goals,
        "history": _sanitize_history(getattr(goal_management, "history", ())),
        "time": _finite(getattr(goal_management, "time", 0.0), field="time"),
    }
    manifest = {
        "goal_fields": sorted(goals[0]) if goals else [
            "abandon_reason",
            "category_sha256",
            "completed_at",
            "created",
            "deadline",
            "goal_id_sha256",
            "last_evaluated",
            "priority",
            "progress",
            "source",
            "status",
        ],
        "owner_type": type(goal_management).__name__,
        "snapshot_schema_version": LEGACY_SNAPSHOT_SCHEMA_VERSION,
        "state_fields": ["counters", "goals", "history", "time"],
    }
    top = active[0] if active else None
    return LegacyGoalStateSnapshot(
        state_digest=_digest(state),
        structural_manifest_digest=_digest(manifest),
        active_count=len(active),
        top_goal_category_sha256=top["category_sha256"] if top else None,
        top_goal_status="active" if top else None,
    )


@dataclass(frozen=True, slots=True)
class ProductionGoalOperation:
    operation_kind: str
    legacy_goal_code: str
    decision_epoch: int
    source_observation_digest: str
    callsite_manifest_digest: str = PRODUCTION_CALLSITE_MANIFEST_DIGEST
    schema_version: str = OPERATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.operation_kind not in _ALLOWED_OPERATIONS:
            raise M3CShadowTapError("unsupported production goal operation")
        _identifier(self.legacy_goal_code, field="legacy_goal_code")
        _non_negative_int(self.decision_epoch, field="decision_epoch")
        _sha256(self.source_observation_digest, field="source_observation_digest")
        if self.callsite_manifest_digest != PRODUCTION_CALLSITE_MANIFEST_DIGEST:
            raise M3CShadowTapError("operation callsite manifest mismatch")
        if self.schema_version != OPERATION_SCHEMA_VERSION:
            raise M3CShadowTapError("unsupported operation schema")

    @classmethod
    def from_source_material(
        cls,
        *,
        operation_kind: str,
        legacy_goal_code: str,
        decision_epoch: int,
        source_material: Mapping[str, Any],
    ) -> "ProductionGoalOperation":
        sanitized = {
            str(key): _sanitize_scalar(value, field=f"source_material.{key}")
            for key, value in sorted(source_material.items(), key=lambda item: str(item[0]))
        }
        return cls(
            operation_kind=operation_kind,
            legacy_goal_code=legacy_goal_code,
            decision_epoch=decision_epoch,
            source_observation_digest=_digest(
                {
                    "operation_kind": operation_kind,
                    "schema_version": OPERATION_SCHEMA_VERSION,
                    "source_material": sanitized,
                }
            ),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "callsite_manifest_digest": self.callsite_manifest_digest,
            "decision_epoch": self.decision_epoch,
            "legacy_goal_code": self.legacy_goal_code,
            "operation_kind": self.operation_kind,
            "schema_version": self.schema_version,
            "source_observation_digest": self.source_observation_digest,
        }


@dataclass(frozen=True, slots=True)
class ProductionGoalComparisonInput:
    operation: ProductionGoalOperation
    legacy_before: LegacyGoalStateSnapshot
    implementation_pin_digest: str
    authorization_artifact_digest: str
    legacy_mapping_digest: str
    v4_evaluator_digest: str
    schema_version: str = COMPARISON_INPUT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.operation, ProductionGoalOperation):
            raise M3CShadowTapError("operation must be ProductionGoalOperation")
        if not isinstance(self.legacy_before, LegacyGoalStateSnapshot):
            raise M3CShadowTapError("legacy_before must be LegacyGoalStateSnapshot")
        for field in (
            "implementation_pin_digest",
            "authorization_artifact_digest",
            "legacy_mapping_digest",
            "v4_evaluator_digest",
        ):
            _sha256(getattr(self, field), field=field)
        if self.schema_version != COMPARISON_INPUT_SCHEMA_VERSION:
            raise M3CShadowTapError("unsupported comparison input schema")

    def to_mapping(self) -> dict[str, Any]:
        return {
            "authorization_artifact_digest": self.authorization_artifact_digest,
            "implementation_pin_digest": self.implementation_pin_digest,
            "legacy_before": self.legacy_before.to_mapping(),
            "legacy_mapping_digest": self.legacy_mapping_digest,
            "operation": self.operation.to_mapping(),
            "schema_version": self.schema_version,
            "v4_evaluator_digest": self.v4_evaluator_digest,
        }

    @property
    def comparison_input_digest(self) -> str:
        return _digest(self.to_mapping())


class V4ProductionShadowEvaluator(Protocol):
    evaluator_digest: str

    def __call__(
        self,
        comparison_input: ProductionGoalComparisonInput,
        legacy_after: LegacyGoalStateSnapshot,
    ) -> V4ShadowGoalObservation: ...


@dataclass(frozen=True, slots=True)
class ShadowTapExecution:
    authoritative_result: Any
    status: str
    operation: ProductionGoalOperation
    legacy_before: LegacyGoalStateSnapshot | None
    legacy_after: LegacyGoalStateSnapshot | None
    comparison_receipt: GoalDualReadComparisonReceipt | None
    failure_code: str | None = None
    authoritative_call_count: int = 1
    legacy_authority: str = LEGACY_AUTHORITY
    v4_authority: str = V4_AUTHORITY
    state_capture_performed: bool = False
    v4_evaluation_performed: bool = False
    comparison_performed: bool = False
    event_append_performed: bool = False
    persistence_write_performed: bool = False
    action_authorized: bool = False
    scheduler_authorized: bool = False
    speech_authorized: bool = False
    legacy_goal_authority_transferred: bool = False
    legacy_migration_authorized: bool = False
    m3_e_authority_open: bool = False
    schema_version: str = EXECUTION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _identifier(self.status, field="status")
        if self.failure_code is not None:
            _identifier(self.failure_code, field="failure_code")
        if self.authoritative_call_count != 1:
            raise M3CShadowTapError("authoritative legacy call count must be exactly one")
        if self.legacy_authority != LEGACY_AUTHORITY or self.v4_authority != V4_AUTHORITY:
            raise M3CShadowTapError("execution authority boundary mismatch")
        if any(
            (
                self.event_append_performed,
                self.persistence_write_performed,
                self.action_authorized,
                self.scheduler_authorized,
                self.speech_authorized,
                self.legacy_goal_authority_transferred,
                self.legacy_migration_authorized,
                self.m3_e_authority_open,
            )
        ):
            raise M3CShadowTapError("shadow execution cannot grant effects or authority")
        if self.schema_version != EXECUTION_SCHEMA_VERSION:
            raise M3CShadowTapError("unsupported execution schema")

    def to_mapping(self) -> dict[str, Any]:
        return {
            "action_authorized": self.action_authorized,
            "authoritative_call_count": self.authoritative_call_count,
            "comparison_performed": self.comparison_performed,
            "comparison_receipt_digest": (
                self.comparison_receipt.receipt_digest
                if self.comparison_receipt is not None
                else None
            ),
            "event_append_performed": self.event_append_performed,
            "failure_code": self.failure_code,
            "legacy_after_digest": self.legacy_after.state_digest if self.legacy_after else None,
            "legacy_authority": self.legacy_authority,
            "legacy_before_digest": self.legacy_before.state_digest if self.legacy_before else None,
            "legacy_goal_authority_transferred": self.legacy_goal_authority_transferred,
            "legacy_migration_authorized": self.legacy_migration_authorized,
            "m3_e_authority_open": self.m3_e_authority_open,
            "operation": self.operation.to_mapping(),
            "persistence_write_performed": self.persistence_write_performed,
            "scheduler_authorized": self.scheduler_authorized,
            "schema_version": self.schema_version,
            "speech_authorized": self.speech_authorized,
            "state_capture_performed": self.state_capture_performed,
            "status": self.status,
            "v4_authority": self.v4_authority,
            "v4_evaluation_performed": self.v4_evaluation_performed,
        }

    @property
    def execution_digest(self) -> str:
        return _digest(self.to_mapping())


class DormantProductionOriginGoalShadowTap:
    """Non-retaining wrapper that keeps legacy authoritative and v4 shadow-only."""

    def __init__(
        self,
        *,
        implementation_pin: ShadowTapImplementationPin | None = None,
        authorization_pin: ShadowTapAuthorizationPin | None = None,
        mapping_table: LegacyGoalMappingTable | None = None,
        v4_evaluator: V4ProductionShadowEvaluator | None = None,
    ) -> None:
        self._implementation_pin = implementation_pin
        self._authorization_pin = authorization_pin
        self._mapping_table = mapping_table
        self._v4_evaluator = v4_evaluator

    def _activation_status(self) -> tuple[str, str | None, str | None]:
        if self._implementation_pin is None:
            return "dormant_missing_implementation_pin", "missing_implementation_pin", None
        if self._authorization_pin is None:
            return "dormant_missing_authorization_pin", "missing_authorization_pin", None
        if self._mapping_table is None:
            return "blocked_missing_mapping_table", "missing_mapping_table", None
        if self._v4_evaluator is None:
            return "blocked_missing_v4_evaluator", "missing_v4_evaluator", None
        try:
            evaluator_digest = _sha256(
                getattr(self._v4_evaluator, "evaluator_digest", ""),
                field="v4_evaluator.evaluator_digest",
            )
        except M3CShadowTapError:
            return "blocked_invalid_v4_evaluator", "invalid_v4_evaluator_digest", None
        if not self._authorization_pin.binds(
            self._implementation_pin,
            self._mapping_table,
            evaluator_digest,
        ):
            return "blocked_exact_pin_mismatch", "exact_pin_mismatch", evaluator_digest
        return "authorized_shadow_only", None, evaluator_digest

    def execute_authoritative_once(
        self,
        *,
        goal_management: Any,
        operation: ProductionGoalOperation,
        authoritative_call: Callable[[], T],
    ) -> ShadowTapExecution:
        if not isinstance(operation, ProductionGoalOperation):
            raise M3CShadowTapError("operation must be ProductionGoalOperation")
        if not callable(authoritative_call):
            raise M3CShadowTapError("authoritative_call must be callable")

        status, failure_code, evaluator_digest = self._activation_status()
        if status != "authorized_shadow_only":
            result = authoritative_call()
            return ShadowTapExecution(
                authoritative_result=result,
                status=status,
                operation=operation,
                legacy_before=None,
                legacy_after=None,
                comparison_receipt=None,
                failure_code=failure_code,
            )

        try:
            before = capture_legacy_goal_state(goal_management)
        except Exception:
            result = authoritative_call()
            return ShadowTapExecution(
                authoritative_result=result,
                status="blocked_before_capture_failed",
                operation=operation,
                legacy_before=None,
                legacy_after=None,
                comparison_receipt=None,
                failure_code="before_capture_failed",
            )

        result = authoritative_call()
        try:
            after = capture_legacy_goal_state(goal_management)
        except Exception:
            return ShadowTapExecution(
                authoritative_result=result,
                status="blocked_after_capture_failed",
                operation=operation,
                legacy_before=before,
                legacy_after=None,
                comparison_receipt=None,
                failure_code="after_capture_failed",
                state_capture_performed=True,
            )

        assert self._implementation_pin is not None
        assert self._authorization_pin is not None
        assert self._mapping_table is not None
        assert self._v4_evaluator is not None
        assert evaluator_digest is not None

        if before.structural_manifest_digest != after.structural_manifest_digest:
            return ShadowTapExecution(
                authoritative_result=result,
                status="blocked_legacy_structure_changed",
                operation=operation,
                legacy_before=before,
                legacy_after=after,
                comparison_receipt=None,
                failure_code="legacy_structure_changed",
                state_capture_performed=True,
            )

        comparison_input = ProductionGoalComparisonInput(
            operation=operation,
            legacy_before=before,
            implementation_pin_digest=self._implementation_pin.pin_digest,
            authorization_artifact_digest=self._authorization_pin.authorization_artifact_digest,
            legacy_mapping_digest=self._mapping_table.table_digest,
            v4_evaluator_digest=evaluator_digest,
        )

        semantic_goal_id: str | None = None
        lifecycle_state: str | None = None
        if after.top_goal_category_sha256 is not None:
            try:
                mapping = self._mapping_table.resolve(
                    legacy_goal_code=operation.legacy_goal_code,
                    category_sha256=after.top_goal_category_sha256,
                    legacy_status=after.top_goal_status or "",
                )
            except M3CShadowTapError:
                return ShadowTapExecution(
                    authoritative_result=result,
                    status="blocked_exact_legacy_mapping_unavailable",
                    operation=operation,
                    legacy_before=before,
                    legacy_after=after,
                    comparison_receipt=None,
                    failure_code="exact_legacy_mapping_unavailable",
                    state_capture_performed=True,
                )
            semantic_goal_id = mapping.semantic_goal_id
            lifecycle_state = mapping.v4_lifecycle_state

        legacy = LegacyGoalObservation(
            comparison_input_digest=comparison_input.comparison_input_digest,
            source_observation_digest=operation.source_observation_digest,
            legacy_goal_code=operation.legacy_goal_code,
            semantic_goal_id=semantic_goal_id,
            lifecycle_state=lifecycle_state,
            decision_epoch=operation.decision_epoch,
            before_state_digest=before.state_digest,
            after_state_digest=after.state_digest,
            structural_manifest_digest=after.structural_manifest_digest,
        )
        try:
            v4 = self._v4_evaluator(comparison_input, after)
            if not isinstance(v4, V4ShadowGoalObservation):
                raise M3CShadowTapError("v4 evaluator returned wrong observation type")
            comparison = compare_goal_observations(legacy, v4)
        except Exception:
            return ShadowTapExecution(
                authoritative_result=result,
                status="blocked_v4_evaluation_or_comparison_failed",
                operation=operation,
                legacy_before=before,
                legacy_after=after,
                comparison_receipt=None,
                failure_code="v4_evaluation_or_comparison_failed",
                state_capture_performed=True,
                v4_evaluation_performed=True,
            )

        return ShadowTapExecution(
            authoritative_result=result,
            status="comparison_ready_in_memory_only",
            operation=operation,
            legacy_before=before,
            legacy_after=after,
            comparison_receipt=comparison,
            state_capture_performed=True,
            v4_evaluation_performed=True,
            comparison_performed=True,
        )
