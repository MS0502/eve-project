"""Pure read-only M3-C-L legacy/v4 goal dual-read comparator preflight.

The comparator accepts immutable fixture observations plus genuine M3-C-B
selection and M3-C-C lifecycle receipts. It derives one canonical comparison
verdict without importing production orchestration, opening persistence,
mutating legacy state, appending events, or granting downstream authority.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Mapping

from core.m3_c_b_goal_selection_kernel import GoalSelectionReceipt
from core.m3_c_c_goal_lifecycle_kernel import (
    LIFECYCLE_STATES,
    LifecycleEvaluationReceipt,
)

COMPARATOR_SCHEMA_VERSION = "eve.m3-c-l.goal-dual-read-comparator.v1"
LEGACY_OBSERVATION_SCHEMA_VERSION = "eve.m3-c-l.legacy-goal-observation.v1"
V4_OBSERVATION_SCHEMA_VERSION = "eve.m3-c-l.v4-shadow-goal-observation.v1"
MAPPING_RULE_SCHEMA_VERSION = "eve.m3-c-l.goal-comparison-rule.v1"
COMPARISON_RECEIPT_SCHEMA_VERSION = "eve.m3-c-l.goal-comparison-receipt.v1"
MAPPING_VERSION = "eve.m3-c-k.legacy-goal-mapping.v1"

LEGACY_AUTHORITY = "legacy_authoritative"
V4_AUTHORITY = "shadow_only"

COMPARISON_VERDICTS = frozenset(
    {
        "exact_equivalent",
        "mapped_equivalent",
        "expected_design_difference",
        "unexplained_divergence",
        "legacy_only_behavior",
        "v4_only_behavior",
        "comparison_unavailable",
    }
)
RULE_RULINGS = frozenset({"mapped_equivalent", "expected_design_difference"})

_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9._:/-]{0,127}$")
_DIGEST = re.compile(r"^[0-9a-f]{64}$")


class M3CGoalComparisonError(ValueError):
    """Fail-closed error for invalid M3-C-L comparison material."""


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


def _sha256(value: str, *, field: str) -> str:
    if not isinstance(value, str) or not _DIGEST.fullmatch(value):
        raise M3CGoalComparisonError(f"{field} must be lowercase SHA-256")
    return value


def _identifier(value: str, *, field: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise M3CGoalComparisonError(
            f"{field} must be a canonical internal identifier"
        )
    return value


def _optional_identifier(value: str | None, *, field: str) -> str | None:
    if value is None:
        return None
    return _identifier(value, field=field)


def _optional_lifecycle(value: str | None, *, field: str) -> str | None:
    if value is None:
        return None
    if value not in LIFECYCLE_STATES:
        raise M3CGoalComparisonError(f"{field} must be a reviewed lifecycle state")
    return value


def _decision_epoch(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise M3CGoalComparisonError(
            "decision_epoch must be a non-negative integer"
        )
    return value


@dataclass(frozen=True, slots=True)
class LegacyGoalObservation:
    comparison_input_digest: str
    source_observation_digest: str
    legacy_goal_code: str
    semantic_goal_id: str | None
    lifecycle_state: str | None
    decision_epoch: int
    before_state_digest: str
    after_state_digest: str
    structural_manifest_digest: str
    authority: str = LEGACY_AUTHORITY
    schema_version: str = LEGACY_OBSERVATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _sha256(self.comparison_input_digest, field="comparison_input_digest")
        _sha256(self.source_observation_digest, field="source_observation_digest")
        _identifier(self.legacy_goal_code, field="legacy_goal_code")
        semantic_goal_id = _optional_identifier(
            self.semantic_goal_id,
            field="semantic_goal_id",
        )
        lifecycle_state = _optional_lifecycle(
            self.lifecycle_state,
            field="lifecycle_state",
        )
        if (semantic_goal_id is None) != (lifecycle_state is None):
            raise M3CGoalComparisonError(
                "legacy semantic goal and lifecycle state must be present together"
            )
        _decision_epoch(self.decision_epoch)
        _sha256(self.before_state_digest, field="before_state_digest")
        _sha256(self.after_state_digest, field="after_state_digest")
        _sha256(
            self.structural_manifest_digest,
            field="structural_manifest_digest",
        )
        if self.authority != LEGACY_AUTHORITY:
            raise M3CGoalComparisonError(
                "legacy observation must remain legacy-authoritative"
            )
        if self.schema_version != LEGACY_OBSERVATION_SCHEMA_VERSION:
            raise M3CGoalComparisonError(
                "unsupported legacy observation schema version"
            )

    @property
    def state_changed(self) -> bool:
        return self.before_state_digest != self.after_state_digest

    def to_mapping(self) -> dict[str, Any]:
        return {
            "after_state_digest": self.after_state_digest,
            "authority": self.authority,
            "before_state_digest": self.before_state_digest,
            "comparison_input_digest": self.comparison_input_digest,
            "decision_epoch": self.decision_epoch,
            "legacy_goal_code": self.legacy_goal_code,
            "lifecycle_state": self.lifecycle_state,
            "schema_version": self.schema_version,
            "semantic_goal_id": self.semantic_goal_id,
            "source_observation_digest": self.source_observation_digest,
            "state_changed": self.state_changed,
            "structural_manifest_digest": self.structural_manifest_digest,
        }

    @property
    def observation_digest(self) -> str:
        return _digest(self.to_mapping())


@dataclass(frozen=True, slots=True)
class V4ShadowGoalObservation:
    comparison_input_digest: str
    source_observation_digest: str
    projected_before_state_digest: str
    projected_after_state_digest: str
    structural_manifest_digest: str
    selection_receipt: GoalSelectionReceipt | None
    lifecycle_receipt: LifecycleEvaluationReceipt | None
    evaluation_available: bool = True
    unavailable_reason_code: str | None = None
    authority: str = V4_AUTHORITY
    schema_version: str = V4_OBSERVATION_SCHEMA_VERSION
    production_integration_performed: bool = False
    persistence_write_performed: bool = False
    event_append_performed: bool = False
    legacy_goal_mutation_performed: bool = False
    action_authorized: bool = False
    scheduler_authorized: bool = False
    speech_authorized: bool = False
    legacy_goal_authority_transferred: bool = False
    m3_e_authority_open: bool = False

    def __post_init__(self) -> None:
        _sha256(self.comparison_input_digest, field="comparison_input_digest")
        _sha256(self.source_observation_digest, field="source_observation_digest")
        _sha256(
            self.projected_before_state_digest,
            field="projected_before_state_digest",
        )
        _sha256(
            self.projected_after_state_digest,
            field="projected_after_state_digest",
        )
        _sha256(
            self.structural_manifest_digest,
            field="structural_manifest_digest",
        )
        if not isinstance(self.evaluation_available, bool):
            raise M3CGoalComparisonError("evaluation_available must be bool")
        if self.authority != V4_AUTHORITY:
            raise M3CGoalComparisonError("v4 observation must remain shadow-only")
        if self.schema_version != V4_OBSERVATION_SCHEMA_VERSION:
            raise M3CGoalComparisonError(
                "unsupported v4 observation schema version"
            )
        if any(
            (
                self.production_integration_performed,
                self.persistence_write_performed,
                self.event_append_performed,
                self.legacy_goal_mutation_performed,
                self.action_authorized,
                self.scheduler_authorized,
                self.speech_authorized,
                self.legacy_goal_authority_transferred,
                self.m3_e_authority_open,
            )
        ):
            raise M3CGoalComparisonError(
                "v4 shadow observation cannot claim effects or authority"
            )

        if not self.evaluation_available:
            if self.selection_receipt is not None or self.lifecycle_receipt is not None:
                raise M3CGoalComparisonError(
                    "unavailable evaluation cannot carry kernel receipts"
                )
            _identifier(
                self.unavailable_reason_code or "",
                field="unavailable_reason_code",
            )
            return

        if self.unavailable_reason_code is not None:
            raise M3CGoalComparisonError(
                "available evaluation cannot carry an unavailable reason"
            )
        if not isinstance(self.selection_receipt, GoalSelectionReceipt):
            raise M3CGoalComparisonError(
                "available evaluation requires GoalSelectionReceipt"
            )

        selected_candidate_id = self.selection_receipt.selected_candidate_id
        if selected_candidate_id is None:
            if self.lifecycle_receipt is not None:
                raise M3CGoalComparisonError(
                    "selection without a selected candidate cannot carry lifecycle receipt"
                )
            return

        if not isinstance(self.lifecycle_receipt, LifecycleEvaluationReceipt):
            raise M3CGoalComparisonError(
                "selected candidate requires LifecycleEvaluationReceipt"
            )
        state = self.lifecycle_receipt.state
        if state.candidate_id != selected_candidate_id:
            raise M3CGoalComparisonError(
                "selection and lifecycle candidate identity mismatch"
            )
        if state.decision_epoch != self.selection_receipt.decision_epoch:
            raise M3CGoalComparisonError(
                "selection and lifecycle decision epoch mismatch"
            )
        semantic_goal_id = self.semantic_goal_id
        if state.semantic_goal_id != semantic_goal_id:
            raise M3CGoalComparisonError(
                "selection and lifecycle semantic goal mismatch"
            )

    @property
    def projected_state_changed(self) -> bool:
        return self.projected_before_state_digest != self.projected_after_state_digest

    @property
    def selected_candidate_id(self) -> str | None:
        if not self.evaluation_available or self.selection_receipt is None:
            return None
        return self.selection_receipt.selected_candidate_id

    @property
    def semantic_goal_id(self) -> str | None:
        selected_candidate_id = self.selected_candidate_id
        if selected_candidate_id is None or self.selection_receipt is None:
            return None
        matches = tuple(
            item
            for item in self.selection_receipt.scored_candidates
            if item.candidate_id == selected_candidate_id
        )
        if len(matches) != 1:
            raise M3CGoalComparisonError(
                "selected candidate must appear exactly once in scored candidates"
            )
        return _identifier(
            matches[0].semantic_goal_id,
            field="v4_semantic_goal_id",
        )

    @property
    def lifecycle_state(self) -> str | None:
        if self.selected_candidate_id is None:
            return None
        if self.lifecycle_receipt is None:
            raise M3CGoalComparisonError(
                "selected candidate has no lifecycle receipt"
            )
        transition = self.lifecycle_receipt.transition
        state = (
            transition.after_state
            if transition is not None
            else self.lifecycle_receipt.state.lifecycle_state
        )
        return _optional_lifecycle(state, field="v4_lifecycle_state")

    @property
    def decision_epoch(self) -> int | None:
        if not self.evaluation_available or self.selection_receipt is None:
            return None
        return self.selection_receipt.decision_epoch

    def to_mapping(self) -> dict[str, Any]:
        return {
            "action_authorized": self.action_authorized,
            "authority": self.authority,
            "comparison_input_digest": self.comparison_input_digest,
            "decision_epoch": self.decision_epoch,
            "evaluation_available": self.evaluation_available,
            "event_append_performed": self.event_append_performed,
            "legacy_goal_authority_transferred": (
                self.legacy_goal_authority_transferred
            ),
            "legacy_goal_mutation_performed": self.legacy_goal_mutation_performed,
            "lifecycle_receipt": (
                self.lifecycle_receipt.to_mapping()
                if self.lifecycle_receipt is not None
                else None
            ),
            "lifecycle_state": self.lifecycle_state,
            "m3_e_authority_open": self.m3_e_authority_open,
            "persistence_write_performed": self.persistence_write_performed,
            "production_integration_performed": (
                self.production_integration_performed
            ),
            "projected_after_state_digest": self.projected_after_state_digest,
            "projected_before_state_digest": self.projected_before_state_digest,
            "projected_state_changed": self.projected_state_changed,
            "scheduler_authorized": self.scheduler_authorized,
            "schema_version": self.schema_version,
            "selected_candidate_id": self.selected_candidate_id,
            "selection_receipt": (
                self.selection_receipt.to_mapping()
                if self.selection_receipt is not None
                else None
            ),
            "semantic_goal_id": self.semantic_goal_id,
            "source_observation_digest": self.source_observation_digest,
            "speech_authorized": self.speech_authorized,
            "structural_manifest_digest": self.structural_manifest_digest,
            "unavailable_reason_code": self.unavailable_reason_code,
        }

    @property
    def observation_digest(self) -> str:
        return _digest(self.to_mapping())


@dataclass(frozen=True, slots=True)
class GoalComparisonRule:
    rule_id: str
    legacy_goal_code: str
    legacy_semantic_goal_id: str
    legacy_lifecycle_state: str
    v4_semantic_goal_id: str
    v4_lifecycle_state: str
    ruling: str
    rationale_code: str
    mapping_version: str = MAPPING_VERSION
    schema_version: str = MAPPING_RULE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _identifier(self.rule_id, field="rule_id")
        _identifier(self.legacy_goal_code, field="legacy_goal_code")
        _identifier(
            self.legacy_semantic_goal_id,
            field="legacy_semantic_goal_id",
        )
        _optional_lifecycle(
            self.legacy_lifecycle_state,
            field="legacy_lifecycle_state",
        )
        _identifier(self.v4_semantic_goal_id, field="v4_semantic_goal_id")
        _optional_lifecycle(
            self.v4_lifecycle_state,
            field="v4_lifecycle_state",
        )
        if self.ruling not in RULE_RULINGS:
            raise M3CGoalComparisonError("unsupported comparison rule ruling")
        _identifier(self.rationale_code, field="rationale_code")
        if self.mapping_version != MAPPING_VERSION:
            raise M3CGoalComparisonError("unsupported mapping version")
        if self.schema_version != MAPPING_RULE_SCHEMA_VERSION:
            raise M3CGoalComparisonError("unsupported mapping rule schema version")
        if (
            self.ruling == "mapped_equivalent"
            and self.legacy_semantic_goal_id == self.v4_semantic_goal_id
            and self.legacy_lifecycle_state == self.v4_lifecycle_state
        ):
            raise M3CGoalComparisonError(
                "identity match must be exact_equivalent without a mapping rule"
            )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "legacy_goal_code": self.legacy_goal_code,
            "legacy_lifecycle_state": self.legacy_lifecycle_state,
            "legacy_semantic_goal_id": self.legacy_semantic_goal_id,
            "mapping_version": self.mapping_version,
            "rationale_code": self.rationale_code,
            "rule_id": self.rule_id,
            "ruling": self.ruling,
            "schema_version": self.schema_version,
            "v4_lifecycle_state": self.v4_lifecycle_state,
            "v4_semantic_goal_id": self.v4_semantic_goal_id,
        }

    @property
    def rule_digest(self) -> str:
        return _digest(self.to_mapping())

    def matches(
        self,
        legacy: LegacyGoalObservation,
        v4: V4ShadowGoalObservation,
    ) -> bool:
        return (
            self.legacy_goal_code == legacy.legacy_goal_code
            and self.legacy_semantic_goal_id == legacy.semantic_goal_id
            and self.legacy_lifecycle_state == legacy.lifecycle_state
            and self.v4_semantic_goal_id == v4.semantic_goal_id
            and self.v4_lifecycle_state == v4.lifecycle_state
        )


@dataclass(frozen=True, slots=True)
class GoalDualReadComparisonReceipt:
    comparison_input_digest: str
    source_observation_digest: str
    legacy_observation_digest: str
    v4_observation_digest: str
    mapping_rule_digest: str | None
    verdict: str
    legacy_goal_code: str
    legacy_semantic_goal_id: str | None
    v4_semantic_goal_id: str | None
    legacy_lifecycle_state: str | None
    v4_lifecycle_state: str | None
    legacy_state_changed: bool
    v4_projected_state_changed: bool
    comparison_available: bool
    comparator_schema_version: str = COMPARATOR_SCHEMA_VERSION
    schema_version: str = COMPARISON_RECEIPT_SCHEMA_VERSION
    event_append_performed: bool = False
    persistence_write_performed: bool = False
    production_integration_performed: bool = False
    legacy_goal_mutation_performed: bool = False
    action_authorized: bool = False
    scheduler_authorized: bool = False
    speech_authorized: bool = False
    legacy_goal_authority_transferred: bool = False
    legacy_migration_authorized: bool = False
    m3_e_authority_open: bool = False

    def __post_init__(self) -> None:
        _sha256(self.comparison_input_digest, field="comparison_input_digest")
        _sha256(self.source_observation_digest, field="source_observation_digest")
        _sha256(self.legacy_observation_digest, field="legacy_observation_digest")
        _sha256(self.v4_observation_digest, field="v4_observation_digest")
        if self.mapping_rule_digest is not None:
            _sha256(self.mapping_rule_digest, field="mapping_rule_digest")
        if self.verdict not in COMPARISON_VERDICTS:
            raise M3CGoalComparisonError("unsupported comparison verdict")
        _identifier(self.legacy_goal_code, field="legacy_goal_code")
        _optional_identifier(
            self.legacy_semantic_goal_id,
            field="legacy_semantic_goal_id",
        )
        _optional_identifier(
            self.v4_semantic_goal_id,
            field="v4_semantic_goal_id",
        )
        _optional_lifecycle(
            self.legacy_lifecycle_state,
            field="legacy_lifecycle_state",
        )
        _optional_lifecycle(
            self.v4_lifecycle_state,
            field="v4_lifecycle_state",
        )
        for field in (
            "legacy_state_changed",
            "v4_projected_state_changed",
            "comparison_available",
        ):
            if not isinstance(getattr(self, field), bool):
                raise M3CGoalComparisonError(f"{field} must be bool")
        if self.comparator_schema_version != COMPARATOR_SCHEMA_VERSION:
            raise M3CGoalComparisonError("unsupported comparator schema version")
        if self.schema_version != COMPARISON_RECEIPT_SCHEMA_VERSION:
            raise M3CGoalComparisonError(
                "unsupported comparison receipt schema version"
            )
        if any(
            (
                self.event_append_performed,
                self.persistence_write_performed,
                self.production_integration_performed,
                self.legacy_goal_mutation_performed,
                self.action_authorized,
                self.scheduler_authorized,
                self.speech_authorized,
                self.legacy_goal_authority_transferred,
                self.legacy_migration_authorized,
                self.m3_e_authority_open,
            )
        ):
            raise M3CGoalComparisonError(
                "comparison receipt cannot claim effects or authority"
            )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "action_authorized": self.action_authorized,
            "comparator_schema_version": self.comparator_schema_version,
            "comparison_available": self.comparison_available,
            "comparison_input_digest": self.comparison_input_digest,
            "event_append_performed": self.event_append_performed,
            "legacy_goal_authority_transferred": (
                self.legacy_goal_authority_transferred
            ),
            "legacy_goal_code": self.legacy_goal_code,
            "legacy_goal_mutation_performed": self.legacy_goal_mutation_performed,
            "legacy_lifecycle_state": self.legacy_lifecycle_state,
            "legacy_migration_authorized": self.legacy_migration_authorized,
            "legacy_observation_digest": self.legacy_observation_digest,
            "legacy_semantic_goal_id": self.legacy_semantic_goal_id,
            "legacy_state_changed": self.legacy_state_changed,
            "m3_e_authority_open": self.m3_e_authority_open,
            "mapping_rule_digest": self.mapping_rule_digest,
            "persistence_write_performed": self.persistence_write_performed,
            "production_integration_performed": (
                self.production_integration_performed
            ),
            "scheduler_authorized": self.scheduler_authorized,
            "schema_version": self.schema_version,
            "source_observation_digest": self.source_observation_digest,
            "speech_authorized": self.speech_authorized,
            "v4_lifecycle_state": self.v4_lifecycle_state,
            "v4_observation_digest": self.v4_observation_digest,
            "v4_projected_state_changed": self.v4_projected_state_changed,
            "v4_semantic_goal_id": self.v4_semantic_goal_id,
            "verdict": self.verdict,
        }

    @property
    def receipt_digest(self) -> str:
        return _digest(self.to_mapping())


def _derive_verdict(
    legacy: LegacyGoalObservation,
    v4: V4ShadowGoalObservation,
    rule: GoalComparisonRule | None,
) -> str:
    if not v4.evaluation_available:
        if rule is not None:
            raise M3CGoalComparisonError(
                "comparison-unavailable result cannot consume a mapping rule"
            )
        return "comparison_unavailable"

    legacy_present = legacy.semantic_goal_id is not None
    v4_present = v4.semantic_goal_id is not None
    if legacy_present and not v4_present:
        if rule is not None:
            raise M3CGoalComparisonError(
                "legacy-only result cannot consume a mapping rule"
            )
        return "legacy_only_behavior"
    if v4_present and not legacy_present:
        if rule is not None:
            raise M3CGoalComparisonError(
                "v4-only result cannot consume a mapping rule"
            )
        return "v4_only_behavior"

    if (
        legacy.semantic_goal_id == v4.semantic_goal_id
        and legacy.lifecycle_state == v4.lifecycle_state
    ):
        if rule is not None:
            raise M3CGoalComparisonError(
                "exact-equivalent result must not consume a mapping rule"
            )
        return "exact_equivalent"

    if rule is None:
        return "unexplained_divergence"
    if not rule.matches(legacy, v4):
        raise M3CGoalComparisonError(
            "comparison rule does not match the exact observed tuple"
        )
    return rule.ruling


def compare_goal_observations(
    legacy: LegacyGoalObservation,
    v4: V4ShadowGoalObservation,
    *,
    rule: GoalComparisonRule | None = None,
) -> GoalDualReadComparisonReceipt:
    """Derive one canonical read-only legacy/v4 comparison receipt."""

    if not isinstance(legacy, LegacyGoalObservation):
        raise M3CGoalComparisonError(
            "legacy must be LegacyGoalObservation"
        )
    if not isinstance(v4, V4ShadowGoalObservation):
        raise M3CGoalComparisonError(
            "v4 must be V4ShadowGoalObservation"
        )
    if rule is not None and not isinstance(rule, GoalComparisonRule):
        raise M3CGoalComparisonError(
            "rule must be GoalComparisonRule"
        )
    if legacy.comparison_input_digest != v4.comparison_input_digest:
        raise M3CGoalComparisonError("comparison input digest mismatch")
    if legacy.source_observation_digest != v4.source_observation_digest:
        raise M3CGoalComparisonError("source observation digest mismatch")
    if v4.evaluation_available and legacy.decision_epoch != v4.decision_epoch:
        raise M3CGoalComparisonError("legacy/v4 decision epoch mismatch")

    verdict = _derive_verdict(legacy, v4, rule)
    return GoalDualReadComparisonReceipt(
        comparison_input_digest=legacy.comparison_input_digest,
        source_observation_digest=legacy.source_observation_digest,
        legacy_observation_digest=legacy.observation_digest,
        v4_observation_digest=v4.observation_digest,
        mapping_rule_digest=rule.rule_digest if rule is not None else None,
        verdict=verdict,
        legacy_goal_code=legacy.legacy_goal_code,
        legacy_semantic_goal_id=legacy.semantic_goal_id,
        v4_semantic_goal_id=v4.semantic_goal_id,
        legacy_lifecycle_state=legacy.lifecycle_state,
        v4_lifecycle_state=v4.lifecycle_state,
        legacy_state_changed=legacy.state_changed,
        v4_projected_state_changed=v4.projected_state_changed,
        comparison_available=v4.evaluation_available,
    )
