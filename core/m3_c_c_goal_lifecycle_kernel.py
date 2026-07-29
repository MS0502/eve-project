"""Pure M3-C-C goal-lifecycle transition-candidate kernel.

This module consumes immutable M3-C-B score/selection evidence and derives at
most one reviewed M3-C-A lifecycle edge. It performs no event append,
persistence write, production integration, action, scheduling, speech,
legacy-goal authority transfer, or M3-E activation.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping

from core.m3_c_b_goal_selection_kernel import (
    CandidateScore,
    GoalSelectionReceipt,
    PROPOSAL_ENTER_THRESHOLD,
    PROPOSAL_EXIT_THRESHOLD,
    SELECTION_MINIMUM_SCORE,
)

LIFECYCLE_KERNEL_VERSION = "eve.m3-c-c.goal-lifecycle-kernel.v1"
LIFECYCLE_STATE_VERSION = "eve.m3-c-c.goal-lifecycle-state.v1"
TRANSITION_CANDIDATE_VERSION = "eve.m3-c-a.goal-transition-candidate.v1"
TRANSITION_PREDICATE_VERSION = "eve.m3-c-a.goal-transition-predicate.v1"
EVALUATION_RECEIPT_VERSION = "eve.m3-c-c.goal-lifecycle-evaluation.v1"

LIFECYCLE_STATES = frozenset(
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
TERMINAL_STATES = frozenset({"rejected", "expired", "withdrawn", "superseded"})
ALLOWED_EDGES = frozenset(
    {
        ("absent", "proposed"),
        ("proposed", "validated"),
        ("proposed", "rejected"),
        ("proposed", "expired"),
        ("validated", "eligible"),
        ("validated", "rejected"),
        ("eligible", "selected"),
        ("eligible", "withdrawn"),
        ("selected", "superseded"),
        ("selected", "expired"),
        ("rejected", "absent"),
        ("expired", "absent"),
        ("withdrawn", "absent"),
        ("superseded", "absent"),
    }
)
VALIDATION_STATUSES = frozenset({"pending", "passed", "failed"})


class M3CGoalLifecycleError(ValueError):
    """Fail-closed lifecycle-contract error."""


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
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise M3CGoalLifecycleError(f"{field} must be lowercase SHA-256")
    return value


def _finite_score(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise M3CGoalLifecycleError("candidate score must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < -1.0 or result > 1.0:
        raise M3CGoalLifecycleError("candidate score must be finite in [-1, 1]")
    return result


@dataclass(frozen=True, slots=True)
class GoalLifecycleState:
    candidate_id: str
    semantic_goal_id: str
    decision_epoch: int
    evidence_digest: str
    lifecycle_state: str = "absent"
    last_transition_id: str | None = None
    state_version: str = LIFECYCLE_STATE_VERSION

    def __post_init__(self) -> None:
        _sha256(self.candidate_id, field="candidate_id")
        _sha256(self.evidence_digest, field="evidence_digest")
        if not isinstance(self.semantic_goal_id, str) or not self.semantic_goal_id:
            raise M3CGoalLifecycleError("semantic_goal_id must be non-empty")
        if (
            isinstance(self.decision_epoch, bool)
            or not isinstance(self.decision_epoch, int)
            or self.decision_epoch < 0
        ):
            raise M3CGoalLifecycleError("decision_epoch must be non-negative")
        if self.lifecycle_state not in LIFECYCLE_STATES:
            raise M3CGoalLifecycleError("unsupported lifecycle state")
        if self.last_transition_id is not None:
            _sha256(self.last_transition_id, field="last_transition_id")
        if self.state_version != LIFECYCLE_STATE_VERSION:
            raise M3CGoalLifecycleError("unsupported lifecycle state version")

    def to_mapping(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "decision_epoch": self.decision_epoch,
            "evidence_digest": self.evidence_digest,
            "last_transition_id": self.last_transition_id,
            "lifecycle_state": self.lifecycle_state,
            "semantic_goal_id": self.semantic_goal_id,
            "state_version": self.state_version,
        }


@dataclass(frozen=True, slots=True)
class LifecycleEvidence:
    candidate_score: CandidateScore
    logical_step: int
    evidence_fresh: bool = True
    validation_status: str = "pending"
    permanent_selection_failure: bool = False
    terminal_acknowledged: bool = False
    selection_receipt: GoalSelectionReceipt | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_score, CandidateScore):
            raise M3CGoalLifecycleError("candidate_score must be CandidateScore")
        _finite_score(self.candidate_score.score)
        if isinstance(self.logical_step, bool) or not isinstance(self.logical_step, int):
            raise M3CGoalLifecycleError("logical_step must be an integer")
        if self.logical_step < 0:
            raise M3CGoalLifecycleError("logical_step must be non-negative")
        if not isinstance(self.evidence_fresh, bool):
            raise M3CGoalLifecycleError("evidence_fresh must be bool")
        if self.validation_status not in VALIDATION_STATUSES:
            raise M3CGoalLifecycleError("unsupported validation status")
        if not isinstance(self.permanent_selection_failure, bool):
            raise M3CGoalLifecycleError("permanent_selection_failure must be bool")
        if not isinstance(self.terminal_acknowledged, bool):
            raise M3CGoalLifecycleError("terminal_acknowledged must be bool")
        if self.selection_receipt is not None and not isinstance(
            self.selection_receipt, GoalSelectionReceipt
        ):
            raise M3CGoalLifecycleError(
                "selection_receipt must be GoalSelectionReceipt"
            )


@dataclass(frozen=True, slots=True)
class GoalLifecycleTransitionCandidate:
    candidate_id: str
    semantic_goal_id: str
    decision_epoch: int
    before_state: str
    after_state: str
    logical_step: int
    evidence_digest: str
    candidate_score: float
    selection_receipt_digest: str | None
    prior_transition_id: str | None
    trigger_code: str
    schema_version: str = TRANSITION_CANDIDATE_VERSION
    predicate_version: str = TRANSITION_PREDICATE_VERSION
    event_eligible: bool = True
    event_append_performed: bool = False
    persistence_write_performed: bool = False
    production_integration_performed: bool = False
    action_authorized: bool = False
    scheduler_authorized: bool = False
    speech_authorized: bool = False
    legacy_goal_authority_transferred: bool = False
    m3_e_authority_open: bool = False

    def __post_init__(self) -> None:
        _sha256(self.candidate_id, field="candidate_id")
        _sha256(self.evidence_digest, field="evidence_digest")
        if self.prior_transition_id is not None:
            _sha256(self.prior_transition_id, field="prior_transition_id")
        if self.selection_receipt_digest is not None:
            _sha256(
                self.selection_receipt_digest,
                field="selection_receipt_digest",
            )
        if (self.before_state, self.after_state) not in ALLOWED_EDGES:
            raise M3CGoalLifecycleError("transition is not an allowed lifecycle edge")
        _finite_score(self.candidate_score)
        if not isinstance(self.trigger_code, str) or not self.trigger_code:
            raise M3CGoalLifecycleError("trigger_code must be non-empty")
        if self.schema_version != TRANSITION_CANDIDATE_VERSION:
            raise M3CGoalLifecycleError("unsupported transition schema version")
        if self.predicate_version != TRANSITION_PREDICATE_VERSION:
            raise M3CGoalLifecycleError("unsupported transition predicate version")
        if not self.event_eligible:
            raise M3CGoalLifecycleError("named transition must remain event-eligible")
        if any(
            (
                self.event_append_performed,
                self.persistence_write_performed,
                self.production_integration_performed,
                self.action_authorized,
                self.scheduler_authorized,
                self.speech_authorized,
                self.legacy_goal_authority_transferred,
                self.m3_e_authority_open,
            )
        ):
            raise M3CGoalLifecycleError(
                "transition candidate cannot claim downstream authority or effects"
            )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "action_authorized": self.action_authorized,
            "after_state": self.after_state,
            "before_state": self.before_state,
            "candidate_id": self.candidate_id,
            "candidate_score": self.candidate_score,
            "decision_epoch": self.decision_epoch,
            "event_append_performed": self.event_append_performed,
            "event_eligible": self.event_eligible,
            "evidence_digest": self.evidence_digest,
            "legacy_goal_authority_transferred": self.legacy_goal_authority_transferred,
            "logical_step": self.logical_step,
            "m3_e_authority_open": self.m3_e_authority_open,
            "persistence_write_performed": self.persistence_write_performed,
            "predicate_version": self.predicate_version,
            "prior_transition_id": self.prior_transition_id,
            "production_integration_performed": self.production_integration_performed,
            "scheduler_authorized": self.scheduler_authorized,
            "schema_version": self.schema_version,
            "selection_receipt_digest": self.selection_receipt_digest,
            "semantic_goal_id": self.semantic_goal_id,
            "speech_authorized": self.speech_authorized,
            "trigger_code": self.trigger_code,
        }

    @property
    def transition_id(self) -> str:
        return _digest(self.to_mapping())

    def next_state(self) -> GoalLifecycleState:
        return GoalLifecycleState(
            candidate_id=self.candidate_id,
            semantic_goal_id=self.semantic_goal_id,
            decision_epoch=self.decision_epoch,
            evidence_digest=self.evidence_digest,
            lifecycle_state=self.after_state,
            last_transition_id=self.transition_id,
        )


@dataclass(frozen=True, slots=True)
class LifecycleEvaluationReceipt:
    state: GoalLifecycleState
    logical_step: int
    decision_code: str
    transition: GoalLifecycleTransitionCandidate | None
    kernel_version: str = LIFECYCLE_KERNEL_VERSION
    event_append_performed: bool = False
    persistence_write_performed: bool = False
    production_integration_performed: bool = False
    legacy_goal_authority_transferred: bool = False
    m3_e_authority_open: bool = False
    schema_version: str = EVALUATION_RECEIPT_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.state, GoalLifecycleState):
            raise M3CGoalLifecycleError("state must be GoalLifecycleState")
        if not isinstance(self.decision_code, str) or not self.decision_code:
            raise M3CGoalLifecycleError("decision_code must be non-empty")
        if self.transition is not None and not isinstance(
            self.transition, GoalLifecycleTransitionCandidate
        ):
            raise M3CGoalLifecycleError(
                "transition must be GoalLifecycleTransitionCandidate"
            )
        if self.kernel_version != LIFECYCLE_KERNEL_VERSION:
            raise M3CGoalLifecycleError("unsupported lifecycle kernel version")
        if self.schema_version != EVALUATION_RECEIPT_VERSION:
            raise M3CGoalLifecycleError("unsupported evaluation receipt version")
        if any(
            (
                self.event_append_performed,
                self.persistence_write_performed,
                self.production_integration_performed,
                self.legacy_goal_authority_transferred,
                self.m3_e_authority_open,
            )
        ):
            raise M3CGoalLifecycleError(
                "evaluation receipt cannot claim downstream effects or authority"
            )

    @property
    def transition_eligible(self) -> bool:
        return self.transition is not None

    def to_mapping(self) -> dict[str, Any]:
        return {
            "decision_code": self.decision_code,
            "event_append_performed": self.event_append_performed,
            "kernel_version": self.kernel_version,
            "legacy_goal_authority_transferred": self.legacy_goal_authority_transferred,
            "logical_step": self.logical_step,
            "m3_e_authority_open": self.m3_e_authority_open,
            "persistence_write_performed": self.persistence_write_performed,
            "production_integration_performed": self.production_integration_performed,
            "schema_version": self.schema_version,
            "state": self.state.to_mapping(),
            "transition": (
                self.transition.to_mapping() if self.transition is not None else None
            ),
            "transition_eligible": self.transition_eligible,
        }

    @property
    def receipt_digest(self) -> str:
        return _digest(self.to_mapping())


def _selection_receipt_digest(
    state: GoalLifecycleState,
    evidence: LifecycleEvidence,
) -> str | None:
    receipt = evidence.selection_receipt
    if receipt is None:
        return None
    if receipt.decision_epoch != state.decision_epoch:
        raise M3CGoalLifecycleError("selection receipt decision epoch mismatch")
    matching_scores = [
        item for item in receipt.scored_candidates if item.candidate_id == state.candidate_id
    ]
    if len(matching_scores) != 1:
        raise M3CGoalLifecycleError(
            "selection receipt must contain candidate exactly once"
        )
    matching = matching_scores[0]
    if matching.semantic_goal_id != state.semantic_goal_id:
        raise M3CGoalLifecycleError("selection receipt semantic goal mismatch")
    if not math.isclose(
        matching.score,
        evidence.candidate_score.score,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise M3CGoalLifecycleError("selection receipt score mismatch")
    return receipt.receipt_digest


def _transition(
    state: GoalLifecycleState,
    evidence: LifecycleEvidence,
    *,
    after_state: str,
    trigger_code: str,
    selection_receipt_digest: str | None,
) -> LifecycleEvaluationReceipt:
    candidate = GoalLifecycleTransitionCandidate(
        candidate_id=state.candidate_id,
        semantic_goal_id=state.semantic_goal_id,
        decision_epoch=state.decision_epoch,
        before_state=state.lifecycle_state,
        after_state=after_state,
        logical_step=evidence.logical_step,
        evidence_digest=state.evidence_digest,
        candidate_score=evidence.candidate_score.score,
        selection_receipt_digest=selection_receipt_digest,
        prior_transition_id=state.last_transition_id,
        trigger_code=trigger_code,
    )
    return LifecycleEvaluationReceipt(
        state=state,
        logical_step=evidence.logical_step,
        decision_code=trigger_code,
        transition=candidate,
    )


def _no_transition(
    state: GoalLifecycleState,
    evidence: LifecycleEvidence,
    *,
    decision_code: str,
) -> LifecycleEvaluationReceipt:
    return LifecycleEvaluationReceipt(
        state=state,
        logical_step=evidence.logical_step,
        decision_code=decision_code,
        transition=None,
    )


def evaluate_lifecycle_transition(
    state: GoalLifecycleState,
    evidence: LifecycleEvidence,
) -> LifecycleEvaluationReceipt:
    """Derive at most one exact named lifecycle edge without appending it."""

    if not isinstance(state, GoalLifecycleState):
        raise M3CGoalLifecycleError("state must be GoalLifecycleState")
    if not isinstance(evidence, LifecycleEvidence):
        raise M3CGoalLifecycleError("evidence must be LifecycleEvidence")
    score = evidence.candidate_score
    if score.candidate_id != state.candidate_id:
        raise M3CGoalLifecycleError("candidate score identity mismatch")
    if score.semantic_goal_id != state.semantic_goal_id:
        raise M3CGoalLifecycleError("candidate score semantic goal mismatch")
    receipt_digest = _selection_receipt_digest(state, evidence)

    current = state.lifecycle_state
    if current == "absent":
        if evidence.evidence_fresh and score.score >= PROPOSAL_ENTER_THRESHOLD:
            return _transition(
                state,
                evidence,
                after_state="proposed",
                trigger_code="proposal_enter_threshold_met",
                selection_receipt_digest=receipt_digest,
            )
        return _no_transition(
            state,
            evidence,
            decision_code="proposal_not_entered",
        )

    if current == "proposed":
        if not evidence.evidence_fresh:
            return _transition(
                state,
                evidence,
                after_state="expired",
                trigger_code="proposed_evidence_stale",
                selection_receipt_digest=receipt_digest,
            )
        if evidence.validation_status == "failed":
            return _transition(
                state,
                evidence,
                after_state="rejected",
                trigger_code="candidate_validation_failed",
                selection_receipt_digest=receipt_digest,
            )
        if evidence.validation_status == "passed":
            return _transition(
                state,
                evidence,
                after_state="validated",
                trigger_code="candidate_validation_passed",
                selection_receipt_digest=receipt_digest,
            )
        return _no_transition(
            state,
            evidence,
            decision_code="candidate_validation_pending",
        )

    if current == "validated":
        if evidence.permanent_selection_failure:
            return _transition(
                state,
                evidence,
                after_state="rejected",
                trigger_code="selection_precondition_permanently_failed",
                selection_receipt_digest=receipt_digest,
            )
        if evidence.evidence_fresh and score.score >= SELECTION_MINIMUM_SCORE:
            return _transition(
                state,
                evidence,
                after_state="eligible",
                trigger_code="selection_minimum_met",
                selection_receipt_digest=receipt_digest,
            )
        return _no_transition(
            state,
            evidence,
            decision_code="selection_minimum_not_met",
        )

    if current == "eligible":
        if score.score <= PROPOSAL_EXIT_THRESHOLD:
            return _transition(
                state,
                evidence,
                after_state="withdrawn",
                trigger_code="proposal_exit_threshold_met",
                selection_receipt_digest=receipt_digest,
            )
        receipt = evidence.selection_receipt
        if (
            receipt is not None
            and receipt.transition_eligible
            and receipt.selected_candidate_id == state.candidate_id
            and receipt.decision_kind in {"initial_selection", "switched_selection"}
        ):
            return _transition(
                state,
                evidence,
                after_state="selected",
                trigger_code="deterministic_selection_confirmed",
                selection_receipt_digest=receipt_digest,
            )
        return _no_transition(
            state,
            evidence,
            decision_code="selection_not_confirmed",
        )

    if current == "selected":
        if not evidence.evidence_fresh:
            return _transition(
                state,
                evidence,
                after_state="expired",
                trigger_code="selected_evidence_stale",
                selection_receipt_digest=receipt_digest,
            )
        receipt = evidence.selection_receipt
        if (
            receipt is not None
            and receipt.transition_eligible
            and receipt.decision_kind == "switched_selection"
            and receipt.prior_selected_candidate_id == state.candidate_id
            and receipt.selected_candidate_id != state.candidate_id
        ):
            return _transition(
                state,
                evidence,
                after_state="superseded",
                trigger_code="selected_candidate_superseded",
                selection_receipt_digest=receipt_digest,
            )
        return _no_transition(
            state,
            evidence,
            decision_code="selected_state_retained",
        )

    if current in TERMINAL_STATES:
        if evidence.terminal_acknowledged:
            return _transition(
                state,
                evidence,
                after_state="absent",
                trigger_code=f"{current}_acknowledged",
                selection_receipt_digest=receipt_digest,
            )
        return _no_transition(
            state,
            evidence,
            decision_code=f"{current}_awaiting_acknowledgement",
        )

    raise M3CGoalLifecycleError("unhandled lifecycle state")
