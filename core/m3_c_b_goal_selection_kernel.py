"""Pure deterministic M3-C-B goal-candidate scoring and selection kernel.

The kernel implements the reviewed M3-C-A arithmetic without integrating it into
legacy GoalManagement or any production loop. It performs no persistence, event
append, action, scheduling, speech, drive/affect mutation, or M3-E cutover.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence

KERNEL_SCHEMA_VERSION = "eve.m3-c-b.goal-selection-kernel.v1"
CANDIDATE_SCHEMA_VERSION = "eve.m3-c-a.goal-candidate.v1"
SCORING_POLICY_VERSION = "eve.m3-c-a.goal-score.v1"
TRANSITION_PREDICATE_VERSION = "eve.m3-c-a.goal-transition-predicate.v1"
SELECTION_RECEIPT_VERSION = "eve.m3-c-a.goal-selection-receipt.v1"
DRIVE_DYNAMICS_VERSION = "eve.m3-a.drive-dynamics.v1"

ALLOWED_DRIVES = (
    "energy",
    "safety",
    "affiliation",
    "curiosity",
    "agency",
    "coherence",
    "competence",
    "expression",
)

PROPOSAL_ENTER_THRESHOLD = 0.20
PROPOSAL_EXIT_THRESHOLD = 0.10
SELECTION_MINIMUM_SCORE = 0.30
INITIAL_WINNER_MARGIN = 0.08
SWITCH_MARGIN = 0.12
SELECTION_COOLDOWN_SECONDS = 30.0

_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9._:/-]{0,127}$")
_DIGEST = re.compile(r"^[0-9a-f]{64}$")


class M3CGoalSelectionError(ValueError):
    """Fail-closed error for invalid M3-C selection material."""


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


def _finite(value: float, *, field: str, lower: float, upper: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise M3CGoalSelectionError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < lower or result > upper:
        raise M3CGoalSelectionError(f"{field} must be finite in [{lower}, {upper}]")
    return result


def _identifier(value: str, *, field: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise M3CGoalSelectionError(f"{field} must be a canonical internal identifier")
    return value


def _sha256(value: str, *, field: str) -> str:
    if not isinstance(value, str) or not _DIGEST.fullmatch(value):
        raise M3CGoalSelectionError(f"{field} must be lowercase SHA-256")
    return value


@dataclass(frozen=True, slots=True)
class DriveSample:
    drive: str
    value: float
    lower_bound: float
    upper_bound: float
    sample_digest: str
    replay_elapsed_seconds: float
    dynamics_version: str = DRIVE_DYNAMICS_VERSION
    predicate_version: str = TRANSITION_PREDICATE_VERSION

    def __post_init__(self) -> None:
        if self.drive not in ALLOWED_DRIVES:
            raise M3CGoalSelectionError("unsupported drive")
        lower = _finite(self.lower_bound, field="lower_bound", lower=-1.0, upper=1.0)
        upper = _finite(self.upper_bound, field="upper_bound", lower=-1.0, upper=1.0)
        if lower >= upper:
            raise M3CGoalSelectionError("drive bounds must be strictly ordered")
        value = _finite(self.value, field="value", lower=lower, upper=upper)
        elapsed = _finite(
            self.replay_elapsed_seconds,
            field="replay_elapsed_seconds",
            lower=0.0,
            upper=1e15,
        )
        _sha256(self.sample_digest, field="sample_digest")
        if self.dynamics_version != DRIVE_DYNAMICS_VERSION:
            raise M3CGoalSelectionError("unsupported drive dynamics version")
        if self.predicate_version != TRANSITION_PREDICATE_VERSION:
            raise M3CGoalSelectionError("unsupported transition predicate version")
        object.__setattr__(self, "lower_bound", lower)
        object.__setattr__(self, "upper_bound", upper)
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "replay_elapsed_seconds", elapsed)

    @property
    def normalized(self) -> float:
        value = (
            2.0 * self.value - (self.lower_bound + self.upper_bound)
        ) / (self.upper_bound - self.lower_bound)
        return max(-1.0, min(1.0, value))

    def to_mapping(self) -> dict[str, Any]:
        return {
            "drive": self.drive,
            "dynamics_version": self.dynamics_version,
            "lower_bound": self.lower_bound,
            "normalized": self.normalized,
            "predicate_version": self.predicate_version,
            "replay_elapsed_seconds": self.replay_elapsed_seconds,
            "sample_digest": self.sample_digest,
            "upper_bound": self.upper_bound,
            "value": self.value,
        }


@dataclass(frozen=True, slots=True)
class GoalCandidate:
    semantic_goal_id: str
    decision_epoch: int
    evidence_digest: str
    base_value: float
    expected_value: float
    urgency: float
    continuity: float
    cost: float
    risk: float
    drive_alignment: Mapping[str, float]
    drive_confidence: Mapping[str, float]
    schema_version: str = CANDIDATE_SCHEMA_VERSION
    scoring_policy_version: str = SCORING_POLICY_VERSION

    def __post_init__(self) -> None:
        _identifier(self.semantic_goal_id, field="semantic_goal_id")
        if (
            isinstance(self.decision_epoch, bool)
            or not isinstance(self.decision_epoch, int)
            or self.decision_epoch < 0
        ):
            raise M3CGoalSelectionError("decision_epoch must be a non-negative integer")
        _sha256(self.evidence_digest, field="evidence_digest")
        if self.schema_version != CANDIDATE_SCHEMA_VERSION:
            raise M3CGoalSelectionError("unsupported candidate schema version")
        if self.scoring_policy_version != SCORING_POLICY_VERSION:
            raise M3CGoalSelectionError("unsupported scoring policy version")

        for field, lower, upper in (
            ("base_value", -1.0, 1.0),
            ("expected_value", -1.0, 1.0),
            ("urgency", 0.0, 1.0),
            ("continuity", -1.0, 1.0),
            ("cost", 0.0, 1.0),
            ("risk", 0.0, 1.0),
        ):
            object.__setattr__(
                self,
                field,
                _finite(getattr(self, field), field=field, lower=lower, upper=upper),
            )

        alignment = dict(self.drive_alignment)
        confidence = dict(self.drive_confidence)
        if set(alignment) != set(ALLOWED_DRIVES):
            raise M3CGoalSelectionError("drive_alignment must contain exact eight drives")
        if set(confidence) != set(ALLOWED_DRIVES):
            raise M3CGoalSelectionError("drive_confidence must contain exact eight drives")
        clean_alignment = {
            drive: _finite(
                alignment[drive],
                field=f"drive_alignment[{drive}]",
                lower=-1.0,
                upper=1.0,
            )
            for drive in ALLOWED_DRIVES
        }
        clean_confidence = {
            drive: _finite(
                confidence[drive],
                field=f"drive_confidence[{drive}]",
                lower=0.0,
                upper=1.0,
            )
            for drive in ALLOWED_DRIVES
        }
        object.__setattr__(self, "drive_alignment", MappingProxyType(clean_alignment))
        object.__setattr__(self, "drive_confidence", MappingProxyType(clean_confidence))

    @property
    def candidate_id(self) -> str:
        material = (
            f"{self.schema_version}|{self.semantic_goal_id}|{self.decision_epoch}|"
            f"{self.evidence_digest}|{self.scoring_policy_version}"
        )
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    def to_mapping(self) -> dict[str, Any]:
        return {
            "base_value": self.base_value,
            "candidate_id": self.candidate_id,
            "continuity": self.continuity,
            "cost": self.cost,
            "decision_epoch": self.decision_epoch,
            "drive_alignment": dict(self.drive_alignment),
            "drive_confidence": dict(self.drive_confidence),
            "evidence_digest": self.evidence_digest,
            "expected_value": self.expected_value,
            "risk": self.risk,
            "schema_version": self.schema_version,
            "scoring_policy_version": self.scoring_policy_version,
            "semantic_goal_id": self.semantic_goal_id,
            "urgency": self.urgency,
        }


@dataclass(frozen=True, slots=True)
class PriorSelection:
    candidate_id: str
    selected_at_replay_seconds: float

    def __post_init__(self) -> None:
        _sha256(self.candidate_id, field="candidate_id")
        object.__setattr__(
            self,
            "selected_at_replay_seconds",
            _finite(
                self.selected_at_replay_seconds,
                field="selected_at_replay_seconds",
                lower=0.0,
                upper=1e15,
            ),
        )


@dataclass(frozen=True, slots=True)
class CandidateScore:
    candidate_id: str
    semantic_goal_id: str
    score: float
    drive_term: float

    def to_mapping(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "drive_term": self.drive_term,
            "score": self.score,
            "semantic_goal_id": self.semantic_goal_id,
        }


@dataclass(frozen=True, slots=True)
class GoalSelectionReceipt:
    decision_epoch: int
    decision_kind: str
    selected_candidate_id: str | None
    prior_selected_candidate_id: str | None
    evaluated_winner_candidate_id: str | None
    evaluated_winner_score: float | None
    comparison_candidate_id: str | None
    comparison_score: float | None
    winner_margin: float | None
    cooldown_elapsed_seconds: float | None
    transition_eligible: bool
    candidate_set_digest: str
    drive_sample_digest: str
    scored_candidates: tuple[CandidateScore, ...]
    schema_version: str = SELECTION_RECEIPT_VERSION
    kernel_schema_version: str = KERNEL_SCHEMA_VERSION
    scoring_policy_version: str = SCORING_POLICY_VERSION
    transition_predicate_version: str = TRANSITION_PREDICATE_VERSION
    action_authorized: bool = False
    speech_authorized: bool = False
    persistence_write_performed: bool = False
    legacy_goal_authority_transferred: bool = False
    m3_e_authority_open: bool = False

    def __post_init__(self) -> None:
        allowed = {
            "no_candidate",
            "below_selection_threshold",
            "insufficient_initial_margin",
            "initial_selection",
            "retained_selection",
            "switch_cooldown",
            "insufficient_switch_margin",
            "switched_selection",
        }
        if self.decision_kind not in allowed:
            raise M3CGoalSelectionError("unsupported decision kind")
        if any((
            self.action_authorized,
            self.speech_authorized,
            self.persistence_write_performed,
            self.legacy_goal_authority_transferred,
            self.m3_e_authority_open,
        )):
            raise M3CGoalSelectionError("selection receipt cannot grant downstream authority")
        _sha256(self.candidate_set_digest, field="candidate_set_digest")
        _sha256(self.drive_sample_digest, field="drive_sample_digest")
        for field in (
            "selected_candidate_id",
            "prior_selected_candidate_id",
            "evaluated_winner_candidate_id",
            "comparison_candidate_id",
        ):
            value = getattr(self, field)
            if value is not None:
                _sha256(value, field=field)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "action_authorized": self.action_authorized,
            "candidate_set_digest": self.candidate_set_digest,
            "comparison_candidate_id": self.comparison_candidate_id,
            "comparison_score": self.comparison_score,
            "cooldown_elapsed_seconds": self.cooldown_elapsed_seconds,
            "decision_epoch": self.decision_epoch,
            "decision_kind": self.decision_kind,
            "drive_sample_digest": self.drive_sample_digest,
            "evaluated_winner_candidate_id": self.evaluated_winner_candidate_id,
            "evaluated_winner_score": self.evaluated_winner_score,
            "kernel_schema_version": self.kernel_schema_version,
            "legacy_goal_authority_transferred": self.legacy_goal_authority_transferred,
            "m3_e_authority_open": self.m3_e_authority_open,
            "persistence_write_performed": self.persistence_write_performed,
            "prior_selected_candidate_id": self.prior_selected_candidate_id,
            "schema_version": self.schema_version,
            "scored_candidates": [item.to_mapping() for item in self.scored_candidates],
            "scoring_policy_version": self.scoring_policy_version,
            "selected_candidate_id": self.selected_candidate_id,
            "speech_authorized": self.speech_authorized,
            "transition_eligible": self.transition_eligible,
            "transition_predicate_version": self.transition_predicate_version,
            "winner_margin": self.winner_margin,
        }

    @property
    def receipt_digest(self) -> str:
        return _digest(self.to_mapping())


def _validate_drive_samples(
    samples: Mapping[str, DriveSample],
) -> tuple[dict[str, DriveSample], float, str]:
    values = dict(samples)
    if set(values) != set(ALLOWED_DRIVES):
        raise M3CGoalSelectionError("drive samples must contain exact eight drives")
    for drive in ALLOWED_DRIVES:
        if not isinstance(values[drive], DriveSample) or values[drive].drive != drive:
            raise M3CGoalSelectionError("drive sample key/object mismatch")
    elapsed_values = {values[drive].replay_elapsed_seconds for drive in ALLOWED_DRIVES}
    if len(elapsed_values) != 1:
        raise M3CGoalSelectionError("drive samples must share one replay time")
    mapping = {drive: values[drive].to_mapping() for drive in ALLOWED_DRIVES}
    return values, next(iter(elapsed_values)), _digest(mapping)


def score_candidate(
    candidate: GoalCandidate,
    drive_samples: Mapping[str, DriveSample],
) -> CandidateScore:
    samples, _, _ = _validate_drive_samples(drive_samples)
    numerator = sum(
        candidate.drive_confidence[drive]
        * candidate.drive_alignment[drive]
        * samples[drive].normalized
        for drive in ALLOWED_DRIVES
    )
    denominator = sum(
        candidate.drive_confidence[drive] * abs(candidate.drive_alignment[drive])
        for drive in ALLOWED_DRIVES
    )
    drive_term = numerator / max(1.0, denominator)
    raw = (
        0.30 * candidate.base_value
        + 0.30 * drive_term
        + 0.15 * candidate.expected_value
        + 0.10 * candidate.urgency
        + 0.10 * candidate.continuity
        - 0.10 * candidate.cost
        - 0.15 * candidate.risk
    )
    return CandidateScore(
        candidate_id=candidate.candidate_id,
        semantic_goal_id=candidate.semantic_goal_id,
        score=max(-1.0, min(1.0, raw)),
        drive_term=drive_term,
    )


def select_goal_proposal(
    candidates: Sequence[GoalCandidate],
    drive_samples: Mapping[str, DriveSample],
    *,
    prior_selection: PriorSelection | None = None,
) -> GoalSelectionReceipt:
    samples, current_elapsed, drive_digest = _validate_drive_samples(drive_samples)
    candidate_values = tuple(candidates)
    if not candidate_values:
        return GoalSelectionReceipt(
            decision_epoch=0,
            decision_kind="no_candidate",
            selected_candidate_id=None,
            prior_selected_candidate_id=(
                prior_selection.candidate_id if prior_selection else None
            ),
            evaluated_winner_candidate_id=None,
            evaluated_winner_score=None,
            comparison_candidate_id=None,
            comparison_score=None,
            winner_margin=None,
            cooldown_elapsed_seconds=None,
            transition_eligible=False,
            candidate_set_digest=_digest({"candidates": []}),
            drive_sample_digest=drive_digest,
            scored_candidates=(),
        )
    if any(not isinstance(candidate, GoalCandidate) for candidate in candidate_values):
        raise M3CGoalSelectionError("candidates must be GoalCandidate values")
    epochs = {candidate.decision_epoch for candidate in candidate_values}
    if len(epochs) != 1:
        raise M3CGoalSelectionError("all candidates must share one decision epoch")
    ids = [candidate.candidate_id for candidate in candidate_values]
    if len(ids) != len(set(ids)):
        raise M3CGoalSelectionError("candidate set contains duplicate identity")

    candidate_set_digest = _digest({
        "candidates": [
            candidate.to_mapping()
            for candidate in sorted(candidate_values, key=lambda item: item.candidate_id)
        ]
    })
    scores = tuple(sorted(
        (score_candidate(candidate, samples) for candidate in candidate_values),
        key=lambda item: (-item.score, item.candidate_id),
    ))
    winner = scores[0]
    runner = scores[1] if len(scores) > 1 else None
    runner_score = runner.score if runner is not None else -1.0
    epoch = next(iter(epochs))

    base = {
        "decision_epoch": epoch,
        "evaluated_winner_candidate_id": winner.candidate_id,
        "evaluated_winner_score": winner.score,
        "candidate_set_digest": candidate_set_digest,
        "drive_sample_digest": drive_digest,
        "scored_candidates": scores,
    }

    if prior_selection is None:
        margin = winner.score - runner_score
        shared = {
            **base,
            "selected_candidate_id": None,
            "prior_selected_candidate_id": None,
            "comparison_candidate_id": runner.candidate_id if runner else None,
            "comparison_score": runner_score,
            "winner_margin": margin,
            "cooldown_elapsed_seconds": None,
            "transition_eligible": False,
        }
        if winner.score < SELECTION_MINIMUM_SCORE:
            return GoalSelectionReceipt(
                decision_kind="below_selection_threshold",
                **shared,
            )
        if margin < INITIAL_WINNER_MARGIN:
            return GoalSelectionReceipt(
                decision_kind="insufficient_initial_margin",
                **shared,
            )
        shared["selected_candidate_id"] = winner.candidate_id
        shared["transition_eligible"] = True
        return GoalSelectionReceipt(decision_kind="initial_selection", **shared)

    by_id = {item.candidate_id: item for item in scores}
    current = by_id.get(prior_selection.candidate_id)
    if current is None:
        raise M3CGoalSelectionError(
            "prior selected candidate must remain in candidate set; expiry is separate"
        )
    cooldown = current_elapsed - prior_selection.selected_at_replay_seconds
    if cooldown < 0.0:
        raise M3CGoalSelectionError("replay time cannot move backwards")

    if winner.candidate_id == current.candidate_id:
        comparison = runner
        margin = current.score - (comparison.score if comparison else -1.0)
        return GoalSelectionReceipt(
            decision_kind="retained_selection",
            selected_candidate_id=current.candidate_id,
            prior_selected_candidate_id=current.candidate_id,
            comparison_candidate_id=comparison.candidate_id if comparison else None,
            comparison_score=comparison.score if comparison else -1.0,
            winner_margin=margin,
            cooldown_elapsed_seconds=cooldown,
            transition_eligible=False,
            **base,
        )

    switch_margin = winner.score - current.score
    shared = {
        **base,
        "selected_candidate_id": current.candidate_id,
        "prior_selected_candidate_id": current.candidate_id,
        "comparison_candidate_id": current.candidate_id,
        "comparison_score": current.score,
        "winner_margin": switch_margin,
        "cooldown_elapsed_seconds": cooldown,
        "transition_eligible": False,
    }
    if cooldown < SELECTION_COOLDOWN_SECONDS:
        return GoalSelectionReceipt(decision_kind="switch_cooldown", **shared)
    if switch_margin < SWITCH_MARGIN:
        return GoalSelectionReceipt(
            decision_kind="insufficient_switch_margin",
            **shared,
        )
    shared["selected_candidate_id"] = winner.candidate_id
    shared["transition_eligible"] = True
    return GoalSelectionReceipt(decision_kind="switched_selection", **shared)
