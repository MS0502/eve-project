from __future__ import annotations

import ast
import hashlib
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from core.m3_c_b_goal_selection_kernel import (
    ALLOWED_DRIVES,
    INITIAL_WINNER_MARGIN,
    PROPOSAL_ENTER_THRESHOLD,
    PROPOSAL_EXIT_THRESHOLD,
    SELECTION_COOLDOWN_SECONDS,
    SELECTION_MINIMUM_SCORE,
    SWITCH_MARGIN,
    DriveSample,
    GoalCandidate,
    M3CGoalSelectionError,
    PriorSelection,
    select_goal_proposal,
)

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "core/m3_c_b_goal_selection_kernel.py"
DESIGN = ROOT / "docs/audit/M3_C_A_GOAL_GENERATION_SELECTION_DESIGN.md"


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _drive_samples(values: dict[str, float], *, elapsed: float):
    return {
        drive: DriveSample(
            drive=drive,
            value=values.get(drive, 0.0),
            lower_bound=-1.0,
            upper_bound=1.0,
            sample_digest=_digest(f"sample:{drive}:{values.get(drive, 0.0)}:{elapsed}"),
            replay_elapsed_seconds=elapsed,
        )
        for drive in ALLOWED_DRIVES
    }


def _candidate(
    semantic_goal_id: str,
    *,
    alignment: dict[str, float],
    evidence: str,
    base_value: float = 0.30,
    expected_value: float = 0.0,
    urgency: float = 0.0,
    continuity: float = 0.0,
    cost: float = 0.0,
    risk: float = 0.0,
):
    exact_alignment = {drive: alignment.get(drive, 0.0) for drive in ALLOWED_DRIVES}
    return GoalCandidate(
        semantic_goal_id=semantic_goal_id,
        decision_epoch=0,
        evidence_digest=_digest(evidence),
        base_value=base_value,
        expected_value=expected_value,
        urgency=urgency,
        continuity=continuity,
        cost=cost,
        risk=risk,
        drive_alignment=exact_alignment,
        drive_confidence={drive: 1.0 for drive in ALLOWED_DRIVES},
    )


def _counterfactual_candidates():
    recovery = _candidate(
        "recover_operating_margin",
        alignment={"energy": -0.90, "safety": -0.80, "curiosity": -0.10},
        evidence="recovery",
    )
    exploration = _candidate(
        "explore_information_gap",
        alignment={"energy": 0.30, "safety": 0.10, "curiosity": 0.90},
        evidence="exploration",
    )
    return recovery, exploration


def test_exact_design_constants_and_candidate_identity():
    assert PROPOSAL_ENTER_THRESHOLD == 0.20
    assert PROPOSAL_EXIT_THRESHOLD == 0.10
    assert SELECTION_MINIMUM_SCORE == 0.30
    assert INITIAL_WINNER_MARGIN == 0.08
    assert SWITCH_MARGIN == 0.12
    assert SELECTION_COOLDOWN_SECONDS == 30.0
    assert len(ALLOWED_DRIVES) == 8

    recovery, _ = _counterfactual_candidates()
    assert len(recovery.candidate_id) == 64
    assert recovery.candidate_id == _counterfactual_candidates()[0].candidate_id
    with pytest.raises(FrozenInstanceError):
        recovery.decision_epoch = 1


def test_invalid_or_raw_text_like_material_fails_closed():
    recovery, _ = _counterfactual_candidates()
    with pytest.raises(M3CGoalSelectionError, match="exact eight drives"):
        GoalCandidate(
            semantic_goal_id="bad",
            decision_epoch=0,
            evidence_digest=_digest("bad"),
            base_value=0.0,
            expected_value=0.0,
            urgency=0.0,
            continuity=0.0,
            cost=0.0,
            risk=0.0,
            drive_alignment={"energy": 0.0},
            drive_confidence={"energy": 1.0},
        )
    with pytest.raises(M3CGoalSelectionError, match="internal identifier"):
        GoalCandidate(
            semantic_goal_id="사용자가 방금 말한 원문",
            decision_epoch=0,
            evidence_digest=_digest("raw"),
            base_value=0.0,
            expected_value=0.0,
            urgency=0.0,
            continuity=0.0,
            cost=0.0,
            risk=0.0,
            drive_alignment=dict(recovery.drive_alignment),
            drive_confidence=dict(recovery.drive_confidence),
        )
    samples = _drive_samples({}, elapsed=0.0)
    samples.pop("expression")
    with pytest.raises(M3CGoalSelectionError, match="exact eight drives"):
        select_goal_proposal([recovery], samples)


def test_design_counterfactual_flips_selected_goal():
    recovery, exploration = _counterfactual_candidates()
    strained = select_goal_proposal(
        [recovery, exploration],
        _drive_samples(
            {"energy": -0.70, "safety": -0.80, "curiosity": -0.20},
            elapsed=0.0,
        ),
    )
    restored = select_goal_proposal(
        [recovery, exploration],
        _drive_samples(
            {"energy": 0.60, "safety": 0.70, "curiosity": 0.90},
            elapsed=0.0,
        ),
    )
    assert strained.decision_kind == restored.decision_kind == "initial_selection"
    assert strained.selected_candidate_id == recovery.candidate_id
    assert restored.selected_candidate_id == exploration.candidate_id
    assert strained.winner_margin >= INITIAL_WINNER_MARGIN
    assert restored.winner_margin >= INITIAL_WINNER_MARGIN
    assert strained.evaluated_winner_score >= SELECTION_MINIMUM_SCORE
    assert restored.evaluated_winner_score >= SELECTION_MINIMUM_SCORE


def test_argmax_tie_break_is_lexical_but_equal_margin_does_not_select():
    left = _candidate("candidate_left", alignment={}, evidence="left", base_value=1.0)
    right = _candidate("candidate_right", alignment={}, evidence="right", base_value=1.0)
    result = select_goal_proposal(
        [right, left],
        _drive_samples({}, elapsed=0.0),
    )
    assert result.decision_kind == "insufficient_initial_margin"
    assert result.transition_eligible is False
    assert [item.candidate_id for item in result.scored_candidates] == sorted(
        [left.candidate_id, right.candidate_id]
    )


def test_switch_requires_both_cooldown_and_larger_margin():
    recovery, exploration = _counterfactual_candidates()
    strained_samples = _drive_samples(
        {"energy": -0.70, "safety": -0.80, "curiosity": -0.20},
        elapsed=0.0,
    )
    first = select_goal_proposal([recovery, exploration], strained_samples)
    prior = PriorSelection(
        candidate_id=first.selected_candidate_id,
        selected_at_replay_seconds=0.0,
    )

    early = select_goal_proposal(
        [recovery, exploration],
        _drive_samples(
            {"energy": 0.60, "safety": 0.70, "curiosity": 0.90},
            elapsed=29.0,
        ),
        prior_selection=prior,
    )
    assert early.decision_kind == "switch_cooldown"
    assert early.selected_candidate_id == recovery.candidate_id
    assert early.transition_eligible is False

    switched = select_goal_proposal(
        [recovery, exploration],
        _drive_samples(
            {"energy": 0.60, "safety": 0.70, "curiosity": 0.90},
            elapsed=30.0,
        ),
        prior_selection=prior,
    )
    assert switched.decision_kind == "switched_selection"
    assert switched.selected_candidate_id == exploration.candidate_id
    assert switched.winner_margin >= SWITCH_MARGIN
    assert switched.transition_eligible is True


def test_repeated_unchanged_selection_is_deterministic_no_event_candidate():
    recovery, exploration = _counterfactual_candidates()
    samples = _drive_samples(
        {"energy": -0.70, "safety": -0.80, "curiosity": -0.20},
        elapsed=60.0,
    )
    prior = PriorSelection(recovery.candidate_id, 0.0)
    first = select_goal_proposal(
        [recovery, exploration], samples, prior_selection=prior
    )
    second = select_goal_proposal(
        [recovery, exploration], samples, prior_selection=prior
    )
    assert first == second
    assert first.receipt_digest == second.receipt_digest
    assert first.decision_kind == "retained_selection"
    assert first.transition_eligible is False


def test_receipt_never_grants_downstream_authority_or_writes():
    recovery, exploration = _counterfactual_candidates()
    result = select_goal_proposal(
        [recovery, exploration],
        _drive_samples(
            {"energy": -0.70, "safety": -0.80, "curiosity": -0.20},
            elapsed=0.0,
        ),
    )
    mapping = result.to_mapping()
    assert mapping["action_authorized"] is False
    assert mapping["speech_authorized"] is False
    assert mapping["persistence_write_performed"] is False
    assert mapping["legacy_goal_authority_transferred"] is False
    assert mapping["m3_e_authority_open"] is False
    assert len(result.receipt_digest) == 64


def test_module_has_no_io_runtime_or_legacy_import_surface():
    tree = ast.parse(MODULE.read_text(encoding="utf-8"), filename=str(MODULE))
    imported = set()
    calls = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".", 1)[0])
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                calls.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                calls.add(node.func.attr)
    assert not imported & {
        "adapters", "language", "main", "os", "pathlib", "sqlite3",
        "subprocess", "threading", "time",
    }
    assert not calls & {
        "open", "write_text", "mkdir", "connect", "append", "emit",
        "schedule", "speak",
    }


def test_design_boundary_is_still_explicit():
    text = DESIGN.read_text(encoding="utf-8")
    required = (
        "legacy goal authority:       unchanged until its own later migration gate",
        "Continuous drive values and continuously recomputed scores are **not events**.",
        "No runtime implementation.",
        "M3-E remains independently closed.",
    )
    assert all(token in text for token in required)
