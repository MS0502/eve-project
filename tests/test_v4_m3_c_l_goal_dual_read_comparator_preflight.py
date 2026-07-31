from __future__ import annotations

import ast
import hashlib
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

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
from core.m3_c_l_goal_dual_read_comparator_preflight import (
    COMPARISON_VERDICTS,
    GoalComparisonRule,
    LegacyGoalObservation,
    M3CGoalComparisonError,
    V4ShadowGoalObservation,
    compare_goal_observations,
)

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "core/m3_c_l_goal_dual_read_comparator_preflight.py"
DESIGN = ROOT / "docs/audit/M3_C_L_GOAL_DUAL_READ_COMPARATOR_PREFLIGHT.md"
REUSE_PIN = ROOT / "docs/audit/M3_C_K_PR235_VALIDATION_REUSE_PIN.json"


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _samples(*, elapsed: float = 0.0):
    return {
        drive: DriveSample(
            drive=drive,
            value=0.0,
            lower_bound=-1.0,
            upper_bound=1.0,
            sample_digest=_digest(f"sample:{drive}:{elapsed}"),
            replay_elapsed_seconds=elapsed,
        )
        for drive in ALLOWED_DRIVES
    }


def _candidate(semantic_goal_id: str, *, evidence: str):
    return GoalCandidate(
        semantic_goal_id=semantic_goal_id,
        decision_epoch=0,
        evidence_digest=_digest(evidence),
        base_value=1.0,
        expected_value=0.0,
        urgency=0.0,
        continuity=0.0,
        cost=0.0,
        risk=0.0,
        drive_alignment={drive: 0.0 for drive in ALLOWED_DRIVES},
        drive_confidence={drive: 1.0 for drive in ALLOWED_DRIVES},
    )


def _selected_kernel_receipts(semantic_goal_id: str, *, evidence: str):
    candidate = _candidate(semantic_goal_id, evidence=evidence)
    selection = select_goal_proposal([candidate], _samples())
    assert selection.decision_kind == "initial_selection"
    score = next(
        item
        for item in selection.scored_candidates
        if item.candidate_id == candidate.candidate_id
    )
    lifecycle = evaluate_lifecycle_transition(
        GoalLifecycleState(
            candidate_id=candidate.candidate_id,
            semantic_goal_id=candidate.semantic_goal_id,
            decision_epoch=candidate.decision_epoch,
            evidence_digest=candidate.evidence_digest,
            lifecycle_state="eligible",
        ),
        LifecycleEvidence(
            candidate_score=score,
            logical_step=1,
            selection_receipt=selection,
        ),
    )
    assert lifecycle.transition is not None
    assert lifecycle.transition.after_state == "selected"
    return candidate, selection, lifecycle


def _v4_selected(
    semantic_goal_id: str = "recover_operating_margin",
    *,
    comparison_input: str = "comparison-input",
    source: str = "source-observation",
    evidence: str = "v4-evidence",
):
    _, selection, lifecycle = _selected_kernel_receipts(
        semantic_goal_id,
        evidence=evidence,
    )
    return V4ShadowGoalObservation(
        comparison_input_digest=_digest(comparison_input),
        source_observation_digest=_digest(source),
        projected_before_state_digest=_digest("v4-before"),
        projected_after_state_digest=_digest("v4-after"),
        structural_manifest_digest=_digest("v4-manifest"),
        selection_receipt=selection,
        lifecycle_receipt=lifecycle,
    )


def _v4_none(
    *,
    comparison_input: str = "comparison-input",
    source: str = "source-observation",
):
    return V4ShadowGoalObservation(
        comparison_input_digest=_digest(comparison_input),
        source_observation_digest=_digest(source),
        projected_before_state_digest=_digest("v4-none-before"),
        projected_after_state_digest=_digest("v4-none-before"),
        structural_manifest_digest=_digest("v4-none-manifest"),
        selection_receipt=select_goal_proposal([], _samples()),
        lifecycle_receipt=None,
    )


def _v4_unavailable(
    *,
    comparison_input: str = "comparison-input",
    source: str = "source-observation",
):
    return V4ShadowGoalObservation(
        comparison_input_digest=_digest(comparison_input),
        source_observation_digest=_digest(source),
        projected_before_state_digest=_digest("v4-unavailable-before"),
        projected_after_state_digest=_digest("v4-unavailable-before"),
        structural_manifest_digest=_digest("v4-unavailable-manifest"),
        selection_receipt=None,
        lifecycle_receipt=None,
        evaluation_available=False,
        unavailable_reason_code="fixture_not_available",
    )


def _legacy(
    semantic_goal_id: str | None = "recover_operating_margin",
    lifecycle_state: str | None = "selected",
    *,
    legacy_goal_code: str = "legacy_recovery",
    comparison_input: str = "comparison-input",
    source: str = "source-observation",
    changed: bool = True,
):
    before = _digest("legacy-before")
    after = _digest("legacy-after") if changed else before
    return LegacyGoalObservation(
        comparison_input_digest=_digest(comparison_input),
        source_observation_digest=_digest(source),
        legacy_goal_code=legacy_goal_code,
        semantic_goal_id=semantic_goal_id,
        lifecycle_state=lifecycle_state,
        decision_epoch=0,
        before_state_digest=before,
        after_state_digest=after,
        structural_manifest_digest=_digest("legacy-manifest"),
    )


def test_verdict_catalog_is_exact_and_closed():
    assert COMPARISON_VERDICTS == {
        "exact_equivalent",
        "mapped_equivalent",
        "expected_design_difference",
        "unexplained_divergence",
        "legacy_only_behavior",
        "v4_only_behavior",
        "comparison_unavailable",
    }


def test_exact_equivalence_is_derived_from_real_kernel_receipts():
    legacy = _legacy()
    v4 = _v4_selected()
    first = compare_goal_observations(legacy, v4)
    second = compare_goal_observations(legacy, v4)

    assert first == second
    assert first.receipt_digest == second.receipt_digest
    assert first.verdict == "exact_equivalent"
    assert first.mapping_rule_digest is None
    assert first.legacy_state_changed is True
    assert first.v4_projected_state_changed is True
    assert len(first.receipt_digest) == 64


def test_mapped_equivalence_requires_exact_versioned_rule():
    legacy = _legacy(
        "legacy_recovery_goal",
        "selected",
        legacy_goal_code="legacy_recovery",
    )
    v4 = _v4_selected()
    rule = GoalComparisonRule(
        rule_id="legacy_recovery_to_v4_recovery",
        legacy_goal_code="legacy_recovery",
        legacy_semantic_goal_id="legacy_recovery_goal",
        legacy_lifecycle_state="selected",
        v4_semantic_goal_id="recover_operating_margin",
        v4_lifecycle_state="selected",
        ruling="mapped_equivalent",
        rationale_code="reviewed_semantic_alias",
    )
    result = compare_goal_observations(legacy, v4, rule=rule)
    assert result.verdict == "mapped_equivalent"
    assert result.mapping_rule_digest == rule.rule_digest


def test_expected_design_difference_requires_explicit_matching_rule():
    legacy = _legacy(
        "legacy_recovery_goal",
        "proposed",
        legacy_goal_code="legacy_recovery",
    )
    v4 = _v4_selected()
    rule = GoalComparisonRule(
        rule_id="legacy_proposed_v4_selected_expected",
        legacy_goal_code="legacy_recovery",
        legacy_semantic_goal_id="legacy_recovery_goal",
        legacy_lifecycle_state="proposed",
        v4_semantic_goal_id="recover_operating_margin",
        v4_lifecycle_state="selected",
        ruling="expected_design_difference",
        rationale_code="reviewed_lifecycle_model_difference",
    )
    result = compare_goal_observations(legacy, v4, rule=rule)
    assert result.verdict == "expected_design_difference"


def test_unknown_difference_fails_closed_as_unexplained_divergence():
    result = compare_goal_observations(
        _legacy("legacy_other_goal", "selected"),
        _v4_selected(),
    )
    assert result.verdict == "unexplained_divergence"
    assert result.mapping_rule_digest is None


@pytest.mark.parametrize(
    ("legacy", "v4", "expected"),
    [
        (_legacy(), _v4_none(), "legacy_only_behavior"),
        (_legacy(None, None, changed=False), _v4_selected(), "v4_only_behavior"),
        (_legacy(None, None, changed=False), _v4_none(), "exact_equivalent"),
        (_legacy(), _v4_unavailable(), "comparison_unavailable"),
    ],
)
def test_presence_and_availability_verdicts_are_distinct(legacy, v4, expected):
    result = compare_goal_observations(legacy, v4)
    assert result.verdict == expected
    assert result.comparison_available is (expected != "comparison_unavailable")


def test_state_change_is_derived_and_observations_are_immutable():
    legacy = _legacy(changed=False)
    v4 = _v4_none()
    result = compare_goal_observations(legacy, v4)
    assert legacy.state_changed is False
    assert v4.projected_state_changed is False
    assert result.legacy_state_changed is False
    assert result.v4_projected_state_changed is False
    with pytest.raises(FrozenInstanceError):
        legacy.decision_epoch = 1


def test_input_source_epoch_and_mapping_mismatches_fail_closed():
    with pytest.raises(M3CGoalComparisonError, match="comparison input"):
        compare_goal_observations(
            _legacy(comparison_input="left"),
            _v4_selected(comparison_input="right"),
        )
    with pytest.raises(M3CGoalComparisonError, match="source observation"):
        compare_goal_observations(
            _legacy(source="left"),
            _v4_selected(source="right"),
        )

    legacy = _legacy("legacy_recovery_goal", "selected")
    v4 = _v4_selected()
    wrong_rule = GoalComparisonRule(
        rule_id="wrong_exact_tuple",
        legacy_goal_code="different_legacy_code",
        legacy_semantic_goal_id="legacy_recovery_goal",
        legacy_lifecycle_state="selected",
        v4_semantic_goal_id="recover_operating_margin",
        v4_lifecycle_state="selected",
        ruling="mapped_equivalent",
        rationale_code="wrong_rule_fixture",
    )
    with pytest.raises(M3CGoalComparisonError, match="exact observed tuple"):
        compare_goal_observations(legacy, v4, rule=wrong_rule)


def test_selection_and_lifecycle_identity_mismatch_fails_closed():
    _, selection, _ = _selected_kernel_receipts(
        "recover_operating_margin",
        evidence="selection-a",
    )
    _, _, other_lifecycle = _selected_kernel_receipts(
        "explore_information_gap",
        evidence="lifecycle-b",
    )
    with pytest.raises(M3CGoalComparisonError, match="identity mismatch"):
        V4ShadowGoalObservation(
            comparison_input_digest=_digest("comparison-input"),
            source_observation_digest=_digest("source-observation"),
            projected_before_state_digest=_digest("v4-before"),
            projected_after_state_digest=_digest("v4-after"),
            structural_manifest_digest=_digest("v4-manifest"),
            selection_receipt=selection,
            lifecycle_receipt=other_lifecycle,
        )


def test_raw_text_like_identifiers_and_manual_authority_fail_closed():
    with pytest.raises(M3CGoalComparisonError, match="internal identifier"):
        _legacy("사용자가 방금 말한 원문", "selected")
    with pytest.raises(M3CGoalComparisonError, match="shadow-only"):
        V4ShadowGoalObservation(
            comparison_input_digest=_digest("comparison-input"),
            source_observation_digest=_digest("source-observation"),
            projected_before_state_digest=_digest("v4-before"),
            projected_after_state_digest=_digest("v4-after"),
            structural_manifest_digest=_digest("v4-manifest"),
            selection_receipt=select_goal_proposal([], _samples()),
            lifecycle_receipt=None,
            authority="authoritative",
        )


def test_comparison_receipt_never_grants_effects_or_authority():
    mapping = compare_goal_observations(_legacy(), _v4_selected()).to_mapping()
    assert mapping["event_append_performed"] is False
    assert mapping["persistence_write_performed"] is False
    assert mapping["production_integration_performed"] is False
    assert mapping["legacy_goal_mutation_performed"] is False
    assert mapping["action_authorized"] is False
    assert mapping["scheduler_authorized"] is False
    assert mapping["speech_authorized"] is False
    assert mapping["legacy_goal_authority_transferred"] is False
    assert mapping["legacy_migration_authorized"] is False
    assert mapping["m3_e_authority_open"] is False


def test_module_has_no_io_persistence_or_production_import_surface():
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
        "adapters",
        "language",
        "main",
        "os",
        "pathlib",
        "sqlite3",
        "subprocess",
        "threading",
        "time",
    }
    assert not calls & {
        "open",
        "write_text",
        "mkdir",
        "connect",
        "append",
        "emit",
        "schedule",
        "speak",
    }


def test_design_and_reuse_pin_preserve_exact_closed_boundary():
    design = DESIGN.read_text(encoding="utf-8")
    reuse = REUSE_PIN.read_text(encoding="utf-8")
    required_design = (
        "legacy executes exactly once as the sole behavior authority",
        "unexplained_divergence",
        "production runtime hook: false",
        "legacy goal-domain authority transferred: false",
        "M3-E authority open: false",
        "M3-C-M dormant production-origin shadow tap",
    )
    assert all(token in design for token in required_design)
    required_reuse = (
        "06a6495089fab4bf7e30ffb5a79180c4b748b6d2",
        "30618763444",
        "e55a84e0e3bc7d96f1e8e73fd0b8144a7f57077759c61d1dcce27ccaeb8f11fc",
        "30618765141",
        "d9c1cf8f615872b6a59ea7e950ccb9ceeb629133",
        "rerun_completed_private_device_operator",
    )
    assert all(token in reuse for token in required_reuse)
