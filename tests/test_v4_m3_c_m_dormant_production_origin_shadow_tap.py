from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass
from pathlib import Path

import pytest

from adapters.goal_adapter import GoalAdapter
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
    PRODUCTION_CALLSITES,
    PRODUCTION_CALLSITE_MANIFEST_DIGEST,
    DormantProductionOriginGoalShadowTap,
    LegacyGoalMappingEntry,
    LegacyGoalMappingTable,
    M3CShadowTapError,
    ProductionGoalOperation,
    ShadowTapAuthorizationPin,
    ShadowTapImplementationPin,
    capture_legacy_goal_state,
)
from utils.types import Meaning

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "core/m3_c_m_dormant_production_origin_shadow_tap.py"
ADAPTER = ROOT / "adapters/goal_adapter.py"
MAIN = ROOT / "main.py"
DESIGN = ROOT / "docs/audit/M3_C_M_DORMANT_PRODUCTION_ORIGIN_SHADOW_TAP.md"
REUSE_PIN = ROOT / "docs/audit/M3_C_L_PR236_VALIDATION_REUSE_PIN.json"


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


@dataclass
class _Goal:
    id: str
    category: str
    priority: float = 0.5
    deadline: float | None = None
    status: str = "active"
    progress: float = 0.0
    created: float = 0.0
    last_evaluated: float = 0.0
    completed_at: float | None = None
    source: str = "command"
    abandon_reason: str | None = None


class _GoalManagement:
    def __init__(self) -> None:
        self.goals = {}
        self.history = []
        self.time = 0.0
        self.tick_count = 0
        self._next_goal_id = 1
        self.set_count = 0
        self.completed_count = 0
        self.abandoned_count = 0
        self.expired_count = 0
        self.suggested_count = 0
        self.proposed_count = 0
        self.goal_set_calls = 0
        self.tick_calls = 0

    def goal_set(self, category, priority=0.5, source="command"):
        self.goal_set_calls += 1
        goal_id = f"goal_{self._next_goal_id:04d}"
        self._next_goal_id += 1
        self.goals[goal_id] = _Goal(
            id=goal_id,
            category=category,
            priority=priority,
            created=self.time,
            last_evaluated=self.time,
            source=source,
        )
        self.set_count += 1
        return goal_id

    def tick(self, dt=1.0):
        self.tick_calls += 1
        self.time += float(dt)
        self.tick_count += 1

    def active_goals(self):
        return sorted(
            self.goals.values(),
            key=lambda goal: (-goal.priority, goal.created, goal.id),
        )


def _samples(epoch: int):
    return {
        drive: DriveSample(
            drive=drive,
            value=0.0,
            lower_bound=-1.0,
            upper_bound=1.0,
            sample_digest=_digest(f"sample:{drive}:{epoch}"),
            replay_elapsed_seconds=float(epoch),
        )
        for drive in ALLOWED_DRIVES
    }


class _Evaluator:
    evaluator_digest = _digest("reviewed-v4-evaluator")

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, comparison_input, legacy_after):
        self.calls += 1
        epoch = comparison_input.operation.decision_epoch
        candidate = GoalCandidate(
            semantic_goal_id="recover_operating_margin",
            decision_epoch=epoch,
            evidence_digest=_digest(f"evidence:{epoch}"),
            base_value=1.0,
            expected_value=0.0,
            urgency=0.0,
            continuity=0.0,
            cost=0.0,
            risk=0.0,
            drive_alignment={drive: 0.0 for drive in ALLOWED_DRIVES},
            drive_confidence={drive: 1.0 for drive in ALLOWED_DRIVES},
        )
        selection = select_goal_proposal([candidate], _samples(epoch))
        score = next(
            item
            for item in selection.scored_candidates
            if item.candidate_id == candidate.candidate_id
        )
        lifecycle = evaluate_lifecycle_transition(
            GoalLifecycleState(
                candidate_id=candidate.candidate_id,
                semantic_goal_id=candidate.semantic_goal_id,
                decision_epoch=epoch,
                evidence_digest=candidate.evidence_digest,
                lifecycle_state="eligible",
            ),
            LifecycleEvidence(
                candidate_score=score,
                logical_step=epoch + 1,
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


def _operation(epoch=0):
    return ProductionGoalOperation.from_source_material(
        operation_kind="goal_set",
        legacy_goal_code="legacy_goal_set_command",
        decision_epoch=epoch,
        source_material={
            "category": "쉬다",
            "priority": 0.5,
            "source": "command",
        },
    )


def _pins_and_mapping(evaluator):
    implementation = ShadowTapImplementationPin(
        exact_head=_digest("implementation-head"),
        exact_run=1,
        artifact_name=f"exact-head-validation-{_digest('implementation-head')}",
        artifact_sha256=_digest("artifact"),
        merge_sha=_digest("merge"),
    )
    mapping = LegacyGoalMappingTable(
        entries=(
            LegacyGoalMappingEntry(
                legacy_goal_code="legacy_goal_set_command",
                category_sha256=_digest("쉬다"),
                legacy_status="active",
                semantic_goal_id="recover_operating_margin",
                v4_lifecycle_state="selected",
            ),
        )
    )
    authorization = ShadowTapAuthorizationPin(
        implementation_pin_digest=implementation.pin_digest,
        legacy_mapping_digest=mapping.table_digest,
        v4_evaluator_digest=evaluator.evaluator_digest,
        authorization_artifact_digest=_digest("authorization-artifact"),
        reviewer_id="kim-minseok",
    )
    return implementation, authorization, mapping


def test_callsite_manifest_is_exact_and_default_engine_has_no_tap():
    assert tuple(item["operation_kind"] for item in PRODUCTION_CALLSITES) == (
        "goal_set",
        "tick",
    )
    assert len(PRODUCTION_CALLSITE_MANIFEST_DIGEST) == 64
    assert "production_origin_shadow_tap=" not in MAIN.read_text(encoding="utf-8")


def test_missing_implementation_pin_runs_legacy_once_without_observation():
    gm = _GoalManagement()
    evaluator = _Evaluator()
    tap = DormantProductionOriginGoalShadowTap(v4_evaluator=evaluator)
    execution = tap.execute_authoritative_once(
        goal_management=gm,
        operation=_operation(),
        authoritative_call=lambda: gm.goal_set("쉬다"),
    )
    assert execution.status == "dormant_missing_implementation_pin"
    assert execution.authoritative_result == "goal_0001"
    assert execution.authoritative_call_count == 1
    assert gm.goal_set_calls == 1
    assert evaluator.calls == 0
    assert execution.legacy_before is None
    assert execution.comparison_receipt is None


def test_missing_authorization_or_exact_pin_mismatch_never_observes():
    gm = _GoalManagement()
    evaluator = _Evaluator()
    implementation, authorization, mapping = _pins_and_mapping(evaluator)
    missing = DormantProductionOriginGoalShadowTap(
        implementation_pin=implementation,
        mapping_table=mapping,
        v4_evaluator=evaluator,
    )
    result = missing.execute_authoritative_once(
        goal_management=gm,
        operation=_operation(),
        authoritative_call=lambda: gm.goal_set("쉬다"),
    )
    assert result.status == "dormant_missing_authorization_pin"
    assert gm.goal_set_calls == 1
    assert evaluator.calls == 0

    wrong = ShadowTapAuthorizationPin(
        implementation_pin_digest=implementation.pin_digest,
        legacy_mapping_digest=mapping.table_digest,
        v4_evaluator_digest=_digest("different-evaluator"),
        authorization_artifact_digest=authorization.authorization_artifact_digest,
        reviewer_id="kim-minseok",
    )
    gm2 = _GoalManagement()
    blocked = DormantProductionOriginGoalShadowTap(
        implementation_pin=implementation,
        authorization_pin=wrong,
        mapping_table=mapping,
        v4_evaluator=evaluator,
    ).execute_authoritative_once(
        goal_management=gm2,
        operation=_operation(),
        authoritative_call=lambda: gm2.goal_set("쉬다"),
    )
    assert blocked.status == "blocked_exact_pin_mismatch"
    assert gm2.goal_set_calls == 1
    assert evaluator.calls == 0


def test_authorized_path_executes_legacy_once_and_compares_in_memory():
    gm = _GoalManagement()
    evaluator = _Evaluator()
    implementation, authorization, mapping = _pins_and_mapping(evaluator)
    tap = DormantProductionOriginGoalShadowTap(
        implementation_pin=implementation,
        authorization_pin=authorization,
        mapping_table=mapping,
        v4_evaluator=evaluator,
    )
    execution = tap.execute_authoritative_once(
        goal_management=gm,
        operation=_operation(),
        authoritative_call=lambda: gm.goal_set("쉬다"),
    )
    assert gm.goal_set_calls == 1
    assert evaluator.calls == 1
    assert execution.status == "comparison_ready_in_memory_only"
    assert execution.comparison_receipt is not None
    assert execution.comparison_receipt.verdict == "exact_equivalent"
    assert execution.state_capture_performed is True
    assert execution.v4_evaluation_performed is True
    assert execution.comparison_performed is True
    assert execution.event_append_performed is False
    assert execution.persistence_write_performed is False
    assert execution.action_authorized is False
    assert execution.scheduler_authorized is False
    assert execution.speech_authorized is False
    assert execution.legacy_goal_authority_transferred is False
    assert execution.legacy_migration_authorized is False
    assert execution.m3_e_authority_open is False
    assert not hasattr(tap, "observations")
    assert not hasattr(tap, "history")


def test_missing_mapping_blocks_after_one_legacy_call_before_v4():
    gm = _GoalManagement()
    evaluator = _Evaluator()
    implementation, _, _ = _pins_and_mapping(evaluator)
    mapping = LegacyGoalMappingTable(
        entries=(
            LegacyGoalMappingEntry(
                legacy_goal_code="legacy_goal_set_command",
                category_sha256=_digest("공부"),
                legacy_status="active",
                semantic_goal_id="build_competence",
                v4_lifecycle_state="selected",
            ),
        )
    )
    authorization = ShadowTapAuthorizationPin(
        implementation_pin_digest=implementation.pin_digest,
        legacy_mapping_digest=mapping.table_digest,
        v4_evaluator_digest=evaluator.evaluator_digest,
        authorization_artifact_digest=_digest("authorization-artifact-2"),
        reviewer_id="kim-minseok",
    )
    execution = DormantProductionOriginGoalShadowTap(
        implementation_pin=implementation,
        authorization_pin=authorization,
        mapping_table=mapping,
        v4_evaluator=evaluator,
    ).execute_authoritative_once(
        goal_management=gm,
        operation=_operation(),
        authoritative_call=lambda: gm.goal_set("쉬다"),
    )
    assert execution.status == "blocked_exact_legacy_mapping_unavailable"
    assert gm.goal_set_calls == 1
    assert evaluator.calls == 0
    assert execution.comparison_receipt is None


def test_snapshot_is_deterministic_and_hides_raw_goal_text():
    gm = _GoalManagement()
    gm.goal_set("비공개 목표 문구")
    first = capture_legacy_goal_state(gm)
    second = capture_legacy_goal_state(gm)
    assert first == second
    assert first.top_goal_category_sha256 == _digest("비공개 목표 문구")
    assert "비공개 목표 문구" not in str(first.to_mapping())


def test_authorization_rejects_any_downstream_authority():
    evaluator = _Evaluator()
    implementation, _, mapping = _pins_and_mapping(evaluator)
    with pytest.raises(M3CShadowTapError, match="downstream authority"):
        ShadowTapAuthorizationPin(
            implementation_pin_digest=implementation.pin_digest,
            legacy_mapping_digest=mapping.table_digest,
            v4_evaluator_digest=evaluator.evaluator_digest,
            authorization_artifact_digest=_digest("bad-authorization"),
            reviewer_id="kim-minseok",
            action_authorized=True,
        )


def test_goal_adapter_default_and_dormant_injection_call_legacy_once():
    gm = _GoalManagement()
    adapter = GoalAdapter(object(), object(), gm=gm)
    adapter.observe_meaning(Meaning(intent="command", raw_text="쉬어"))
    assert gm.goal_set_calls == 1

    gm2 = _GoalManagement()
    evaluator = _Evaluator()
    adapter2 = GoalAdapter(
        object(),
        object(),
        gm=gm2,
        production_origin_shadow_tap=(
            DormantProductionOriginGoalShadowTap(v4_evaluator=evaluator)
        ),
    )
    adapter2.tick(2.0)
    assert gm2.tick_calls == 1
    assert evaluator.calls == 0


def test_module_has_no_io_network_persistence_or_private_database_surface():
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
        "os",
        "pathlib",
        "sqlite3",
        "subprocess",
        "socket",
        "requests",
        "urllib",
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


def test_adapter_lazy_import_and_documents_pin_closed_boundary():
    adapter_text = ADAPTER.read_text(encoding="utf-8")
    assert "production_origin_shadow_tap=None" in adapter_text
    assert "from core.m3_c_m_dormant_production_origin_shadow_tap" in adapter_text
    design = DESIGN.read_text(encoding="utf-8")
    reuse = REUSE_PIN.read_text(encoding="utf-8")
    for token in (
        "tap reachable by default: false",
        "legacy authoritative call count: exactly one",
        "retention / persistence write: false",
        "M3-C-N",
        "73e2fdcf9e4006c726a27304f39c7efb0826bc9f",
        "30620402653",
        "44678184335a8a5f4c25efd3b0e7085914554e7bf54bdc81409b1b966606e065",
        "30620402663",
        "dd524a820a58947f0b589cd0cd521ee35eda73da",
    ):
        assert token in design or token in reuse
    assert "rerun_completed_private_device_operator" in reuse
    assert "retained_sequences_1_through_5_rerun" in reuse
