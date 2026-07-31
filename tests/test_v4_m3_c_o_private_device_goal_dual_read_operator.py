from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

import core.m3_c_o_private_device_goal_dual_read_operator as operator_module
from adapters.goal_adapter import GoalAdapter
from core.m3_c_b_goal_selection_kernel import ALLOWED_DRIVES, DriveSample, GoalCandidate
from core.m3_c_m_dormant_production_origin_shadow_tap import (
    LegacyGoalMappingEntry,
    LegacyGoalMappingTable,
    capture_legacy_goal_state,
)
from core.m3_c_n_bounded_private_device_goal_dual_read_window_preflight import (
    ACCEPTED_M3_C_M_EVIDENCE,
    BoundedDualReadWindowAuthorizationPacket,
    BoundedDualReadWindowPolicy,
    M3CNDualReadWindowError,
    PrivateDeviceWindowRollbackPlan,
)
from core.m3_c_o_private_device_goal_dual_read_operator import (
    LocalHumanReviewArtifact,
    M3COOperatorAuthorizationError,
    M3COOperatorExecutionError,
    PrivateDeviceGoalDualReadPackage,
    ReviewedGoalProbe,
    build_private_path_binding,
    execute_private_device_goal_dual_read_window,
    operator_manifest,
    read_canonical_private_package,
    require_single_use_private_paths,
)

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "core/m3_c_o_private_device_goal_dual_read_operator.py"
SCRIPT = ROOT / "scripts/operator/m3_c_o_private_device_goal_dual_read_window.py"
DESIGN = ROOT / "docs/audit/M3_C_O_PRIVATE_DEVICE_GOAL_DUAL_READ_OPERATOR.md"
REUSE = ROOT / "docs/audit/M3_C_N_PR238_VALIDATION_REUSE_PIN.json"
OPERATOR_HEAD = "a" * 40
LAUNCH_HEAD = "b" * 40


def _d(label: str) -> str:
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
    source: str = "m3-c-o-private-operator"
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
        self.authoritative_calls = 0

    def goal_set(self, category, priority=0.5, source="manual"):
        self.authoritative_calls += 1
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
        self.authoritative_calls += 1
        self.time += float(dt)
        self.tick_count += 1

    def active_goals(self):
        return sorted(
            self.goals.values(),
            key=lambda goal: (-goal.priority, goal.created, goal.id),
        )


def _candidate(semantic_goal_id: str, epoch: int, label: str) -> GoalCandidate:
    return GoalCandidate(
        semantic_goal_id=semantic_goal_id,
        decision_epoch=epoch,
        evidence_digest=_d(f"evidence:{label}"),
        base_value=1.0,
        expected_value=0.0,
        urgency=0.0,
        continuity=0.0,
        cost=0.0,
        risk=0.0,
        drive_alignment={drive: 0.0 for drive in ALLOWED_DRIVES},
        drive_confidence={drive: 1.0 for drive in ALLOWED_DRIVES},
    )


def _samples(epoch: int, label: str) -> tuple[DriveSample, ...]:
    return tuple(
        DriveSample(
            drive=drive,
            value=0.0,
            lower_bound=-1.0,
            upper_bound=1.0,
            sample_digest=_d(f"sample:{label}:{drive}"),
            replay_elapsed_seconds=float(epoch),
        )
        for drive in ALLOWED_DRIVES
    )


def _probes(*, mismatch: bool = False) -> tuple[ReviewedGoalProbe, ...]:
    semantic_a = "build_competence" if mismatch else "recover_operating_margin"
    return (
        ReviewedGoalProbe(
            operation_kind="goal_set",
            legacy_goal_code="legacy_goal_set_command",
            expected_decision_epoch=0,
            logical_step=1,
            category="private-alpha",
            priority=0.5,
            candidate=_candidate(semantic_a, 0, "alpha-set"),
            drive_samples=_samples(0, "alpha-set"),
        ),
        ReviewedGoalProbe(
            operation_kind="tick",
            legacy_goal_code="legacy_goal_tick",
            expected_decision_epoch=0,
            logical_step=2,
            dt=1.0,
            candidate=_candidate("recover_operating_margin", 0, "alpha-tick"),
            drive_samples=_samples(0, "alpha-tick"),
        ),
        ReviewedGoalProbe(
            operation_kind="goal_set",
            legacy_goal_code="legacy_goal_set_command",
            expected_decision_epoch=1,
            logical_step=3,
            category="private-beta",
            priority=0.9,
            candidate=_candidate("build_competence", 1, "beta-set"),
            drive_samples=_samples(1, "beta-set"),
        ),
        ReviewedGoalProbe(
            operation_kind="tick",
            legacy_goal_code="legacy_goal_tick",
            expected_decision_epoch=1,
            logical_step=4,
            dt=2.0,
            candidate=_candidate("build_competence", 1, "beta-tick"),
            drive_samples=_samples(1, "beta-tick"),
        ),
    )


def _mapping() -> LegacyGoalMappingTable:
    return LegacyGoalMappingTable(
        entries=(
            LegacyGoalMappingEntry(
                legacy_goal_code="legacy_goal_set_command",
                category_sha256=_d("private-alpha"),
                legacy_status="active",
                semantic_goal_id="recover_operating_margin",
                v4_lifecycle_state="selected",
            ),
            LegacyGoalMappingEntry(
                legacy_goal_code="legacy_goal_tick",
                category_sha256=_d("private-alpha"),
                legacy_status="active",
                semantic_goal_id="recover_operating_margin",
                v4_lifecycle_state="selected",
            ),
            LegacyGoalMappingEntry(
                legacy_goal_code="legacy_goal_set_command",
                category_sha256=_d("private-beta"),
                legacy_status="active",
                semantic_goal_id="build_competence",
                v4_lifecycle_state="selected",
            ),
            LegacyGoalMappingEntry(
                legacy_goal_code="legacy_goal_tick",
                category_sha256=_d("private-beta"),
                legacy_status="active",
                semantic_goal_id="build_competence",
                v4_lifecycle_state="selected",
            ),
        )
    )


def _material(
    tmp_path: Path,
    gm: _GoalManagement,
    *,
    mismatch: bool = False,
):
    package_path = tmp_path / "private" / "reviewed-package.json"
    store_path = tmp_path / "private" / "window.jsonl"
    backup_path = tmp_path / "private" / "baseline.backup"
    restore_path = tmp_path / "separate" / "baseline.restore"
    package_path.parent.mkdir(parents=True)
    path_binding = build_private_path_binding(
        package_path=package_path,
        working_store_path=store_path,
        baseline_backup_path=backup_path,
        separate_restore_path=restore_path,
        forbidden_existing_path_digests=tuple(
            sorted((_d("m3-c-j-database"), _d("m3-c-j-backup")))
        ),
    )
    policy = BoundedDualReadWindowPolicy()
    rollback = PrivateDeviceWindowRollbackPlan(
        path_binding_digest=path_binding.path_binding_digest,
        baseline_state_digest=capture_legacy_goal_state(gm).state_digest,
    )
    review = LocalHumanReviewArtifact(
        reviewer_id="kim-minseok",
        review_statement_digest=_d("reviewed-m3-c-o-private-window"),
    )
    probes = _probes(mismatch=mismatch)
    evaluator_digest = operator_module._digest(
        {
            "probes": [item.evaluator_mapping() for item in probes],
            "schema_version": "eve.m3-c-o.reviewed-v4-evaluator.v1",
        }
    )
    mapping = _mapping()
    authorization = BoundedDualReadWindowAuthorizationPacket(
        window_implementation_head=OPERATOR_HEAD,
        accepted_m3_c_m_evidence_digest=ACCEPTED_M3_C_M_EVIDENCE.evidence_digest,
        compatibility_shadow_pin_digest=(
            ACCEPTED_M3_C_M_EVIDENCE.compatibility_shadow_pin.pin_digest
        ),
        legacy_mapping_digest=mapping.table_digest,
        v4_evaluator_digest=evaluator_digest,
        policy_digest=policy.policy_digest,
        path_binding_digest=path_binding.path_binding_digest,
        rollback_digest=rollback.rollback_digest,
        authorization_artifact_digest=review.review_digest,
        reviewer_id=review.reviewer_id,
    )
    package = PrivateDeviceGoalDualReadPackage(
        authorization=authorization,
        policy=policy,
        rollback=rollback,
        mapping_table=mapping,
        probes=probes,
        review_artifact=review,
    )
    package_path.write_text(
        operator_module._canonical(package.to_mapping()) + "\n",
        encoding="utf-8",
    )
    return (
        package,
        path_binding,
        package_path,
        store_path,
        backup_path,
        restore_path,
    )


def _activate(package, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        operator_module,
        "_ACTIVE_REVIEWED_OPERATOR_IMPLEMENTATION_HEAD",
        OPERATOR_HEAD,
    )
    monkeypatch.setattr(
        operator_module,
        "_ACTIVE_REVIEWED_OPERATOR_AUTHORIZATION_DIGEST",
        package.authorization.authorization_digest,
    )


def test_checked_in_operator_is_default_absent_and_manifest_is_closed():
    with pytest.raises(M3COOperatorAuthorizationError, match="no active reviewed"):
        operator_module.active_reviewed_operator_pin()
    manifest = operator_manifest()
    assert manifest["active_authorization_present"] is False
    assert manifest["execution_available_in_this_slice"] is False
    assert manifest["default_runtime_integration"] is False
    assert manifest["existing_m3_c_j_database_access"] is False
    assert manifest["single_use"] is True


def test_private_package_round_trip_binds_review_mapping_evaluator_and_no_raw_paths(
    tmp_path: Path,
):
    gm = _GoalManagement()
    package, _, package_path, *_ = _material(tmp_path, gm)
    restored = read_canonical_private_package(package_path)
    assert restored.to_mapping() == package.to_mapping()
    assert restored.package_digest == package.package_digest
    assert restored.authorization.legacy_mapping_digest == restored.mapping_table.table_digest
    assert restored.authorization.v4_evaluator_digest == restored.evaluator_digest
    assert restored.authorization.authorization_artifact_digest == (
        restored.review_artifact.review_digest
    )
    assert restored.raw_paths_embedded is False
    assert restored.existing_m3_c_j_material_embedded is False


def test_new_distinct_paths_and_prior_private_digest_exclusion_fail_closed(
    tmp_path: Path,
):
    package = tmp_path / "package.json"
    package.write_text("{}")
    existing_digest = operator_module.private_path_digest(tmp_path / "existing")
    with pytest.raises(M3CNDualReadWindowError, match="overlap"):
        build_private_path_binding(
            package_path=package,
            working_store_path=tmp_path / "existing",
            baseline_backup_path=tmp_path / "backup",
            separate_restore_path=tmp_path / "restore",
            forbidden_existing_path_digests=(existing_digest,),
        )


def test_four_production_origin_records_are_digest_only_and_legacy_runs_once_each(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    gm = _GoalManagement()
    (
        package,
        path_binding,
        package_path,
        store_path,
        backup_path,
        restore_path,
    ) = _material(tmp_path, gm)
    _activate(package, monkeypatch)
    require_single_use_private_paths(
        package_path=package_path,
        working_store_path=store_path,
        baseline_backup_path=backup_path,
        separate_restore_path=restore_path,
        path_binding=path_binding,
    )
    adapter = GoalAdapter(object(), object(), gm=gm)
    receipt, window = execute_private_device_goal_dual_read_window(
        package,
        goal_adapter=adapter,
        path_binding=path_binding,
        working_store_path=store_path,
        baseline_backup_path=backup_path,
        separate_restore_path=restore_path,
        launch_repository_head=LAUNCH_HEAD,
    )
    assert gm.authoritative_calls == 4
    assert receipt.record_count == 4
    assert window.record_count == 4
    assert window.human_gate_review_eligible is True
    assert receipt.human_gate_review_eligible is True
    assert adapter.production_origin_shadow_tap is None
    assert backup_path.is_file()
    assert restore_path.is_file()
    assert backup_path.read_bytes() == restore_path.read_bytes()
    stored = store_path.read_text(encoding="utf-8")
    assert "private-alpha" not in stored
    assert "private-beta" not in stored
    assert '"stage":"digest_record"' in stored
    assert '"legacy_goal_authority_transferred":false' in stored
    assert receipt.existing_m3_c_j_database_accessed is False
    assert receipt.default_runtime_integration_performed is False
    assert receipt.event_append_performed is False
    assert receipt.legacy_migration_authorized is False
    assert receipt.m3_e_authority_open is False


def test_blocking_divergence_is_retained_but_not_promoted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    gm = _GoalManagement()
    (
        package,
        path_binding,
        _package_path,
        store_path,
        backup_path,
        restore_path,
    ) = _material(tmp_path, gm, mismatch=True)
    _activate(package, monkeypatch)
    adapter = GoalAdapter(object(), object(), gm=gm)
    receipt, window = execute_private_device_goal_dual_read_window(
        package,
        goal_adapter=adapter,
        path_binding=path_binding,
        working_store_path=store_path,
        baseline_backup_path=backup_path,
        separate_restore_path=restore_path,
        launch_repository_head=LAUNCH_HEAD,
    )
    assert window.blocking_verdict_count >= 1
    assert window.human_gate_review_eligible is False
    assert receipt.human_gate_review_eligible is False
    assert receipt.legacy_goal_authority_transferred is False
    assert receipt.legacy_migration_authorized is False


def test_existing_store_refuses_cross_chat_rerun_and_preserves_first_window(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    gm = _GoalManagement()
    (
        package,
        path_binding,
        package_path,
        store_path,
        backup_path,
        restore_path,
    ) = _material(tmp_path, gm)
    _activate(package, monkeypatch)
    adapter = GoalAdapter(object(), object(), gm=gm)
    execute_private_device_goal_dual_read_window(
        package,
        goal_adapter=adapter,
        path_binding=path_binding,
        working_store_path=store_path,
        baseline_backup_path=backup_path,
        separate_restore_path=restore_path,
        launch_repository_head=LAUNCH_HEAD,
    )
    first = store_path.read_bytes()
    with pytest.raises(M3COOperatorExecutionError, match="target already exists"):
        require_single_use_private_paths(
            package_path=package_path,
            working_store_path=store_path,
            baseline_backup_path=backup_path,
            separate_restore_path=restore_path,
            path_binding=path_binding,
        )
    assert store_path.read_bytes() == first
    assert gm.authoritative_calls == 4


def test_scope_escape_and_wrong_exact_pin_fail_before_legacy_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    gm = _GoalManagement()
    package, path_binding, _, store, backup, restore = _material(tmp_path, gm)
    adapter = GoalAdapter(object(), object(), gm=gm)
    monkeypatch.setattr(
        operator_module,
        "_ACTIVE_REVIEWED_OPERATOR_IMPLEMENTATION_HEAD",
        "c" * 40,
    )
    monkeypatch.setattr(
        operator_module,
        "_ACTIVE_REVIEWED_OPERATOR_AUTHORIZATION_DIGEST",
        package.authorization.authorization_digest,
    )
    with pytest.raises(M3COOperatorAuthorizationError, match="implementation head"):
        execute_private_device_goal_dual_read_window(
            package,
            goal_adapter=adapter,
            path_binding=path_binding,
            working_store_path=store,
            baseline_backup_path=backup,
            separate_restore_path=restore,
            launch_repository_head=LAUNCH_HEAD,
        )
    assert gm.authoritative_calls == 0
    with pytest.raises(M3CNDualReadWindowError, match="escaped"):
        replace(package.authorization, action_authorized=True)


def test_operator_script_is_explicit_and_prerequisite_reuse_is_cross_chat_durable():
    module_tree = ast.parse(MODULE.read_text(encoding="utf-8"), filename=str(MODULE))
    assignments = {}
    for node in module_tree.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            assignments[node.target.id] = node.value
    for name in (
        "_ACTIVE_REVIEWED_OPERATOR_IMPLEMENTATION_HEAD",
        "_ACTIVE_REVIEWED_OPERATOR_AUTHORIZATION_DIGEST",
    ):
        assert isinstance(assignments[name], ast.Constant)
        assert assignments[name].value is None

    script = SCRIPT.read_text(encoding="utf-8")
    assert "--expected-head" in script
    assert "--package-file" in script
    assert "--working-store" in script
    assert "--baseline-backup" in script
    assert "--separate-restore" in script
    assert "--forbidden-path-digest" in script
    assert "active_reviewed_operator_pin()" in script
    assert "build_full_engine()" in script
    assert "production_origin_shadow_tap is not None" in script
    assert "m3_c_j" not in script.lower()
    assert "phone_witness" not in script
    assert "retained_sequences" not in script

    design = DESIGN.read_text(encoding="utf-8")
    reuse = REUSE.read_text(encoding="utf-8")
    for token in (
        operator_module.M3_C_N_EXACT_HEAD,
        str(operator_module.M3_C_N_EXACT_RUN),
        operator_module.M3_C_N_ARTIFACT_SHA256,
        str(operator_module.M3_C_N_M2E_RUN),
        operator_module.M3_C_N_MERGE_SHA,
        "actual private-device execution: false",
        "default runtime integration: false",
    ):
        assert token in design or token in reuse
    assert "chat change" in reuse
    assert "rerun_phone_witness_pr_211" in reuse
    assert "rerun_retention_sequences_1_through_5" in reuse
    assert "rerun_completed_private_device_operator" in reuse
