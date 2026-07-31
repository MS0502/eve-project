from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import pytest

from core.m3_c_l_goal_dual_read_comparator_preflight import (
    GoalDualReadComparisonReceipt,
)
from core.m3_c_m_dormant_production_origin_shadow_tap import (
    LegacyGoalStateSnapshot,
    ProductionGoalOperation,
    ShadowTapExecution,
)
from core.m3_c_n_bounded_private_device_goal_dual_read_window_preflight import (
    ACCEPTED_M3_C_M_EVIDENCE,
    GENESIS_RECORD_DIGEST,
    M3_C_M_ARTIFACT_SHA256,
    M3_C_M_EXACT_HEAD,
    M3_C_M_EXACT_RUN,
    M3_C_M_MERGE_SHA,
    BoundedDualReadWindowAuthorizationPacket,
    BoundedDualReadWindowPolicy,
    GoalDualReadWindowRecord,
    M3CNDualReadWindowError,
    PrivateDeviceWindowPathBinding,
    PrivateDeviceWindowRollbackPlan,
    active_reviewed_window_authorization,
    evaluate_bounded_dual_read_window,
    private_device_operator_manifest,
)

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "core/m3_c_n_bounded_private_device_goal_dual_read_window_preflight.py"
DESIGN = ROOT / "docs/audit/M3_C_N_BOUNDED_PRIVATE_DEVICE_GOAL_DUAL_READ_WINDOW_PREFLIGHT.md"
REUSE = ROOT / "docs/audit/M3_C_M_PR237_VALIDATION_REUSE_PIN.json"


def _d(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _paths() -> PrivateDeviceWindowPathBinding:
    return PrivateDeviceWindowPathBinding(
        operator_input_path_digest=_d("new-operator-input"),
        working_store_path_digest=_d("new-window-store"),
        baseline_backup_path_digest=_d("new-window-backup"),
        separate_restore_path_digest=_d("new-window-restore"),
        forbidden_existing_path_digests=tuple(
            sorted((_d("m3-c-j-private-database"), _d("m3-c-j-backup")))
        ),
    )


def _material():
    policy = BoundedDualReadWindowPolicy()
    paths = _paths()
    rollback = PrivateDeviceWindowRollbackPlan(
        path_binding_digest=paths.path_binding_digest,
        baseline_state_digest=_d("legacy-baseline"),
    )
    authorization = BoundedDualReadWindowAuthorizationPacket(
        window_implementation_head="a" * 40,
        accepted_m3_c_m_evidence_digest=ACCEPTED_M3_C_M_EVIDENCE.evidence_digest,
        compatibility_shadow_pin_digest=(
            ACCEPTED_M3_C_M_EVIDENCE.compatibility_shadow_pin.pin_digest
        ),
        legacy_mapping_digest=_d("reviewed-mapping"),
        v4_evaluator_digest=_d("reviewed-evaluator"),
        policy_digest=policy.policy_digest,
        path_binding_digest=paths.path_binding_digest,
        rollback_digest=rollback.rollback_digest,
        authorization_artifact_digest=_d("reviewed-authorization-artifact"),
        reviewer_id="kim-minseok",
    )
    return policy, paths, rollback, authorization


def _execution(sequence: int, verdict: str = "exact_equivalent") -> ShadowTapExecution:
    operation = ProductionGoalOperation.from_source_material(
        operation_kind="goal_set",
        legacy_goal_code="legacy_goal_set_command",
        decision_epoch=sequence,
        source_material={"category": f"private-category-{sequence}"},
    )
    before = LegacyGoalStateSnapshot(
        state_digest=_d(f"before:{sequence}"),
        structural_manifest_digest=_d("stable-manifest"),
        active_count=sequence - 1,
        top_goal_category_sha256=(
            _d(f"previous-category:{sequence}") if sequence > 1 else None
        ),
        top_goal_status="active" if sequence > 1 else None,
    )
    after = LegacyGoalStateSnapshot(
        state_digest=_d(f"after:{sequence}"),
        structural_manifest_digest=_d("stable-manifest"),
        active_count=sequence,
        top_goal_category_sha256=_d(f"private-category-{sequence}"),
        top_goal_status="active",
    )
    receipt = GoalDualReadComparisonReceipt(
        comparison_input_digest=_d(f"comparison-input:{sequence}"),
        source_observation_digest=operation.source_observation_digest,
        legacy_observation_digest=_d(f"legacy-observation:{sequence}"),
        v4_observation_digest=_d(f"v4-observation:{sequence}"),
        mapping_rule_digest=None,
        verdict=verdict,
        legacy_goal_code="legacy_goal_set_command",
        legacy_semantic_goal_id="recover_operating_margin",
        v4_semantic_goal_id=(
            "recover_operating_margin"
            if verdict not in {"legacy_only_behavior", "comparison_unavailable"}
            else None
        ),
        legacy_lifecycle_state="selected",
        v4_lifecycle_state=(
            "selected"
            if verdict not in {"legacy_only_behavior", "comparison_unavailable"}
            else None
        ),
        legacy_state_changed=True,
        v4_projected_state_changed=True,
        comparison_available=(verdict != "comparison_unavailable"),
    )
    return ShadowTapExecution(
        authoritative_result=f"legacy-result-{sequence}",
        status="comparison_ready_in_memory_only",
        operation=operation,
        legacy_before=before,
        legacy_after=after,
        comparison_receipt=receipt,
        state_capture_performed=True,
        v4_evaluation_performed=True,
        comparison_performed=True,
    )


def _records(verdicts):
    result = []
    previous = GENESIS_RECORD_DIGEST
    for sequence, verdict in enumerate(verdicts, start=1):
        record = GoalDualReadWindowRecord.from_shadow_execution(
            sequence=sequence,
            previous_record_digest=previous,
            execution=_execution(sequence, verdict),
        )
        result.append(record)
        previous = record.record_digest
    return tuple(result)


def test_accepted_m3_c_m_evidence_binds_raw_git_shas_and_compatibility_pin():
    evidence = ACCEPTED_M3_C_M_EVIDENCE
    assert evidence.exact_head == M3_C_M_EXACT_HEAD
    assert evidence.exact_run == M3_C_M_EXACT_RUN
    assert evidence.artifact_sha256 == M3_C_M_ARTIFACT_SHA256
    assert evidence.merge_sha == M3_C_M_MERGE_SHA
    assert len(evidence.exact_head) == 40
    assert len(evidence.merge_sha) == 40
    assert len(evidence.compatibility_shadow_pin.exact_head) == 64
    assert len(evidence.compatibility_shadow_pin.merge_sha) == 64
    assert evidence.exact_head not in str(evidence.compatibility_shadow_pin.to_mapping())


def test_no_active_authorization_or_executable_operator_is_checked_in():
    with pytest.raises(M3CNDualReadWindowError, match="no active reviewed"):
        active_reviewed_window_authorization()
    manifest = private_device_operator_manifest()
    assert manifest["active_authorization_present"] is False
    assert manifest["execution_available_in_this_slice"] is False
    assert manifest["default_runtime_integration"] is False
    assert manifest["existing_m3_c_j_database_access"] is False
    assert manifest["single_use"] is True


def test_policy_is_bounded_digest_only_and_rejects_scope_escape():
    policy = BoundedDualReadWindowPolicy()
    assert (policy.min_observations, policy.max_observations) == (4, 16)
    assert policy.raw_text_retention_authorized is False
    assert policy.existing_private_database_access_authorized is False
    with pytest.raises(M3CNDualReadWindowError, match="raw text"):
        BoundedDualReadWindowPolicy(raw_text_retention_authorized=True)
    with pytest.raises(M3CNDualReadWindowError, match="bound"):
        BoundedDualReadWindowPolicy(max_observations=65)


def test_path_binding_requires_new_distinct_paths_and_prior_digest_exclusion():
    paths = _paths()
    assert paths.working_store_path_digest not in paths.forbidden_existing_path_digests
    with pytest.raises(M3CNDualReadWindowError, match="overlap"):
        PrivateDeviceWindowPathBinding(
            operator_input_path_digest=_d("one"),
            working_store_path_digest=_d("existing"),
            baseline_backup_path_digest=_d("three"),
            separate_restore_path_digest=_d("four"),
            forbidden_existing_path_digests=(_d("existing"),),
        )
    with pytest.raises(M3CNDualReadWindowError, match="must differ"):
        PrivateDeviceWindowPathBinding(
            operator_input_path_digest=_d("same"),
            working_store_path_digest=_d("same"),
            baseline_backup_path_digest=_d("three"),
            separate_restore_path_digest=_d("four"),
            forbidden_existing_path_digests=(_d("existing"),),
        )


def test_authorization_binds_exact_prerequisite_and_grants_only_bounded_shadow():
    policy, paths, rollback, authorization = _material()
    assert authorization.policy_digest == policy.policy_digest
    assert authorization.path_binding_digest == paths.path_binding_digest
    assert authorization.rollback_digest == rollback.rollback_digest
    assert authorization.private_device_shadow_observation_authorized is True
    assert authorization.bounded_private_retention_authorized is True
    assert authorization.default_runtime_integration_authorized is False
    assert authorization.legacy_goal_authority_transferred is False
    assert authorization.legacy_migration_authorized is False
    assert authorization.m3_e_authority_open is False
    values = authorization.to_mapping()
    values["legacy_migration_authorized"] = True
    with pytest.raises(M3CNDualReadWindowError, match="escaped"):
        BoundedDualReadWindowAuthorizationPacket(**values)


def test_shadow_execution_converts_to_digest_only_single_call_record():
    execution = _execution(1)
    record = GoalDualReadWindowRecord.from_shadow_execution(
        sequence=1,
        previous_record_digest=GENESIS_RECORD_DIGEST,
        execution=execution,
    )
    assert record.authoritative_call_count == 1
    assert record.previous_record_digest == GENESIS_RECORD_DIGEST
    assert record.verdict == "exact_equivalent"
    assert record.raw_text_retained is False
    assert "private-category-1" not in str(record.to_mapping())
    assert record.legacy_goal_authority_transferred is False
    assert record.legacy_migration_authorized is False
    assert record.m3_e_authority_open is False


def test_four_clean_records_are_only_eligible_for_later_human_gate_review():
    policy, _, _, authorization = _material()
    records = _records(["exact_equivalent"] * 4)
    receipt = evaluate_bounded_dual_read_window(
        records,
        policy=policy,
        authorization=authorization,
    )
    assert receipt.record_count == 4
    assert receipt.blocking_verdict_count == 0
    assert receipt.human_gate_review_eligible is True
    assert receipt.final_record_digest == records[-1].record_digest
    assert receipt.legacy_goal_authority_transferred is False
    assert receipt.legacy_migration_authorized is False
    assert receipt.m3_e_authority_open is False


def test_blocking_verdict_is_retained_but_never_promotes_authority():
    policy, _, _, authorization = _material()
    records = _records(
        ["exact_equivalent", "mapped_equivalent", "expected_design_difference", "unexplained_divergence"]
    )
    receipt = evaluate_bounded_dual_read_window(
        records,
        policy=policy,
        authorization=authorization,
    )
    assert receipt.blocking_verdict_count == 1
    assert receipt.human_gate_review_eligible is False
    assert receipt.legacy_migration_authorized is False


def test_sequence_chain_duplicate_and_bound_fail_closed():
    policy, _, _, authorization = _material()
    records = list(_records(["exact_equivalent"] * 4))
    wrong = GoalDualReadWindowRecord(
        **{
            **records[1].to_mapping(),
            "previous_record_digest": GENESIS_RECORD_DIGEST,
        }
    )
    records[1] = wrong
    with pytest.raises(M3CNDualReadWindowError, match="chain"):
        evaluate_bounded_dual_read_window(
            records,
            policy=policy,
            authorization=authorization,
        )
    with pytest.raises(M3CNDualReadWindowError, match="outside reviewed bounds"):
        evaluate_bounded_dual_read_window(
            _records(["exact_equivalent"] * 3),
            policy=policy,
            authorization=authorization,
        )


def test_module_has_no_io_network_database_or_operator_execution_surface():
    tree = ast.parse(MODULE.read_text(encoding="utf-8"), filename=str(MODULE))
    imported = set()
    called = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".", 1)[0])
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                called.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                called.add(node.func.attr)
    assert not imported & {
        "os",
        "pathlib",
        "sqlite3",
        "subprocess",
        "socket",
        "requests",
        "urllib",
    }
    assert not called & {
        "open",
        "write_text",
        "mkdir",
        "connect",
        "unlink",
        "rmtree",
        "emit",
        "schedule",
        "speak",
    }


def test_design_and_reuse_pin_make_chat_changes_non_invalidating():
    design = DESIGN.read_text(encoding="utf-8")
    reuse = REUSE.read_text(encoding="utf-8")
    for token in (
        "execution available in this slice: false",
        "existing M3-C-J database access: false",
        "M3-C-O",
        M3_C_M_EXACT_HEAD,
        str(M3_C_M_EXACT_RUN),
        M3_C_M_ARTIFACT_SHA256,
        str(30635460203),
        M3_C_M_MERGE_SHA,
    ):
        assert token in design or token in reuse
    assert "chat change" in reuse
    assert "retained_sequences_1_through_5_rerun" in reuse
    assert "rerun_completed_private_device_operator" in reuse
