from adapters.runtime_mapping_persistence_approval import round103_operator_unblocked_validation_status, runtime_mapping_persistence_approval_gate
from adapters.runtime_mapping_persistence_decision import (
    checkpoint_rollback_audit_requirements,
    default_operator_approval_schema,
    runtime_mapping_persistence_activation_dry_run,
    runtime_mapping_persistence_activation_prerequisites,
)


def test_round106_prerequisites_keep_operator_approval_pending() -> None:
    gate = runtime_mapping_persistence_approval_gate(round103_operator_unblocked_validation_status())
    prereq = runtime_mapping_persistence_activation_prerequisites(gate)

    assert prereq["round"] == 106
    assert prereq["all_non_approval_prerequisites_satisfied"] is True
    assert prereq["operator_approval_pending"] is True
    pending = [row for row in prereq["prerequisites"] if row.get("intentionally_pending")]
    assert pending and pending[0]["id"] == "operator_explicit_approval_pending"


def test_round106_dry_run_does_not_activate_without_approval() -> None:
    dry_run = runtime_mapping_persistence_activation_dry_run()

    assert dry_run["activation_dry_run_only"] is True
    assert dry_run["runtime_mapping_enabled_default"] is False
    assert dry_run["enforcement_enabled_default"] is False
    assert dry_run["runtime_mapping_persisted_now"] is False
    assert dry_run["operator_approval_valid"] is False
    assert dry_run["would_allow_future_activation_patch"] is False
    assert dry_run["mutation_checks"]["agp_bypassed"] is False
    assert dry_run["mutation_checks"]["vectors_npy_committed"] is False


def test_round106_valid_approval_allows_future_patch_but_still_no_mutation() -> None:
    approval = {
        "operator_id": "operator",
        "approval_id": "approval-round106-test",
        "approved_at_utc": "2026-06-03T00:00:00Z",
        "scope": "runtime_mapping_persistence_only",
        "approved_tokens": ["민석"],
        "validation_status": "round103_104_105_validation_reviewed",
        "rollback_acknowledged": True,
        "binary_artifact_boundary_acknowledged": True,
        "agp_anchor_boundary_acknowledged": True,
    }

    dry_run = runtime_mapping_persistence_activation_dry_run(operator_approval=approval)

    assert dry_run["operator_approval_valid"] is True
    assert dry_run["would_allow_future_activation_patch"] is True
    assert dry_run["runtime_mapping_persisted_now"] is False
    assert dry_run["mutation_checks"]["runtime_mapping_enabled_changed"] is False


def test_round106_requirements_and_schema_are_structured() -> None:
    requirements = checkpoint_rollback_audit_requirements()
    schema = default_operator_approval_schema()

    assert requirements["required_before_activation"] is True
    assert "restore runtime_mapping_enabled to false" in requirements["rollback_requirements"]
    assert "surface runtime_mapping_enabled default state" in requirements["state_debug_requirements"]
    assert schema["approval_required_before_mutation"] is True
    assert schema["required_fields"]["scope"] == "runtime_mapping_persistence_only"
    assert schema["forbidden_fields"]["bypass_agp"] is True
