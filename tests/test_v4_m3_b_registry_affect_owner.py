from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from adapters.affect_hormone_neural_rhythm_registry import affect_hormone_axis_registry
from core.event_kernel import SHADOW_AUTHORITY
from core.m3_b_registry_affect_owner import (
    REGISTRY_AXIS_ORDER,
    RegistryAffectOwnerError,
    RegistryAffectOwnerState,
    advance_registry_affect_owner,
    apply_validated_registry_proposal,
    create_registry_affect_owner,
)

ROOT = Path(__file__).resolve().parents[1]
CORE_MODULE = ROOT / "core/m3_b_registry_affect_owner.py"


def owner() -> RegistryAffectOwnerState:
    return create_registry_affect_owner(
        owner_instance_id="test:registry-owner:v1",
        genesis_source_id="test:registry-genesis:v1",
    )


def apply(
    current: RegistryAffectOwnerState,
    *,
    proposal_id: str,
    event_category: str = "praise",
    deltas: dict[str, float] | None = None,
    confidence: float = 0.8,
) -> RegistryAffectOwnerState:
    return apply_validated_registry_proposal(
        current,
        event_category=event_category,
        proposed_axis_deltas=deltas or {"competence_drive": 0.04},
        proposal_id=proposal_id,
        proposal_sequence=current.state_sequence + 1,
        proposal_confidence=confidence,
        expected_owner_digest=current.state_digest,
        operator_authorization_id=f"operator:{proposal_id}",
    )


def test_genesis_materializes_exact_37_axis_current_state_without_calling_it_observation():
    current = owner()
    registry = affect_hormone_axis_registry()
    assert len(REGISTRY_AXIS_ORDER) == len(set(REGISTRY_AXIS_ORDER)) == 37
    assert tuple(axis.axis for axis in current.axes) == REGISTRY_AXIS_ORDER
    assert current.authority == SHADOW_AUTHORITY
    assert current.genesis_is_observation_evidence is False
    assert current.proposal_metadata_is_current_state is False
    assert current.logical_tick == current.state_sequence == 0
    assert all(axis.value == registry[axis.axis]["baseline"] for axis in current.axes)
    assert all(axis.last_source_kind.endswith("genesis_not_observation") for axis in current.axes)
    registry["energy_budget"]["baseline"] = 0.01
    assert current.value_for("energy_budget") != 0.01


def test_genesis_and_observation_snapshot_are_deterministic_and_complete():
    first = owner()
    second = owner()
    assert first.to_mapping() == second.to_mapping()
    assert first.state_digest == second.state_digest
    observations = first.to_axis_observations()
    assert len(observations) == 37
    assert tuple(item.axis for item in observations) == REGISTRY_AXIS_ORDER
    assert all(item.source_family == "read_only_affect_registry" for item in observations)
    assert all(item.source_snapshot_id == first.source_snapshot_id for item in observations)
    assert all(item.source_integrity_digest == first.state_digest for item in observations)
    assert all(item.confidence == 1.0 for item in observations)


def test_validated_proposal_returns_new_owner_and_preserves_old_owner():
    before = owner()
    after = apply(
        before,
        proposal_id="proposal:praise:1",
        deltas={"competence_drive": 0.04, "social_trust": 0.03},
    )
    assert before.value_for("competence_drive") == 0.50
    assert before.value_for("social_trust") == 0.65
    assert after.value_for("competence_drive") == pytest.approx(0.54)
    assert after.value_for("social_trust") == pytest.approx(0.68)
    assert after.value_for("energy_budget") == before.value_for("energy_budget")
    assert after.prior_state_digest == before.state_digest
    assert after.state_sequence == 1
    assert after.applied_proposal_ids == ("proposal:praise:1",)
    assert after.last_transition_kind == "validated_detached_event_proposal"
    assert after.event_append_performed is False
    assert after.live_affect_mutated is False
    assert after.live_drive_mutated is False


def test_proposal_replay_stale_digest_and_sequence_fail_closed():
    first = owner()
    second = apply(first, proposal_id="proposal:praise:1")
    with pytest.raises(RegistryAffectOwnerError, match="duplicate proposal id"):
        apply_validated_registry_proposal(
            second,
            event_category="praise",
            proposed_axis_deltas={"competence_drive": 0.04},
            proposal_id="proposal:praise:1",
            proposal_sequence=2,
            proposal_confidence=0.8,
            expected_owner_digest=second.state_digest,
            operator_authorization_id="operator:duplicate",
        )
    with pytest.raises(RegistryAffectOwnerError, match="expected owner digest"):
        apply_validated_registry_proposal(
            second,
            event_category="praise",
            proposed_axis_deltas={"competence_drive": 0.04},
            proposal_id="proposal:praise:2",
            proposal_sequence=2,
            proposal_confidence=0.8,
            expected_owner_digest=first.state_digest,
            operator_authorization_id="operator:stale",
        )
    with pytest.raises(RegistryAffectOwnerError, match="sequence"):
        apply_validated_registry_proposal(
            second,
            event_category="praise",
            proposed_axis_deltas={"competence_drive": 0.04},
            proposal_id="proposal:praise:2",
            proposal_sequence=3,
            proposal_confidence=0.8,
            expected_owner_digest=second.state_digest,
            operator_authorization_id="operator:gap",
        )


def test_existing_validator_boundary_rejects_unknown_or_outside_proposals():
    current = owner()
    with pytest.raises(RegistryAffectOwnerError, match="existing validator"):
        apply_validated_registry_proposal(
            current,
            event_category="unknown_event",
            proposed_axis_deltas={"competence_drive": 0.01},
            proposal_id="proposal:unknown:1",
            proposal_sequence=1,
            proposal_confidence=0.8,
            expected_owner_digest=current.state_digest,
            operator_authorization_id="operator:unknown",
        )
    with pytest.raises(RegistryAffectOwnerError, match="existing validator"):
        apply_validated_registry_proposal(
            current,
            event_category="praise",
            proposed_axis_deltas={"threat_pressure": 0.01},
            proposal_id="proposal:outside:1",
            proposal_sequence=1,
            proposal_confidence=0.8,
            expected_owner_digest=current.state_digest,
            operator_authorization_id="operator:outside",
        )
    with pytest.raises(RegistryAffectOwnerError, match="unknown registry axis"):
        apply_validated_registry_proposal(
            current,
            event_category="praise",
            proposed_axis_deltas={"not_an_axis": 0.01},
            proposal_id="proposal:bad-axis:1",
            proposal_sequence=1,
            proposal_confidence=0.8,
            expected_owner_digest=current.state_digest,
            operator_authorization_id="operator:bad-axis",
        )


def test_saturation_is_deterministic_and_bounded():
    current = owner()
    for index in range(20):
        current = apply(current, proposal_id=f"proposal:praise:{index}")
    assert current.value_for("competence_drive") == 1.0
    assert current.axes[REGISTRY_AXIS_ORDER.index("competence_drive")].update_count == 20
    assert current.state_sequence == 20


def test_caller_invoked_cadence_honors_refractory_then_decays_toward_baseline():
    genesis = owner()
    stimulated = apply(genesis, proposal_id="proposal:praise:1")
    early = advance_registry_affect_owner(
        stimulated,
        target_tick=1,
        cadence_id="cadence:1",
        expected_owner_digest=stimulated.state_digest,
    )
    assert early.value_for("competence_drive") == stimulated.value_for("competence_drive")
    later = advance_registry_affect_owner(
        early,
        target_tick=20,
        cadence_id="cadence:20",
        expected_owner_digest=early.state_digest,
    )
    assert 0.50 < later.value_for("competence_drive") < stimulated.value_for("competence_drive")
    assert later.logical_tick == 20
    assert later.last_transition_kind == "deterministic_caller_invoked_cadence"
    assert later.scheduler_installed is False


def test_cadence_replay_and_non_monotonic_ticks_fail_closed():
    current = owner()
    advanced = advance_registry_affect_owner(
        current,
        target_tick=5,
        cadence_id="cadence:5",
        expected_owner_digest=current.state_digest,
    )
    with pytest.raises(RegistryAffectOwnerError, match="expected owner digest"):
        advance_registry_affect_owner(
            advanced,
            target_tick=10,
            cadence_id="cadence:stale",
            expected_owner_digest=current.state_digest,
        )
    with pytest.raises(RegistryAffectOwnerError, match="greater than current"):
        advance_registry_affect_owner(
            advanced,
            target_tick=5,
            cadence_id="cadence:replay",
            expected_owner_digest=advanced.state_digest,
        )


def test_owner_and_axis_states_are_frozen_and_authority_flags_remain_false():
    current = owner()
    with pytest.raises(FrozenInstanceError):
        current.logical_tick = 1  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        current.axes[0].value = 0.1  # type: ignore[misc]
    assert current.runtime_hook_installed is False
    assert current.scheduler_installed is False
    assert current.persistence_accessed is False
    assert current.event_append_performed is False
    assert current.goal_memory_self_expression_mutated is False
    assert current.observation_window_started is False
    assert current.m3_b_complete is False
    assert current.m3_c_open is False
    assert current.m3_e_authority_open is False
    assert current.cutover_authorized is False


def test_core_module_has_no_io_persistence_scheduler_event_or_runtime_activation_surface():
    tree = ast.parse(CORE_MODULE.read_text(encoding="utf-8"))
    imported: set[str] = set()
    called: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                called.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                called.add(node.func.attr)
    assert not imported & {
        "os",
        "pathlib",
        "persistence",
        "sqlite3",
        "subprocess",
        "threading",
        "time",
    }
    assert not called & {
        "append_event",
        "connect",
        "emit",
        "mkdir",
        "open",
        "save",
        "start",
        "write",
        "write_text",
    }
