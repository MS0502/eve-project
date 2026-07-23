from __future__ import annotations

import ast
import copy
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from hormone_system import HormoneSystem

from core.event_kernel import SHADOW_AUTHORITY
from core.m3_b_affect_projection import AffectProjectionError
from core.m3_b_observation_packet import (
    EXPECTED_AXIS_ORDER,
    M3BObservationPacketError,
    WINDOW_BLOCKER_REGISTRY_CONFIDENCE,
    build_m3_b_observation_packet,
)
from core.m3_b_registry_affect_owner import (
    apply_validated_registry_proposal,
    create_registry_affect_owner,
)

ROOT = Path(__file__).resolve().parents[1]
CORE_MODULE = ROOT / "core/m3_b_observation_packet.py"


def registry_owner():
    return create_registry_affect_owner(
        owner_instance_id="test:registry-owner:v1",
        genesis_source_id="test:registry-genesis:v1",
    )


def packet(source: HormoneSystem, owner, *, packet_id: str = "test:packet:1", sequence: int = 1):
    return build_m3_b_observation_packet(
        source,
        owner,
        packet_id=packet_id,
        packet_sequence=sequence,
        logical_tick=owner.logical_tick,
        legacy_source_instance_id="test:legacy-owner:v1",
        legacy_source_snapshot_id=f"test:legacy-snapshot:{sequence}",
    )


def test_genesis_packet_is_exact_63_axis_structurally_complete_and_window_blocked():
    result = packet(HormoneSystem(developmental_stage="adult"), registry_owner())
    assert result.axis_count == 63
    assert result.legacy_axis_count == 26
    assert result.registry_axis_count == 37
    assert tuple(item.axis for item in result.observations) == EXPECTED_AXIS_ORDER
    assert result.positive_confidence_count == 26
    assert result.zero_confidence_count == 37
    assert result.positive_confidence_axes == EXPECTED_AXIS_ORDER[:26]
    assert result.zero_confidence_axes == EXPECTED_AXIS_ORDER[26:]
    assert result.structurally_complete is True
    assert result.strict_projection_input_ready is True
    assert result.observation_window_start_eligible is False
    assert result.window_blockers == (WINDOW_BLOCKER_REGISTRY_CONFIDENCE,)


def test_packet_is_deterministic_and_preserves_both_sources():
    source = HormoneSystem(developmental_stage="adult")
    owner = registry_owner()
    source_before = copy.deepcopy(source.__dict__)
    owner_before = owner.to_mapping()
    first = packet(source, owner)
    middle_source = copy.deepcopy(source.__dict__)
    middle_owner = owner.to_mapping()
    second = packet(source, owner)
    source_after = copy.deepcopy(source.__dict__)
    owner_after = owner.to_mapping()
    assert source_before == middle_source == source_after
    assert owner_before == middle_owner == owner_after
    assert first.to_mapping() == second.to_mapping()
    assert first.packet_digest == second.packet_digest
    assert len(first.packet_digest) == 64
    assert len(first.source_set.digest) == 64
    assert first.source_set.registry_owner_state_digest == owner.state_digest


def test_validated_registry_evidence_promotes_only_touched_axes_and_window_remains_blocked():
    genesis = registry_owner()
    observed = apply_validated_registry_proposal(
        genesis,
        event_category="praise",
        proposed_axis_deltas={"competence_drive": 0.04, "social_trust": 0.03},
        proposal_id="test:proposal:praise:1",
        proposal_sequence=1,
        proposal_confidence=0.8,
        expected_owner_digest=genesis.state_digest,
        operator_authorization_id="test:operator:praise:1",
    )
    result = packet(HormoneSystem(developmental_stage="adult"), observed)
    assert result.positive_confidence_count == 28
    assert result.zero_confidence_count == 35
    assert "competence_drive" in result.positive_confidence_axes
    assert "social_trust" in result.positive_confidence_axes
    assert "energy_budget" in result.zero_confidence_axes
    assert result.window_blockers == (WINDOW_BLOCKER_REGISTRY_CONFIDENCE,)
    assert result.observation_window_start_eligible is False


def test_boundary_baseline_legacy_source_fails_closed_before_packet_emission():
    with pytest.raises((AffectProjectionError, M3BObservationPacketError), match="floor < baseline < ceiling|strict v1"):
        packet(HormoneSystem(developmental_stage="newborn"), registry_owner())


def test_exact_source_types_and_identity_fields_are_required():
    class DerivedHormoneSystem(HormoneSystem):
        pass

    with pytest.raises(M3BObservationPacketError, match="exact HormoneSystem"):
        packet(DerivedHormoneSystem(), registry_owner())
    with pytest.raises(M3BObservationPacketError, match="exact RegistryAffectOwnerState"):
        packet(HormoneSystem(), object())
    with pytest.raises(M3BObservationPacketError, match="packet_id"):
        build_m3_b_observation_packet(
            HormoneSystem(),
            registry_owner(),
            packet_id="",
            packet_sequence=1,
            logical_tick=0,
            legacy_source_instance_id="test:legacy-owner:v1",
            legacy_source_snapshot_id="test:legacy-snapshot:1",
        )


def test_packet_is_frozen_and_all_authority_effect_flags_remain_false():
    result = packet(HormoneSystem(developmental_stage="adult"), registry_owner())
    with pytest.raises(FrozenInstanceError):
        result.packet_sequence = 2  # type: ignore[misc]
    assert result.authority == SHADOW_AUTHORITY
    assert result.projection_performed is False
    assert result.observation_window_started is False
    assert result.observation_window_satisfied is False
    assert result.persistence_accessed is False
    assert result.event_append_performed is False
    assert result.live_affect_mutated is False
    assert result.live_drive_mutated is False
    assert result.named_state_mutated is False
    assert result.goal_memory_self_expression_mutated is False
    assert result.m3_b_complete is False
    assert result.m3_c_open is False
    assert result.m3_e_authority_open is False
    assert result.cutover_authorized is False


def test_core_module_has_no_io_persistence_scheduler_event_projection_or_runtime_activation_surface():
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
        "project_shadow_affect",
        "save",
        "start",
        "write",
        "write_text",
    }
