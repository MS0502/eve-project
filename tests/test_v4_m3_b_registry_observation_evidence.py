from __future__ import annotations

import ast
import hashlib
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from adapters.affect_hormone_neural_rhythm_registry import affect_hormone_axis_registry
from hormone_system import HormoneSystem
from core.m3_b_observation_packet import build_m3_b_observation_packet
from core.m3_b_registry_affect_owner import (
    REGISTRY_AXIS_ORDER,
    create_registry_affect_owner,
)
from core.m3_b_registry_observation_evidence import (
    RegistryAxisPositiveConfidenceEvidence,
    RegistryObservationEvidenceError,
    VERIFIED_OBSERVATION_KIND,
    build_registry_positive_confidence_evidence_bundle,
    materialize_registry_observed_owner,
)

ROOT = Path(__file__).resolve().parents[1]
CORE_MODULE = ROOT / "core/m3_b_registry_observation_evidence.py"


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _owner():
    return create_registry_affect_owner(
        owner_instance_id="test:registry-owner:v1",
        genesis_source_id="test:registry-genesis:v1",
    )


def _evidence(*, observed_tick: int = 1):
    registry = affect_hormone_axis_registry()
    return tuple(
        RegistryAxisPositiveConfidenceEvidence(
            axis=axis,
            value=float(registry[axis]["baseline"]),
            confidence=0.75 + (index % 3) * 0.05,
            observed_tick=observed_tick,
            observation_id=f"test:registry-observation:{index:02d}",
            source_family="verified_internal_appraisal_observation",
            source_instance_id="test:appraisal-observer:v1",
            source_snapshot_id=f"test:appraisal-snapshot:{observed_tick}",
            source_schema_version="test.registry-observation-source.v1",
            source_integrity_digest=_digest(
                f"source:{axis}:{observed_tick}"
            ),
            raw_observation_digest=_digest(
                f"raw:{axis}:{observed_tick}"
            ),
            acquisition_method="explicit_recalculable_observation_capture",
            verification_method="deterministic_schema_and_range_verification",
            model_or_rule_version="test:observation-rule:v1",
        )
        for index, axis in enumerate(REGISTRY_AXIS_ORDER)
    )


def _bundle(owner, *, observations=None, logical_tick: int = 1):
    return build_registry_positive_confidence_evidence_bundle(
        owner,
        _evidence(observed_tick=logical_tick)
        if observations is None
        else observations,
        bundle_id="test:registry-evidence-bundle:v1",
        logical_tick=logical_tick,
        source_manifest_schema_version="test.registry-source-manifest.v1",
        source_manifest_digest=_digest("test:source-manifest:v1"),
        verification_authorization_id="test:verification-authorization:v1",
        acceptance_policy_version="test:acceptance-policy:v1",
    )


def test_exact_37_axis_positive_confidence_bundle_materializes_detached_owner():
    owner = _owner()
    before = owner.to_mapping()
    bundle = _bundle(owner)
    observed = materialize_registry_observed_owner(owner, bundle)
    assert owner.to_mapping() == before
    assert bundle.positive_confidence_count == 37
    assert bundle.exact_positive_confidence_coverage is True
    assert tuple(item.axis for item in bundle.observations) == REGISTRY_AXIS_ORDER
    assert observed.owner_instance_id == owner.owner_instance_id
    assert observed.logical_tick == 1
    assert observed.state_sequence == 1
    assert observed.prior_state_digest == owner.state_digest
    assert observed.last_transition_digest == bundle.bundle_digest
    assert observed.last_transition_kind == "detached_verified_observation_bundle"
    assert all(axis.confidence > 0.0 for axis in observed.axes)
    assert all(axis.update_count == 1 for axis in observed.axes)
    assert all(axis.last_source_kind == VERIFIED_OBSERVATION_KIND for axis in observed.axes)
    assert observed.runtime_hook_installed is False
    assert observed.scheduler_installed is False
    assert observed.persistence_accessed is False
    assert observed.event_append_performed is False
    assert observed.observation_window_started is False
    assert observed.m3_b_complete is False
    assert observed.m3_c_open is False
    assert observed.m3_e_authority_open is False
    assert observed.cutover_authorized is False


def test_verified_owner_resolves_packet_confidence_blocker_without_starting_window():
    owner = _owner()
    observed = materialize_registry_observed_owner(owner, _bundle(owner))
    packet = build_m3_b_observation_packet(
        HormoneSystem(developmental_stage="adult"),
        observed,
        packet_id="test:combined-packet:verified-registry:v1",
        packet_sequence=1,
        logical_tick=observed.logical_tick,
        legacy_source_instance_id="test:legacy-owner:v1",
        legacy_source_snapshot_id="test:legacy-snapshot:v1",
    )
    assert packet.axis_count == 63
    assert packet.positive_confidence_count == 63
    assert packet.zero_confidence_count == 0
    assert packet.window_blockers == ()
    assert packet.observation_window_start_eligible is True
    assert packet.observation_window_started is False
    assert packet.observation_window_satisfied is False
    assert packet.m3_b_complete is False
    assert packet.m3_c_open is False
    assert packet.m3_e_authority_open is False
    assert packet.cutover_authorized is False


def test_bundle_and_materialized_owner_are_deterministic():
    owner = _owner()
    first_bundle = _bundle(owner)
    second_bundle = _bundle(owner)
    first_owner = materialize_registry_observed_owner(owner, first_bundle)
    second_owner = materialize_registry_observed_owner(owner, second_bundle)
    assert first_bundle.to_mapping() == second_bundle.to_mapping()
    assert first_bundle.bundle_digest == second_bundle.bundle_digest
    assert first_owner.to_mapping() == second_owner.to_mapping()
    assert first_owner.state_digest == second_owner.state_digest


def test_zero_confidence_and_out_of_bounds_values_fail_closed():
    valid = _evidence()
    with pytest.raises(RegistryObservationEvidenceError, match="strictly positive"):
        replace(valid[0], confidence=0.0)
    with pytest.raises(RegistryObservationEvidenceError, match="outside declared bounds"):
        replace(valid[0], value=2.0)


@pytest.mark.parametrize(
    "field",
    ("genesis_derived", "baseline_derived", "default_derived", "proposal_only", "synthetic"),
)
def test_non_observation_derivations_cannot_masquerade_as_evidence(field: str):
    with pytest.raises(RegistryObservationEvidenceError, match="not observation evidence"):
        replace(_evidence()[0], **{field: True})


def test_registry_owner_cannot_be_a_circular_source_and_raw_reference_is_required():
    valid = _evidence()[0]
    with pytest.raises(RegistryObservationEvidenceError, match="own observation source"):
        replace(valid, source_family="read_only_affect_registry")
    with pytest.raises(RegistryObservationEvidenceError, match="recalculable raw reference"):
        replace(valid, recalculable_reference_present=False)
    with pytest.raises(RegistryObservationEvidenceError, match="placeholder digest"):
        replace(valid, raw_observation_digest="0" * 64)


def test_missing_duplicate_reordered_and_future_tick_coverage_fail_closed():
    owner = _owner()
    valid = _evidence()
    with pytest.raises(RegistryObservationEvidenceError, match="exactly 37"):
        _bundle(owner, observations=valid[:-1])
    duplicate = valid[:-1] + (replace(valid[-1], axis=valid[-2].axis),)
    with pytest.raises(RegistryObservationEvidenceError, match="canonical 37-axis order"):
        _bundle(owner, observations=duplicate)
    with pytest.raises(RegistryObservationEvidenceError, match="canonical 37-axis order"):
        _bundle(owner, observations=(valid[1], valid[0], *valid[2:]))
    future = tuple(replace(item, observed_tick=2) for item in valid)
    with pytest.raises(RegistryObservationEvidenceError, match="exceed bundle logical tick"):
        _bundle(owner, observations=future, logical_tick=1)


def test_materialization_is_bound_to_owner_digest_sequence_identity_and_time():
    owner = _owner()
    bundle = _bundle(owner)
    other = create_registry_affect_owner(
        owner_instance_id="test:other-owner:v1",
        genesis_source_id="test:other-genesis:v1",
    )
    with pytest.raises(RegistryObservationEvidenceError, match="target owner"):
        materialize_registry_observed_owner(other, bundle)
    with pytest.raises(RegistryObservationEvidenceError, match="prior digest"):
        materialize_registry_observed_owner(
            owner,
            replace(bundle, expected_prior_owner_digest=_digest("wrong-owner")),
        )
    with pytest.raises(RegistryObservationEvidenceError, match="next owner state"):
        materialize_registry_observed_owner(
            owner,
            replace(bundle, target_state_sequence=2),
        )
    advanced = replace(owner, logical_tick=2)
    backward_bundle = _bundle(
        advanced,
        observations=_evidence(observed_tick=1),
        logical_tick=1,
    )
    with pytest.raises(RegistryObservationEvidenceError, match="move backward"):
        materialize_registry_observed_owner(advanced, backward_bundle)


def test_bundle_is_frozen_and_cannot_claim_window_mutation_or_authority():
    bundle = _bundle(_owner())
    with pytest.raises(FrozenInstanceError):
        bundle.logical_tick = 2  # type: ignore[misc]
    for field in (
        "runtime_hook_installed",
        "scheduler_installed",
        "persistence_accessed",
        "event_append_performed",
        "live_affect_mutated",
        "live_drive_mutated",
        "named_state_mutated",
        "goal_memory_self_expression_mutated",
        "observation_window_started",
        "observation_window_satisfied",
        "m3_b_complete",
        "m3_c_open",
        "m3_e_authority_open",
        "cutover_authorized",
    ):
        with pytest.raises(RegistryObservationEvidenceError, match="cannot grant"):
            replace(bundle, **{field: True})


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
