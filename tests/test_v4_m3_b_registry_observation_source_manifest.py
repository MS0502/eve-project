from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from adapters.affect_hormone_neural_rhythm_registry import (
    AXIS_GROUPS,
    affect_hormone_axis_registry,
)
from core.m3_b_registry_affect_owner import REGISTRY_AXIS_ORDER
from core.m3_b_registry_observation_source_manifest import (
    POSITIVE_CONFIDENCE_BLOCKER,
    SOURCE_BINDING_BLOCKER,
    RegistryObservationSourceManifest,
    RegistryObservationSourceManifestError,
    registry_observation_source_manifest,
)

ROOT = Path(__file__).resolve().parents[1]
CORE_MODULE = ROOT / "core/m3_b_registry_observation_source_manifest.py"


def test_manifest_has_exact_canonical_37_axis_source_plan():
    manifest = registry_observation_source_manifest()
    assert manifest.axis_count == 37
    assert tuple(item.axis for item in manifest.entries) == REGISTRY_AXIS_ORDER
    assert len({item.axis for item in manifest.entries}) == 37
    assert len({item.source_contract_id for item in manifest.entries}) == 37
    assert manifest.structurally_complete is True
    assert manifest.real_source_binding_count == 0
    assert manifest.blockers == (
        SOURCE_BINDING_BLOCKER,
        POSITIVE_CONFIDENCE_BLOCKER,
    )
    assert len(manifest.manifest_digest) == 64
    assert all(len(item.entry_digest) == 64 for item in manifest.entries)


def test_manifest_matches_registry_group_evidence_quarantine_and_hardware_boundaries():
    manifest = registry_observation_source_manifest()
    registry = affect_hormone_axis_registry()
    group_by_axis = {
        axis: group for group, axes in AXIS_GROUPS.items() for axis in axes
    }
    for entry in manifest.entries:
        definition = registry[entry.axis]
        assert entry.group == group_by_axis[entry.axis]
        assert entry.registry_evidence_requirement == definition["evidence_required"]
        assert entry.quarantine_required is bool(
            definition["requires_quarantine_for_social_feedback"]
        )
        assert entry.hardware_direct_input_allowed is bool(
            definition["hardware_direct_input_allowed"]
        )
        assert entry.required_raw_fields == tuple(
            sorted(set(entry.required_raw_fields))
        )
        assert entry.minimum_raw_record_count > 0
        assert entry.minimum_logical_span_ticks >= 0
        assert entry.raw_reference_required is True
        assert entry.source_schema_version_required is True
        assert entry.source_integrity_digest_required is True


def test_only_registry_declared_operational_axes_allow_direct_hardware_input():
    manifest = registry_observation_source_manifest()
    hardware_axes = tuple(
        item.axis for item in manifest.entries if item.hardware_direct_input_allowed
    )
    assert hardware_axes == (
        "energy_budget",
        "fatigue_pressure",
        "recovery_need",
        "overload_risk",
    )
    assert all(
        item.appraisal_required is False
        for item in manifest.entries
        if item.hardware_direct_input_allowed
    )
    assert all(
        item.appraisal_required is True
        for item in manifest.entries
        if not item.hardware_direct_input_allowed
    )


def test_long_horizon_social_and_self_axes_require_multi_record_spans():
    manifest = registry_observation_source_manifest()
    for entry in manifest.entries:
        if entry.group == "social_relationship":
            assert entry.minimum_raw_record_count >= 2
            assert entry.minimum_logical_span_ticks >= 2
        if entry.group == "self_identity":
            assert entry.minimum_raw_record_count >= 3
            assert entry.minimum_logical_span_ticks >= 8


def test_source_plan_is_deterministic():
    first = registry_observation_source_manifest()
    second = registry_observation_source_manifest()
    assert first.to_mapping() == second.to_mapping()
    assert first.manifest_digest == second.manifest_digest
    assert tuple(item.entry_digest for item in first.entries) == tuple(
        item.entry_digest for item in second.entries
    )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("source_family", "tampered_source_family"),
        ("observation_class", "tampered_observation_class"),
        ("minimum_raw_record_count", 99),
        ("minimum_logical_span_ticks", 99),
        ("appraisal_required", False),
    ),
)
def test_axis_entry_rejects_noncanonical_plan_mutation(field: str, value):
    entry = registry_observation_source_manifest().entries[10]
    with pytest.raises(
        RegistryObservationSourceManifestError,
        match="canonical axis plan",
    ):
        replace(entry, **{field: value})


def test_axis_entry_rejects_raw_field_and_registry_boundary_mutation():
    manifest = registry_observation_source_manifest()
    entry = manifest.entries[0]
    with pytest.raises(
        RegistryObservationSourceManifestError,
        match="canonical axis plan",
    ):
        replace(entry, required_raw_fields=("fake_a", "fake_b"))
    with pytest.raises(
        RegistryObservationSourceManifestError,
        match="evidence requirement",
    ):
        replace(entry, registry_evidence_requirement="none")
    with pytest.raises(
        RegistryObservationSourceManifestError,
        match="hardware boundary",
    ):
        replace(entry, hardware_direct_input_allowed=False)


def test_entry_rejects_proposal_synthetic_circular_binding_and_runtime_claims():
    entry = registry_observation_source_manifest().entries[0]
    for field in (
        "proposal_only_allowed",
        "synthetic_values_allowed",
        "registry_owner_as_source_allowed",
        "real_source_binding_present",
        "runtime_capture_installed",
    ):
        with pytest.raises(
            RegistryObservationSourceManifestError,
            match="cannot claim real bindings or live authority",
        ):
            replace(entry, **{field: True})


def test_manifest_rejects_missing_reordered_duplicate_and_wrong_types():
    manifest = registry_observation_source_manifest()
    with pytest.raises(
        RegistryObservationSourceManifestError,
        match="exactly 37",
    ):
        RegistryObservationSourceManifest(entries=manifest.entries[:-1])
    with pytest.raises(
        RegistryObservationSourceManifestError,
        match="canonical 37-axis order",
    ):
        RegistryObservationSourceManifest(
            entries=(manifest.entries[1], manifest.entries[0], *manifest.entries[2:])
        )
    duplicate = manifest.entries[:-1] + (manifest.entries[-2],)
    with pytest.raises(
        RegistryObservationSourceManifestError,
        match="canonical 37-axis order",
    ):
        RegistryObservationSourceManifest(entries=duplicate)
    with pytest.raises(
        RegistryObservationSourceManifestError,
        match="exact immutable entry type",
    ):
        RegistryObservationSourceManifest(
            entries=(*manifest.entries[:-1], object())  # type: ignore[arg-type]
        )


def test_manifest_is_frozen_and_cannot_claim_capture_window_or_authority():
    manifest = registry_observation_source_manifest()
    with pytest.raises(FrozenInstanceError):
        manifest.capture_ready = True  # type: ignore[misc]
    for field in (
        "real_observation_values_present",
        "real_source_bindings_present",
        "capture_ready",
        "runtime_capture_installed",
        "hardware_polling_installed",
        "scheduler_installed",
        "persistence_accessed",
        "event_append_performed",
        "observation_window_started",
        "observation_window_satisfied",
        "m3_b_complete",
        "m3_c_open",
        "m3_e_authority_open",
        "cutover_authorized",
    ):
        with pytest.raises(
            RegistryObservationSourceManifestError,
            match="cannot claim bindings, capture, window, or authority",
        ):
            replace(manifest, **{field: True})


def test_core_module_has_no_io_hardware_polling_scheduler_event_or_runtime_surface():
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
        "psutil",
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
        "poll",
        "save",
        "start",
        "write",
        "write_text",
    }
