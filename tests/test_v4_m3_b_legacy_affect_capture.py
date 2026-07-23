from __future__ import annotations

import ast
import copy
from pathlib import Path

import pytest

import core.m3_b_legacy_affect_capture as capture_module
from hormone_system import Hormone, HormoneSystem

from core.event_kernel import SHADOW_AUTHORITY
from core.m3_b_affect_projection import AffectProjectionError
from core.m3_b_legacy_affect_capture import (
    CAPTURE_SCHEMA_VERSION,
    LEGACY_AXIS_ORDER,
    SOURCE_SCHEMA_VERSION,
    LegacyAffectCaptureError,
    capture_legacy_hormone_state,
)
from scripts.audit.m3_b_legacy_affect_capture import (
    ROOT,
    audit_repository,
    parse_authoritative_axis_order,
)

CORE_MODULE = ROOT / "core/m3_b_legacy_affect_capture.py"


def capture(source: HormoneSystem):
    return capture_legacy_hormone_state(
        source,
        source_instance_id="test:legacy-source:v1",
        source_snapshot_id="test:legacy-snapshot:v1",
    )


def test_capture_axis_catalog_matches_authoritative_source_exactly():
    assert len(LEGACY_AXIS_ORDER) == len(set(LEGACY_AXIS_ORDER)) == 26
    assert parse_authoritative_axis_order(ROOT) == LEGACY_AXIS_ORDER
    source = HormoneSystem()
    assert tuple(source.hormones) == LEGACY_AXIS_ORDER


def test_capture_is_detached_deterministic_and_does_not_mutate_source():
    source = HormoneSystem()
    source.hormones["dopamine"].level = 0.47
    before = copy.deepcopy(source.__dict__)
    first = capture(source)
    middle = copy.deepcopy(source.__dict__)
    second = capture(source)
    after = copy.deepcopy(source.__dict__)
    assert before == middle == after
    assert first.to_mapping() == second.to_mapping()
    assert first.capture_digest == second.capture_digest
    assert len(first.capture_digest) == 64
    assert len(first.source_integrity_digest) == 64
    assert first.schema_version == CAPTURE_SCHEMA_VERSION
    assert first.source_schema_version == SOURCE_SCHEMA_VERSION
    assert first.authority == SHADOW_AUTHORITY
    assert first.axes[LEGACY_AXIS_ORDER.index("dopamine")].value == 0.47


def test_capture_preserves_exact_axis_fields_and_builds_m3_b_observations():
    source = HormoneSystem(phase=3, developmental_stage="adult")
    result = capture(source)
    assert tuple(axis.axis for axis in result.axes) == LEGACY_AXIS_ORDER
    assert result.source_phase == 3
    assert result.source_stage == "adult"
    assert result.active_hormones == tuple(
        axis for axis in LEGACY_AXIS_ORDER if source.hormones[axis].phase <= 3
    )
    observations = result.to_axis_observations()
    assert len(observations) == 26
    assert tuple(observation.axis for observation in observations) == LEGACY_AXIS_ORDER
    assert all(observation.source_family == "legacy_mutable_hormone" for observation in observations)
    assert all(observation.source_snapshot_id == "test:legacy-snapshot:v1" for observation in observations)
    assert all(observation.source_integrity_digest == result.source_integrity_digest for observation in observations)
    assert all(observation.floor == 0.0 and observation.ceiling == 1.0 for observation in observations)
    assert all(observation.confidence == 1.0 for observation in observations)


def test_capture_preserves_boundary_baselines_and_projection_bridge_fails_closed():
    source = HormoneSystem(phase=3, developmental_stage="newborn")
    result = capture(source)
    by_axis = {axis.axis: axis for axis in result.axes}
    assert by_axis["estrogen"].baseline == by_axis["estrogen"].floor == 0.0
    assert by_axis["testosterone"].baseline == by_axis["testosterone"].floor == 0.0
    with pytest.raises(AffectProjectionError, match="floor < baseline < ceiling"):
        result.to_axis_observations()


def test_capture_calls_no_update_or_stimulate_surface(monkeypatch: pytest.MonkeyPatch):
    source = HormoneSystem()

    def forbidden(*_args, **_kwargs):
        raise AssertionError("mutation surface was called")

    monkeypatch.setattr(HormoneSystem, "update", forbidden)
    monkeypatch.setattr(Hormone, "stimulate", forbidden)
    result = capture(source)
    assert result.source_mutated is False
    assert result.persistence_accessed is False
    assert result.event_append_performed is False
    assert result.live_behavior_changed is False
    assert result.observation_window_started is False
    assert result.m3_c_open is False
    assert result.m3_e_authority_open is False
    assert result.cutover_authorized is False


def test_capture_rejects_non_exact_source_type_and_catalog_drift():
    class DerivedHormoneSystem(HormoneSystem):
        pass

    with pytest.raises(LegacyAffectCaptureError, match="exact HormoneSystem"):
        capture(DerivedHormoneSystem())

    source = HormoneSystem()
    source.hormones.pop("dopamine")
    with pytest.raises(LegacyAffectCaptureError, match="exact 26 axes"):
        capture(source)


def test_capture_rejects_invalid_numeric_or_identity_fields():
    source = HormoneSystem()
    source.hormones["dopamine"].level = 1.01
    with pytest.raises(LegacyAffectCaptureError, match=r"outside \[0,1\]"):
        capture(source)

    source = HormoneSystem()
    source.hormones["dopamine"].name = "other"
    with pytest.raises(LegacyAffectCaptureError, match="identity contract"):
        capture(source)

    with pytest.raises(LegacyAffectCaptureError, match="source_instance_id"):
        capture_legacy_hormone_state(
            HormoneSystem(),
            source_instance_id="",
            source_snapshot_id="test:snapshot:v1",
        )


def test_capture_fails_closed_when_source_value_changes_between_reads(monkeypatch: pytest.MonkeyPatch):
    source = HormoneSystem()
    original = capture_module._read_source_state
    calls = 0

    def changing(current: HormoneSystem):
        nonlocal calls
        calls += 1
        material = original(current)
        if calls == 1:
            current.hormones["dopamine"].level += 0.05
        return material

    monkeypatch.setattr(capture_module, "_read_source_state", changing)
    with pytest.raises(LegacyAffectCaptureError, match="state changed"):
        capture(source)


def test_capture_fails_closed_when_axis_object_identity_is_replaced(monkeypatch: pytest.MonkeyPatch):
    source = HormoneSystem()
    original = capture_module._read_source_state
    calls = 0

    def replacing(current: HormoneSystem):
        nonlocal calls
        calls += 1
        material = original(current)
        if calls == 1:
            prior = current.hormones["dopamine"]
            current.hormones["dopamine"] = Hormone(
                name=prior.name,
                level=prior.level,
                baseline=prior.baseline,
                reactivity=prior.reactivity,
                decay_rate=prior.decay_rate,
                tier=prior.tier,
                phase=prior.phase,
            )
        return material

    monkeypatch.setattr(capture_module, "_read_source_state", replacing)
    with pytest.raises(LegacyAffectCaptureError, match="object identity changed"):
        capture(source)


def test_audit_report_is_recalculable_and_leaves_only_registry_blocker():
    first = audit_repository(ROOT)
    second = audit_repository(ROOT)
    assert first == second
    assert first["errors"] == []
    assert first["axis_count"] == 26
    assert first["axis_observation_count"] == 26
    assert first["axis_order_matches_authoritative_source"] is True
    assert first["before_after_source_equal"] is True
    assert first["deterministic_repeat_equal"] is True
    assert first["legacy_capture_ready"] is True
    assert first["remaining_blockers"] == ["REGISTRY_OBSERVED_VALUE_OWNER_ABSENT"]
    assert first["m3_b_complete"] is False
    assert first["m3_c_open"] is False
    assert len(first["report_digest"]) == 64


def test_core_module_has_no_io_persistence_observer_or_runtime_activation_surface():
    tree = ast.parse(CORE_MODULE.read_text(encoding="utf-8"))
    imported: set[str] = set()
    calls: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                calls.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                calls.add(node.func.attr)
    assert not imported & {
        "main",
        "os",
        "pathlib",
        "persistence",
        "sqlite3",
        "subprocess",
        "threading",
        "time",
    }
    assert not calls & {
        "append_event",
        "connect",
        "mkdir",
        "open",
        "save",
        "start",
        "stimulate",
        "tick",
        "update",
        "write",
        "write_text",
    }
