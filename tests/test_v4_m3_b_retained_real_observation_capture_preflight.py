from __future__ import annotations

import ast
import inspect
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

import core.m3_b_retained_real_observation_capture_preflight as preflight_module
from core.m3_b_registry_affect_owner import REGISTRY_AXIS_ORDER
from core.m3_b_retained_real_observation_capture_preflight import (
    OBSERVATION_WINDOW_NOT_STARTED_BLOCKER,
    POSITIVE_CONFIDENCE_COVERAGE_BLOCKER,
    PRODUCTION_CAPTURE_COMPONENT_ID,
    PRODUCTION_CAPTURE_FUTURE_PATH,
    RETAINED_REAL_OBSERVATION_CAPTURE_BLOCKER,
    RETENTION_SINK_COMPONENT_ID,
    RETENTION_SINK_FUTURE_PATH,
    RequiredProductionCaptureComponent,
    RetainedRealObservationCapturePreflightError,
    retained_real_observation_capture_preflight,
)

ROOT = Path(__file__).resolve().parents[1]


def test_preflight_reassembles_all_seven_binding_groups_into_exact_37_axis_order():
    preflight = retained_real_observation_capture_preflight()
    axes = tuple(axis for group in preflight.source_binding_groups for axis in group.axes)
    assert axes == REGISTRY_AXIS_ORDER
    assert len(axes) == 37
    assert len(set(axes)) == 37
    assert tuple(group.group_binding_count for group in preflight.source_binding_groups) == (
        4,
        2,
        6,
        7,
        6,
        6,
        6,
    )
    assert tuple(
        group.cumulative_bound_axis_count for group in preflight.source_binding_groups
    ) == (4, 6, 12, 19, 25, 31, 37)
    assert all(len(group.binding_set_digest) == 64 for group in preflight.source_binding_groups)
    assert preflight.source_binding_count == 37
    assert preflight.source_binding_complete is True


def test_preflight_keeps_production_capture_and_retained_real_observation_absent():
    preflight = retained_real_observation_capture_preflight()
    assert preflight.production_capture_adapter_present is False
    assert preflight.retention_sink_present is False
    assert preflight.retained_real_observation_count == 0
    assert preflight.positive_confidence_real_observation_count == 0
    assert preflight.observation_window_eligible is False
    assert preflight.observation_window_started is False
    assert preflight.observation_window_satisfied is False
    assert preflight.runtime_hook_installed is False
    assert preflight.scheduler_installed is False
    assert preflight.persistence_accessed is False
    assert preflight.event_append_performed is False
    assert preflight.registry_owner_mutated is False
    assert preflight.live_affect_mutated is False
    assert preflight.live_drive_mutated is False
    assert preflight.named_state_mutated is False
    assert preflight.goal_memory_self_expression_mutated is False
    assert preflight.m3_b_complete is False
    assert preflight.m3_c_open is False
    assert preflight.m3_e_authority_open is False
    assert preflight.cutover_authorized is False


def test_preflight_has_exact_three_blockers_at_this_boundary():
    preflight = retained_real_observation_capture_preflight()
    assert preflight.blockers == (
        RETAINED_REAL_OBSERVATION_CAPTURE_BLOCKER,
        POSITIVE_CONFIDENCE_COVERAGE_BLOCKER,
        OBSERVATION_WINDOW_NOT_STARTED_BLOCKER,
    )
    assert preflight.blockers == (
        "REGISTRY_RETAINED_REAL_OBSERVATION_CAPTURE_ABSENT",
        "REGISTRY_POSITIVE_CONFIDENCE_COVERAGE_INCOMPLETE",
        "REGISTRY_OBSERVATION_WINDOW_NOT_STARTED",
    )


def test_future_production_components_are_enumerated_but_not_present_or_installed():
    preflight = retained_real_observation_capture_preflight()
    assert tuple(item.component_id for item in preflight.required_production_components) == (
        PRODUCTION_CAPTURE_COMPONENT_ID,
        RETENTION_SINK_COMPONENT_ID,
    )
    assert tuple(item.future_path for item in preflight.required_production_components) == (
        PRODUCTION_CAPTURE_FUTURE_PATH,
        RETENTION_SINK_FUTURE_PATH,
    )
    assert all(item.present is False for item in preflight.required_production_components)
    assert all(item.installed is False for item in preflight.required_production_components)
    assert all(item.enabled is False for item in preflight.required_production_components)
    assert not (ROOT / PRODUCTION_CAPTURE_FUTURE_PATH).exists()
    assert not (ROOT / RETENTION_SINK_FUTURE_PATH).exists()


def test_preflight_is_deterministic_and_recalculable():
    first = retained_real_observation_capture_preflight()
    second = retained_real_observation_capture_preflight()
    assert first == second
    assert first.to_mapping() == second.to_mapping()
    assert first.preflight_digest == second.preflight_digest
    assert len(first.preflight_digest) == 64


def test_preflight_and_components_are_frozen_and_fail_closed_on_capability_claims():
    preflight = retained_real_observation_capture_preflight()
    with pytest.raises(FrozenInstanceError):
        preflight.m3_b_complete = True  # type: ignore[misc]
    with pytest.raises(RetainedRealObservationCapturePreflightError, match="cannot claim production"):
        replace(preflight, production_capture_adapter_present=True)
    with pytest.raises(RetainedRealObservationCapturePreflightError, match="cannot fabricate"):
        replace(preflight, retained_real_observation_count=1)
    with pytest.raises(RetainedRealObservationCapturePreflightError, match="cannot become eligible"):
        replace(preflight, observation_window_eligible=True)
    with pytest.raises(RetainedRealObservationCapturePreflightError, match="cannot grant runtime"):
        replace(preflight, m3_e_authority_open=True)
    with pytest.raises(RetainedRealObservationCapturePreflightError, match="cannot claim"):
        replace(preflight.required_production_components[0], present=True)


def test_component_constructor_cannot_smuggle_an_installed_or_enabled_surface():
    kwargs = {
        "component_id": PRODUCTION_CAPTURE_COMPONENT_ID,
        "future_path": PRODUCTION_CAPTURE_FUTURE_PATH,
        "responsibility": "future-only test component",
    }
    with pytest.raises(RetainedRealObservationCapturePreflightError, match="cannot claim"):
        RequiredProductionCaptureComponent(**kwargs, installed=True)
    with pytest.raises(RetainedRealObservationCapturePreflightError, match="cannot claim"):
        RequiredProductionCaptureComponent(**kwargs, enabled=True)


def test_core_preflight_has_no_filesystem_network_thread_scheduler_or_persistence_surface():
    tree = ast.parse(inspect.getsource(preflight_module))
    imported_roots: set[str] = set()
    calls: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".")[0])
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                calls.add(func.id)
            elif isinstance(func, ast.Attribute):
                calls.add(func.attr)
    assert imported_roots.isdisjoint(
        {"os", "pathlib", "socket", "subprocess", "threading", "asyncio", "sqlite3"}
    )
    assert calls.isdisjoint(
        {"open", "write", "write_text", "write_bytes", "connect", "send", "start", "schedule"}
    )
