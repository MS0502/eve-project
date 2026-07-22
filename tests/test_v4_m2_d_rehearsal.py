from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import pytest

from core.event_kernel import EventEnvelope, SHADOW_AUTHORITY
from core.m2_c_migration import StateEvidence
from core.m2_d_rehearsal import (
    ACCEPTED_M2_C_ARTIFACT_SHA256,
    ACCEPTED_M2_C_HEAD,
    ACCEPTED_M2_C_PR,
    ACCEPTED_M2_C_WORKFLOW,
    HUMAN_REVIEW_REQUIRED,
    REQUIRED_SCENARIOS,
    REHEARSAL_AUTHORITY,
    M2DRehearsalError,
    ObservationWindowSpec,
    run_recovery_rehearsal,
)
from core.shadow_observer import (
    ACTIVATION_LEARN_PAIR_TARGET,
    OBSERVER_PRODUCER,
    OBSERVER_VERSION,
    SUCCESS_EVENT_TYPE,
)

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "core/m2_d_rehearsal.py"
EMPTY = {"calls": [], "learned": []}
PAIR_ONE = ["alpha", "beta", 0.4]
PAIR_TWO = ["gamma", "delta", 0.6]
PAIR_PROBE = ["epsilon", "zeta", 0.8]
AFTER_ONE = {"calls": [PAIR_ONE], "learned": [PAIR_ONE]}
AFTER_TWO = {"calls": [PAIR_ONE, PAIR_TWO], "learned": [PAIR_ONE, PAIR_TWO]}
AFTER_PROBE = {
    "calls": [PAIR_ONE, PAIR_TWO, PAIR_PROBE],
    "learned": [PAIR_ONE, PAIR_TWO, PAIR_PROBE],
}


def observed_event(
    sequence: int,
    *,
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    event_id: str,
) -> EventEnvelope:
    return EventEnvelope.create(
        event_id=event_id,
        event_type=SUCCESS_EVENT_TYPE,
        stream_id=ACTIVATION_LEARN_PAIR_TARGET.stream_id,
        sequence=sequence,
        producer=OBSERVER_PRODUCER,
        producer_version=OBSERVER_VERSION,
        correlation_id="m2d:test-window",
        causation_id=None,
        payload={
            "after": dict(after),
            "before": dict(before),
            "legacy_outcome": {"error_type": None, "succeeded": True},
            "target": {
                "callable": ACTIVATION_LEARN_PAIR_TARGET.callable_name,
                "disposition": ACTIVATION_LEARN_PAIR_TARGET.module_disposition,
                "module_path": ACTIVATION_LEARN_PAIR_TARGET.module_path,
                "target_id": ACTIVATION_LEARN_PAIR_TARGET.target_id,
            },
        },
        causal_context={
            "arguments_captured": False,
            "legacy_result_captured": False,
            "observation_phase": "after_the_fact",
            "source_evidence_range": ACTIVATION_LEARN_PAIR_TARGET.evidence_range,
        },
    )


def fixture_values():
    events = (
        observed_event(1, before=EMPTY, after=AFTER_ONE, event_id="m2d:event:1"),
        observed_event(2, before=AFTER_ONE, after=AFTER_TWO, event_id="m2d:event:2"),
    )
    probe = observed_event(
        3,
        before=AFTER_TWO,
        after=AFTER_PROBE,
        event_id="m2d:event:rollback-probe",
    )
    window = ObservationWindowSpec(
        window_id="m2-d:activation-learn-pair:v1",
        baseline_event_count=2,
        snapshot_sequence=1,
        expected_final_state=StateEvidence.from_snapshot(AFTER_TWO),
    )
    return window, events, probe


def run_fixture(tmp_path: Path, name: str = "rehearsal"):
    window, events, probe = fixture_values()
    return run_recovery_rehearsal(
        workspace=tmp_path / name,
        window=window,
        baseline_events=events,
        rollback_probe_event=probe,
        initial_snapshot=EMPTY,
    )


def test_rehearsal_packet_covers_all_required_scenarios_without_authority(tmp_path: Path):
    packet = run_fixture(tmp_path)
    assert tuple(value.scenario_id for value in packet.scenarios) == REQUIRED_SCENARIOS
    assert packet.passed_count == len(REQUIRED_SCENARIOS)
    assert packet.failed_count == 0
    assert packet.machine_passed is True
    assert packet.eligible_for_human_review is True
    assert packet.human_review_status == HUMAN_REVIEW_REQUIRED
    assert packet.human_accepted is False
    assert packet.authority == REHEARSAL_AUTHORITY
    assert packet.shadow_authority == SHADOW_AUTHORITY
    assert packet.legacy_authority_retained is True
    assert packet.runtime_integrated is False
    assert packet.production_dual_read is False
    assert packet.authoritative_recovery is False
    assert packet.cutover_authorized is False
    assert len(packet.packet_digest) == 64
    assert all(value.passed for value in packet.scenarios)


def test_rehearsal_packet_is_deterministic_across_independent_workspaces(tmp_path: Path):
    left = run_fixture(tmp_path, "left")
    right = run_fixture(tmp_path, "right")
    assert left.packet_digest == right.packet_digest
    assert left.canonical_record == right.canonical_record


def test_observation_window_is_pinned_to_accepted_m2_c_evidence():
    window, _events, _probe = fixture_values()
    assert window.m2_c_pr == ACCEPTED_M2_C_PR == 164
    assert window.m2_c_head == ACCEPTED_M2_C_HEAD
    assert window.m2_c_workflow == ACCEPTED_M2_C_WORKFLOW
    assert window.m2_c_artifact_sha256 == ACCEPTED_M2_C_ARTIFACT_SHA256
    with pytest.raises(M2DRehearsalError, match="accepted M2-C scope"):
        replace(window, m2_c_workflow=1)


def test_corrupt_state_forced_termination_and_rollback_are_recalculable(tmp_path: Path):
    packet = run_fixture(tmp_path)
    by_id = {value.scenario_id: value for value in packet.scenarios}

    snapshot = by_id["corrupt_snapshot_fallback"]
    assert snapshot.checks["corruption_visible"] is True
    assert snapshot.observations["selected_snapshot_id"] == "m2d:snapshot:good"
    assert snapshot.observations["rejected_snapshot_ids"] == ["m2d:snapshot:newest"]

    event = by_id["corrupt_event_fail_closed"]
    assert event.observations["read_error_type"] == "PersistedEventCorruption"
    assert event.observations["restore_error_type"] == "PersistedEventCorruption"

    forced = by_id["forced_termination"]
    assert forced.observations["forced_exit_code"] == 97
    assert forced.checks["uncommitted_event_absent"] is True

    rollback = by_id["rollback_rehearsal"]
    assert rollback.checks["probe_changed_state"] is True
    assert rollback.checks["probe_removed_after_rollback"] is True
    assert rollback.observations["rollback_method"] == (
        "verified_backup_file_replacement_in_disposable_workspace"
    )


def test_packet_and_scenario_values_cannot_self_promote(tmp_path: Path):
    packet = run_fixture(tmp_path)
    with pytest.raises(M2DRehearsalError, match="cannot self-promote"):
        replace(packet, human_accepted=True)
    with pytest.raises(M2DRehearsalError, match="cannot self-promote"):
        replace(packet, authoritative_recovery=True)
    with pytest.raises(M2DRehearsalError, match="cannot self-promote"):
        replace(packet, cutover_authorized=True)
    with pytest.raises(M2DRehearsalError, match="authority boundary"):
        replace(packet.scenarios[0], runtime_integrated=True)


def test_invalid_window_fails_before_workspace_creation(tmp_path: Path):
    window, events, probe = fixture_values()
    workspace = tmp_path / "must-not-exist"
    with pytest.raises(M2DRehearsalError, match="baseline events"):
        run_recovery_rehearsal(
            workspace=workspace,
            window=window,
            baseline_events=events[:1],
            rollback_probe_event=probe,
            initial_snapshot=EMPTY,
        )
    assert workspace.exists() is False


def test_existing_workspace_is_rejected_without_touching_contents(tmp_path: Path):
    workspace = tmp_path / "existing"
    workspace.mkdir()
    marker = workspace / "marker.txt"
    marker.write_text("untouched", encoding="utf-8")
    window, events, probe = fixture_values()
    with pytest.raises(M2DRehearsalError, match="new concrete path"):
        run_recovery_rehearsal(
            workspace=workspace,
            window=window,
            baseline_events=events,
            rollback_probe_event=probe,
            initial_snapshot=EMPTY,
        )
    assert marker.read_text(encoding="utf-8") == "untouched"


def test_module_has_no_runtime_bridge_default_activation_or_unsafe_decoder_surface():
    source = MODULE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports: set[str] = set()
    calls: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                calls.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                calls.add(node.func.attr)
    assert not imports & {
        "adapters",
        "asyncio",
        "gzip",
        "language",
        "main",
        "pickle",
        "random",
        "secrets",
        "threading",
        "time",
        "uuid",
    }
    assert not calls & {"eval", "exec", "start", "sleep"}
    assert "authoritative_recovery: bool = False" in source
    assert "cutover_authorized: bool = False" in source
    assert "production_dual_read: bool = False" in source
