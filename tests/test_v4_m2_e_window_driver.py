from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path

import pytest

from core.event_kernel import SHADOW_AUTHORITY
from core.m2_e_window_driver import (
    CIRCADIAN_RUNTIME_SECONDS,
    EVENT_CAP_PER_SIM_HOUR,
    REQUIRED_ACTUAL_MIDNIGHTS,
    WINDOW_BASELINE_SHA,
    WINDOW_EVENT_QUOTA,
    WindowConfig,
    WindowContractError,
    WindowState,
    acceptance_checks,
    advance_runtime,
    apply_watchdog,
    evidence_record,
    mark_notification_sent,
    maybe_seal,
    one_line_status,
    record_discrete_stimulus,
    record_power_cycle,
    record_recovery,
    record_tick_sample_event,
    record_unauthorized_effect,
)

ROOT = Path(__file__).resolve().parents[1]
CORE_MODULE = ROOT / "core/m2_e_window_driver.py"
HABITAT_RUNTIME = ROOT / "scripts/habitat/m2_e_window_runtime.py"
SUPERVISOR = ROOT / "scripts/habitat/supervisor.sh"
SETUP = ROOT / "scripts/habitat/setup_window.sh"


def state() -> WindowState:
    return WindowState.create(window_id="m2-e:test-window:v1", local_date="2026-07-22")


def test_fixed_quota_is_one_cumulative_circadian_cycle_and_cannot_promote():
    config = WindowConfig()
    assert config.event_quota == WINDOW_EVENT_QUOTA == 288
    assert config.stimuli_per_sim_hour == config.event_cap_per_sim_hour == 12
    assert config.circadian_runtime_seconds == CIRCADIAN_RUNTIME_SECONDS
    assert config.authority == SHADOW_AUTHORITY
    assert config.cutover_authorized is False
    assert config.m3_authority_open is False
    assert config.legacy_runtime_authoritative is True
    with pytest.raises(WindowContractError, match="promotion authority"):
        replace(config, cutover_authorized=True)
    with pytest.raises(WindowContractError, match="quota"):
        replace(config, event_quota=287)


def test_a9_allows_only_bounded_discrete_events_and_rejects_tick_sampling():
    config = WindowConfig()
    current = state()
    for _ in range(EVENT_CAP_PER_SIM_HOUR):
        current = record_discrete_stimulus(current, config=config)
    assert current.event_count == EVENT_CAP_PER_SIM_HOUR
    assert current.frozen is False
    overflow = record_discrete_stimulus(current, config=config)
    assert overflow.freeze_reason == "a9_event_cap_exceeded"
    tick = record_tick_sample_event(state())
    assert tick.tick_sample_events == 1
    assert tick.freeze_reason == "a9_tick_sampling_detected"


def test_power_off_gap_is_continuity_evidence_but_not_a_false_running_midnight():
    current = record_power_cycle(state(), local_date="2026-07-24")
    assert current.power_cycle_count == 1
    assert current.actual_midnights == 0
    assert current.last_local_date == "2026-07-24"
    running = advance_runtime(current, elapsed_seconds=60, local_date="2026-07-25")
    assert running.actual_midnights == 1
    assert running.cumulative_runtime_seconds == 60


def test_sealer_waits_for_target_day_and_never_authorizes_cutover():
    config = WindowConfig()
    current = state()
    hour_counts = tuple((hour, 12) for hour in range(24))
    current = replace(
        current,
        cumulative_runtime_seconds=CIRCADIAN_RUNTIME_SECONDS,
        event_count=WINDOW_EVENT_QUOTA,
        hour_counts=hour_counts,
        actual_midnights=REQUIRED_ACTUAL_MIDNIGHTS,
        last_local_date="2026-07-25",
    )
    assert maybe_seal(current, config=config, local_date="2026-07-25").sealed is False
    sealed = maybe_seal(current, config=config, local_date="2026-07-26")
    assert sealed.sealed is True
    assert sealed.awaiting_human_review is True
    assert sealed.seal_reason == "quota_and_acceptance_criteria_met"
    assert sealed.cutover_authorized is False
    assert sealed.m3_authority_open is False


def test_maximum_day_seals_incomplete_for_review_without_self_promotion():
    sealed = maybe_seal(state(), config=WindowConfig(), local_date="2026-07-28")
    assert sealed.sealed is True
    assert sealed.seal_reason == "maximum_deadline_reached_incomplete"
    assert sealed.awaiting_human_review is True
    assert acceptance_checks(sealed, config=WindowConfig())["quota_met"] is False


def test_every_unclean_death_requires_matching_recovery_digest():
    current = record_recovery(state(), expected_digest="a" * 64, recovered_digest="a" * 64)
    assert current.death_count == 1
    assert current.recovery_match_count == 1
    assert current.frozen is False
    mismatch = record_recovery(current, expected_digest="b" * 64, recovered_digest="c" * 64)
    assert mismatch.death_count == 2
    assert mismatch.recovery_match_count == 1
    assert mismatch.freeze_reason == "recovery_digest_mismatch"


def test_watchdog_freezes_shadow_only_and_notification_is_idempotent():
    config = WindowConfig()
    corrupt = apply_watchdog(state(), config=config, disk_bytes=0, integrity_valid=False)
    assert corrupt.freeze_reason == "unrecoverable_corruption"
    full = apply_watchdog(
        state(), config=config, disk_bytes=config.disk_budget_bytes + 1, integrity_valid=True
    )
    assert full.freeze_reason == "disk_budget_exceeded"
    notified = mark_notification_sent(corrupt)
    assert notified.notification_sent is True
    assert mark_notification_sent(notified) == notified
    assert notified.legacy_runtime_authoritative is True


def test_evidence_and_status_preserve_shadow_authority_and_fixed_baseline():
    current = record_unauthorized_effect(state())
    evidence = evidence_record(current, config=WindowConfig())
    assert evidence["baseline_sha"] == WINDOW_BASELINE_SHA
    assert evidence["authority"] == SHADOW_AUTHORITY
    assert evidence["legacy_runtime_authoritative"] is True
    assert evidence["cutover_authorized"] is False
    assert evidence["m3_authority_open"] is False
    assert len(evidence["evidence_digest"]) == 64
    line = one_line_status(current, config=WindowConfig())
    assert "health=frozen" in line
    assert "authority=shadow_only" in line


def test_habitat_has_no_intentional_kill_scheduler_or_legacy_runtime_import():
    tree = ast.parse(HABITAT_RUNTIME.read_text(encoding="utf-8"))
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
    assert not imports & {"adapters", "language", "legacy", "main"}
    assert "kill" not in calls
    supervisor = SUPERVISOR.read_text(encoding="utf-8")
    setup = SETUP.read_text(encoding="utf-8")
    assert "kill -0" in supervisor
    assert "kill -9" not in supervisor
    assert "pkill" not in supervisor
    assert "termux-job-scheduler" not in supervisor + setup
    assert "check-ignore --no-index" in setup
    assert "termux-wake-lock" in supervisor + setup


def test_core_contract_import_has_no_io_thread_or_runtime_activation_surface():
    tree = ast.parse(CORE_MODULE.read_text(encoding="utf-8"))
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
    assert not imports & {"os", "pathlib", "sqlite3", "subprocess", "threading"}
    assert not calls & {"open", "write_text", "mkdir", "connect", "start"}
