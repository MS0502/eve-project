import json

import pytest

from core.m3_b_phone_recovery_need_witness import (
    COOLDOWN_METHOD,
    DEFAULT_SOURCE_INSTANCE_ID,
    OVERLOAD_COUNT_METHOD,
    PROCESS_CPU_METHOD,
    QUEUE_METHOD,
    RECOVERY_COUNT_METHOD,
    PhoneRecoveryNeedRuntimeSnapshot,
    PhoneRecoveryNeedWitnessError,
    build_phone_recovery_need_witness,
    derive_detached_recovery_need_evidence,
)
from scripts.operator import m3_b_phone_recovery_need_witness as operator_cli


def _snapshot(
    tick: int,
    *,
    active_cpu: float = 0.8,
    active_wall: float = 2.0,
    cooldown_cpu: float = 0.05,
    cooldown_wall: float = 1.0,
    load_before: float = 3.0,
    load_after_active: float = 2.5,
    load_after_cooldown: float = 1.5,
) -> PhoneRecoveryNeedRuntimeSnapshot:
    return PhoneRecoveryNeedRuntimeSnapshot(
        source_instance_id=DEFAULT_SOURCE_INSTANCE_ID,
        logical_tick=tick,
        active_process_cpu_seconds=active_cpu,
        active_wall_seconds=active_wall,
        cooldown_process_cpu_seconds=cooldown_cpu,
        cooldown_wall_seconds=cooldown_wall,
        cpu_count=2,
        load_average_1m_before=load_before,
        load_average_1m_after_active=load_after_active,
        load_average_1m_after_cooldown=load_after_cooldown,
    )


def _snapshots() -> tuple[PhoneRecoveryNeedRuntimeSnapshot, ...]:
    return (
        _snapshot(0),
        _snapshot(
            1,
            active_cpu=1.0,
            cooldown_cpu=0.08,
            load_before=2.4,
            load_after_active=2.2,
            load_after_cooldown=1.9,
        ),
        _snapshot(
            2,
            active_cpu=0.5,
            cooldown_cpu=0.12,
            load_before=1.2,
            load_after_active=1.4,
            load_after_cooldown=1.6,
        ),
    )


def test_snapshot_derives_exact_governed_recovery_fields() -> None:
    snapshot = _snapshot(0)

    assert snapshot.active_window_ticks == 2_000_000
    assert snapshot.cooldown_ticks == 1_000_000
    assert snapshot.sampling_window_ticks == 3_000_000
    assert snapshot.active_processing_ticks == 400_000
    assert snapshot.active_process_cpu_ratio == pytest.approx(0.2)
    assert snapshot.cooldown_process_cpu_ratio == pytest.approx(0.025)
    assert snapshot.recent_overload_count == 2
    assert snapshot.successful_recovery_count == 2

    fields = tuple(field for field, _ in snapshot.raw_values)
    raw = dict(snapshot.raw_values)
    assert fields == (
        "active_processing_ticks",
        "cooldown_ticks",
        "recent_overload_count",
        "sampling_window_ticks",
        "successful_recovery_count",
    )
    assert raw["active_processing_ticks"] == 400_000
    assert raw["cooldown_ticks"] == 1_000_000
    assert raw["recent_overload_count"] == 2
    assert raw["sampling_window_ticks"] == 3_000_000
    assert raw["successful_recovery_count"] == 2

    record = snapshot.to_operational_raw_record()
    assert record.axis == "recovery_need"
    assert record.source_instance_id == DEFAULT_SOURCE_INSTANCE_ID
    assert record.synthetic is False
    assert record.runtime_polled is False
    assert record.recalculated_raw_observation_digest == record.raw_observation_digest


def test_three_real_shape_snapshots_derive_positive_confidence() -> None:
    evidence = derive_detached_recovery_need_evidence(_snapshots())

    assert evidence.axis == "recovery_need"
    assert evidence.source_instance_id == DEFAULT_SOURCE_INSTANCE_ID
    assert 0.0 <= evidence.value <= 1.0
    assert 0.5 <= evidence.confidence <= 1.0
    assert evidence.observed_tick == 2
    assert evidence.synthetic is False
    assert evidence.proposal_only is False


def test_public_review_is_digest_only_and_claims_no_promotion() -> None:
    witness = build_phone_recovery_need_witness(
        private_nonce=b"recovery-private-nonce-material-0001",
        runtime_instance_id="phone-runtime-recovery-001",
        source_instance_id=DEFAULT_SOURCE_INSTANCE_ID,
        repository_head_sha="2" * 40,
        launch_attestation_id="recovery-launch-001",
        snapshots=_snapshots(),
    )

    public = witness.public_review_mapping()
    encoded = json.dumps(public, sort_keys=True)

    assert public["axis"] == "recovery_need"
    assert public["raw_record_count"] == 3
    assert public["private_raw_location"] == "operator_private_companion_only"
    assert public["fixture_only"] is False
    assert public["reviewed_attestation_registered"] is False
    assert public["runtime_provenance_verifier_registered"] is False
    assert public["production_source_verifier_registered"] is False
    assert public["retained_real_observation"] is False
    assert public["observation_window_started"] is False
    assert public["m3_b_complete"] is False
    assert public["m3_c_open"] is False
    assert public["m3_e_authority_open"] is False
    assert public["cutover_authorized"] is False
    assert public["process_cpu_measurement_methods"] == [PROCESS_CPU_METHOD]
    assert public["queue_measurement_methods"] == [QUEUE_METHOD]
    assert public["cooldown_measurement_methods"] == [COOLDOWN_METHOD]
    assert public["overload_count_methods"] == [OVERLOAD_COUNT_METHOD]
    assert public["recovery_count_methods"] == [RECOVERY_COUNT_METHOD]
    for private_field in (
        "active_process_cpu_seconds",
        "active_wall_seconds",
        "cooldown_process_cpu_seconds",
        "cooldown_wall_seconds",
        "load_average_1m_before",
        "load_average_1m_after_active",
        "load_average_1m_after_cooldown",
    ):
        assert private_field not in encoded
    assert len(public["public_review_digest"]) == 64


def test_snapshot_fails_closed_when_active_cpu_exceeds_visible_capacity() -> None:
    with pytest.raises(
        PhoneRecoveryNeedWitnessError,
        match="active process CPU exceeds visible CPU capacity",
    ):
        _snapshot(0, active_cpu=4.1, active_wall=2.0)


def test_snapshot_fails_closed_when_cooldown_cpu_exceeds_visible_capacity() -> None:
    with pytest.raises(
        PhoneRecoveryNeedWitnessError,
        match="cooldown process CPU exceeds visible CPU capacity",
    ):
        _snapshot(0, cooldown_cpu=2.1, cooldown_wall=1.0)


def test_snapshot_sequence_fails_closed_on_duplicate_tick() -> None:
    snapshots = (_snapshot(0), _snapshot(1), _snapshot(1))

    with pytest.raises(
        PhoneRecoveryNeedWitnessError,
        match="strictly increasing",
    ):
        derive_detached_recovery_need_evidence(snapshots)


def test_operator_capture_uses_fixed_quiet_cooldown(monkeypatch) -> None:
    cpu_values = iter((1.0, 1.4, 1.4, 1.45))
    wall_values = iter((10.0, 12.0, 12.0, 13.0))
    load_values = iter((3.0, 2.5, 1.5))
    slept: list[float] = []

    monkeypatch.setattr(operator_cli, "_process_cpu_seconds", lambda: next(cpu_values))
    monkeypatch.setattr(operator_cli.time, "monotonic", lambda: next(wall_values))
    monkeypatch.setattr(operator_cli, "_load_average_1m", lambda: next(load_values))
    monkeypatch.setattr(operator_cli, "_run_interaction", lambda engine, text: None)
    monkeypatch.setattr(operator_cli.time, "sleep", lambda seconds: slept.append(seconds))
    monkeypatch.setattr(operator_cli.os, "cpu_count", lambda: 2)

    snapshot = operator_cli._capture_window(
        object(),
        "실제 입력",
        logical_tick=0,
        source_instance_id=DEFAULT_SOURCE_INSTANCE_ID,
    )

    assert slept == [1.0]
    assert snapshot.active_process_cpu_seconds == pytest.approx(0.4)
    assert snapshot.active_wall_seconds == pytest.approx(2.0)
    assert snapshot.cooldown_process_cpu_seconds == pytest.approx(0.05)
    assert snapshot.cooldown_wall_seconds == pytest.approx(1.0)
