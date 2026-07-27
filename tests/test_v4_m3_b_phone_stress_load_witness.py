import json

import pytest

from core.m3_b_appraised_survival_source_binding import APPRAISAL_SCHEMA_VERSION
from core.m3_b_phone_stress_load_witness import (
    APPRAISAL_OUTPUT_KIND,
    APPRAISAL_POLICY_VERSION,
    CONTROLLABILITY_METHOD,
    DEFAULT_SOURCE_INSTANCE_ID,
    DEMAND_METHOD,
    OVERLOAD_METHOD,
    PROCESS_CPU_METHOD,
    QUEUE_METHOD,
    RUNTIME_INPUT_KIND,
    UNCERTAINTY_METHOD,
    PhoneStressLoadRuntimeSnapshot,
    PhoneStressLoadWitnessError,
    appraisal_bridge_provenance_boundary,
    build_phone_stress_load_witness,
    derive_detached_stress_load_evidence,
)
from scripts.operator import m3_b_phone_stress_load_witness as operator_cli


def _snapshot(
    tick: int,
    *,
    process_cpu: float = 0.8,
    wall: float = 2.0,
    load_before: float = 0.6,
    load_after: float = 0.8,
) -> PhoneStressLoadRuntimeSnapshot:
    return PhoneStressLoadRuntimeSnapshot(
        source_instance_id=DEFAULT_SOURCE_INSTANCE_ID,
        logical_tick=tick,
        process_cpu_seconds=process_cpu,
        wall_seconds=wall,
        cpu_count=2,
        load_average_1m_before=load_before,
        load_average_1m_after=load_after,
    )


def _snapshots() -> tuple[PhoneStressLoadRuntimeSnapshot, ...]:
    return (
        _snapshot(0),
        _snapshot(
            1,
            process_cpu=1.0,
            wall=2.5,
            load_before=0.8,
            load_after=1.0,
        ),
        _snapshot(
            2,
            process_cpu=0.6,
            wall=2.0,
            load_before=0.5,
            load_after=0.7,
        ),
    )


def test_runtime_metrics_are_bridged_to_detached_appraisal_without_relabeling() -> None:
    snapshot = _snapshot(0)
    boundary = appraisal_bridge_provenance_boundary()

    assert snapshot.process_cpu_ratio == pytest.approx(0.2)
    assert snapshot.queue_ratio_before == pytest.approx(0.3)
    assert snapshot.queue_ratio_after == pytest.approx(0.4)
    assert snapshot.uncertainty_score == pytest.approx(0.1)
    assert snapshot.demand_score == pytest.approx(0.3)
    assert snapshot.overload_score == pytest.approx(0.4)
    assert snapshot.controllability_score == pytest.approx(0.75)

    assert boundary == {
        "appraisal_bridge_output_detached": True,
        "appraisal_output_kind": APPRAISAL_OUTPUT_KIND,
        "canonical_appraised_record_hardware_direct_input": False,
        "canonical_appraised_record_runtime_polled": False,
        "raw_runtime_metrics_publicly_retained": False,
        "runtime_input_kind": RUNTIME_INPUT_KIND,
        "runtime_metrics_used_as_appraisal_input": True,
    }
    assert snapshot.appraisal_input_mapping["runtime_input_kind"] == RUNTIME_INPUT_KIND
    assert snapshot.appraisal_trace_mapping["appraisal_output_kind"] == APPRAISAL_OUTPUT_KIND
    assert snapshot.appraisal_trace_mapping["appraisal_version"] == APPRAISAL_SCHEMA_VERSION
    assert snapshot.appraisal_trace_mapping["appraisal_policy_version"] == APPRAISAL_POLICY_VERSION

    record = snapshot.to_appraised_raw_record()
    assert record.axis == "stress_load"
    assert record.source_instance_id == DEFAULT_SOURCE_INSTANCE_ID
    assert record.appraisal_verified is True
    assert record.synthetic is False
    assert record.runtime_polled is False
    assert record.hardware_direct_input is False
    assert tuple(record.raw_mapping) == (
        "appraisal_version",
        "controllability_score",
        "demand_score",
        "overload_score",
        "uncertainty_score",
    )
    assert "process_cpu_seconds" not in record.raw_mapping
    assert "wall_seconds" not in record.raw_mapping
    assert "load_average_1m_before" not in record.raw_mapping
    assert "load_average_1m_after" not in record.raw_mapping
    assert record.recalculated_raw_observation_digest == record.raw_observation_digest


def test_three_real_shape_snapshots_derive_positive_confidence() -> None:
    evidence = derive_detached_stress_load_evidence(_snapshots())

    assert evidence.axis == "stress_load"
    assert evidence.source_instance_id == DEFAULT_SOURCE_INSTANCE_ID
    assert 0.0 <= evidence.value <= 1.0
    assert 0.5 <= evidence.confidence <= 1.0
    assert evidence.observed_tick == 2
    assert evidence.synthetic is False
    assert evidence.proposal_only is False


def test_public_review_discloses_bridge_but_not_private_runtime_values() -> None:
    witness = build_phone_stress_load_witness(
        private_nonce=b"stress-private-nonce-material-00000001",
        runtime_instance_id="phone-runtime-stress-001",
        source_instance_id=DEFAULT_SOURCE_INSTANCE_ID,
        repository_head_sha="3" * 40,
        launch_attestation_id="stress-launch-001",
        snapshots=_snapshots(),
    )

    public = witness.public_review_mapping()
    encoded = json.dumps(public, sort_keys=True)

    assert public["axis"] == "stress_load"
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
    assert public["provenance_boundary"] == appraisal_bridge_provenance_boundary()
    assert public["process_cpu_measurement_methods"] == [PROCESS_CPU_METHOD]
    assert public["queue_measurement_methods"] == [QUEUE_METHOD]
    assert public["controllability_methods"] == [CONTROLLABILITY_METHOD]
    assert public["demand_methods"] == [DEMAND_METHOD]
    assert public["overload_methods"] == [OVERLOAD_METHOD]
    assert public["uncertainty_methods"] == [UNCERTAINTY_METHOD]
    for private_field in (
        "process_cpu_seconds",
        "wall_seconds",
        "cpu_count",
        "load_average_1m_before",
        "load_average_1m_after",
    ):
        assert private_field not in encoded
    assert len(public["public_review_digest"]) == 64


def test_snapshot_fails_closed_when_process_cpu_exceeds_visible_capacity() -> None:
    with pytest.raises(
        PhoneStressLoadWitnessError,
        match="process CPU exceeds visible CPU capacity",
    ):
        _snapshot(0, process_cpu=4.1, wall=2.0)


def test_snapshot_sequence_fails_closed_on_duplicate_tick() -> None:
    snapshots = (_snapshot(0), _snapshot(1), _snapshot(1))

    with pytest.raises(
        PhoneStressLoadWitnessError,
        match="strictly increasing",
    ):
        derive_detached_stress_load_evidence(snapshots)


def test_operator_capture_uses_new_full_engine_window_measurements(monkeypatch) -> None:
    cpu_values = iter((1.0, 1.4))
    wall_values = iter((10.0, 12.0))
    load_values = iter((0.6, 0.8))
    interactions: list[str] = []

    monkeypatch.setattr(operator_cli, "_process_cpu_seconds", lambda: next(cpu_values))
    monkeypatch.setattr(operator_cli.time, "monotonic", lambda: next(wall_values))
    monkeypatch.setattr(operator_cli, "_load_average_1m", lambda: next(load_values))
    monkeypatch.setattr(
        operator_cli,
        "_run_interaction",
        lambda engine, text: interactions.append(text),
    )
    monkeypatch.setattr(operator_cli.os, "cpu_count", lambda: 2)

    snapshot = operator_cli._capture_interaction(
        object(),
        "새 stress witness 입력",
        logical_tick=0,
        source_instance_id=DEFAULT_SOURCE_INSTANCE_ID,
    )

    assert interactions == ["새 stress witness 입력"]
    assert snapshot.process_cpu_seconds == pytest.approx(0.4)
    assert snapshot.wall_seconds == pytest.approx(2.0)
    assert snapshot.load_average_1m_before == pytest.approx(0.6)
    assert snapshot.load_average_1m_after == pytest.approx(0.8)
