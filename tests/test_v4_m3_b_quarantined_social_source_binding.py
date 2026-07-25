from __future__ import annotations

import ast
import hashlib
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from core.m3_b_quarantined_social_source_binding import (
    ACQUISITION_METHOD,
    APPRAISAL_SCHEMA_VERSION,
    BINDING_SCHEMA_VERSION,
    RAW_SCHEMA_VERSION,
    SOCIAL_RELATIONSHIP_AXES,
    SOURCE_FAMILY,
    VERIFICATION_METHOD,
    QuarantinedSocialRawRecord,
    QuarantinedSocialSourceBindingError,
    derive_quarantined_social_axis_evidence,
    quarantined_social_raw_observation_digest,
    quarantined_social_source_bindings,
)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _ticks(axis: str) -> tuple[int, ...]:
    if axis == "care_drive":
        return (1, 3)
    if axis == "attachment":
        return (1, 7, 13)
    return (1, 5, 9)


def _offset(tick: int) -> float:
    return min(0.08, ((tick - 1) // 4) * 0.025)


def _raw_values(axis: str, tick: int) -> tuple[tuple[str, object], ...]:
    offset = _offset(tick)
    if axis == "social_pain":
        return (
            ("appraisal_version", APPRAISAL_SCHEMA_VERSION),
            ("injury_evidence_score", 0.28 + offset),
            ("intent_confidence", 0.62 + offset),
            ("recurrence_count", (tick - 1) // 4),
            ("source_trust", 0.78 - offset),
        )
    if axis == "social_trust":
        return (
            ("contradiction_count", (tick - 1) // 8),
            ("fulfilled_commitment_count", 2 + (tick - 1) // 4),
            ("observation_span_ticks", 8 + tick),
            ("repair_count", 1 + (tick - 1) // 8),
            ("source_trust", 0.82 - offset),
        )
    if axis == "attachment":
        return (
            ("appraisal_version", APPRAISAL_SCHEMA_VERSION),
            ("interaction_continuity", 0.68 + offset),
            ("mutual_reliability", 0.76 - offset / 2),
            ("relationship_span_ticks", 20 + tick),
            ("separation_tolerance", 0.72 - offset / 2),
        )
    if axis == "care_drive":
        return (
            ("appraisal_version", APPRAISAL_SCHEMA_VERSION),
            ("capability_to_help", 0.70 - offset),
            ("consent_status", "granted" if tick == 1 else "limited"),
            ("cost_boundary", 0.24 + offset),
            ("welfare_need_score", 0.56 + offset),
        )
    if axis == "loneliness_pressure":
        return (
            ("appraisal_version", APPRAISAL_SCHEMA_VERSION),
            ("available_relationship_context", 0.66 - offset),
            ("chosen_solitude_flag", tick == 1),
            ("meaningful_contact_gap_ticks", 2 + tick),
            ("unmet_connection_signal_count", (tick - 1) // 4),
        )
    if axis == "belonging_need":
        return (
            ("appraisal_version", APPRAISAL_SCHEMA_VERSION),
            ("context_span_ticks", 8 + tick),
            ("group_continuity", 0.74 - offset),
            ("reciprocal_inclusion_count", 2 + (tick - 1) // 4),
            ("role_clarity", 0.71 - offset / 2),
        )
    if axis == "rejection_sensitivity":
        return (
            ("ambiguous_signal_count", 1 + (tick - 1) // 4),
            ("false_positive_count", (tick - 1) // 8),
            ("observation_span_ticks", 8 + tick),
            ("source_trust", 0.79 - offset),
            ("verified_rejection_count", (tick - 1) // 8),
        )
    raise AssertionError(axis)


def _record(axis: str, tick: int) -> QuarantinedSocialRawRecord:
    observation_id = f"test:{axis}:observation:{tick}"
    source_instance_id = "test:quarantined-social-source:v1"
    source_snapshot_id = f"test:{axis}:snapshot:{tick}"
    source_schema_version = "test.quarantined-social-source.v1"
    source_integrity_digest = _sha(f"source:{axis}:{tick}")
    quarantine_trace_id = f"test:{axis}:quarantine:{tick}"
    quarantine_input_digest = _sha(f"quarantine-input:{axis}:{tick}")
    quarantine_integrity_digest = _sha(f"quarantine-integrity:{axis}:{tick}")
    appraisal_trace_id = f"test:{axis}:appraisal:{tick}"
    appraisal_input_digest = quarantine_integrity_digest
    appraisal_integrity_digest = _sha(f"appraisal-integrity:{axis}:{tick}")
    raw_values = _raw_values(axis, tick)
    raw_observation_digest = quarantined_social_raw_observation_digest(
        axis=axis,
        logical_tick=tick,
        observation_id=observation_id,
        source_instance_id=source_instance_id,
        source_snapshot_id=source_snapshot_id,
        source_schema_version=source_schema_version,
        source_integrity_digest=source_integrity_digest,
        quarantine_trace_id=quarantine_trace_id,
        quarantine_input_digest=quarantine_input_digest,
        quarantine_integrity_digest=quarantine_integrity_digest,
        appraisal_trace_id=appraisal_trace_id,
        appraisal_input_digest=appraisal_input_digest,
        appraisal_integrity_digest=appraisal_integrity_digest,
        raw_values=raw_values,
    )
    return QuarantinedSocialRawRecord(
        axis=axis,
        logical_tick=tick,
        observation_id=observation_id,
        source_instance_id=source_instance_id,
        source_snapshot_id=source_snapshot_id,
        source_schema_version=source_schema_version,
        source_integrity_digest=source_integrity_digest,
        quarantine_trace_id=quarantine_trace_id,
        quarantine_input_digest=quarantine_input_digest,
        quarantine_integrity_digest=quarantine_integrity_digest,
        appraisal_trace_id=appraisal_trace_id,
        appraisal_input_digest=appraisal_input_digest,
        appraisal_integrity_digest=appraisal_integrity_digest,
        raw_observation_digest=raw_observation_digest,
        raw_values=raw_values,
    )


def _rebuilt(record: QuarantinedSocialRawRecord, **changes: object) -> QuarantinedSocialRawRecord:
    values = {
        "axis": changes.get("axis", record.axis),
        "logical_tick": changes.get("logical_tick", record.logical_tick),
        "observation_id": changes.get("observation_id", record.observation_id),
        "source_instance_id": changes.get("source_instance_id", record.source_instance_id),
        "source_snapshot_id": changes.get("source_snapshot_id", record.source_snapshot_id),
        "source_schema_version": changes.get("source_schema_version", record.source_schema_version),
        "source_integrity_digest": changes.get("source_integrity_digest", record.source_integrity_digest),
        "quarantine_trace_id": changes.get("quarantine_trace_id", record.quarantine_trace_id),
        "quarantine_input_digest": changes.get("quarantine_input_digest", record.quarantine_input_digest),
        "quarantine_integrity_digest": changes.get("quarantine_integrity_digest", record.quarantine_integrity_digest),
        "appraisal_trace_id": changes.get("appraisal_trace_id", record.appraisal_trace_id),
        "appraisal_input_digest": changes.get("appraisal_input_digest", record.appraisal_input_digest),
        "appraisal_integrity_digest": changes.get("appraisal_integrity_digest", record.appraisal_integrity_digest),
        "raw_values": changes.get("raw_values", record.raw_values),
    }
    digest = quarantined_social_raw_observation_digest(**values)  # type: ignore[arg-type]
    return replace(record, **changes, raw_observation_digest=digest)


def test_binding_set_has_exact_seven_axes_and_total_progress_is_nineteen_of_thirty_seven():
    binding_set = quarantined_social_source_bindings()
    assert tuple(item.axis for item in binding_set.bindings) == SOCIAL_RELATIONSHIP_AXES
    assert binding_set.appraised_binding_count == 7
    assert binding_set.total_bound_axis_count == 19
    assert binding_set.remaining_axis_count == 18
    assert binding_set.blockers == (
        "REGISTRY_APPRAISED_18_AXIS_SOURCE_BINDINGS_INCOMPLETE",
        "REGISTRY_POSITIVE_CONFIDENCE_COVERAGE_INCOMPLETE",
    )
    expected_minima = {
        "attachment": (3, 12),
        "care_drive": (2, 2),
    }
    for binding in binding_set.bindings:
        if binding.axis in expected_minima:
            assert (binding.minimum_raw_record_count, binding.minimum_logical_span_ticks) == expected_minima[binding.axis]
        else:
            assert (binding.minimum_raw_record_count, binding.minimum_logical_span_ticks) == (3, 8)
        assert binding.appraisal_required is True
        assert binding.quarantine_required is True
        assert binding.hardware_direct_input_allowed is False
        assert binding.authority == "shadow_only"
        assert binding.production_capture_present is False
        assert binding.observation_window_started is False
        assert binding.m3_b_complete is False
        assert binding.m3_c_open is False
        assert binding.m3_e_authority_open is False
        assert binding.cutover_authorized is False


@pytest.mark.parametrize("axis", SOCIAL_RELATIONSHIP_AXES)
def test_quarantined_social_records_derive_deterministic_positive_confidence_evidence(axis: str):
    records = tuple(_record(axis, tick) for tick in _ticks(axis))
    first = derive_quarantined_social_axis_evidence(records)
    second = derive_quarantined_social_axis_evidence(records)
    assert first == second
    assert first.axis == axis
    assert 0.0 <= first.value <= 1.0
    assert 0.0 < first.confidence <= 1.0
    assert first.observed_tick == _ticks(axis)[-1]
    assert first.source_family == SOURCE_FAMILY
    assert first.source_schema_version == RAW_SCHEMA_VERSION
    assert first.acquisition_method == ACQUISITION_METHOD
    assert first.verification_method == VERIFICATION_METHOD
    assert first.model_or_rule_version == f"{BINDING_SCHEMA_VERSION}:{axis}:mean.v1"
    assert first.observation_kind == "verified_current_value_observation"
    assert first.verification_status == "verified"
    assert first.genesis_derived is False
    assert first.baseline_derived is False
    assert first.default_derived is False
    assert first.synthetic is False
    assert first.proposal_only is False
    assert first.recalculable_reference_present is True


def test_raw_digest_binds_social_identity_time_quarantine_appraisal_and_values():
    record = _record("social_pain", 1)
    assert record.raw_observation_digest == record.recalculated_raw_observation_digest
    for field, value in (
        ("logical_tick", 2),
        ("observation_id", "changed-observation"),
        ("source_snapshot_id", "changed-snapshot"),
        ("quarantine_trace_id", "changed-quarantine"),
        ("appraisal_trace_id", "changed-appraisal"),
        ("source_integrity_digest", _sha("changed-source")),
    ):
        with pytest.raises(
            QuarantinedSocialSourceBindingError,
            match="raw observation digest does not match|social appraisal input must be",
        ):
            replace(record, **{field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("acquisition_method", "unverified"),
        ("verification_method", "none"),
        ("quarantine_method", "bypassed"),
        ("quarantine_outcome", "accepted_without_quarantine"),
        ("appraisal_method", "caller_claim"),
        ("appraisal_outcome", "accepted_without_review"),
        ("source_family", "raw_social_feedback"),
    ),
)
def test_noncanonical_social_provenance_fails_closed(field: str, value: str):
    with pytest.raises(
        QuarantinedSocialSourceBindingError,
        match="canonical social provenance contract",
    ):
        replace(_record("social_pain", 1), **{field: value})


@pytest.mark.parametrize("field", ("quarantine_verified", "appraisal_verified"))
def test_unverified_social_quarantine_or_appraisal_fails_closed(field: str):
    with pytest.raises(
        QuarantinedSocialSourceBindingError,
        match="requires exact quarantine and appraisal verification",
    ):
        replace(_record("social_pain", 1), **{field: False})


def test_social_appraisal_must_bind_exact_quarantine_output():
    with pytest.raises(
        QuarantinedSocialSourceBindingError,
        match="social appraisal input must be the exact verified quarantine output",
    ):
        replace(
            _record("social_pain", 1),
            appraisal_input_digest=_sha("not-quarantine-output"),
        )


@pytest.mark.parametrize(
    "field",
    (
        "raw_social_feedback_source",
        "hardware_direct_input",
        "synthetic",
        "proposal_only",
        "registry_owner_source",
        "runtime_polled",
    ),
)
def test_forbidden_social_origins_fail_closed(field: str):
    with pytest.raises(QuarantinedSocialSourceBindingError, match="cannot use"):
        replace(_record("social_pain", 1), **{field: True})


def test_social_field_order_appraisal_versions_boolean_consent_and_counts_are_exact():
    social_pain = _record("social_pain", 1)
    with pytest.raises(QuarantinedSocialSourceBindingError, match="canonical social source plan"):
        replace(social_pain, raw_values=tuple(reversed(social_pain.raw_values)))
    bad_version = tuple(
        (field, "other") if field == "appraisal_version" else (field, value)
        for field, value in social_pain.raw_values
    )
    with pytest.raises(QuarantinedSocialSourceBindingError, match="appraisal_version"):
        _rebuilt(social_pain, raw_values=bad_version)
    loneliness = _record("loneliness_pressure", 1)
    bad_solitude = tuple(
        (field, 1) if field == "chosen_solitude_flag" else (field, value)
        for field, value in loneliness.raw_values
    )
    with pytest.raises(QuarantinedSocialSourceBindingError, match="chosen_solitude_flag"):
        _rebuilt(loneliness, raw_values=bad_solitude)
    care = _record("care_drive", 1)
    bad_consent = tuple(
        (field, "assumed") if field == "consent_status" else (field, value)
        for field, value in care.raw_values
    )
    with pytest.raises(QuarantinedSocialSourceBindingError, match="consent_status"):
        _rebuilt(care, raw_values=bad_consent)
    trust = _record("social_trust", 1)
    bad_count = tuple(
        (field, -1) if field == "contradiction_count" else (field, value)
        for field, value in trust.raw_values
    )
    with pytest.raises(QuarantinedSocialSourceBindingError, match="non-negative integer"):
        _rebuilt(trust, raw_values=bad_count)


def test_derivation_enforces_social_minima_unique_ids_span_and_one_source_contract():
    pain = tuple(_record("social_pain", tick) for tick in _ticks("social_pain"))
    with pytest.raises(QuarantinedSocialSourceBindingError, match="insufficient raw record count"):
        derive_quarantined_social_axis_evidence(pain[:2])
    too_short = (_record("social_pain", 1), _record("social_pain", 2), _record("social_pain", 3))
    with pytest.raises(QuarantinedSocialSourceBindingError, match="insufficient logical observation span"):
        derive_quarantined_social_axis_evidence(too_short)
    with pytest.raises(QuarantinedSocialSourceBindingError, match="ticks must be sorted"):
        derive_quarantined_social_axis_evidence(tuple(reversed(pain)))
    duplicate = _rebuilt(pain[1], observation_id=pain[0].observation_id)
    with pytest.raises(QuarantinedSocialSourceBindingError, match="observation_id values must be unique"):
        derive_quarantined_social_axis_evidence((pain[0], duplicate, pain[2]))
    changed_source = _rebuilt(pain[1], source_instance_id="other-social-source")
    with pytest.raises(QuarantinedSocialSourceBindingError, match="share one source_instance_id"):
        derive_quarantined_social_axis_evidence((pain[0], changed_source, pain[2]))
    mixed = (_record("social_pain", 1), _record("social_trust", 5), _record("social_pain", 9))
    with pytest.raises(QuarantinedSocialSourceBindingError, match="cannot mix axes"):
        derive_quarantined_social_axis_evidence(mixed)


def test_social_binding_objects_are_frozen_and_cannot_claim_authority():
    binding_set = quarantined_social_source_bindings()
    with pytest.raises(FrozenInstanceError):
        binding_set.m3_b_complete = True  # type: ignore[misc]
    with pytest.raises(QuarantinedSocialSourceBindingError, match="cannot claim"):
        replace(binding_set, cutover_authorized=True)
    with pytest.raises(QuarantinedSocialSourceBindingError, match="cannot claim"):
        replace(binding_set.bindings[0], runtime_capture_installed=True)


def test_social_core_module_has_no_io_polling_scheduler_event_or_runtime_surface():
    path = Path("core/m3_b_quarantined_social_source_binding.py")
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=path.as_posix())
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
    assert not imported.intersection(
        {"os", "pathlib", "socket", "requests", "urllib", "subprocess"}
    )
    assert not called.intersection(
        {"open", "read", "read_text", "write", "write_text", "poll"}
    )
