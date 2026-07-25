from __future__ import annotations

import ast
import hashlib
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from core.m3_b_quarantined_risk_source_binding import (
    ACQUISITION_METHOD,
    APPRAISAL_METHOD,
    APPRAISAL_OUTCOME,
    APPRAISAL_SCHEMA_VERSION,
    BINDING_SCHEMA_VERSION,
    QUARANTINE_METHOD,
    QUARANTINE_OUTCOME,
    QUARANTINE_SCHEMA_VERSION,
    QUARANTINE_STATUS,
    RAW_MODEL_OR_RULE_VERSION,
    RAW_SCHEMA_VERSION,
    RISK_DEFENSE_AXES,
    SOURCE_FAMILY,
    VERIFICATION_METHOD,
    QuarantinedRiskRawRecord,
    QuarantinedRiskSourceBindingError,
    derive_quarantined_risk_axis_evidence,
    quarantined_risk_raw_observation_digest,
    quarantined_risk_source_bindings,
)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _raw_values(axis: str, tick: int) -> tuple[tuple[str, object], ...]:
    offset = (tick - 1) * 0.04
    if axis == "threat_pressure":
        return (
            ("appraisal_version", APPRAISAL_SCHEMA_VERSION),
            ("impact_score", 0.42 + offset),
            ("source_trust", 0.82 - offset),
            ("threat_probability", 0.31 + offset),
            ("verification_status", "verified"),
        )
    if axis == "uncertainty_pressure":
        return (
            ("appraisal_version", APPRAISAL_SCHEMA_VERSION),
            ("conflict_count", tick - 1),
            ("missing_evidence_ratio", 0.24 + offset),
            ("source_reliability", 0.80 - offset),
            ("verification_gap", 0.18 + offset),
        )
    if axis == "self_protection":
        return (
            ("appraisal_version", APPRAISAL_SCHEMA_VERSION),
            ("capability_limit", 0.28 + offset),
            ("exposure_scope", 0.33 + offset),
            ("reversibility", 0.82 - offset),
            ("threat_pressure_input", 0.29 + offset),
        )
    if axis == "boundary_defense":
        return (
            ("appraisal_version", APPRAISAL_SCHEMA_VERSION),
            ("boundary_violation_count", tick - 1),
            ("intent_confidence", 0.68 + offset),
            ("persistence_score", 0.38 + offset),
            ("remedy_available", 0.78 - offset),
        )
    if axis == "trust_risk":
        return (
            ("appraisal_version", APPRAISAL_SCHEMA_VERSION),
            ("contradiction_count", tick - 1),
            ("reversibility", 0.80 - offset),
            ("source_reliability", 0.76 - offset),
            ("verification_depth", 0.72 - offset),
        )
    if axis == "exposure_risk":
        return (
            ("audience_scope", 0.30 + offset),
            ("authorization_status", "authorized" if tick == 1 else "restricted"),
            ("persistence_risk", 0.35 + offset),
            ("reversibility", 0.84 - offset),
            ("sensitivity_class", "internal" if tick == 1 else "sensitive"),
        )
    raise AssertionError(axis)


def _record(axis: str, tick: int) -> QuarantinedRiskRawRecord:
    observation_id = f"test:{axis}:observation:{tick}"
    source_instance_id = "test:quarantined-risk-source:v1"
    source_snapshot_id = f"test:{axis}:snapshot:{tick}"
    source_schema_version = "test.quarantined-risk-source.v1"
    source_integrity_digest = _sha(f"source:{axis}:{tick}")
    quarantine_trace_id = f"test:{axis}:quarantine:{tick}"
    quarantine_input_digest = _sha(f"quarantine-input:{axis}:{tick}")
    quarantine_integrity_digest = _sha(f"quarantine-integrity:{axis}:{tick}")
    appraisal_trace_id = f"test:{axis}:appraisal:{tick}"
    appraisal_input_digest = quarantine_integrity_digest
    appraisal_integrity_digest = _sha(f"appraisal-integrity:{axis}:{tick}")
    raw_values = _raw_values(axis, tick)
    raw_observation_digest = quarantined_risk_raw_observation_digest(
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
    return QuarantinedRiskRawRecord(
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


def _rebuilt(
    record: QuarantinedRiskRawRecord,
    **changes: object,
) -> QuarantinedRiskRawRecord:
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
    digest = quarantined_risk_raw_observation_digest(**values)  # type: ignore[arg-type]
    return replace(record, **changes, raw_observation_digest=digest)


def test_binding_set_has_exact_six_axes_and_total_progress_is_twelve_of_thirty_seven():
    binding_set = quarantined_risk_source_bindings()
    assert tuple(item.axis for item in binding_set.bindings) == RISK_DEFENSE_AXES
    assert binding_set.appraised_binding_count == 6
    assert binding_set.total_bound_axis_count == 12
    assert binding_set.remaining_axis_count == 25
    assert binding_set.blockers == (
        "REGISTRY_APPRAISED_25_AXIS_SOURCE_BINDINGS_INCOMPLETE",
        "REGISTRY_POSITIVE_CONFIDENCE_COVERAGE_INCOMPLETE",
    )
    for binding in binding_set.bindings:
        assert binding.appraisal_required is True
        assert binding.quarantine_required is True
        assert binding.hardware_direct_input_allowed is False
        assert binding.authority == "shadow_only"
        assert binding.production_capture_present is False
        assert binding.runtime_capture_installed is False
        assert binding.observation_window_started is False
        assert binding.m3_b_complete is False
        assert binding.m3_c_open is False
        assert binding.m3_e_authority_open is False
        assert binding.cutover_authorized is False


@pytest.mark.parametrize("axis", RISK_DEFENSE_AXES)
def test_quarantined_verified_appraisals_derive_deterministic_positive_confidence_evidence(axis: str):
    records = tuple(_record(axis, tick) for tick in (1, 2))
    first = derive_quarantined_risk_axis_evidence(records)
    second = derive_quarantined_risk_axis_evidence(records)
    assert first == second
    assert first.axis == axis
    assert 0.0 <= first.value <= 1.0
    assert 0.0 < first.confidence <= 1.0
    assert first.observed_tick == 2
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


def test_raw_digest_binds_identity_time_source_quarantine_appraisal_and_values():
    record = _record("threat_pressure", 1)
    assert record.raw_observation_digest == record.recalculated_raw_observation_digest
    for field, value in (
        ("logical_tick", 2),
        ("observation_id", "changed-observation"),
        ("source_snapshot_id", "changed-snapshot"),
        ("source_schema_version", "changed-source-schema"),
        ("quarantine_trace_id", "changed-quarantine-trace"),
        ("quarantine_input_digest", _sha("changed-quarantine-input")),
        ("appraisal_trace_id", "changed-appraisal-trace"),
        ("appraisal_integrity_digest", _sha("changed-appraisal-integrity")),
        ("source_integrity_digest", _sha("changed-source-integrity")),
    ):
        with pytest.raises(
            QuarantinedRiskSourceBindingError,
            match="raw observation digest does not match|appraisal input must be",
        ):
            replace(record, **{field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("acquisition_method", "unverified"),
        ("verification_method", "none"),
        ("quarantine_schema_version", "unversioned"),
        ("quarantine_method", "bypassed"),
        ("quarantine_outcome", "accepted_without_quarantine"),
        ("quarantine_status", "bypassed"),
        ("appraisal_schema_version", "unversioned"),
        ("appraisal_method", "caller_claim"),
        ("appraisal_outcome", "accepted_without_review"),
        ("model_or_rule_version", "unversioned"),
        ("source_family", "synthetic"),
    ),
)
def test_noncanonical_provenance_fails_closed(field: str, value: str):
    with pytest.raises(
        QuarantinedRiskSourceBindingError,
        match="canonical quarantined-risk provenance contract",
    ):
        replace(_record("threat_pressure", 1), **{field: value})


@pytest.mark.parametrize("field", ("quarantine_verified", "appraisal_verified"))
def test_unverified_quarantine_or_appraisal_fails_closed(field: str):
    with pytest.raises(
        QuarantinedRiskSourceBindingError,
        match="requires exact quarantine and appraisal verification",
    ):
        replace(_record("threat_pressure", 1), **{field: False})


def test_appraisal_must_bind_exact_quarantine_output():
    with pytest.raises(
        QuarantinedRiskSourceBindingError,
        match="appraisal input must be the exact verified quarantine output",
    ):
        replace(
            _record("threat_pressure", 1),
            appraisal_input_digest=_sha("not-the-quarantine-output"),
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
def test_forbidden_source_origins_fail_closed(field: str):
    with pytest.raises(QuarantinedRiskSourceBindingError, match="cannot use"):
        replace(_record("threat_pressure", 1), **{field: True})


def test_raw_field_order_versions_enums_ranges_and_counts_are_exact():
    threat = _record("threat_pressure", 1)
    with pytest.raises(QuarantinedRiskSourceBindingError, match="canonical axis source plan"):
        replace(threat, raw_values=tuple(reversed(threat.raw_values)))
    bad_status = tuple(
        (field, "claimed") if field == "verification_status" else (field, value)
        for field, value in threat.raw_values
    )
    with pytest.raises(QuarantinedRiskSourceBindingError, match="verification_status"):
        _rebuilt(threat, raw_values=bad_status)
    uncertainty = _record("uncertainty_pressure", 1)
    bad_count = tuple(
        (field, -1) if field == "conflict_count" else (field, value)
        for field, value in uncertainty.raw_values
    )
    with pytest.raises(QuarantinedRiskSourceBindingError, match="non-negative integer"):
        _rebuilt(uncertainty, raw_values=bad_count)
    exposure = _record("exposure_risk", 1)
    bad_authorization = tuple(
        (field, "maybe") if field == "authorization_status" else (field, value)
        for field, value in exposure.raw_values
    )
    with pytest.raises(QuarantinedRiskSourceBindingError, match="authorization_status"):
        _rebuilt(exposure, raw_values=bad_authorization)


def test_derivation_requires_minimum_count_unique_ids_and_one_source_contract():
    first = _record("threat_pressure", 1)
    second = _record("threat_pressure", 2)
    with pytest.raises(QuarantinedRiskSourceBindingError, match="insufficient raw record count"):
        derive_quarantined_risk_axis_evidence((first,))
    with pytest.raises(QuarantinedRiskSourceBindingError, match="ticks must be sorted"):
        derive_quarantined_risk_axis_evidence((second, first))
    duplicate = _rebuilt(second, observation_id=first.observation_id)
    with pytest.raises(QuarantinedRiskSourceBindingError, match="observation_id values must be unique"):
        derive_quarantined_risk_axis_evidence((first, duplicate))
    duplicate_quarantine = _rebuilt(second, quarantine_trace_id=first.quarantine_trace_id)
    with pytest.raises(QuarantinedRiskSourceBindingError, match="quarantine_trace_id values must be unique"):
        derive_quarantined_risk_axis_evidence((first, duplicate_quarantine))
    changed_source = _rebuilt(second, source_instance_id="other-source")
    with pytest.raises(QuarantinedRiskSourceBindingError, match="share one source_instance_id"):
        derive_quarantined_risk_axis_evidence((first, changed_source))
    mixed_axis = (_record("threat_pressure", 1), _record("uncertainty_pressure", 2))
    with pytest.raises(QuarantinedRiskSourceBindingError, match="cannot mix axes"):
        derive_quarantined_risk_axis_evidence(mixed_axis)


def test_binding_and_binding_set_are_frozen_and_cannot_claim_authority():
    binding_set = quarantined_risk_source_bindings()
    with pytest.raises(FrozenInstanceError):
        binding_set.m3_b_complete = True  # type: ignore[misc]
    with pytest.raises(QuarantinedRiskSourceBindingError, match="cannot claim"):
        replace(binding_set, cutover_authorized=True)
    with pytest.raises(QuarantinedRiskSourceBindingError, match="cannot claim"):
        replace(binding_set.bindings[0], production_capture_present=True)


def test_core_module_has_no_io_polling_scheduler_event_or_runtime_surface():
    path = Path("core/m3_b_quarantined_risk_source_binding.py")
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
        {"open", "read", "read_text", "write", "write_text", "append", "poll"}
    )
