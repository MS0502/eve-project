from __future__ import annotations

import ast
import hashlib
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from core.m3_b_appraised_survival_source_binding import (
    ACQUISITION_METHOD,
    APPRAISAL_METHOD,
    APPRAISAL_OUTCOME,
    APPRAISAL_SCHEMA_VERSION,
    APPRAISED_SURVIVAL_AXES,
    BINDING_SCHEMA_VERSION,
    QUARANTINE_STATUS,
    RAW_MODEL_OR_RULE_VERSION,
    RAW_SCHEMA_VERSION,
    SOURCE_FAMILY,
    VERIFICATION_METHOD,
    AppraisedSurvivalRawRecord,
    AppraisedSurvivalSourceBindingError,
    appraised_survival_raw_observation_digest,
    appraised_survival_source_bindings,
    derive_appraised_survival_axis_evidence,
)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _raw_values(axis: str, tick: int) -> tuple[tuple[str, object], ...]:
    offset = (tick - 1) * 0.03
    if axis == "stress_load":
        return (
            ("appraisal_version", APPRAISAL_SCHEMA_VERSION),
            ("controllability_score", 0.72 - offset),
            ("demand_score", 0.42 + offset),
            ("overload_score", 0.28 + offset),
            ("uncertainty_score", 0.34 + offset),
        )
    if axis == "stability_need":
        return (
            ("invariant_failure_count", tick - 1),
            ("pending_migration_count", tick),
            ("replay_divergence_count", tick - 1),
            ("rollback_readiness_score", 0.82 - offset),
            ("sampling_window_ticks", 10),
        )
    raise AssertionError(axis)


def _record(axis: str, tick: int) -> AppraisedSurvivalRawRecord:
    observation_id = f"test:{axis}:observation:{tick}"
    source_instance_id = "test:appraised-survival-source:v1"
    source_snapshot_id = f"test:{axis}:snapshot:{tick}"
    source_schema_version = "test.appraised-survival-source.v1"
    source_integrity_digest = _sha(f"source:{axis}:{tick}")
    appraisal_trace_id = f"test:{axis}:appraisal:{tick}"
    appraisal_input_digest = _sha(f"appraisal-input:{axis}:{tick}")
    appraisal_integrity_digest = _sha(f"appraisal-integrity:{axis}:{tick}")
    raw_values = _raw_values(axis, tick)
    raw_observation_digest = appraised_survival_raw_observation_digest(
        axis=axis,
        logical_tick=tick,
        observation_id=observation_id,
        source_instance_id=source_instance_id,
        source_snapshot_id=source_snapshot_id,
        source_schema_version=source_schema_version,
        source_integrity_digest=source_integrity_digest,
        appraisal_trace_id=appraisal_trace_id,
        appraisal_input_digest=appraisal_input_digest,
        appraisal_integrity_digest=appraisal_integrity_digest,
        raw_values=raw_values,
    )
    return AppraisedSurvivalRawRecord(
        axis=axis,
        logical_tick=tick,
        observation_id=observation_id,
        source_instance_id=source_instance_id,
        source_snapshot_id=source_snapshot_id,
        source_schema_version=source_schema_version,
        source_integrity_digest=source_integrity_digest,
        appraisal_trace_id=appraisal_trace_id,
        appraisal_input_digest=appraisal_input_digest,
        appraisal_integrity_digest=appraisal_integrity_digest,
        raw_observation_digest=raw_observation_digest,
        raw_values=raw_values,
    )


def _rebuilt(
    record: AppraisedSurvivalRawRecord,
    **changes: object,
) -> AppraisedSurvivalRawRecord:
    values = {
        "axis": changes.get("axis", record.axis),
        "logical_tick": changes.get("logical_tick", record.logical_tick),
        "observation_id": changes.get("observation_id", record.observation_id),
        "source_instance_id": changes.get(
            "source_instance_id", record.source_instance_id
        ),
        "source_snapshot_id": changes.get(
            "source_snapshot_id", record.source_snapshot_id
        ),
        "source_schema_version": changes.get(
            "source_schema_version", record.source_schema_version
        ),
        "source_integrity_digest": changes.get(
            "source_integrity_digest", record.source_integrity_digest
        ),
        "appraisal_trace_id": changes.get(
            "appraisal_trace_id", record.appraisal_trace_id
        ),
        "appraisal_input_digest": changes.get(
            "appraisal_input_digest", record.appraisal_input_digest
        ),
        "appraisal_integrity_digest": changes.get(
            "appraisal_integrity_digest", record.appraisal_integrity_digest
        ),
        "raw_values": changes.get("raw_values", record.raw_values),
    }
    digest = appraised_survival_raw_observation_digest(**values)  # type: ignore[arg-type]
    return replace(record, **changes, raw_observation_digest=digest)


def test_binding_set_has_exact_two_axes_and_total_progress_is_six_of_thirty_seven():
    binding_set = appraised_survival_source_bindings()
    assert tuple(item.axis for item in binding_set.bindings) == APPRAISED_SURVIVAL_AXES
    assert binding_set.appraised_binding_count == 2
    assert binding_set.total_bound_axis_count == 6
    assert binding_set.remaining_axis_count == 31
    assert binding_set.blockers == (
        "REGISTRY_APPRAISED_31_AXIS_SOURCE_BINDINGS_INCOMPLETE",
        "REGISTRY_POSITIVE_CONFIDENCE_COVERAGE_INCOMPLETE",
    )
    for binding in binding_set.bindings:
        assert binding.appraisal_required is True
        assert binding.quarantine_required is False
        assert binding.hardware_direct_input_allowed is False
        assert binding.authority == "shadow_only"
        assert binding.production_capture_present is False
        assert binding.runtime_capture_installed is False
        assert binding.observation_window_started is False
        assert binding.m3_b_complete is False
        assert binding.m3_c_open is False
        assert binding.m3_e_authority_open is False
        assert binding.cutover_authorized is False


@pytest.mark.parametrize("axis", APPRAISED_SURVIVAL_AXES)
def test_exact_verified_appraisals_derive_deterministic_positive_confidence_evidence(axis: str):
    records = tuple(_record(axis, tick) for tick in (1, 2, 3))
    first = derive_appraised_survival_axis_evidence(records)
    second = derive_appraised_survival_axis_evidence(records)
    assert first == second
    assert first.axis == axis
    assert 0.0 <= first.value <= 1.0
    assert 0.0 < first.confidence <= 1.0
    assert first.observed_tick == 3
    assert first.source_family == SOURCE_FAMILY
    assert first.source_schema_version == RAW_SCHEMA_VERSION
    assert first.acquisition_method == ACQUISITION_METHOD
    assert first.verification_method == VERIFICATION_METHOD
    assert first.model_or_rule_version == f"{BINDING_SCHEMA_VERSION}:{axis}:mean.v1"
    assert first.synthetic is False
    assert first.proposal_only is False
    assert first.registry_owner_source is False
    assert first.observation_window_started is False
    assert first.affect_impulse_applied is False
    assert first.memory_write_performed is False


def test_raw_digest_binds_identity_time_source_appraisal_and_values():
    record = _record("stress_load", 1)
    assert record.raw_observation_digest == record.recalculated_raw_observation_digest
    for field, value in (
        ("logical_tick", 2),
        ("observation_id", "changed-observation"),
        ("source_snapshot_id", "changed-snapshot"),
        ("source_schema_version", "changed-source-schema"),
        ("appraisal_trace_id", "changed-appraisal-trace"),
        ("appraisal_input_digest", _sha("changed-appraisal-input")),
        ("appraisal_integrity_digest", _sha("changed-appraisal-integrity")),
        ("source_integrity_digest", _sha("changed-source-integrity")),
    ):
        with pytest.raises(
            AppraisedSurvivalSourceBindingError,
            match="raw observation digest does not match",
        ):
            replace(record, **{field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("acquisition_method", "unverified"),
        ("verification_method", "none"),
        ("appraisal_schema_version", "unversioned"),
        ("appraisal_method", "caller_claim"),
        ("appraisal_outcome", "accepted_without_review"),
        ("quarantine_status", "bypassed"),
        ("model_or_rule_version", "unversioned"),
        ("source_family", "synthetic"),
    ),
)
def test_noncanonical_provenance_fails_closed(field: str, value: str):
    with pytest.raises(
        AppraisedSurvivalSourceBindingError,
        match="canonical appraised-survival provenance contract",
    ):
        replace(_record("stress_load", 1), **{field: value})


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
    with pytest.raises(AppraisedSurvivalSourceBindingError, match="cannot use"):
        replace(_record("stress_load", 1), **{field: True})


def test_unverified_appraisal_fails_closed():
    with pytest.raises(AppraisedSurvivalSourceBindingError, match="exactly verified appraisal"):
        replace(_record("stress_load", 1), appraisal_verified=False)


def test_raw_field_order_appraisal_version_ranges_and_counts_are_exact():
    stress = _record("stress_load", 1)
    with pytest.raises(AppraisedSurvivalSourceBindingError, match="canonical axis source plan"):
        replace(stress, raw_values=tuple(reversed(stress.raw_values)))
    bad_version = tuple(
        (field, "wrong") if field == "appraisal_version" else (field, value)
        for field, value in stress.raw_values
    )
    with pytest.raises(AppraisedSurvivalSourceBindingError, match="appraisal_version"):
        _rebuilt(stress, raw_values=bad_version)
    stability = _record("stability_need", 1)
    too_many = tuple(
        (field, 11) if field == "invariant_failure_count" else (field, value)
        for field, value in stability.raw_values
    )
    with pytest.raises(AppraisedSurvivalSourceBindingError, match="cannot exceed"):
        _rebuilt(stability, raw_values=too_many)


def test_derivation_requires_minimum_count_unique_ids_and_one_source_contract():
    first = _record("stress_load", 1)
    second = _record("stress_load", 2)
    third = _record("stress_load", 3)
    with pytest.raises(AppraisedSurvivalSourceBindingError, match="insufficient raw record count"):
        derive_appraised_survival_axis_evidence((first, second))
    with pytest.raises(AppraisedSurvivalSourceBindingError, match="ticks must be sorted"):
        derive_appraised_survival_axis_evidence((third, second, first))
    duplicate = _rebuilt(third, observation_id=second.observation_id)
    with pytest.raises(AppraisedSurvivalSourceBindingError, match="observation_id values must be unique"):
        derive_appraised_survival_axis_evidence((first, second, duplicate))
    duplicate_trace = _rebuilt(third, appraisal_trace_id=second.appraisal_trace_id)
    with pytest.raises(AppraisedSurvivalSourceBindingError, match="appraisal_trace_id values must be unique"):
        derive_appraised_survival_axis_evidence((first, second, duplicate_trace))
    changed_source = _rebuilt(third, source_instance_id="other-source")
    with pytest.raises(AppraisedSurvivalSourceBindingError, match="share one source_instance_id"):
        derive_appraised_survival_axis_evidence((first, second, changed_source))
    mixed_axis = (
        _record("stress_load", 1),
        _record("stress_load", 2),
        _record("stability_need", 3),
    )
    with pytest.raises(AppraisedSurvivalSourceBindingError, match="cannot mix axes"):
        derive_appraised_survival_axis_evidence(mixed_axis)


def test_binding_and_binding_set_are_frozen_and_cannot_claim_authority():
    binding_set = appraised_survival_source_bindings()
    with pytest.raises(FrozenInstanceError):
        binding_set.m3_b_complete = True  # type: ignore[misc]
    with pytest.raises(AppraisedSurvivalSourceBindingError, match="cannot claim"):
        replace(binding_set, cutover_authorized=True)
    with pytest.raises(AppraisedSurvivalSourceBindingError, match="cannot claim"):
        replace(binding_set.bindings[0], production_capture_present=True)


def test_core_module_has_no_io_polling_scheduler_event_or_runtime_surface():
    path = Path("core/m3_b_appraised_survival_source_binding.py")
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
        {"os", "pathlib", "psutil", "sqlite3", "subprocess", "threading", "time"}
    )
    assert not called.intersection(
        {"append_event", "connect", "emit", "open", "poll", "save", "start", "write", "write_text"}
    )
    assert RAW_MODEL_OR_RULE_VERSION == BINDING_SCHEMA_VERSION
    assert APPRAISAL_METHOD
    assert APPRAISAL_OUTCOME
    assert QUARANTINE_STATUS
