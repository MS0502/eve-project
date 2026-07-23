from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest

from core.m3_b_operational_registry_source_binding import (
    ACQUISITION_METHOD,
    BINDING_SCHEMA_VERSION,
    RAW_MODEL_OR_RULE_VERSION,
    SOURCE_FAMILY,
    VERIFICATION_METHOD,
    OperationalRegistryRawRecord,
    OperationalRegistrySourceBindingError,
    derive_operational_axis_evidence,
    operational_raw_observation_digest,
)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _record(tick: int) -> OperationalRegistryRawRecord:
    raw_values = (
        ("available_cpu_budget", 0.8 - tick * 0.01),
        ("available_memory_budget", 0.75 - tick * 0.01),
        ("battery_governor_band", 0.7 - tick * 0.01),
        ("foreground_load", 0.2 + tick * 0.01),
        ("sampling_window_ticks", 10),
    )
    observation_id = f"provenance:energy:{tick}"
    source_instance_id = "provenance:operational-source:v1"
    source_snapshot_id = f"provenance:snapshot:{tick}"
    source_schema_version = "provenance.operational-source.v1"
    source_integrity_digest = _sha(f"source-integrity:{tick}")
    raw_observation_digest = operational_raw_observation_digest(
        axis="energy_budget",
        logical_tick=tick,
        observation_id=observation_id,
        source_instance_id=source_instance_id,
        source_snapshot_id=source_snapshot_id,
        source_schema_version=source_schema_version,
        source_integrity_digest=source_integrity_digest,
        raw_values=raw_values,
    )
    return OperationalRegistryRawRecord(
        axis="energy_budget",
        logical_tick=tick,
        observation_id=observation_id,
        source_instance_id=source_instance_id,
        source_snapshot_id=source_snapshot_id,
        source_schema_version=source_schema_version,
        source_integrity_digest=source_integrity_digest,
        raw_observation_digest=raw_observation_digest,
        raw_values=raw_values,
    )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("acquisition_method", "unverified_caller_claim"),
        ("verification_method", "none"),
        ("model_or_rule_version", "unversioned"),
        ("source_family", "synthetic_operational_metrics"),
    ),
)
def test_raw_record_rejects_noncanonical_provenance_metadata(
    field: str,
    value: str,
):
    with pytest.raises(
        OperationalRegistrySourceBindingError,
        match="canonical operational provenance contract",
    ):
        replace(_record(1), **{field: value})


def test_derived_evidence_uses_only_canonical_verified_provenance():
    evidence = derive_operational_axis_evidence(tuple(_record(tick) for tick in (1, 2, 3)))
    assert evidence.source_family == SOURCE_FAMILY
    assert evidence.acquisition_method == ACQUISITION_METHOD
    assert evidence.verification_method == VERIFICATION_METHOD
    assert evidence.model_or_rule_version == (
        f"{BINDING_SCHEMA_VERSION}:energy_budget:mean.v1"
    )
    assert RAW_MODEL_OR_RULE_VERSION == BINDING_SCHEMA_VERSION
