"""Immutable retained-real-observation sink backed by the existing M2-A store.

The sink performs durable append only when the caller explicitly provides an
exact ``ProductionCaptureRecord``. Construction performs no I/O and never
initializes a database. The caller must explicitly initialize the concrete
``SQLiteShadowStore`` first. Persisted events remain ``shadow_only`` and cannot
start an observation window, mutate the registry owner, authorize cutover, or
open M3-E runtime authority.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Mapping

from core.event_kernel import EventEnvelope, SHADOW_AUTHORITY, canonical_json_object
from core.m3_b_registry_production_capture_adapter import ProductionCaptureRecord
from core.sqlite_shadow_store import AppendReceipt as StoreAppendReceipt
from core.sqlite_shadow_store import SQLiteShadowStore

SINK_SCHEMA_VERSION = "eve.m3-b.registry-retained-real-observation-sink.v1"
RECEIPT_SCHEMA_VERSION = "eve.m3-b.registry-retained-real-observation-receipt.v1"
CAPABILITY_SCHEMA_VERSION = "eve.m3-b.registry-retained-real-observation-sink-capability.v1"
SINK_VERSION = "eve.m3-b.registry-retained-real-observation-sink.v1"
RETENTION_EVENT_TYPE = "m3_b.registry.retained_real_observation"
RETENTION_STREAM_ID = "shadow:m3-b:registry-retained-real-observation"
RETENTION_PRODUCER = "eve.m3-b.registry-retention-sink"
ZERO_DIGEST = "0" * 64
_DIGEST = re.compile(r"^[0-9a-f]{64}$")


class RegistryRetainedObservationSinkError(ValueError):
    """Raised when a capture cannot be retained without weakening provenance."""


def _identifier(value: Any, field: str, *, limit: int = 256) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise RegistryRetainedObservationSinkError(
            f"{field} must be a bounded non-empty string"
        )
    return value


def _digest_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None or value == ZERO_DIGEST:
        raise RegistryRetainedObservationSinkError(
            f"{field} must be a non-placeholder lowercase SHA-256 digest"
        )
    return value


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RegistryRetainedObservationSinkError(f"{field} must be a positive integer")
    return value


def _canonical(value: Mapping[str, Any], field: str) -> str:
    try:
        return canonical_json_object(value, field=field)
    except (TypeError, ValueError) as exc:
        raise RegistryRetainedObservationSinkError(f"{field} is not canonical JSON") from exc


def _digest(value: Mapping[str, Any], field: str) -> str:
    return hashlib.sha256(_canonical(value, field).encode("utf-8")).hexdigest()


def _retention_payload(capture: ProductionCaptureRecord) -> dict[str, Any]:
    return {
        "axis": capture.axis,
        "capture_digest": capture.capture_digest,
        "capture_record": capture.to_mapping(),
        "classification": "retained_real_observation",
        "hash_algorithm": "sha256",
        "immutable": True,
        "retention_schema_version": SINK_SCHEMA_VERSION,
        "source_evidence_digest": capture.evidence.evidence_digest,
        "source_verification_digest": capture.verification.verification_digest,
    }


def build_retained_real_observation_event(
    capture: ProductionCaptureRecord,
    *,
    event_id: str,
    sequence: int,
    correlation_id: str,
    causation_id: str | None = None,
) -> EventEnvelope:
    """Build the exact shadow-only event that the durable sink will append."""
    if type(capture) is not ProductionCaptureRecord:
        raise RegistryRetainedObservationSinkError(
            "retention requires exact immutable ProductionCaptureRecord"
        )
    if not capture.retained_real_observation_eligible or not capture.verification.counts_as_real:
        raise RegistryRetainedObservationSinkError(
            "capture is not eligible for retained-real-observation persistence"
        )
    _identifier(event_id, "event_id")
    _positive_int(sequence, "sequence")
    _identifier(correlation_id, "correlation_id")
    if causation_id is not None:
        _identifier(causation_id, "causation_id")
    return EventEnvelope.create(
        event_id=event_id,
        event_type=RETENTION_EVENT_TYPE,
        stream_id=RETENTION_STREAM_ID,
        sequence=sequence,
        producer=RETENTION_PRODUCER,
        producer_version=SINK_VERSION,
        correlation_id=correlation_id,
        causation_id=causation_id,
        payload=_retention_payload(capture),
        causal_context={
            "capture_id": capture.capture_id,
            "capture_tick": capture.capture_tick,
            "source_contract_id": capture.verification.source_contract_id,
            "source_snapshot_id": capture.verification.source_snapshot_id,
            "verifier_id": capture.verification.verifier_id,
            "verifier_trace_digest": capture.verification.verifier_trace_digest,
        },
        authority=SHADOW_AUTHORITY,
    )


@dataclass(frozen=True, slots=True)
class RetainedRealObservationReceipt:
    axis: str
    capture_id: str
    capture_digest: str
    verification_digest: str
    event_id: str
    sequence: int
    event_envelope_digest: str
    store_ordinal: int
    store_before_count: int
    store_after_count: int
    store_before_chain_digest: str
    store_after_chain_digest: str
    store_transition_hash: str
    readback_verified: bool
    schema_version: str = RECEIPT_SCHEMA_VERSION
    sink_version: str = SINK_VERSION
    authority: str = SHADOW_AUTHORITY
    retained_real_observation_delta: int = 1
    observation_window_started: bool = False
    registry_owner_mutated: bool = False
    m3_b_complete: bool = False
    m3_c_open: bool = False
    m3_e_authority_open: bool = False
    cutover_authorized: bool = False

    def __post_init__(self) -> None:
        for field in ("axis", "capture_id", "event_id"):
            _identifier(getattr(self, field), field)
        for field in (
            "capture_digest",
            "verification_digest",
            "event_envelope_digest",
            "store_before_chain_digest",
            "store_after_chain_digest",
            "store_transition_hash",
        ):
            _digest_string(getattr(self, field), field)
        _positive_int(self.sequence, "sequence")
        if isinstance(self.store_ordinal, bool) or not isinstance(self.store_ordinal, int) or self.store_ordinal < 1:
            raise RegistryRetainedObservationSinkError("store_ordinal must be positive")
        if (
            isinstance(self.store_before_count, bool)
            or not isinstance(self.store_before_count, int)
            or self.store_before_count < 0
            or isinstance(self.store_after_count, bool)
            or not isinstance(self.store_after_count, int)
            or self.store_after_count != self.store_before_count + 1
        ):
            raise RegistryRetainedObservationSinkError(
                "store receipt must prove exactly one append"
            )
        if self.store_before_chain_digest == self.store_after_chain_digest:
            raise RegistryRetainedObservationSinkError(
                "retention append must advance the immutable chain"
            )
        if self.readback_verified is not True:
            raise RegistryRetainedObservationSinkError(
                "retention receipt requires verified durable readback"
            )
        if self.schema_version != RECEIPT_SCHEMA_VERSION or self.sink_version != SINK_VERSION:
            raise RegistryRetainedObservationSinkError("unsupported retention receipt schema")
        if self.authority != SHADOW_AUTHORITY or self.retained_real_observation_delta != 1:
            raise RegistryRetainedObservationSinkError(
                "retention receipt must remain one shadow-only append"
            )
        if any(
            (
                self.observation_window_started,
                self.registry_owner_mutated,
                self.m3_b_complete,
                self.m3_c_open,
                self.m3_e_authority_open,
                self.cutover_authorized,
            )
        ):
            raise RegistryRetainedObservationSinkError(
                "retention receipt cannot grant window, mutation, completion, or authority"
            )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "authority": self.authority,
            "axis": self.axis,
            "capture_digest": self.capture_digest,
            "capture_id": self.capture_id,
            "cutover_authorized": self.cutover_authorized,
            "event_envelope_digest": self.event_envelope_digest,
            "event_id": self.event_id,
            "m3_b_complete": self.m3_b_complete,
            "m3_c_open": self.m3_c_open,
            "m3_e_authority_open": self.m3_e_authority_open,
            "observation_window_started": self.observation_window_started,
            "readback_verified": self.readback_verified,
            "registry_owner_mutated": self.registry_owner_mutated,
            "retained_real_observation_delta": self.retained_real_observation_delta,
            "schema_version": self.schema_version,
            "sequence": self.sequence,
            "sink_version": self.sink_version,
            "store_after_chain_digest": self.store_after_chain_digest,
            "store_after_count": self.store_after_count,
            "store_before_chain_digest": self.store_before_chain_digest,
            "store_before_count": self.store_before_count,
            "store_ordinal": self.store_ordinal,
            "store_transition_hash": self.store_transition_hash,
            "verification_digest": self.verification_digest,
        }

    @property
    def receipt_digest(self) -> str:
        return _digest(self.to_mapping(), "retained_real_observation_receipt")


@dataclass(frozen=True, slots=True)
class RetentionSinkCapabilityStatus:
    schema_version: str = CAPABILITY_SCHEMA_VERSION
    authority: str = SHADOW_AUTHORITY
    immutable_retention_sink_present: bool = True
    durable_store_type: str = "SQLiteShadowStore"
    append_only_chain_required: bool = True
    readback_verification_required: bool = True
    auto_initialize: bool = False
    auto_append: bool = False
    retained_real_observation_count: int = 0
    positive_confidence_real_observation_count: int = 0
    observation_window_started: bool = False
    m3_b_complete: bool = False
    m3_c_open: bool = False
    m3_e_authority_open: bool = False
    cutover_authorized: bool = False

    def __post_init__(self) -> None:
        if self.schema_version != CAPABILITY_SCHEMA_VERSION or self.authority != SHADOW_AUTHORITY:
            raise RegistryRetainedObservationSinkError("unsupported retention capability status")
        if (
            self.immutable_retention_sink_present is not True
            or self.durable_store_type != "SQLiteShadowStore"
            or self.append_only_chain_required is not True
            or self.readback_verification_required is not True
        ):
            raise RegistryRetainedObservationSinkError("retention sink capability contract is incomplete")
        if self.auto_initialize or self.auto_append:
            raise RegistryRetainedObservationSinkError("retention sink cannot perform implicit I/O")
        if self.retained_real_observation_count != 0 or self.positive_confidence_real_observation_count != 0:
            raise RegistryRetainedObservationSinkError(
                "sink code presence cannot fabricate retained observations"
            )
        if any(
            (
                self.observation_window_started,
                self.m3_b_complete,
                self.m3_c_open,
                self.m3_e_authority_open,
                self.cutover_authorized,
            )
        ):
            raise RegistryRetainedObservationSinkError(
                "sink capability cannot grant window, completion, cutover, or authority"
            )


class RetainedRealObservationSink:
    """Explicit durable append boundary over an initialized M2-A shadow store."""

    def __init__(self, store: SQLiteShadowStore) -> None:
        if type(store) is not SQLiteShadowStore:
            raise RegistryRetainedObservationSinkError(
                "retention sink requires exact SQLiteShadowStore"
            )
        self._store = store

    @property
    def database_path(self) -> str:
        return str(self._store.database_path)

    def append(
        self,
        capture: ProductionCaptureRecord,
        *,
        event_id: str,
        sequence: int,
        correlation_id: str,
        causation_id: str | None = None,
    ) -> RetainedRealObservationReceipt:
        event = build_retained_real_observation_event(
            capture,
            event_id=event_id,
            sequence=sequence,
            correlation_id=correlation_id,
            causation_id=causation_id,
        )
        receipt: StoreAppendReceipt = self._store.append(event)
        if (
            receipt.event_id != event.event_id
            or receipt.stream_id != event.stream_id
            or receipt.sequence != event.sequence
            or receipt.envelope_digest != event.digest
            or receipt.readback_verified is not True
            or receipt.state_changed is not True
        ):
            raise RegistryRetainedObservationSinkError(
                "underlying append receipt does not prove exact retained event"
            )
        return RetainedRealObservationReceipt(
            axis=capture.axis,
            capture_id=capture.capture_id,
            capture_digest=capture.capture_digest,
            verification_digest=capture.verification.verification_digest,
            event_id=event.event_id,
            sequence=event.sequence,
            event_envelope_digest=event.digest,
            store_ordinal=receipt.ordinal,
            store_before_count=receipt.before_count,
            store_after_count=receipt.after_count,
            store_before_chain_digest=receipt.before_chain_digest,
            store_after_chain_digest=receipt.after_chain_digest,
            store_transition_hash=receipt.transition_hash,
            readback_verified=receipt.readback_verified,
        )


def retention_sink_capability_status() -> RetentionSinkCapabilityStatus:
    """Report code presence only; no database or retained observation is implied."""
    return RetentionSinkCapabilityStatus()
