"""Explicit durable retention activation for the reviewed C2 phone witness.

CI may exercise this module against a disposable SQLite database, but only an
operator execution against the post-merge phone checkout and operator-private
companion can produce the real retained-observation receipt later pinned by the
repository. Importing this module performs no I/O.
"""
from __future__ import annotations

import hashlib
from dataclasses import InitVar, dataclass
from typing import Any, Mapping

from core.event_kernel import EventEnvelope, SHADOW_AUTHORITY, canonical_json_object
from core.m3_b_c2_reviewed_phone_integration import (
    AXIS,
    ATTESTATION_DIGEST,
    EVIDENCE_DIGEST,
    PUBLIC_REVIEW_DIGEST,
    SOURCE_CONTRACT_ID,
    C2ReviewedProductionCapture,
    build_reviewed_capture,
)
from core.m3_b_registry_retained_real_observation_sink import (
    RETENTION_EVENT_TYPE,
    RETENTION_PRODUCER,
    RETENTION_STREAM_ID,
)
from core.sqlite_shadow_store import AppendReceipt as StoreAppendReceipt
from core.sqlite_shadow_store import SQLiteShadowStore

RETENTION_SCHEMA_VERSION = "eve.m3-b.c2-reviewed-retention-activation.v1"
RECEIPT_SCHEMA_VERSION = "eve.m3-b.c2-reviewed-retention-receipt.v1"
RETENTION_PRODUCER_VERSION = "eve.m3-b.c2-reviewed-retention-activation.v1"
EVENT_ID = "m3b:c2:retained:prediction_error_pressure:000001"
CORRELATION_ID = "m3b:c2:reviewed-phone-witness:b4968be9aeb6"
_RECEIPT_TOKEN = object()


class C2RetentionActivationError(ValueError):
    """Raised when the reviewed C2 observation cannot be retained exactly once."""


def _canonical(value: Mapping[str, Any], field: str) -> str:
    try:
        return canonical_json_object(value, field=field)
    except (TypeError, ValueError) as exc:
        raise C2RetentionActivationError(f"{field} is not canonical JSON") from exc


def _digest(value: Mapping[str, Any], field: str) -> str:
    return hashlib.sha256(_canonical(value, field).encode("utf-8")).hexdigest()


def _payload(capture: C2ReviewedProductionCapture) -> dict[str, Any]:
    return {
        "attestation_digest": ATTESTATION_DIGEST,
        "axis": AXIS,
        "capture_digest": capture.capture_digest,
        "capture_record": capture.to_mapping(),
        "classification": "retained_real_observation",
        "evidence_digest": EVIDENCE_DIGEST,
        "hash_algorithm": "sha256",
        "immutable": True,
        "public_review_digest": PUBLIC_REVIEW_DIGEST,
        "retention_schema_version": RETENTION_SCHEMA_VERSION,
        "runtime_provenance_verification_digest": (
            capture.runtime_verification.verification_digest
        ),
        "source_contract_id": SOURCE_CONTRACT_ID,
        "source_verification_digest": capture.source_verification.verification_digest,
    }


def build_retention_event(capture: C2ReviewedProductionCapture) -> EventEnvelope:
    if type(capture) is not C2ReviewedProductionCapture:
        raise C2RetentionActivationError("retention requires exact C2 reviewed capture")
    if (
        not capture.retained_real_observation_eligible
        or not capture.runtime_verification.counts_as_production
        or not capture.source_verification.counts_as_real
    ):
        raise C2RetentionActivationError("reviewed capture is not retention eligible")
    return EventEnvelope.create(
        event_id=EVENT_ID,
        event_type=RETENTION_EVENT_TYPE,
        stream_id=RETENTION_STREAM_ID,
        sequence=1,
        producer=RETENTION_PRODUCER,
        producer_version=RETENTION_PRODUCER_VERSION,
        correlation_id=CORRELATION_ID,
        causation_id=None,
        payload=_payload(capture),
        causal_context={
            "attestation_digest": ATTESTATION_DIGEST,
            "capture_id": capture.capture_id,
            "capture_tick": capture.capture_tick,
            "public_review_digest": PUBLIC_REVIEW_DIGEST,
            "source_contract_id": SOURCE_CONTRACT_ID,
            "source_snapshot_id": capture.evidence.source_snapshot_id,
            "source_verifier_id": capture.source_verification.verifier_id,
        },
        authority=SHADOW_AUTHORITY,
    )


@dataclass(frozen=True, slots=True)
class C2RetentionReceipt:
    axis: str
    public_review_digest: str
    attestation_digest: str
    evidence_digest: str
    capture_digest: str
    runtime_provenance_verification_digest: str
    source_verification_digest: str
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
    retained_real_observation_delta: int = 1
    observation_window_started: bool = False
    m3_b_complete: bool = False
    m3_c_open: bool = False
    m3_e_authority_open: bool = False
    cutover_authorized: bool = False
    authority: str = SHADOW_AUTHORITY
    schema_version: str = RECEIPT_SCHEMA_VERSION
    _issuance_token: InitVar[object | None] = None

    def __post_init__(self, _issuance_token: object | None) -> None:
        if _issuance_token is not _RECEIPT_TOKEN:
            raise C2RetentionActivationError("retention receipt must be issued by durable append")
        if (
            self.axis != AXIS
            or self.public_review_digest != PUBLIC_REVIEW_DIGEST
            or self.attestation_digest != ATTESTATION_DIGEST
            or self.evidence_digest != EVIDENCE_DIGEST
            or self.event_id != EVENT_ID
            or self.sequence != 1
            or self.store_ordinal < 1
            or self.store_before_count != 0
            or self.store_after_count != 1
            or self.store_before_chain_digest == self.store_after_chain_digest
            or self.readback_verified is not True
            or self.retained_real_observation_delta != 1
            or self.authority != SHADOW_AUTHORITY
            or self.schema_version != RECEIPT_SCHEMA_VERSION
        ):
            raise C2RetentionActivationError("retention receipt does not prove exact one-event append")
        if any((
            self.observation_window_started,
            self.m3_b_complete,
            self.m3_c_open,
            self.m3_e_authority_open,
            self.cutover_authorized,
        )):
            raise C2RetentionActivationError("retention receipt cannot open later authority")

    def to_mapping(self) -> dict[str, Any]:
        return {
            "attestation_digest": self.attestation_digest,
            "authority": self.authority,
            "axis": self.axis,
            "capture_digest": self.capture_digest,
            "cutover_authorized": self.cutover_authorized,
            "event_envelope_digest": self.event_envelope_digest,
            "event_id": self.event_id,
            "evidence_digest": self.evidence_digest,
            "m3_b_complete": self.m3_b_complete,
            "m3_c_open": self.m3_c_open,
            "m3_e_authority_open": self.m3_e_authority_open,
            "observation_window_started": self.observation_window_started,
            "public_review_digest": self.public_review_digest,
            "readback_verified": self.readback_verified,
            "retained_real_observation_delta": self.retained_real_observation_delta,
            "runtime_provenance_verification_digest": (
                self.runtime_provenance_verification_digest
            ),
            "schema_version": self.schema_version,
            "sequence": self.sequence,
            "source_verification_digest": self.source_verification_digest,
            "store_after_chain_digest": self.store_after_chain_digest,
            "store_after_count": self.store_after_count,
            "store_before_chain_digest": self.store_before_chain_digest,
            "store_before_count": self.store_before_count,
            "store_ordinal": self.store_ordinal,
            "store_transition_hash": self.store_transition_hash,
        }

    @property
    def receipt_digest(self) -> str:
        return _digest(self.to_mapping(), "c2_retention_receipt")


def append_reviewed_observation(
    store: SQLiteShadowStore,
    public_review: Mapping[str, Any],
) -> C2RetentionReceipt:
    """Append exactly one reviewed real observation to an initialized private store."""

    if type(store) is not SQLiteShadowStore:
        raise C2RetentionActivationError("retention requires exact SQLiteShadowStore")
    existing = store.events(stream_id=RETENTION_STREAM_ID)
    if existing:
        raise C2RetentionActivationError(
            "reviewed C2 retention stream is already non-empty; duplicate append refused"
        )
    capture = build_reviewed_capture(public_review)
    event = build_retention_event(capture)
    receipt: StoreAppendReceipt = store.append(event)
    if (
        receipt.event_id != event.event_id
        or receipt.stream_id != event.stream_id
        or receipt.sequence != event.sequence
        or receipt.envelope_digest != event.digest
        or receipt.before_count != 0
        or receipt.after_count != 1
        or receipt.readback_verified is not True
        or receipt.state_changed is not True
    ):
        raise C2RetentionActivationError(
            "underlying SQLite append receipt does not prove exact reviewed retention"
        )
    readback = store.events(stream_id=RETENTION_STREAM_ID)
    if len(readback) != 1 or readback[0] != event:
        raise C2RetentionActivationError("retained event readback does not match exact envelope")
    return C2RetentionReceipt(
        axis=AXIS,
        public_review_digest=PUBLIC_REVIEW_DIGEST,
        attestation_digest=ATTESTATION_DIGEST,
        evidence_digest=EVIDENCE_DIGEST,
        capture_digest=capture.capture_digest,
        runtime_provenance_verification_digest=(
            capture.runtime_verification.verification_digest
        ),
        source_verification_digest=capture.source_verification.verification_digest,
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
        _issuance_token=_RECEIPT_TOKEN,
    )
