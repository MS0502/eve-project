"""Sequence-three durable retention activation for reviewed ``fatigue_pressure``.

The append is permitted only when the operator-private retention stream already
contains the exact immutable sequence-1 prediction-error event and sequence-2
energy-budget event pinned by repository receipts. Importing this module does no
I/O.
"""
from __future__ import annotations

import hashlib
from dataclasses import InitVar, dataclass
from typing import Any, Mapping

from core.event_kernel import EventEnvelope, SHADOW_AUTHORITY, canonical_json_object
from core.m3_b_c2_reviewed_fatigue_pressure_integration import (
    AXIS, ATTESTATION_DIGEST, EVIDENCE_DIGEST, PUBLIC_REVIEW_DIGEST,
    SOURCE_CONTRACT_ID, C2ReviewedFatiguePressureCapture, build_reviewed_capture,
)
from core.m3_b_registry_retained_real_observation_sink import (
    RETENTION_EVENT_TYPE, RETENTION_PRODUCER, RETENTION_STREAM_ID,
)
from core.sqlite_shadow_store import AppendReceipt as StoreAppendReceipt
from core.sqlite_shadow_store import SQLiteShadowStore

RETENTION_SCHEMA_VERSION = "eve.m3-b.c2-fatigue-pressure-retention-activation.v1"
RECEIPT_SCHEMA_VERSION = "eve.m3-b.c2-fatigue-pressure-retention-receipt.v1"
RETENTION_PRODUCER_VERSION = RETENTION_SCHEMA_VERSION
EVENT_ID = "m3b:c2:retained:fatigue_pressure:000003"
SEQUENCE = 3
CORRELATION_ID = "m3b:c2:reviewed-fatigue-pressure-witness:1ac94c402d6f"
PRIOR_EVENT_ID = "m3b:c2:retained:energy_budget:000002"
PRIOR_EVENT_ENVELOPE_DIGEST = "1e4bd659ef348ac39588ba2bc13440bd96a81a9c24a4cdf804bf9ef48b23f664"
PRIOR_STORE_CHAIN_DIGEST = "d4660b5cef058bad1b9d1b6b1cb2987c78ef9dbbee403c85562ab945535883e0"
PRIOR_AXIS = "energy_budget"
PRIOR_PUBLIC_REVIEW_DIGEST = "a2ce3d84111224e2009bf22d1e03a8f92acab0506e42515aac185ae05ff54ab4"
FIRST_EVENT_ID = "m3b:c2:retained:prediction_error_pressure:000001"
FIRST_EVENT_ENVELOPE_DIGEST = "07deb0e7345db33ac7655229044c8d62e7b14198bd7d80611ace6f5352adb493"
FIRST_PUBLIC_REVIEW_DIGEST = "6a3d34120d9773f28544aa82d963cf2e65220f6f899aeab42c132660f87ad81e"
_RECEIPT_TOKEN = object()


class C2FatiguePressureRetentionActivationError(ValueError):
    pass


def _canonical(value: Mapping[str, Any], field: str) -> str:
    try:
        return canonical_json_object(value, field=field)
    except (TypeError, ValueError) as exc:
        raise C2FatiguePressureRetentionActivationError(f"{field} is not canonical JSON") from exc


def _digest(value: Mapping[str, Any], field: str) -> str:
    return hashlib.sha256(_canonical(value, field).encode("utf-8")).hexdigest()


def _payload(capture: C2ReviewedFatiguePressureCapture) -> dict[str, Any]:
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
        "runtime_provenance_verification_digest": capture.runtime_verification.verification_digest,
        "source_contract_id": SOURCE_CONTRACT_ID,
        "source_verification_digest": capture.source_verification.verification_digest,
    }


def build_retention_event(capture: C2ReviewedFatiguePressureCapture) -> EventEnvelope:
    if type(capture) is not C2ReviewedFatiguePressureCapture:
        raise C2FatiguePressureRetentionActivationError("retention requires exact reviewed fatigue-pressure capture")
    if not capture.retained_real_observation_eligible or not capture.runtime_verification.counts_as_production or not capture.source_verification.counts_as_real:
        raise C2FatiguePressureRetentionActivationError("reviewed fatigue-pressure capture is not retention eligible")
    return EventEnvelope.create(
        event_id=EVENT_ID, event_type=RETENTION_EVENT_TYPE, stream_id=RETENTION_STREAM_ID,
        sequence=SEQUENCE, producer=RETENTION_PRODUCER, producer_version=RETENTION_PRODUCER_VERSION,
        correlation_id=CORRELATION_ID, causation_id=PRIOR_EVENT_ID, payload=_payload(capture),
        causal_context={
            "attestation_digest": ATTESTATION_DIGEST,
            "capture_id": capture.capture_id,
            "capture_tick": capture.capture_tick,
            "prior_event_envelope_digest": PRIOR_EVENT_ENVELOPE_DIGEST,
            "public_review_digest": PUBLIC_REVIEW_DIGEST,
            "source_contract_id": SOURCE_CONTRACT_ID,
            "source_snapshot_id": capture.evidence.source_snapshot_id,
            "source_verifier_id": capture.source_verification.to_mapping()["verifier_id"],
        },
        authority=SHADOW_AUTHORITY,
    )


def _require_exact_prior_retention(store: SQLiteShadowStore) -> tuple[EventEnvelope, EventEnvelope]:
    existing = store.events(stream_id=RETENTION_STREAM_ID)
    if len(existing) != 2:
        raise C2FatiguePressureRetentionActivationError("fatigue-pressure retention requires exactly two prior retained events")
    first, prior = existing
    if (
        first.event_id != FIRST_EVENT_ID or first.sequence != 1 or first.digest != FIRST_EVENT_ENVELOPE_DIGEST
        or first.payload.get("axis") != "prediction_error_pressure"
        or first.payload.get("public_review_digest") != FIRST_PUBLIC_REVIEW_DIGEST
        or prior.event_id != PRIOR_EVENT_ID or prior.sequence != 2 or prior.digest != PRIOR_EVENT_ENVELOPE_DIGEST
        or prior.payload.get("axis") != PRIOR_AXIS or prior.payload.get("public_review_digest") != PRIOR_PUBLIC_REVIEW_DIGEST
        or any(event.event_type != RETENTION_EVENT_TYPE or event.stream_id != RETENTION_STREAM_ID or event.producer != RETENTION_PRODUCER or event.authority != SHADOW_AUTHORITY for event in existing)
        or any(event.payload.get("classification") != "retained_real_observation" for event in existing)
    ):
        raise C2FatiguePressureRetentionActivationError("existing retention history does not match pinned sequence-one/two receipts")
    return first, prior


@dataclass(frozen=True, slots=True)
class C2FatiguePressureRetentionReceipt:
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
    prior_event_id: str
    prior_event_envelope_digest: str
    store_ordinal: int
    store_before_count: int
    store_after_count: int
    store_before_chain_digest: str
    store_after_chain_digest: str
    store_transition_hash: str
    readback_verified: bool
    retained_real_observation_delta: int = 1
    retained_real_observation_count_after_append: int = 3
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
            raise C2FatiguePressureRetentionActivationError("fatigue-pressure receipt must be issued by durable append")
        if (
            self.axis != AXIS or self.public_review_digest != PUBLIC_REVIEW_DIGEST
            or self.attestation_digest != ATTESTATION_DIGEST or self.evidence_digest != EVIDENCE_DIGEST
            or self.event_id != EVENT_ID or self.sequence != SEQUENCE or self.prior_event_id != PRIOR_EVENT_ID
            or self.prior_event_envelope_digest != PRIOR_EVENT_ENVELOPE_DIGEST or self.store_ordinal != 3
            or self.store_before_count != 2 or self.store_after_count != 3
            or self.store_before_chain_digest != PRIOR_STORE_CHAIN_DIGEST
            or self.store_before_chain_digest == self.store_after_chain_digest or self.readback_verified is not True
            or self.retained_real_observation_delta != 1 or self.retained_real_observation_count_after_append != 3
            or self.authority != SHADOW_AUTHORITY or self.schema_version != RECEIPT_SCHEMA_VERSION
        ):
            raise C2FatiguePressureRetentionActivationError("fatigue-pressure receipt does not prove exact sequence-three append")
        if any((self.observation_window_started, self.m3_b_complete, self.m3_c_open, self.m3_e_authority_open, self.cutover_authorized)):
            raise C2FatiguePressureRetentionActivationError("fatigue-pressure retention receipt cannot open later authority")

    def to_mapping(self) -> dict[str, Any]:
        return {
            "attestation_digest": self.attestation_digest, "authority": self.authority, "axis": self.axis,
            "capture_digest": self.capture_digest, "cutover_authorized": self.cutover_authorized,
            "event_envelope_digest": self.event_envelope_digest, "event_id": self.event_id,
            "evidence_digest": self.evidence_digest, "m3_b_complete": self.m3_b_complete,
            "m3_c_open": self.m3_c_open, "m3_e_authority_open": self.m3_e_authority_open,
            "observation_window_started": self.observation_window_started,
            "prior_event_envelope_digest": self.prior_event_envelope_digest, "prior_event_id": self.prior_event_id,
            "public_review_digest": self.public_review_digest, "readback_verified": self.readback_verified,
            "retained_real_observation_count_after_append": self.retained_real_observation_count_after_append,
            "retained_real_observation_delta": self.retained_real_observation_delta,
            "runtime_provenance_verification_digest": self.runtime_provenance_verification_digest,
            "schema_version": self.schema_version, "sequence": self.sequence,
            "source_verification_digest": self.source_verification_digest,
            "store_after_chain_digest": self.store_after_chain_digest, "store_after_count": self.store_after_count,
            "store_before_chain_digest": self.store_before_chain_digest, "store_before_count": self.store_before_count,
            "store_ordinal": self.store_ordinal, "store_transition_hash": self.store_transition_hash,
        }

    @property
    def receipt_digest(self) -> str:
        return _digest(self.to_mapping(), "c2_fatigue_pressure_retention_receipt")


def append_reviewed_fatigue_pressure_observation(store: SQLiteShadowStore, public_review: Mapping[str, Any]) -> C2FatiguePressureRetentionReceipt:
    if type(store) is not SQLiteShadowStore:
        raise C2FatiguePressureRetentionActivationError("fatigue-pressure retention requires exact SQLiteShadowStore")
    _require_exact_prior_retention(store)
    capture = build_reviewed_capture(public_review)
    event = build_retention_event(capture)
    receipt: StoreAppendReceipt = store.append(event)
    if (
        receipt.event_id != event.event_id or receipt.stream_id != event.stream_id or receipt.sequence != event.sequence
        or receipt.envelope_digest != event.digest or receipt.ordinal != 3 or receipt.before_count != 2 or receipt.after_count != 3
        or receipt.before_chain_digest != PRIOR_STORE_CHAIN_DIGEST or receipt.readback_verified is not True or receipt.state_changed is not True
    ):
        raise C2FatiguePressureRetentionActivationError("underlying SQLite receipt does not prove exact fatigue-pressure append")
    readback = store.events(stream_id=RETENTION_STREAM_ID)
    if len(readback) != 3 or readback[0].digest != FIRST_EVENT_ENVELOPE_DIGEST or readback[1].digest != PRIOR_EVENT_ENVELOPE_DIGEST or readback[2] != event:
        raise C2FatiguePressureRetentionActivationError("sequence-three retained readback does not match exact event history")
    return C2FatiguePressureRetentionReceipt(
        axis=AXIS, public_review_digest=PUBLIC_REVIEW_DIGEST, attestation_digest=ATTESTATION_DIGEST,
        evidence_digest=EVIDENCE_DIGEST, capture_digest=capture.capture_digest,
        runtime_provenance_verification_digest=capture.runtime_verification.verification_digest,
        source_verification_digest=capture.source_verification.verification_digest,
        event_id=event.event_id, sequence=event.sequence, event_envelope_digest=event.digest,
        prior_event_id=PRIOR_EVENT_ID, prior_event_envelope_digest=PRIOR_EVENT_ENVELOPE_DIGEST,
        store_ordinal=receipt.ordinal, store_before_count=receipt.before_count, store_after_count=receipt.after_count,
        store_before_chain_digest=receipt.before_chain_digest, store_after_chain_digest=receipt.after_chain_digest,
        store_transition_hash=receipt.transition_hash, readback_verified=receipt.readback_verified,
        _issuance_token=_RECEIPT_TOKEN,
    )
