"""Sequence-five durable retention staging for reviewed ``stress_load``.

The append is permitted only when the operator-private retention stream already
contains the exact immutable sequence-1 through sequence-4 retained events pinned
by repository receipts. Importing this module performs no I/O and does not append.
"""
from __future__ import annotations

import hashlib
from dataclasses import InitVar, dataclass
from typing import Any, Mapping

from core.event_kernel import EventEnvelope, SHADOW_AUTHORITY, canonical_json_object
from core.m3_b_c2_reviewed_stress_load_integration import (
    AXIS,
    ATTESTATION_DIGEST,
    EVIDENCE_DIGEST,
    PUBLIC_REVIEW_DIGEST,
    SOURCE_CONTRACT_ID,
    C2ReviewedStressLoadCapture,
    build_reviewed_capture,
)
from core.m3_b_registry_retained_real_observation_sink import (
    RETENTION_EVENT_TYPE,
    RETENTION_PRODUCER,
    RETENTION_STREAM_ID,
)
from core.sqlite_shadow_store import AppendReceipt as StoreAppendReceipt
from core.sqlite_shadow_store import SQLiteShadowStore

RETENTION_SCHEMA_VERSION = "eve.m3-b.c2-stress-load-retention-activation.v1"
RECEIPT_SCHEMA_VERSION = "eve.m3-b.c2-stress-load-retention-receipt.v1"
RETENTION_PRODUCER_VERSION = RETENTION_SCHEMA_VERSION
EVENT_ID = "m3b:c2:retained:stress_load:000005"
SEQUENCE = 5
CORRELATION_ID = "m3b:c2:reviewed-stress-load-witness:3298d3b9911c"

FIRST_EVENT_ID = "m3b:c2:retained:prediction_error_pressure:000001"
FIRST_EVENT_ENVELOPE_DIGEST = "07deb0e7345db33ac7655229044c8d62e7b14198bd7d80611ace6f5352adb493"
FIRST_PUBLIC_REVIEW_DIGEST = "6a3d34120d9773f28544aa82d963cf2e65220f6f899aeab42c132660f87ad81e"
SECOND_EVENT_ID = "m3b:c2:retained:energy_budget:000002"
SECOND_EVENT_ENVELOPE_DIGEST = "1e4bd659ef348ac39588ba2bc13440bd96a81a9c24a4cdf804bf9ef48b23f664"
SECOND_PUBLIC_REVIEW_DIGEST = "a2ce3d84111224e2009bf22d1e03a8f92acab0506e42515aac185ae05ff54ab4"
THIRD_EVENT_ID = "m3b:c2:retained:fatigue_pressure:000003"
THIRD_EVENT_ENVELOPE_DIGEST = "f81d43bf40b4dc76130767f91b65ad2503bc70e61ef718fe3d0e446528d1a7e3"
THIRD_PUBLIC_REVIEW_DIGEST = "4b88c7734234ac2982836b95bf392fe143bc928119d4af515e576b39e480af61"
PRIOR_EVENT_ID = "m3b:c2:retained:recovery_need:000004"
PRIOR_EVENT_ENVELOPE_DIGEST = "7619663391db95dc59951a3d12bba58af1bd1e01bb3cabbb89e862b55f3f9691"
PRIOR_STORE_CHAIN_DIGEST = "16efec6a9f775175fc99c252411d2e0ca6b3504799c824e8e5a70cf2697f1e0f"
PRIOR_AXIS = "recovery_need"
PRIOR_PUBLIC_REVIEW_DIGEST = "e46df034d01b13e768ce37d14261b8ed20fdec30101945bea492d97e482e4c33"
_RECEIPT_TOKEN = object()


class C2StressLoadRetentionActivationError(ValueError):
    pass


def _canonical(value: Mapping[str, Any], field: str) -> str:
    try:
        return canonical_json_object(value, field=field)
    except (TypeError, ValueError) as exc:
        raise C2StressLoadRetentionActivationError(
            f"{field} is not canonical JSON"
        ) from exc


def _digest(value: Mapping[str, Any], field: str) -> str:
    return hashlib.sha256(_canonical(value, field).encode("utf-8")).hexdigest()


def _payload(capture: C2ReviewedStressLoadCapture) -> dict[str, Any]:
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


def build_retention_event(capture: C2ReviewedStressLoadCapture) -> EventEnvelope:
    if type(capture) is not C2ReviewedStressLoadCapture:
        raise C2StressLoadRetentionActivationError(
            "retention requires exact reviewed stress-load capture"
        )
    if (
        not capture.retained_real_observation_eligible
        or not capture.runtime_verification.counts_as_production
        or not capture.source_verification.counts_as_real
    ):
        raise C2StressLoadRetentionActivationError(
            "reviewed stress-load capture is not retention eligible"
        )
    return EventEnvelope.create(
        event_id=EVENT_ID,
        event_type=RETENTION_EVENT_TYPE,
        stream_id=RETENTION_STREAM_ID,
        sequence=SEQUENCE,
        producer=RETENTION_PRODUCER,
        producer_version=RETENTION_PRODUCER_VERSION,
        correlation_id=CORRELATION_ID,
        causation_id=PRIOR_EVENT_ID,
        payload=_payload(capture),
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


def _require_exact_prior_retention(
    store: SQLiteShadowStore,
) -> tuple[EventEnvelope, EventEnvelope, EventEnvelope, EventEnvelope]:
    existing = store.events(stream_id=RETENTION_STREAM_ID)
    if len(existing) != 4:
        raise C2StressLoadRetentionActivationError(
            "stress-load retention requires exactly four prior retained events"
        )
    first, second, third, prior = existing
    if (
        first.event_id != FIRST_EVENT_ID
        or first.sequence != 1
        or first.digest != FIRST_EVENT_ENVELOPE_DIGEST
        or first.payload.get("axis") != "prediction_error_pressure"
        or first.payload.get("public_review_digest") != FIRST_PUBLIC_REVIEW_DIGEST
        or second.event_id != SECOND_EVENT_ID
        or second.sequence != 2
        or second.digest != SECOND_EVENT_ENVELOPE_DIGEST
        or second.payload.get("axis") != "energy_budget"
        or second.payload.get("public_review_digest") != SECOND_PUBLIC_REVIEW_DIGEST
        or third.event_id != THIRD_EVENT_ID
        or third.sequence != 3
        or third.digest != THIRD_EVENT_ENVELOPE_DIGEST
        or third.payload.get("axis") != "fatigue_pressure"
        or third.payload.get("public_review_digest") != THIRD_PUBLIC_REVIEW_DIGEST
        or prior.event_id != PRIOR_EVENT_ID
        or prior.sequence != 4
        or prior.digest != PRIOR_EVENT_ENVELOPE_DIGEST
        or prior.payload.get("axis") != PRIOR_AXIS
        or prior.payload.get("public_review_digest") != PRIOR_PUBLIC_REVIEW_DIGEST
        or any(
            event.event_type != RETENTION_EVENT_TYPE
            or event.stream_id != RETENTION_STREAM_ID
            or event.producer != RETENTION_PRODUCER
            or event.authority != SHADOW_AUTHORITY
            for event in existing
        )
        or any(
            event.payload.get("classification") != "retained_real_observation"
            for event in existing
        )
    ):
        raise C2StressLoadRetentionActivationError(
            "existing retention history does not match pinned sequence-one/two/three/four receipts"
        )
    return first, second, third, prior


@dataclass(frozen=True, slots=True)
class C2StressLoadRetentionReceipt:
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
    retained_real_observation_count_after_append: int = 5
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
            raise C2StressLoadRetentionActivationError(
                "stress-load receipt must be issued by durable append"
            )
        if (
            self.axis != AXIS
            or self.public_review_digest != PUBLIC_REVIEW_DIGEST
            or self.attestation_digest != ATTESTATION_DIGEST
            or self.evidence_digest != EVIDENCE_DIGEST
            or self.event_id != EVENT_ID
            or self.sequence != SEQUENCE
            or self.prior_event_id != PRIOR_EVENT_ID
            or self.prior_event_envelope_digest != PRIOR_EVENT_ENVELOPE_DIGEST
            or self.store_ordinal != 5
            or self.store_before_count != 4
            or self.store_after_count != 5
            or self.store_before_chain_digest != PRIOR_STORE_CHAIN_DIGEST
            or self.store_before_chain_digest == self.store_after_chain_digest
            or self.readback_verified is not True
            or self.retained_real_observation_delta != 1
            or self.retained_real_observation_count_after_append != 5
            or self.authority != SHADOW_AUTHORITY
            or self.schema_version != RECEIPT_SCHEMA_VERSION
        ):
            raise C2StressLoadRetentionActivationError(
                "stress-load receipt does not prove exact sequence-five append"
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
            raise C2StressLoadRetentionActivationError(
                "stress-load retention receipt cannot open later authority"
            )

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
            "prior_event_envelope_digest": self.prior_event_envelope_digest,
            "prior_event_id": self.prior_event_id,
            "public_review_digest": self.public_review_digest,
            "readback_verified": self.readback_verified,
            "retained_real_observation_count_after_append": self.retained_real_observation_count_after_append,
            "retained_real_observation_delta": self.retained_real_observation_delta,
            "runtime_provenance_verification_digest": self.runtime_provenance_verification_digest,
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
        return _digest(self.to_mapping(), "c2_stress_load_retention_receipt")


def append_reviewed_stress_load_observation(
    store: SQLiteShadowStore,
    public_review: Mapping[str, Any],
) -> C2StressLoadRetentionReceipt:
    if type(store) is not SQLiteShadowStore:
        raise C2StressLoadRetentionActivationError(
            "stress-load retention requires exact SQLiteShadowStore"
        )
    _require_exact_prior_retention(store)
    capture = build_reviewed_capture(public_review)
    event = build_retention_event(capture)
    receipt: StoreAppendReceipt = store.append(event)
    if (
        receipt.event_id != event.event_id
        or receipt.stream_id != event.stream_id
        or receipt.sequence != event.sequence
        or receipt.envelope_digest != event.digest
        or receipt.ordinal != 5
        or receipt.before_count != 4
        or receipt.after_count != 5
        or receipt.before_chain_digest != PRIOR_STORE_CHAIN_DIGEST
        or receipt.readback_verified is not True
        or receipt.state_changed is not True
    ):
        raise C2StressLoadRetentionActivationError(
            "underlying SQLite receipt does not prove exact stress-load append"
        )
    readback = store.events(stream_id=RETENTION_STREAM_ID)
    if (
        len(readback) != 5
        or readback[0].digest != FIRST_EVENT_ENVELOPE_DIGEST
        or readback[1].digest != SECOND_EVENT_ENVELOPE_DIGEST
        or readback[2].digest != THIRD_EVENT_ENVELOPE_DIGEST
        or readback[3].digest != PRIOR_EVENT_ENVELOPE_DIGEST
        or readback[4] != event
    ):
        raise C2StressLoadRetentionActivationError(
            "sequence-five retained readback does not match exact event history"
        )
    return C2StressLoadRetentionReceipt(
        axis=AXIS,
        public_review_digest=PUBLIC_REVIEW_DIGEST,
        attestation_digest=ATTESTATION_DIGEST,
        evidence_digest=EVIDENCE_DIGEST,
        capture_digest=capture.capture_digest,
        runtime_provenance_verification_digest=capture.runtime_verification.verification_digest,
        source_verification_digest=capture.source_verification.verification_digest,
        event_id=event.event_id,
        sequence=event.sequence,
        event_envelope_digest=event.digest,
        prior_event_id=PRIOR_EVENT_ID,
        prior_event_envelope_digest=PRIOR_EVENT_ENVELOPE_DIGEST,
        store_ordinal=receipt.ordinal,
        store_before_count=receipt.before_count,
        store_after_count=receipt.after_count,
        store_before_chain_digest=receipt.before_chain_digest,
        store_after_chain_digest=receipt.after_chain_digest,
        store_transition_hash=receipt.transition_hash,
        readback_verified=receipt.readback_verified,
        _issuance_token=_RECEIPT_TOKEN,
    )
