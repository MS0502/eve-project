"""Sequence-two durable retention activation for the reviewed energy-budget witness.

The activation may append only after the exact already-retained
``prediction_error_pressure`` event from the first C2 retention boundary.  It
therefore cannot replay sequence 1 or silently start a new retention history.
Importing this module performs no I/O.
"""
from __future__ import annotations

import hashlib
from dataclasses import InitVar, dataclass
from typing import Any, Mapping

from core.event_kernel import EventEnvelope, SHADOW_AUTHORITY, canonical_json_object
from core.m3_b_c2_reviewed_energy_budget_integration import (
    AXIS,
    ATTESTATION_DIGEST,
    EVIDENCE_DIGEST,
    PUBLIC_REVIEW_DIGEST,
    SOURCE_CONTRACT_ID,
    C2ReviewedEnergyBudgetCapture,
    build_reviewed_capture,
)
from core.m3_b_registry_retained_real_observation_sink import (
    RETENTION_EVENT_TYPE,
    RETENTION_PRODUCER,
    RETENTION_STREAM_ID,
)
from core.sqlite_shadow_store import AppendReceipt as StoreAppendReceipt
from core.sqlite_shadow_store import SQLiteShadowStore

RETENTION_SCHEMA_VERSION = "eve.m3-b.c2-energy-budget-retention-activation.v1"
RECEIPT_SCHEMA_VERSION = "eve.m3-b.c2-energy-budget-retention-receipt.v1"
RETENTION_PRODUCER_VERSION = "eve.m3-b.c2-energy-budget-retention-activation.v1"
EVENT_ID = "m3b:c2:retained:energy_budget:000002"
SEQUENCE = 2
CORRELATION_ID = "m3b:c2:reviewed-energy-budget-witness:1161bb15d7bb"
PRIOR_EVENT_ID = "m3b:c2:retained:prediction_error_pressure:000001"
PRIOR_EVENT_ENVELOPE_DIGEST = "07deb0e7345db33ac7655229044c8d62e7b14198bd7d80611ace6f5352adb493"
PRIOR_STORE_CHAIN_DIGEST = "d51406d84dc755f72bd2ab661563c75cf19244710bf98376dbe3174ff101c8ce"
PRIOR_AXIS = "prediction_error_pressure"
PRIOR_PUBLIC_REVIEW_DIGEST = "6a3d34120d9773f28544aa82d963cf2e65220f6f899aeab42c132660f87ad81e"
_RECEIPT_TOKEN = object()


class C2EnergyBudgetRetentionActivationError(ValueError):
    """Raised when the second reviewed observation cannot be retained exactly once."""


def _canonical(value: Mapping[str, Any], field: str) -> str:
    try:
        return canonical_json_object(value, field=field)
    except (TypeError, ValueError) as exc:
        raise C2EnergyBudgetRetentionActivationError(
            f"{field} is not canonical JSON"
        ) from exc


def _digest(value: Mapping[str, Any], field: str) -> str:
    return hashlib.sha256(_canonical(value, field).encode("utf-8")).hexdigest()


def _payload(capture: C2ReviewedEnergyBudgetCapture) -> dict[str, Any]:
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


def build_retention_event(capture: C2ReviewedEnergyBudgetCapture) -> EventEnvelope:
    if type(capture) is not C2ReviewedEnergyBudgetCapture:
        raise C2EnergyBudgetRetentionActivationError(
            "retention requires exact reviewed energy-budget capture"
        )
    if (
        not capture.retained_real_observation_eligible
        or not capture.runtime_verification.counts_as_production
        or not capture.source_verification.counts_as_real
    ):
        raise C2EnergyBudgetRetentionActivationError(
            "reviewed energy-budget capture is not retention eligible"
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
            "source_verifier_id": capture.source_verification.verifier_id,
        },
        authority=SHADOW_AUTHORITY,
    )


def _require_exact_prior_retention(store: SQLiteShadowStore) -> EventEnvelope:
    existing = store.events(stream_id=RETENTION_STREAM_ID)
    if len(existing) != 1:
        raise C2EnergyBudgetRetentionActivationError(
            "energy-budget retention requires exactly the prior sequence-1 event"
        )
    prior = existing[0]
    if (
        prior.event_id != PRIOR_EVENT_ID
        or prior.sequence != 1
        or prior.event_type != RETENTION_EVENT_TYPE
        or prior.stream_id != RETENTION_STREAM_ID
        or prior.producer != RETENTION_PRODUCER
        or prior.authority != SHADOW_AUTHORITY
        or prior.digest != PRIOR_EVENT_ENVELOPE_DIGEST
        or prior.payload.get("axis") != PRIOR_AXIS
        or prior.payload.get("public_review_digest") != PRIOR_PUBLIC_REVIEW_DIGEST
        or prior.payload.get("classification") != "retained_real_observation"
    ):
        raise C2EnergyBudgetRetentionActivationError(
            "existing retention event does not match the pinned first C2 receipt"
        )
    return prior


@dataclass(frozen=True, slots=True)
class C2EnergyBudgetRetentionReceipt:
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
    retained_real_observation_count_after_append: int = 2
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
            raise C2EnergyBudgetRetentionActivationError(
                "energy-budget retention receipt must be issued by durable append"
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
            or self.store_ordinal != 2
            or self.store_before_count != 1
            or self.store_after_count != 2
            or self.store_before_chain_digest != PRIOR_STORE_CHAIN_DIGEST
            or self.store_before_chain_digest == self.store_after_chain_digest
            or self.readback_verified is not True
            or self.retained_real_observation_delta != 1
            or self.retained_real_observation_count_after_append != 2
            or self.authority != SHADOW_AUTHORITY
            or self.schema_version != RECEIPT_SCHEMA_VERSION
        ):
            raise C2EnergyBudgetRetentionActivationError(
                "energy-budget receipt does not prove exact sequence-two append"
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
            raise C2EnergyBudgetRetentionActivationError(
                "energy-budget retention receipt cannot open later authority"
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
            "retained_real_observation_count_after_append": (
                self.retained_real_observation_count_after_append
            ),
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
        return _digest(self.to_mapping(), "c2_energy_budget_retention_receipt")


def append_reviewed_energy_budget_observation(
    store: SQLiteShadowStore,
    public_review: Mapping[str, Any],
) -> C2EnergyBudgetRetentionReceipt:
    """Append energy_budget as sequence 2 after the exact pinned sequence 1."""

    if type(store) is not SQLiteShadowStore:
        raise C2EnergyBudgetRetentionActivationError(
            "energy-budget retention requires exact SQLiteShadowStore"
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
        or receipt.ordinal != 2
        or receipt.before_count != 1
        or receipt.after_count != 2
        or receipt.before_chain_digest != PRIOR_STORE_CHAIN_DIGEST
        or receipt.readback_verified is not True
        or receipt.state_changed is not True
    ):
        raise C2EnergyBudgetRetentionActivationError(
            "underlying SQLite receipt does not prove exact energy-budget append"
        )
    readback = store.events(stream_id=RETENTION_STREAM_ID)
    if len(readback) != 2 or readback[0].digest != PRIOR_EVENT_ENVELOPE_DIGEST or readback[1] != event:
        raise C2EnergyBudgetRetentionActivationError(
            "sequence-two retained readback does not match exact event history"
        )
    return C2EnergyBudgetRetentionReceipt(
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
