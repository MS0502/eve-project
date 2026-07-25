"""Habitat-only A11 EventEnvelope factory.

The accepted M1-A event kernel stays byte-for-byte unchanged. This subclass is
used only by the bounded M2-E phone habitat runtime. It preserves the frozen
65,536-byte EventEnvelope limit; when an otherwise-valid append-only
``before``/``after`` payload itself reaches that limit, it replaces the large
state mappings with versioned digest+manifest references plus a deterministic
append delta. Arbitrary oversized events remain fail-closed.
"""
from __future__ import annotations

from typing import Any, Mapping

from core.canonical_content import CanonicalContentError, compact_append_state_payload
from core.event_kernel import (
    EventEnvelope as _M1AEventEnvelope,
    InvalidEventEnvelope,
    SHADOW_AUTHORITY,
    canonical_json_object,
)


class HabitatEventEnvelope(_M1AEventEnvelope):
    __slots__ = ()

    @classmethod
    def create(
        cls,
        *,
        event_id: str,
        event_type: str,
        stream_id: str,
        sequence: int,
        producer: str,
        producer_version: str,
        correlation_id: str,
        causation_id: str | None = None,
        payload: Mapping[str, Any],
        causal_context: Mapping[str, Any],
        authority: str = SHADOW_AUTHORITY,
    ) -> "HabitatEventEnvelope":
        try:
            payload_json = canonical_json_object(payload, field="payload")
        except InvalidEventEnvelope as original:
            if str(original) != "payload exceeds canonical size limit":
                raise
            try:
                compact = compact_append_state_payload(payload)
                payload_json = canonical_json_object(compact, field="payload")
            except (CanonicalContentError, InvalidEventEnvelope, TypeError, ValueError):
                raise original
        return cls(
            event_id=event_id,
            event_type=event_type,
            stream_id=stream_id,
            sequence=sequence,
            producer=producer,
            producer_version=producer_version,
            correlation_id=correlation_id,
            causation_id=causation_id,
            payload_json=payload_json,
            causal_context_json=canonical_json_object(causal_context, field="causal_context"),
            authority=authority,
        )


# Wrapper consumers intentionally import this name to replace only their local
# EventEnvelope binding; the global M1-A module is untouched.
EventEnvelope = HabitatEventEnvelope
