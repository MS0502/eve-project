"""EVE event-kernel public facade with A11 large-state compaction.

The frozen M1-A kernel remains unchanged in ``event_kernel_v1``. EventEnvelope
still enforces the same 65,536-byte canonical payload limit. If and only if an
otherwise valid oversized payload contains append-only ``before``/``after``
state mappings, A11 replaces those large mappings with content digest +
structural manifest references and a deterministic append delta. Arbitrary
oversized events remain fail-closed.
"""
from __future__ import annotations

from typing import Any, Mapping

from core.canonical_content import CanonicalContentError, compact_append_state_payload
from core import event_kernel_v1 as _v1
from core.event_kernel_v1 import *  # noqa: F401,F403 - frozen compatibility surface


class EventEnvelope(_v1.EventEnvelope):
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
    ) -> "EventEnvelope":
        """Create the same bounded envelope, with A11 append-state fallback only."""
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
