"""Type-preserving habitat facade over the A11 SQLite store.

SQLite persistence validates/reconstructs the accepted base EventEnvelope. The
phone habitat factory uses a narrow subclass solely to add A11 construction for
future oversized append-state payloads. This facade converts verified readback
to that subclass so deterministic equality checks used by reviewed resume keep
their historical semantics.
"""
from __future__ import annotations

import sqlite3

from core.habitat_event_kernel_a11 import HabitatEventEnvelope
from core.sqlite_shadow_store_a11 import SQLiteShadowStore as _A11SQLiteShadowStore


class SQLiteShadowStore(_A11SQLiteShadowStore):
    def _event_from_row_with_connection(
        self,
        connection: sqlite3.Connection,
        row: sqlite3.Row,
    ) -> HabitatEventEnvelope:
        envelope = super()._event_from_row_with_connection(connection, row)
        if isinstance(envelope, HabitatEventEnvelope):
            return envelope
        return HabitatEventEnvelope(
            event_id=envelope.event_id,
            event_type=envelope.event_type,
            stream_id=envelope.stream_id,
            sequence=envelope.sequence,
            producer=envelope.producer,
            producer_version=envelope.producer_version,
            correlation_id=envelope.correlation_id,
            causation_id=envelope.causation_id,
            payload_json=envelope.payload_json,
            causal_context_json=envelope.causal_context_json,
            schema_version=envelope.schema_version,
            authority=envelope.authority,
        )
