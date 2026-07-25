"""Lazy A11 activation over the frozen M2-A shadow-store surface.

Small events/snapshots keep the accepted v1 representation byte-for-byte.
The additive content-addressed schema appears only when a growing material
actually crosses a canonical persistence boundary. This preserves existing
M2-D evidence while fixing the habitat growth wall without raising limits.
"""
from __future__ import annotations

import sqlite3
from typing import Any, Mapping

from core.event_kernel import InvalidEventEnvelope
from core import sqlite_shadow_store_v1 as _v1
from core.sqlite_shadow_store_a11 import (
    CONTENT_TABLE_DDL,
    CONTENT_TRIGGER_DDL,
    SQLiteShadowStore as _EagerA11SQLiteShadowStore,
    SnapshotReceipt,
)


class SQLiteShadowStore(_EagerA11SQLiteShadowStore):
    """Activate the A11 storage extension only at the first large material."""

    def _initialize_empty_database(self, connection: sqlite3.Connection) -> None:
        # Preserve the exact accepted M2-A/M2-D fresh-store schema until A11 is needed.
        _v1.SQLiteShadowStore._initialize_empty_database(self, connection)

    def _ensure_content_extension(self, connection: sqlite3.Connection) -> None:
        # ``initialize`` must be observationally identical for exact v1 stores.
        # The first actual content write installs the additive extension atomically.
        return None

    @staticmethod
    def _validate_schema(connection: sqlite3.Connection) -> None:
        has_content = connection.execute(
            "SELECT 1 FROM sqlite_schema WHERE type='table' AND name='content_materials'"
        ).fetchone()
        if has_content is None:
            _v1.SQLiteShadowStore._validate_schema(connection)
            return
        _EagerA11SQLiteShadowStore._validate_schema(connection)

    @staticmethod
    def _put_content(
        connection: sqlite3.Connection,
        reference: Mapping[str, Any],
        material_json: str,
    ) -> None:
        has_content = connection.execute(
            "SELECT 1 FROM sqlite_schema WHERE type='table' AND name='content_materials'"
        ).fetchone()
        if has_content is None:
            # Fail closed unless this is an exact frozen-v1 store before extension.
            _v1.SQLiteShadowStore._validate_schema(connection)
            connection.execute(CONTENT_TABLE_DDL)
            for statement in CONTENT_TRIGGER_DDL.values():
                connection.execute(statement)
        _EagerA11SQLiteShadowStore._put_content(connection, reference, material_json)

    def write_snapshot(
        self,
        *,
        snapshot_id: str,
        stream_id: str,
        through_sequence: int,
        state: Mapping[str, Any],
        state_schema_version: str,
    ) -> SnapshotReceipt:
        try:
            _v1._canon(state, "snapshot_state")
        except InvalidEventEnvelope as exc:
            if "snapshot_state exceeds canonical size limit" not in str(exc):
                raise
        else:
            # Small accepted snapshots retain exact v1 bytes/digests/evidence.
            return _v1.SQLiteShadowStore.write_snapshot(
                self,
                snapshot_id=snapshot_id,
                stream_id=stream_id,
                through_sequence=through_sequence,
                state=state,
                state_schema_version=state_schema_version,
            )
        return super().write_snapshot(
            snapshot_id=snapshot_id,
            stream_id=stream_id,
            through_sequence=through_sequence,
            state=state,
            state_schema_version=state_schema_version,
        )
