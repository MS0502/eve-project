"""Habitat-only A11 extension of the accepted M2-A SQLite shadow store.

The global ``core.sqlite_shadow_store`` stays byte-for-byte unchanged so M2-D
and M2-B accepted evidence remains stable. This subclass is selected only by
the bounded M2-E habitat wrapper. Small historical material keeps the v1
representation. At the first canonical growth boundary an additive append-only
content table is installed transactionally and large material is represented by
SHA-256 content digest + versioned structural manifest.
"""
from __future__ import annotations

import json
import sqlite3
from typing import Any, Iterable, Mapping

import core.sqlite_shadow_store as _base
from core.canonical_content import (
    CONTENT_REFERENCE_SCHEMA_VERSION,
    build_content_reference,
    canonical_content_json,
    verify_content_reference,
)
from core.event_kernel import EventEnvelope, InvalidEventEnvelope, SHADOW_AUTHORITY
from core.sqlite_shadow_store import *  # noqa: F401,F403 - preserve public error/receipt types

CONTENT_TABLE_DDL = (
    "CREATE TABLE content_materials(content_digest TEXT PRIMARY KEY,material_json TEXT NOT NULL,"
    "material_bytes INTEGER NOT NULL)"
)
CONTENT_TRIGGER_DDL = {
    "content_materials_no_update": (
        "CREATE TRIGGER content_materials_no_update BEFORE UPDATE ON content_materials "
        "BEGIN SELECT RAISE(ABORT,'append-only content material'); END"
    ),
    "content_materials_no_delete": (
        "CREATE TRIGGER content_materials_no_delete BEFORE DELETE ON content_materials "
        "BEGIN SELECT RAISE(ABORT,'append-only content material'); END"
    ),
}
EVENT_STORAGE_REFERENCE_SCHEMA_VERSION = "eve.sqlite-shadow-event-storage-ref.v1"
EVENT_PAYLOAD_CONTENT_SCHEMA_VERSION = "eve.event-payload-material.v1"


def _event_reference_material(
    envelope: EventEnvelope,
    reference: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "authority": envelope.authority,
        "causal_context_json": envelope.causal_context_json,
        "causation_id": envelope.causation_id,
        "correlation_id": envelope.correlation_id,
        "event_id": envelope.event_id,
        "event_type": envelope.event_type,
        "payload_reference": dict(reference),
        "producer": envelope.producer,
        "producer_version": envelope.producer_version,
        "schema_version": envelope.schema_version,
        "sequence": envelope.sequence,
        "storage_schema_version": EVENT_STORAGE_REFERENCE_SCHEMA_VERSION,
        "stream_id": envelope.stream_id,
    }


class SQLiteShadowStore(_base.SQLiteShadowStore):
    """M2-E habitat store with lazy A11 large-material persistence."""

    @staticmethod
    def _validate_schema(connection: sqlite3.Connection) -> None:
        has_content = connection.execute(
            "SELECT 1 FROM sqlite_schema WHERE type='table' AND name='content_materials'"
        ).fetchone()
        if has_content is None:
            _base.SQLiteShadowStore._validate_schema(connection)
            return

        table_rows = connection.execute(
            "SELECT name,sql FROM sqlite_schema WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
        trigger_rows = connection.execute(
            "SELECT name,sql FROM sqlite_schema WHERE type='trigger'"
        ).fetchall()
        actual_tables = {
            str(row["name"]): _base._normalize_sql(str(row["sql"])) for row in table_rows
        }
        actual_triggers = {
            str(row["name"]): _base._normalize_sql(str(row["sql"])) for row in trigger_rows
        }
        expected_tables = {
            name: _base._normalize_sql(sql) for name, sql in _base._TABLE_DDL.items()
        }
        expected_tables["content_materials"] = _base._normalize_sql(CONTENT_TABLE_DDL)
        expected_triggers = {
            name: _base._normalize_sql(sql) for name, sql in _base._TRIGGER_DDL.items()
        }
        expected_triggers.update(
            {name: _base._normalize_sql(sql) for name, sql in CONTENT_TRIGGER_DDL.items()}
        )
        if actual_tables != expected_tables:
            raise SchemaMismatch("table schema differs from M2-A v1 + habitat A11 extension")
        if actual_triggers != expected_triggers:
            raise SchemaMismatch("append-only trigger schema differs from M2-A v1 + habitat A11 extension")
        metadata = dict(connection.execute("SELECT key,value FROM metadata"))
        expected_metadata = {
            "authority": SHADOW_AUTHORITY,
            "event_schema_version": EventEnvelope.__dataclass_fields__["schema_version"].default,
            "snapshot_schema_version": _base.SNAPSHOT_SCHEMA_VERSION,
            "store_schema_version": _base.STORE_SCHEMA_VERSION,
        }
        if metadata != expected_metadata:
            raise SchemaMismatch("metadata differs from M2-A v1")
        rows = connection.execute("SELECT * FROM migrations ORDER BY ordinal").fetchall()
        expected_migration = (
            1,
            "m2-a.initial.v1",
            _base.MIGRATION_SCHEMA_VERSION,
            _base._MIGRATION,
            _base._MIGRATION_DIGEST,
        )
        if len(rows) != 1 or tuple(rows[0]) != expected_migration:
            raise SchemaMismatch("migration history differs from frozen M2-A v1")

    @staticmethod
    def _ensure_content_table(connection: sqlite3.Connection) -> None:
        existing = connection.execute(
            "SELECT 1 FROM sqlite_schema WHERE type='table' AND name='content_materials'"
        ).fetchone()
        if existing is not None:
            return
        # The extension may be added only on top of an exact accepted v1 store.
        _base.SQLiteShadowStore._validate_schema(connection)
        connection.execute(CONTENT_TABLE_DDL)
        for statement in CONTENT_TRIGGER_DDL.values():
            connection.execute(statement)

    @classmethod
    def _put_content(
        cls,
        connection: sqlite3.Connection,
        reference: Mapping[str, Any],
        material_json: str,
    ) -> None:
        cls._ensure_content_table(connection)
        digest = str(reference["content_digest"])
        material_bytes = len(material_json.encode("utf-8"))
        existing = connection.execute(
            "SELECT material_json,material_bytes FROM content_materials WHERE content_digest=?",
            (digest,),
        ).fetchone()
        if existing is None:
            connection.execute(
                "INSERT INTO content_materials(content_digest,material_json,material_bytes) VALUES(?,?,?)",
                (digest, material_json, material_bytes),
            )
            return
        if str(existing["material_json"]) != material_json or int(existing["material_bytes"]) != material_bytes:
            raise PersistedEventCorruption("content-addressed digest collision or corrupt material")

    @staticmethod
    def _resolve_content(
        connection: sqlite3.Connection,
        reference: Mapping[str, Any],
        *,
        expected_schema_version: str,
        error_type: type[Exception],
    ) -> str:
        try:
            if reference.get("reference_schema_version") != CONTENT_REFERENCE_SCHEMA_VERSION:
                raise ValueError("reference schema mismatch")
            digest = str(reference["content_digest"])
            _base._require_digest(digest, "content_digest")
            row = connection.execute(
                "SELECT material_json,material_bytes FROM content_materials WHERE content_digest=?",
                (digest,),
            ).fetchone()
            if row is None:
                raise ValueError("referenced content is missing")
            material_json = str(row["material_json"])
            material = json.loads(material_json)
            if not isinstance(material, dict):
                raise ValueError("content material must encode an object")
            if canonical_content_json(material, field="content_material") != material_json:
                raise ValueError("content material is noncanonical")
            if int(row["material_bytes"]) != len(material_json.encode("utf-8")):
                raise ValueError("content byte count mismatch")
            verify_content_reference(
                reference,
                material,
                expected_schema_version=expected_schema_version,
            )
            return material_json
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise error_type(f"content-addressed material is corrupt: {exc}") from exc

    def _event_from_row_with_connection(
        self,
        connection: sqlite3.Connection,
        row: sqlite3.Row,
    ) -> EventEnvelope:
        text = str(row["event_json"])
        try:
            value = json.loads(text)
            if not isinstance(value, dict) or _base._canon(value, "persisted_event") != text:
                raise PersistedEventCorruption("persisted event is not canonical")
            if value.get("storage_schema_version") == EVENT_STORAGE_REFERENCE_SCHEMA_VERSION:
                reference = value.get("payload_reference")
                if not isinstance(reference, dict):
                    raise PersistedEventCorruption("event payload reference is missing")
                payload_json = self._resolve_content(
                    connection,
                    reference,
                    expected_schema_version=EVENT_PAYLOAD_CONTENT_SCHEMA_VERSION,
                    error_type=PersistedEventCorruption,
                )
                material = dict(value)
                material.pop("payload_reference")
                material.pop("storage_schema_version")
                material["payload_json"] = payload_json
                envelope = EventEnvelope(**material)
            else:
                envelope = _base._event_from_json(text, str(row["envelope_digest"]))
            if envelope.authority != SHADOW_AUTHORITY:
                raise PersistedEventCorruption("persisted event authority mismatch")
            if envelope.digest != str(row["envelope_digest"]):
                raise PersistedEventCorruption("persisted event digest mismatch")
            actual = (str(row["event_id"]), str(row["stream_id"]), int(row["sequence"]))
            if actual != (envelope.event_id, envelope.stream_id, envelope.sequence):
                raise PersistedEventCorruption("event index columns disagree with canonical envelope")
            if int(row["event_bytes"]) != len(text.encode("utf-8")):
                raise PersistedEventCorruption("event byte count mismatch")
            return envelope
        except (KeyError, TypeError, ValueError, json.JSONDecodeError, InvalidEventEnvelope) as exc:
            if isinstance(exc, PersistedEventCorruption):
                raise
            raise PersistedEventCorruption("persisted event row is malformed") from exc

    def _event_state(
        self,
        connection: sqlite3.Connection,
    ) -> tuple[int, int, str, set[str], dict[str, int]]:
        rows = connection.execute("SELECT * FROM events ORDER BY ordinal").fetchall()
        chain = _base.GENESIS_DIGEST
        known: set[str] = set()
        stream_sequences: dict[str, int] = {}
        total_bytes = 0
        for expected_ordinal, row in enumerate(rows, 1):
            if int(row["ordinal"]) != expected_ordinal or str(row["previous_chain_digest"]) != chain:
                raise PersistedEventCorruption("ordinal or previous chain mismatch")
            envelope = self._event_from_row_with_connection(connection, row)
            expected_sequence = stream_sequences.get(envelope.stream_id, 0) + 1
            if envelope.sequence != expected_sequence:
                raise PersistedEventCorruption("stream sequence mismatch")
            if envelope.causation_id is not None and envelope.causation_id not in known:
                raise PersistedEventCorruption("causation points outside prior history")
            expected_chain = _base._digest(
                {
                    "envelope_digest": envelope.digest,
                    "ordinal": expected_ordinal,
                    "previous_chain_digest": chain,
                },
                "event_chain",
            )
            if str(row["chain_digest"]) != expected_chain:
                raise PersistedEventCorruption("event chain digest mismatch")
            chain = expected_chain
            known.add(envelope.event_id)
            stream_sequences[envelope.stream_id] = envelope.sequence
            total_bytes += int(row["event_bytes"])
        return len(rows), total_bytes, chain, known, stream_sequences

    def append_many(self, envelopes: Iterable[EventEnvelope]) -> tuple[AppendReceipt, ...]:
        values = tuple(envelopes)
        if not values:
            return ()
        if any(not isinstance(item, EventEnvelope) for item in values):
            raise AppendOnlyViolation("store accepts EventEnvelope only")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                before_count, before_bytes, chain, known, last_seq = self._event_state(connection)
                batch_ids: set[str] = set()
                materials: list[tuple[EventEnvelope, str, int, tuple[dict[str, Any], str] | None]] = []
                for envelope in values:
                    if envelope.authority != SHADOW_AUTHORITY:
                        raise AppendOnlyViolation("only shadow_only envelopes may persist")
                    if envelope.event_id in known or envelope.event_id in batch_ids:
                        raise AppendOnlyViolation(f"duplicate event_id: {envelope.event_id}")
                    expected = last_seq.get(envelope.stream_id, 0) + 1
                    if envelope.sequence != expected:
                        raise AppendOnlyViolation(f"expected sequence {expected} for {envelope.stream_id}")
                    if envelope.causation_id is not None and envelope.causation_id not in known | batch_ids:
                        raise AppendOnlyViolation(f"unknown causation: {envelope.causation_id}")
                    content: tuple[dict[str, Any], str] | None = None
                    try:
                        text = _base._canon(_base._event_material(envelope), "event_material")
                    except InvalidEventEnvelope as exc:
                        if "event_material exceeds canonical size limit" not in str(exc):
                            raise
                        payload = json.loads(envelope.payload_json)
                        if not isinstance(payload, dict):
                            raise PersistedEventCorruption("event payload must be an object") from exc
                        content = build_content_reference(
                            payload,
                            content_schema_version=EVENT_PAYLOAD_CONTENT_SCHEMA_VERSION,
                        )
                        text = _base._canon(
                            _event_reference_material(envelope, content[0]),
                            "event_material_reference",
                        )
                    materials.append((envelope, text, len(text.encode("utf-8")), content))
                    batch_ids.add(envelope.event_id)
                    last_seq[envelope.stream_id] = envelope.sequence
                total_bytes = sum(item[2] for item in materials)
                if before_count + len(materials) > self._policy.max_event_count or before_bytes + total_bytes > self._policy.max_event_bytes:
                    raise StoragePolicyExceeded("event append exceeds bounded storage policy")

                receipts: list[AppendReceipt] = []
                count = before_count
                for envelope, text, event_bytes, content in materials:
                    if content is not None:
                        self._put_content(connection, content[0], content[1])
                    ordinal = count + 1
                    next_chain = _base._digest(
                        {
                            "envelope_digest": envelope.digest,
                            "ordinal": ordinal,
                            "previous_chain_digest": chain,
                        },
                        "event_chain",
                    )
                    connection.execute(
                        "INSERT INTO events(ordinal,event_id,stream_id,sequence,event_json,envelope_digest,event_bytes,previous_chain_digest,chain_digest) VALUES(?,?,?,?,?,?,?,?,?)",
                        (
                            ordinal,
                            envelope.event_id,
                            envelope.stream_id,
                            envelope.sequence,
                            text,
                            envelope.digest,
                            event_bytes,
                            chain,
                            next_chain,
                        ),
                    )
                    row = connection.execute("SELECT * FROM events WHERE ordinal=?", (ordinal,)).fetchone()
                    if row is None or self._event_from_row_with_connection(connection, row) != envelope:
                        raise PersistedEventCorruption("event readback failed before commit")
                    transition = _base._digest(
                        {
                            "after_chain_digest": next_chain,
                            "after_count": count + 1,
                            "before_chain_digest": chain,
                            "before_count": count,
                            "event_id": envelope.event_id,
                        },
                        "append_transition",
                    )
                    receipts.append(
                        AppendReceipt(
                            ordinal,
                            envelope.event_id,
                            envelope.stream_id,
                            envelope.sequence,
                            envelope.digest,
                            count,
                            count + 1,
                            chain,
                            next_chain,
                            transition,
                            True,
                            True,
                        )
                    )
                    count += 1
                    chain = next_chain
                connection.commit()
                return tuple(receipts)
            except (sqlite3.DatabaseError, ShadowPersistenceError):
                connection.rollback()
                raise

    def events(
        self,
        *,
        stream_id: str | None = None,
        after_sequence: int = 0,
    ) -> tuple[EventEnvelope, ...]:
        if isinstance(after_sequence, bool) or not isinstance(after_sequence, int) or after_sequence < 0:
            raise ValueError("after_sequence must be non-negative")
        if stream_id is None and after_sequence:
            raise ValueError("after_sequence requires an explicit stream_id")
        with self._connect() as connection:
            self._event_state(connection)
            if stream_id is None:
                rows = connection.execute("SELECT * FROM events ORDER BY ordinal").fetchall()
            else:
                rows = connection.execute(
                    "SELECT * FROM events WHERE stream_id=? AND sequence>? ORDER BY sequence",
                    (stream_id, after_sequence),
                ).fetchall()
            return tuple(self._event_from_row_with_connection(connection, row) for row in rows)

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
            _base._canon(state, "snapshot_state")
            inline_fits = True
        except InvalidEventEnvelope as exc:
            if "snapshot_state exceeds canonical size limit" not in str(exc):
                raise
            inline_fits = False

        with self._connect() as probe:
            content_active = probe.execute(
                "SELECT 1 FROM sqlite_schema WHERE type='table' AND name='content_materials'"
            ).fetchone() is not None
        if inline_fits and not content_active:
            # Accepted small snapshots remain byte-for-byte v1 until A11 is needed.
            return _base.SQLiteShadowStore.write_snapshot(
                self,
                snapshot_id=snapshot_id,
                stream_id=stream_id,
                through_sequence=through_sequence,
                state=state,
                state_schema_version=state_schema_version,
            )

        for field_name, value in (
            ("snapshot_id", snapshot_id),
            ("stream_id", stream_id),
            ("state_schema_version", state_schema_version),
        ):
            if not isinstance(value, str) or not value.strip():
                raise SnapshotCorruption(f"{field_name} must be a non-empty string")
        if isinstance(through_sequence, bool) or not isinstance(through_sequence, int) or through_sequence < 0:
            raise SnapshotCorruption("through_sequence must be a non-negative integer")
        if not isinstance(state, Mapping):
            raise SnapshotCorruption("state must be a mapping")
        try:
            reference, material_json = build_content_reference(
                state,
                content_schema_version=state_schema_version,
            )
            state_json = _base._canon(reference, "snapshot_state_reference")
            state_digest = _base._sha(material_json)
            manifest_json = _base._canon(reference["manifest"], "snapshot_manifest")
            manifest_digest = _base._sha(manifest_json)
        except (TypeError, ValueError, InvalidEventEnvelope) as exc:
            raise SnapshotCorruption("state must be canonical JSON-compatible mapping") from exc

        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                self._event_state(connection)
                snapshot_rows = connection.execute("SELECT * FROM snapshots ORDER BY ordinal").fetchall()
                count = len(snapshot_rows)
                if count >= self._policy.max_snapshot_count:
                    raise StoragePolicyExceeded("snapshot count exceeds bounded policy")
                existing_snapshot_bytes = sum(
                    _base._snapshot_storage_bytes(dict(row)) for row in snapshot_rows
                )
                head = connection.execute(
                    "SELECT * FROM events WHERE stream_id=? ORDER BY sequence DESC LIMIT 1",
                    (stream_id,),
                ).fetchone()
                if head is None:
                    if through_sequence != 0:
                        raise SnapshotCorruption("empty stream snapshot must bind sequence zero")
                    event_id = event_digest = None
                else:
                    envelope = self._event_from_row_with_connection(connection, head)
                    if through_sequence != envelope.sequence:
                        raise SnapshotCorruption("snapshot must bind the current stream head")
                    event_id, event_digest = envelope.event_id, envelope.digest
                snapshot_digest = _base._digest(
                    {
                        "authority": SHADOW_AUTHORITY,
                        "manifest_digest": manifest_digest,
                        "schema_version": _base.SNAPSHOT_SCHEMA_VERSION,
                        "snapshot_id": snapshot_id,
                        "state_digest": state_digest,
                        "state_schema_version": state_schema_version,
                        "stream_id": stream_id,
                        "through_event_digest": event_digest,
                        "through_event_id": event_id,
                        "through_sequence": through_sequence,
                    },
                    "snapshot_material",
                )
                ordinal = count + 1
                row_material = {
                    "ordinal": ordinal,
                    "snapshot_id": snapshot_id,
                    "stream_id": stream_id,
                    "through_sequence": through_sequence,
                    "through_event_id": event_id,
                    "through_event_digest": event_digest,
                    "state_schema_version": state_schema_version,
                    "state_json": state_json,
                    "state_digest": state_digest,
                    "manifest_json": manifest_json,
                    "manifest_digest": manifest_digest,
                    "snapshot_digest": snapshot_digest,
                }
                if existing_snapshot_bytes + _base._snapshot_storage_bytes(row_material) > self._policy.max_snapshot_bytes:
                    raise StoragePolicyExceeded("snapshot bytes exceed bounded policy")
                self._put_content(connection, reference, material_json)
                connection.execute(
                    "INSERT INTO snapshots(ordinal,snapshot_id,stream_id,through_sequence,through_event_id,through_event_digest,state_schema_version,state_json,state_digest,manifest_json,manifest_digest,snapshot_digest) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        ordinal,
                        snapshot_id,
                        stream_id,
                        through_sequence,
                        event_id,
                        event_digest,
                        state_schema_version,
                        state_json,
                        state_digest,
                        manifest_json,
                        manifest_digest,
                        snapshot_digest,
                    ),
                )
                persisted = self._snapshot_from_row(
                    connection,
                    connection.execute("SELECT * FROM snapshots WHERE ordinal=?", (ordinal,)).fetchone(),
                )
                transition = _base._digest(
                    {"snapshot_digest": persisted.snapshot_digest, "snapshot_id": snapshot_id},
                    "snapshot_transition",
                )
                connection.commit()
                return SnapshotReceipt(
                    snapshot_id,
                    stream_id,
                    through_sequence,
                    state_digest,
                    manifest_digest,
                    snapshot_digest,
                    transition,
                    True,
                )
            except (sqlite3.DatabaseError, ShadowPersistenceError):
                connection.rollback()
                raise

    def _snapshot_from_row(
        self,
        connection: sqlite3.Connection,
        row: sqlite3.Row | None,
    ) -> Snapshot:
        if row is None:
            raise SnapshotCorruption("snapshot row is missing")
        try:
            state_value = json.loads(str(row["state_json"]))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise SnapshotCorruption("snapshot row is malformed") from exc
        if not isinstance(state_value, dict) or state_value.get("reference_schema_version") != CONTENT_REFERENCE_SCHEMA_VERSION:
            return _base.SQLiteShadowStore._snapshot_from_row(self, connection, row)
        try:
            material_json = self._resolve_content(
                connection,
                state_value,
                expected_schema_version=str(row["state_schema_version"]),
                error_type=SnapshotCorruption,
            )
            manifest_json = str(row["manifest_json"])
            if _base._canon(state_value["manifest"], "snapshot_manifest") != manifest_json:
                raise SnapshotCorruption("snapshot manifest differs from content reference")
            if _base._sha(material_json) != str(row["state_digest"]):
                raise SnapshotCorruption("snapshot state digest mismatch")
            if _base._sha(manifest_json) != str(row["manifest_digest"]):
                raise SnapshotCorruption("snapshot manifest digest mismatch")
            snap = Snapshot(
                int(row["ordinal"]),
                str(row["snapshot_id"]),
                str(row["stream_id"]),
                int(row["through_sequence"]),
                None if row["through_event_id"] is None else str(row["through_event_id"]),
                None if row["through_event_digest"] is None else str(row["through_event_digest"]),
                str(row["state_schema_version"]),
                material_json,
                str(row["state_digest"]),
                manifest_json,
                str(row["manifest_digest"]),
                str(row["snapshot_digest"]),
            )
            if snap.through_sequence == 0:
                if snap.through_event_id is not None or snap.through_event_digest is not None:
                    raise SnapshotCorruption("sequence-zero snapshot names an event")
            else:
                event_row = connection.execute(
                    "SELECT * FROM events WHERE stream_id=? AND sequence=?",
                    (snap.stream_id, snap.through_sequence),
                ).fetchone()
                if event_row is None:
                    raise SnapshotCorruption("snapshot boundary event is missing")
                envelope = self._event_from_row_with_connection(connection, event_row)
                if (snap.through_event_id, snap.through_event_digest) != (
                    envelope.event_id,
                    envelope.digest,
                ):
                    raise SnapshotCorruption("snapshot boundary event mismatch")
            expected_digest = _base._digest(
                {
                    "authority": SHADOW_AUTHORITY,
                    "manifest_digest": snap.manifest_digest,
                    "schema_version": _base.SNAPSHOT_SCHEMA_VERSION,
                    "snapshot_id": snap.snapshot_id,
                    "state_digest": snap.state_digest,
                    "state_schema_version": snap.state_schema_version,
                    "stream_id": snap.stream_id,
                    "through_event_digest": snap.through_event_digest,
                    "through_event_id": snap.through_event_id,
                    "through_sequence": snap.through_sequence,
                },
                "snapshot_material",
            )
            if expected_digest != snap.snapshot_digest:
                raise SnapshotCorruption("snapshot digest mismatch")
            return snap
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            if isinstance(exc, SnapshotCorruption):
                raise
            raise SnapshotCorruption("snapshot row is malformed") from exc
