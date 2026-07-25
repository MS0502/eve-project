"""A11 content-addressed extension for the immutable M2-A SQLite shadow store.

The frozen v1 implementation is retained verbatim in ``sqlite_shadow_store_v1``.
This module preserves its public API while replacing only persistence surfaces
that can grow with habitat state. EventEnvelope validation and its 65,536-byte
canonical JSON limit are deliberately unchanged.
"""
from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, TypeVar

from core.event_kernel import EventEnvelope, InvalidEventEnvelope, SHADOW_AUTHORITY
from core import sqlite_shadow_store_v1 as _v1
from core.sqlite_shadow_store_v1 import *  # noqa: F401,F403 - compatibility surface

StateT = TypeVar("StateT")

CONTENT_REFERENCE_SCHEMA_VERSION = "eve.content-addressed-json-reference.v1"
CONTENT_SERIALIZATION_SCHEMA_VERSION = "eve.canonical-json-content.v1"
EVENT_STORAGE_REFERENCE_SCHEMA_VERSION = "eve.sqlite-shadow-event-storage-ref.v1"
EVENT_PAYLOAD_CONTENT_SCHEMA_VERSION = "eve.event-payload-material.v1"
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

# Explicit aliases used by focused compatibility/diagnostic tests.
_event_material = _v1._event_material
_V1_TABLE_DDL = _v1._TABLE_DDL
_V1_TRIGGER_DDL = _v1._TRIGGER_DDL


def _validate_content_value(value: Any, *, depth: int = 0) -> None:
    if depth > 32:
        raise ValueError("content JSON exceeds maximum nesting depth")
    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, int) and not isinstance(value, bool):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("content JSON numbers must be finite")
        return
    if isinstance(value, list):
        for item in value:
            _validate_content_value(item, depth=depth + 1)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("content JSON object keys must be strings")
            _validate_content_value(item, depth=depth + 1)
        return
    raise ValueError(f"unsupported content JSON value type: {type(value).__name__}")


def _content_json(value: Mapping[str, Any], *, field_name: str) -> str:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping")
    plain = dict(value)
    _validate_content_value(plain)
    return json.dumps(plain, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))


def _collection_counts(value: Mapping[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for key in sorted(value):
        item = value[key]
        if isinstance(item, (list, dict)):
            counts[key] = len(item)
        if isinstance(item, dict):
            for nested_key in sorted(item):
                nested = item[nested_key]
                if isinstance(nested, (list, dict)):
                    counts[f"{key}.{nested_key}"] = len(nested)
    return counts


def _manifest(value: Mapping[str, Any], material_json: str, content_schema_version: str) -> dict[str, Any]:
    return {
        "canonical_bytes": len(material_json.encode("utf-8")),
        "collection_counts": _collection_counts(value),
        "content_schema_version": content_schema_version,
        "hash_algorithm": "sha256",
        "reference_schema_version": CONTENT_REFERENCE_SCHEMA_VERSION,
        "serialization_schema": CONTENT_SERIALIZATION_SCHEMA_VERSION,
        "top_level_key_count": len(value),
        "top_level_keys": sorted(value),
    }


def _content_reference(value: Mapping[str, Any], content_schema_version: str) -> tuple[dict[str, Any], str]:
    material_json = _content_json(value, field_name="content_material")
    reference = {
        "content_digest": _v1._sha(material_json),
        "manifest": _manifest(value, material_json, content_schema_version),
        "reference_schema_version": CONTENT_REFERENCE_SCHEMA_VERSION,
    }
    _v1._canon(reference, "content_reference")
    return reference, material_json


def _event_reference_material(envelope: EventEnvelope, payload_reference: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "authority": envelope.authority,
        "causal_context_json": envelope.causal_context_json,
        "causation_id": envelope.causation_id,
        "correlation_id": envelope.correlation_id,
        "event_id": envelope.event_id,
        "event_type": envelope.event_type,
        "payload_reference": dict(payload_reference),
        "producer": envelope.producer,
        "producer_version": envelope.producer_version,
        "schema_version": envelope.schema_version,
        "sequence": envelope.sequence,
        "storage_schema_version": EVENT_STORAGE_REFERENCE_SCHEMA_VERSION,
        "stream_id": envelope.stream_id,
    }


class SQLiteShadowStore(_v1.SQLiteShadowStore):
    """v1 API with A11 content-addressed persistence for growing material."""

    def initialize(self) -> InitializationReport:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect(require_initialized=False, validate_schema=False) as connection:
            journal = str(connection.execute("PRAGMA journal_mode=WAL").fetchone()[0]).lower()
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute("PRAGMA wal_autocheckpoint=1000")
            objects = int(
                connection.execute("SELECT COUNT(*) FROM sqlite_schema WHERE name NOT LIKE 'sqlite_%'").fetchone()[0]
            )
            if objects == 0:
                self._initialize_empty_database(connection)
            else:
                self._ensure_content_extension(connection)
            self._validate_schema(connection)
        self._initialized = True
        return InitializationReport(str(self._path), journal, journal == "wal")

    def _initialize_empty_database(self, connection: sqlite3.Connection) -> None:
        connection.execute("BEGIN IMMEDIATE")
        try:
            for statement in _v1._TABLE_DDL.values():
                connection.execute(statement)
            connection.execute(CONTENT_TABLE_DDL)
            for statement in _v1._TRIGGER_DDL.values():
                connection.execute(statement)
            for statement in CONTENT_TRIGGER_DDL.values():
                connection.execute(statement)
            self._insert_initial_records(connection)
            connection.commit()
        except Exception:
            connection.rollback()
            raise

    def _ensure_content_extension(self, connection: sqlite3.Connection) -> None:
        names = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_schema WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            ).fetchall()
        }
        if "content_materials" in names:
            return
        # Migrate only an exact frozen v1 store; malformed/partial stores remain fail-closed.
        _v1.SQLiteShadowStore._validate_schema(connection)
        connection.execute("BEGIN IMMEDIATE")
        try:
            connection.execute(CONTENT_TABLE_DDL)
            for statement in CONTENT_TRIGGER_DDL.values():
                connection.execute(statement)
            connection.commit()
        except Exception:
            connection.rollback()
            raise

    @staticmethod
    def _validate_schema(connection: sqlite3.Connection) -> None:
        table_rows = connection.execute(
            "SELECT name,sql FROM sqlite_schema WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
        trigger_rows = connection.execute("SELECT name,sql FROM sqlite_schema WHERE type='trigger'").fetchall()
        actual_tables = {str(row["name"]): _v1._normalize_sql(str(row["sql"])) for row in table_rows}
        actual_triggers = {str(row["name"]): _v1._normalize_sql(str(row["sql"])) for row in trigger_rows}
        expected_tables = {name: _v1._normalize_sql(sql) for name, sql in _v1._TABLE_DDL.items()}
        expected_tables["content_materials"] = _v1._normalize_sql(CONTENT_TABLE_DDL)
        expected_triggers = {name: _v1._normalize_sql(sql) for name, sql in _v1._TRIGGER_DDL.items()}
        expected_triggers.update({name: _v1._normalize_sql(sql) for name, sql in CONTENT_TRIGGER_DDL.items()})
        if actual_tables != expected_tables:
            raise SchemaMismatch("table schema differs from M2-A v1 + A11 content extension")
        if actual_triggers != expected_triggers:
            raise SchemaMismatch("append-only trigger schema differs from M2-A v1 + A11 content extension")
        metadata = dict(connection.execute("SELECT key,value FROM metadata"))
        expected_metadata = {
            "authority": SHADOW_AUTHORITY,
            "event_schema_version": EventEnvelope.__dataclass_fields__["schema_version"].default,
            "snapshot_schema_version": SNAPSHOT_SCHEMA_VERSION,
            "store_schema_version": STORE_SCHEMA_VERSION,
        }
        if metadata != expected_metadata:
            raise SchemaMismatch("metadata differs from M2-A v1")
        rows = connection.execute("SELECT * FROM migrations ORDER BY ordinal").fetchall()
        expected_migration = (1, "m2-a.initial.v1", MIGRATION_SCHEMA_VERSION, _v1._MIGRATION, _v1._MIGRATION_DIGEST)
        if len(rows) != 1 or tuple(rows[0]) != expected_migration:
            raise SchemaMismatch("migration history differs from frozen M2-A v1")

    @staticmethod
    def _put_content(connection: sqlite3.Connection, reference: Mapping[str, Any], material_json: str) -> None:
        digest = str(reference["content_digest"])
        material_bytes = len(material_json.encode("utf-8"))
        existing = connection.execute(
            "SELECT material_json,material_bytes FROM content_materials WHERE content_digest=?", (digest,)
        ).fetchone()
        if existing is None:
            connection.execute(
                "INSERT INTO content_materials(content_digest,material_json,material_bytes) VALUES(?,?,?)",
                (digest, material_json, material_bytes),
            )
        elif str(existing["material_json"]) != material_json or int(existing["material_bytes"]) != material_bytes:
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
            _v1._require_digest(digest, "content_digest")
            manifest = reference["manifest"]
            if not isinstance(manifest, dict):
                raise ValueError("manifest must be an object")
            if manifest.get("content_schema_version") != expected_schema_version:
                raise ValueError("content schema mismatch")
            if manifest.get("hash_algorithm") != "sha256":
                raise ValueError("hash algorithm mismatch")
            if manifest.get("reference_schema_version") != CONTENT_REFERENCE_SCHEMA_VERSION:
                raise ValueError("manifest reference schema mismatch")
            if manifest.get("serialization_schema") != CONTENT_SERIALIZATION_SCHEMA_VERSION:
                raise ValueError("serialization schema mismatch")
            row = connection.execute(
                "SELECT material_json,material_bytes FROM content_materials WHERE content_digest=?", (digest,)
            ).fetchone()
            if row is None:
                raise ValueError("referenced content is missing")
            material_json = str(row["material_json"])
            decoded = json.loads(material_json)
            if not isinstance(decoded, dict):
                raise ValueError("content material must encode an object")
            if _content_json(decoded, field_name="content_material") != material_json:
                raise ValueError("content material is noncanonical")
            if _v1._sha(material_json) != digest or int(row["material_bytes"]) != len(material_json.encode("utf-8")):
                raise ValueError("content digest or byte count mismatch")
            expected_manifest = _manifest(decoded, material_json, expected_schema_version)
            if _v1._canon(expected_manifest, "content_manifest") != _v1._canon(manifest, "content_manifest"):
                raise ValueError("content manifest mismatch")
            return material_json
        except (KeyError, TypeError, ValueError, json.JSONDecodeError, ShadowPersistenceError) as exc:
            raise error_type(f"content-addressed material is corrupt: {exc}") from exc

    def _event_from_row_with_connection(self, connection: sqlite3.Connection, row: sqlite3.Row) -> EventEnvelope:
        text = str(row["event_json"])
        try:
            value = json.loads(text)
            if not isinstance(value, dict) or _v1._canon(value, "persisted_event") != text:
                raise PersistedEventCorruption("persisted event is not canonical")
            if value.get("storage_schema_version") == EVENT_STORAGE_REFERENCE_SCHEMA_VERSION:
                payload_ref = value.get("payload_reference")
                if not isinstance(payload_ref, dict):
                    raise PersistedEventCorruption("event payload reference is missing")
                payload_json = self._resolve_content(
                    connection,
                    payload_ref,
                    expected_schema_version=EVENT_PAYLOAD_CONTENT_SCHEMA_VERSION,
                    error_type=PersistedEventCorruption,
                )
                material = dict(value)
                material.pop("payload_reference")
                material.pop("storage_schema_version")
                material["payload_json"] = payload_json
                envelope = EventEnvelope(**material)
            else:
                envelope = _v1._event_from_json(text, str(row["envelope_digest"]))
            if envelope.authority != SHADOW_AUTHORITY or envelope.digest != str(row["envelope_digest"]):
                raise PersistedEventCorruption("persisted event digest or authority mismatch")
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

    def _event_state(self, connection: sqlite3.Connection) -> tuple[int, int, str, set[str], dict[str, int]]:
        rows = connection.execute("SELECT * FROM events ORDER BY ordinal").fetchall()
        chain = GENESIS_DIGEST
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
            expected_chain = _v1._digest(
                {"envelope_digest": envelope.digest, "ordinal": expected_ordinal, "previous_chain_digest": chain},
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
                        text = _v1._canon(_v1._event_material(envelope), "event_material")
                    except InvalidEventEnvelope as exc:
                        if "event_material exceeds canonical size limit" not in str(exc):
                            raise
                        payload = json.loads(envelope.payload_json)
                        if not isinstance(payload, dict):
                            raise PersistedEventCorruption("event payload must be an object") from exc
                        content = _content_reference(payload, EVENT_PAYLOAD_CONTENT_SCHEMA_VERSION)
                        text = _v1._canon(_event_reference_material(envelope, content[0]), "event_material_reference")
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
                    next_chain = _v1._digest(
                        {"envelope_digest": envelope.digest, "ordinal": ordinal, "previous_chain_digest": chain},
                        "event_chain",
                    )
                    connection.execute(
                        "INSERT INTO events(ordinal,event_id,stream_id,sequence,event_json,envelope_digest,event_bytes,previous_chain_digest,chain_digest) VALUES(?,?,?,?,?,?,?,?,?)",
                        (ordinal, envelope.event_id, envelope.stream_id, envelope.sequence, text, envelope.digest, event_bytes, chain, next_chain),
                    )
                    row = connection.execute("SELECT * FROM events WHERE ordinal=?", (ordinal,)).fetchone()
                    if row is None or self._event_from_row_with_connection(connection, row) != envelope:
                        raise PersistedEventCorruption("event readback failed before commit")
                    transition = _v1._digest(
                        {"after_chain_digest": next_chain, "after_count": count + 1, "before_chain_digest": chain, "before_count": count, "event_id": envelope.event_id},
                        "append_transition",
                    )
                    receipts.append(AppendReceipt(ordinal, envelope.event_id, envelope.stream_id, envelope.sequence, envelope.digest, count, count + 1, chain, next_chain, transition, True, True))
                    count += 1
                    chain = next_chain
                connection.commit()
                return tuple(receipts)
            except (sqlite3.DatabaseError, ShadowPersistenceError):
                connection.rollback()
                raise

    def events(self, *, stream_id: str | None = None, after_sequence: int = 0) -> tuple[EventEnvelope, ...]:
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
                    "SELECT * FROM events WHERE stream_id=? AND sequence>? ORDER BY sequence", (stream_id, after_sequence)
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
        for field_name, value in (("snapshot_id", snapshot_id), ("stream_id", stream_id), ("state_schema_version", state_schema_version)):
            if not isinstance(value, str) or not value.strip():
                raise SnapshotCorruption(f"{field_name} must be a non-empty string")
        if isinstance(through_sequence, bool) or not isinstance(through_sequence, int) or through_sequence < 0:
            raise SnapshotCorruption("through_sequence must be a non-negative integer")
        if not isinstance(state, Mapping):
            raise SnapshotCorruption("state must be a mapping")
        try:
            reference, material_json = _content_reference(state, state_schema_version)
            state_json = _v1._canon(reference, "snapshot_state_reference")
            state_digest = _v1._sha(material_json)
            manifest_json = _v1._canon(reference["manifest"], "snapshot_manifest")
            manifest_digest = _v1._sha(manifest_json)
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
                existing_snapshot_bytes = sum(_v1._snapshot_storage_bytes(dict(row)) for row in snapshot_rows)
                head = connection.execute(
                    "SELECT * FROM events WHERE stream_id=? ORDER BY sequence DESC LIMIT 1", (stream_id,)
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
                snapshot_digest = _v1._digest(
                    {
                        "authority": SHADOW_AUTHORITY,
                        "manifest_digest": manifest_digest,
                        "schema_version": SNAPSHOT_SCHEMA_VERSION,
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
                content_exists = connection.execute(
                    "SELECT 1 FROM content_materials WHERE content_digest=?", (state_digest,)
                ).fetchone() is not None
                added_content_bytes = 0 if content_exists else len(material_json.encode("utf-8"))
                if existing_snapshot_bytes + _v1._snapshot_storage_bytes(row_material) + added_content_bytes > self._policy.max_snapshot_bytes:
                    raise StoragePolicyExceeded("snapshot bytes exceed bounded policy")
                self._put_content(connection, reference, material_json)
                connection.execute(
                    "INSERT INTO snapshots(ordinal,snapshot_id,stream_id,through_sequence,through_event_id,through_event_digest,state_schema_version,state_json,state_digest,manifest_json,manifest_digest,snapshot_digest) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                    (ordinal, snapshot_id, stream_id, through_sequence, event_id, event_digest, state_schema_version, state_json, state_digest, manifest_json, manifest_digest, snapshot_digest),
                )
                persisted = self._snapshot_from_row(
                    connection, connection.execute("SELECT * FROM snapshots WHERE ordinal=?", (ordinal,)).fetchone()
                )
                transition = _v1._digest({"snapshot_digest": persisted.snapshot_digest, "snapshot_id": snapshot_id}, "snapshot_transition")
                connection.commit()
                return SnapshotReceipt(snapshot_id, stream_id, through_sequence, state_digest, manifest_digest, snapshot_digest, transition, True)
            except (sqlite3.DatabaseError, ShadowPersistenceError):
                connection.rollback()
                raise

    def _snapshot_from_row(self, connection: sqlite3.Connection, row: sqlite3.Row | None) -> Snapshot:
        if row is None:
            raise SnapshotCorruption("snapshot row is missing")
        try:
            state_value = json.loads(str(row["state_json"]))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise SnapshotCorruption("snapshot row is malformed") from exc
        if not isinstance(state_value, dict) or state_value.get("reference_schema_version") != CONTENT_REFERENCE_SCHEMA_VERSION:
            return super()._snapshot_from_row(connection, row)
        try:
            material_json = self._resolve_content(
                connection,
                state_value,
                expected_schema_version=str(row["state_schema_version"]),
                error_type=SnapshotCorruption,
            )
            manifest_json = str(row["manifest_json"])
            if _v1._canon(state_value["manifest"], "snapshot_manifest") != manifest_json:
                raise SnapshotCorruption("snapshot manifest differs from content reference")
            if _v1._sha(material_json) != str(row["state_digest"]) or _v1._sha(manifest_json) != str(row["manifest_digest"]):
                raise SnapshotCorruption("snapshot state or manifest digest mismatch")
            snap = Snapshot(
                int(row["ordinal"]), str(row["snapshot_id"]), str(row["stream_id"]), int(row["through_sequence"]),
                None if row["through_event_id"] is None else str(row["through_event_id"]),
                None if row["through_event_digest"] is None else str(row["through_event_digest"]),
                str(row["state_schema_version"]), material_json, str(row["state_digest"]), manifest_json,
                str(row["manifest_digest"]), str(row["snapshot_digest"]),
            )
            if snap.through_sequence == 0:
                if snap.through_event_id is not None or snap.through_event_digest is not None:
                    raise SnapshotCorruption("sequence-zero snapshot names an event")
            else:
                event_row = connection.execute(
                    "SELECT * FROM events WHERE stream_id=? AND sequence=?", (snap.stream_id, snap.through_sequence)
                ).fetchone()
                if event_row is None:
                    raise SnapshotCorruption("snapshot boundary event is missing")
                envelope = self._event_from_row_with_connection(connection, event_row)
                if (snap.through_event_id, snap.through_event_digest) != (envelope.event_id, envelope.digest):
                    raise SnapshotCorruption("snapshot boundary event mismatch")
            expected_digest = _v1._digest(
                {
                    "authority": SHADOW_AUTHORITY,
                    "manifest_digest": snap.manifest_digest,
                    "schema_version": SNAPSHOT_SCHEMA_VERSION,
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
            raise SnapshotCorruption("snapshot row is malformed") from exc

    def restore_verified(
        self,
        *,
        stream_id: str,
        initial_state: StateT,
        reducer: Callable[[StateT, EventEnvelope], StateT],
        state_to_mapping: Callable[[StateT], Mapping[str, Any]],
        state_from_mapping: Callable[[Mapping[str, Any]], StateT],
    ) -> RestoreReport[StateT]:
        if not callable(reducer) or not callable(state_to_mapping) or not callable(state_from_mapping):
            raise RestoreVerificationError("restore requires explicit state codecs and reducer")
        try:
            initial_json = _content_json(state_to_mapping(initial_state), field_name="restore_initial_state")
        except (TypeError, ValueError) as exc:
            raise RestoreVerificationError("restore initial state must be a canonical mapping") from exc
        all_events = self.events(stream_id=stream_id)

        def replay(start_json: str, replay_events: tuple[EventEnvelope, ...]):
            decoded = json.loads(start_json)
            if not isinstance(decoded, dict):
                raise RestoreVerificationError("restore start state must be an object")
            current = state_from_mapping(decoded)
            trace: list[tuple[int, str]] = []
            for event in replay_events:
                current = reducer(current, event)
                if current is None:
                    raise RestoreVerificationError("reducer returned None")
                try:
                    digest = _v1._sha(_content_json(state_to_mapping(current), field_name="restored_state"))
                except (TypeError, ValueError) as exc:
                    raise RestoreVerificationError("restored state must be a canonical mapping") from exc
                trace.append((event.sequence, digest))
            final_digest = _v1._sha(_content_json(state_to_mapping(current), field_name="restored_state"))
            return current, final_digest, tuple(trace)

        full_left, full_digest, full_trace = replay(initial_json, all_events)
        _full_right, repeated_full_digest, repeated_full_trace = replay(initial_json, all_events)
        if full_digest != repeated_full_digest or full_trace != repeated_full_trace:
            raise RestoreVerificationError("repeated full replay produced different history")
        prefix_digests = {0: _v1._sha(initial_json)}
        prefix_digests.update(full_trace)
        rejected: list[str] = []
        selected: Snapshot | None = None
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM snapshots WHERE stream_id=? ORDER BY ordinal DESC", (stream_id,)
            ).fetchall()
            for row in rows:
                try:
                    snapshot = self._snapshot_from_row(connection, row)
                except SnapshotCorruption:
                    rejected.append(str(row["snapshot_id"]))
                    continue
                if prefix_digests.get(snapshot.through_sequence) != snapshot.state_digest:
                    rejected.append(snapshot.snapshot_id)
                    continue
                selected = snapshot
                break
        selection_material = {
            "rejected_snapshot_ids": rejected,
            "selected_snapshot_id": None if selected is None else selected.snapshot_id,
        }
        selection_digest = _v1._digest(selection_material, "snapshot_selection")
        if selected is None:
            current, restored_digest, repeated_digest = full_left, full_digest, repeated_full_digest
            replayed_events, snapshot_id = all_events, None
        else:
            replayed_events = tuple(event for event in all_events if event.sequence > selected.through_sequence)
            current, restored_digest, trace = replay(selected.state_json, replayed_events)
            _repeat, repeated_digest, repeated_trace = replay(selected.state_json, replayed_events)
            if restored_digest != repeated_digest or trace != repeated_trace:
                raise RestoreVerificationError("repeated snapshot replay produced different history")
            if restored_digest != full_digest:
                raise RestoreVerificationError("snapshot restore diverges from full replay")
            snapshot_id = selected.snapshot_id
        transition = _v1._digest(
            {"final_state_digest": restored_digest, "replayed_event_count": len(replayed_events), "selection_digest": selection_digest, "stream_id": stream_id},
            "restore_transition",
        )
        return RestoreReport(current, restored_digest, repeated_digest, len(replayed_events), snapshot_id, tuple(rejected), transition, True)
