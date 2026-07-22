"""M2-A append-only SQLite shadow persistence.

Import and construction perform no I/O. The caller must explicitly initialize a
concrete file and explicitly append ``shadow_only`` event envelopes or validated
snapshots. Nothing here installs a bridge, changes a default, reads legacy sidecars,
or grants dual-read, recovery, cutover, mutation, or production authority.
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Generic, Iterable, Iterator, Mapping, TypeVar

from core.event_kernel import EventEnvelope, InvalidEventEnvelope, SHADOW_AUTHORITY, canonical_json_object

STORE_SCHEMA_VERSION = "eve.sqlite-shadow-store.v1"
MIGRATION_SCHEMA_VERSION = "eve.sqlite-shadow-migration.v1"
SNAPSHOT_SCHEMA_VERSION = "eve.sqlite-shadow-snapshot.v1"
RECEIPT_SCHEMA_VERSION = "eve.sqlite-shadow-append-receipt.v1"
SNAPSHOT_RECEIPT_SCHEMA_VERSION = "eve.sqlite-shadow-snapshot-receipt.v1"
INTEGRITY_SCHEMA_VERSION = "eve.sqlite-shadow-integrity-report.v1"
RESTORE_SCHEMA_VERSION = "eve.sqlite-shadow-restore-report.v1"
BACKUP_SCHEMA_VERSION = "eve.sqlite-shadow-backup-receipt.v1"
STATE_ENCODING_VERSION = "eve.canonical-json-state.v1"
GENESIS_DIGEST = "0" * 64
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_BACKUP = re.compile(r"^shadow-backup-([0-9]{8})\.sqlite3$")
StateT = TypeVar("StateT")


class ShadowPersistenceError(RuntimeError):
    pass


class StoreNotInitialized(ShadowPersistenceError):
    pass


class SchemaMismatch(ShadowPersistenceError):
    pass


class AppendOnlyViolation(ShadowPersistenceError):
    pass


class StoragePolicyExceeded(ShadowPersistenceError):
    pass


class PersistedEventCorruption(ShadowPersistenceError):
    pass


class SnapshotCorruption(ShadowPersistenceError):
    pass


class RestoreVerificationError(ShadowPersistenceError):
    pass


class BackupPolicyError(ShadowPersistenceError):
    pass


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canon(value: Mapping[str, Any], field: str) -> str:
    return canonical_json_object(value, field=field)


def _digest(value: Mapping[str, Any], field: str) -> str:
    return _sha(_canon(value, field))


def _require_digest(value: str, field: str) -> None:
    if not isinstance(value, str) or not _DIGEST.fullmatch(value):
        raise ShadowPersistenceError(f"{field} is not a SHA-256 digest")


def _normalize_sql(value: str) -> str:
    return " ".join(value.strip().rstrip(";").split())


@dataclass(frozen=True, slots=True)
class ShadowStoragePolicy:
    snapshot_interval_events: int = 100
    max_event_count: int = 1_000_000
    max_event_bytes: int = 268_435_456
    max_snapshot_count: int = 10_000
    max_backups: int = 3

    def __post_init__(self) -> None:
        for field in self.__slots__:
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{field} must be a positive integer")


@dataclass(frozen=True, slots=True)
class InitializationReport:
    database_path: str
    journal_mode: str
    wal_enabled: bool
    schema_version: str = STORE_SCHEMA_VERSION
    authority: str = SHADOW_AUTHORITY


@dataclass(frozen=True, slots=True)
class AppendReceipt:
    ordinal: int
    event_id: str
    stream_id: str
    sequence: int
    envelope_digest: str
    before_count: int
    after_count: int
    before_chain_digest: str
    after_chain_digest: str
    transition_hash: str
    readback_verified: bool
    state_changed: bool
    schema_version: str = RECEIPT_SCHEMA_VERSION
    authority: str = SHADOW_AUTHORITY

    def __post_init__(self) -> None:
        if self.authority != SHADOW_AUTHORITY or not self.readback_verified:
            raise ShadowPersistenceError("append receipt is not verified shadow evidence")
        expected = self.after_count == self.before_count + 1 and self.before_chain_digest != self.after_chain_digest
        if self.state_changed != expected:
            raise ShadowPersistenceError("state_changed disagrees with before/after evidence")
        for field in ("envelope_digest", "before_chain_digest", "after_chain_digest", "transition_hash"):
            _require_digest(getattr(self, field), field)


@dataclass(frozen=True, slots=True)
class Snapshot:
    ordinal: int
    snapshot_id: str
    stream_id: str
    through_sequence: int
    through_event_id: str | None
    through_event_digest: str | None
    state_schema_version: str
    state_json: str
    state_digest: str
    manifest_json: str
    manifest_digest: str
    snapshot_digest: str
    schema_version: str = SNAPSHOT_SCHEMA_VERSION
    authority: str = SHADOW_AUTHORITY

    @property
    def state(self) -> dict[str, Any]:
        return json.loads(self.state_json)


@dataclass(frozen=True, slots=True)
class SnapshotReceipt:
    snapshot_id: str
    stream_id: str
    through_sequence: int
    state_digest: str
    manifest_digest: str
    snapshot_digest: str
    transition_hash: str
    readback_verified: bool
    schema_version: str = SNAPSHOT_RECEIPT_SCHEMA_VERSION
    authority: str = SHADOW_AUTHORITY

    def __post_init__(self) -> None:
        if self.authority != SHADOW_AUTHORITY or not self.readback_verified:
            raise SnapshotCorruption("snapshot receipt is not verified shadow evidence")
        for field in ("state_digest", "manifest_digest", "snapshot_digest", "transition_hash"):
            _require_digest(getattr(self, field), field)


@dataclass(frozen=True, slots=True)
class SnapshotSelection:
    selected: Snapshot | None
    rejected_snapshot_ids: tuple[str, ...]
    fallback_used: bool
    selection_digest: str


@dataclass(frozen=True, slots=True)
class IntegrityReport:
    valid: bool
    errors: tuple[str, ...]
    event_count: int
    snapshot_count: int
    chain_head_digest: str
    report_digest: str
    schema_version: str = INTEGRITY_SCHEMA_VERSION
    authority: str = SHADOW_AUTHORITY


@dataclass(frozen=True, slots=True)
class RestoreReport(Generic[StateT]):
    state: StateT
    state_digest: str
    repeated_state_digest: str
    replayed_event_count: int
    snapshot_id: str | None
    rejected_snapshot_ids: tuple[str, ...]
    transition_hash: str
    verified: bool
    schema_version: str = RESTORE_SCHEMA_VERSION
    authority: str = SHADOW_AUTHORITY


@dataclass(frozen=True, slots=True)
class BackupReceipt:
    backup_path: str
    backup_ordinal: int
    backup_sha256: str
    pruned_backup_names: tuple[str, ...]
    integrity_verified: bool
    schema_version: str = BACKUP_SCHEMA_VERSION
    authority: str = SHADOW_AUTHORITY

    def __post_init__(self) -> None:
        if self.authority != SHADOW_AUTHORITY or not self.integrity_verified:
            raise BackupPolicyError("backup receipt is not verified shadow evidence")
        _require_digest(self.backup_sha256, "backup_sha256")


_TABLE_DDL = {
    "metadata": "CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL)",
    "migrations": (
        "CREATE TABLE migrations(ordinal INTEGER PRIMARY KEY,migration_id TEXT UNIQUE NOT NULL,"
        "schema_version TEXT NOT NULL,manifest_json TEXT NOT NULL,migration_digest TEXT NOT NULL)"
    ),
    "events": (
        "CREATE TABLE events(ordinal INTEGER PRIMARY KEY,event_id TEXT UNIQUE NOT NULL,"
        "stream_id TEXT NOT NULL,sequence INTEGER NOT NULL,event_json TEXT NOT NULL,"
        "envelope_digest TEXT NOT NULL,event_bytes INTEGER NOT NULL,"
        "previous_chain_digest TEXT NOT NULL,chain_digest TEXT NOT NULL,UNIQUE(stream_id,sequence))"
    ),
    "snapshots": (
        "CREATE TABLE snapshots(ordinal INTEGER PRIMARY KEY,snapshot_id TEXT UNIQUE NOT NULL,"
        "stream_id TEXT NOT NULL,through_sequence INTEGER NOT NULL,through_event_id TEXT,"
        "through_event_digest TEXT,state_schema_version TEXT NOT NULL,state_json TEXT NOT NULL,"
        "state_digest TEXT NOT NULL,manifest_json TEXT NOT NULL,manifest_digest TEXT NOT NULL,"
        "snapshot_digest TEXT NOT NULL)"
    ),
}
_TRIGGER_DDL = {
    "metadata_no_update": "CREATE TRIGGER metadata_no_update BEFORE UPDATE ON metadata BEGIN SELECT RAISE(ABORT,'append-only metadata'); END",
    "metadata_no_delete": "CREATE TRIGGER metadata_no_delete BEFORE DELETE ON metadata BEGIN SELECT RAISE(ABORT,'append-only metadata'); END",
    "migrations_no_update": "CREATE TRIGGER migrations_no_update BEFORE UPDATE ON migrations BEGIN SELECT RAISE(ABORT,'append-only migrations'); END",
    "migrations_no_delete": "CREATE TRIGGER migrations_no_delete BEFORE DELETE ON migrations BEGIN SELECT RAISE(ABORT,'append-only migrations'); END",
    "events_no_update": "CREATE TRIGGER events_no_update BEFORE UPDATE ON events BEGIN SELECT RAISE(ABORT,'append-only events'); END",
    "events_no_delete": "CREATE TRIGGER events_no_delete BEFORE DELETE ON events BEGIN SELECT RAISE(ABORT,'append-only events'); END",
    "snapshots_no_update": "CREATE TRIGGER snapshots_no_update BEFORE UPDATE ON snapshots BEGIN SELECT RAISE(ABORT,'append-only snapshots'); END",
    "snapshots_no_delete": "CREATE TRIGGER snapshots_no_delete BEFORE DELETE ON snapshots BEGIN SELECT RAISE(ABORT,'append-only snapshots'); END",
}
_MIGRATION = _canon(
    {
        "migration_id": "m2-a.initial.v1",
        "from_version": None,
        "to_version": STORE_SCHEMA_VERSION,
        "tables": ["events", "metadata", "migrations", "snapshots"],
        "append_only": True,
    },
    "migration_manifest",
)
_MIGRATION_DIGEST = _sha(_MIGRATION)


def _event_material(envelope: EventEnvelope) -> dict[str, Any]:
    return {
        "authority": envelope.authority,
        "causal_context_json": envelope.causal_context_json,
        "causation_id": envelope.causation_id,
        "correlation_id": envelope.correlation_id,
        "event_id": envelope.event_id,
        "event_type": envelope.event_type,
        "payload_json": envelope.payload_json,
        "producer": envelope.producer,
        "producer_version": envelope.producer_version,
        "schema_version": envelope.schema_version,
        "sequence": envelope.sequence,
        "stream_id": envelope.stream_id,
    }


def _event_from_json(text: str, expected_digest: str) -> EventEnvelope:
    try:
        value = json.loads(text)
        if not isinstance(value, dict) or _canon(value, "persisted_event") != text:
            raise PersistedEventCorruption("persisted event is not canonical")
        envelope = EventEnvelope(**value)
    except (json.JSONDecodeError, TypeError, ValueError, InvalidEventEnvelope) as exc:
        raise PersistedEventCorruption("persisted event is malformed") from exc
    if envelope.authority != SHADOW_AUTHORITY or envelope.digest != expected_digest:
        raise PersistedEventCorruption("persisted event digest or authority mismatch")
    return envelope


class SQLiteShadowStore:
    """Explicit, disconnected, append-only SQLite store for M2-A evidence."""

    def __init__(self, database_path: str | Path, *, policy: ShadowStoragePolicy | None = None) -> None:
        self._path = Path(database_path)
        if str(database_path) == ":memory:" or self._path.name == "":
            raise ValueError("M2-A requires a concrete SQLite file path")
        self._policy = policy or ShadowStoragePolicy()
        self._initialized = False

    @property
    def database_path(self) -> Path:
        return self._path

    def initialize(self) -> InitializationReport:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect(require_initialized=False, validate_schema=False) as connection:
            journal = str(connection.execute("PRAGMA journal_mode=WAL").fetchone()[0]).lower()
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute("PRAGMA wal_autocheckpoint=1000")
            objects = connection.execute(
                "SELECT COUNT(*) FROM sqlite_schema WHERE name NOT LIKE 'sqlite_%'"
            ).fetchone()[0]
            if int(objects) == 0:
                self._initialize_empty_database(connection)
            self._validate_schema(connection)
        self._initialized = True
        return InitializationReport(str(self._path), journal, journal == "wal")

    def _initialize_empty_database(self, connection: sqlite3.Connection) -> None:
        connection.execute("BEGIN IMMEDIATE")
        try:
            for statement in _TABLE_DDL.values():
                connection.execute(statement)
            for statement in _TRIGGER_DDL.values():
                connection.execute(statement)
            self._insert_initial_records(connection)
            connection.commit()
        except Exception:
            connection.rollback()
            raise

    def _insert_initial_records(self, connection: sqlite3.Connection) -> None:
        metadata = {
            "authority": SHADOW_AUTHORITY,
            "event_schema_version": EventEnvelope.__dataclass_fields__["schema_version"].default,
            "snapshot_schema_version": SNAPSHOT_SCHEMA_VERSION,
            "store_schema_version": STORE_SCHEMA_VERSION,
        }
        connection.executemany("INSERT INTO metadata(key,value) VALUES(?,?)", sorted(metadata.items()))
        connection.execute(
            "INSERT INTO migrations VALUES(1,?,?,?,?)",
            ("m2-a.initial.v1", MIGRATION_SCHEMA_VERSION, _MIGRATION, _MIGRATION_DIGEST),
        )

    @contextmanager
    def _connect(
        self,
        *,
        require_initialized: bool = True,
        validate_schema: bool = True,
    ) -> Iterator[sqlite3.Connection]:
        if require_initialized and not self._initialized:
            raise StoreNotInitialized("store requires explicit initialize()")
        connection = sqlite3.connect(self._path, isolation_level=None, timeout=5.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA wal_autocheckpoint=1000")
        connection.execute("PRAGMA busy_timeout=5000")
        try:
            if validate_schema:
                self._validate_schema(connection)
            yield connection
        finally:
            connection.close()

    @staticmethod
    def _validate_schema(connection: sqlite3.Connection) -> None:
        table_rows = connection.execute(
            "SELECT name,sql FROM sqlite_schema WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
        trigger_rows = connection.execute(
            "SELECT name,sql FROM sqlite_schema WHERE type='trigger'"
        ).fetchall()
        actual_tables = {str(row["name"]): _normalize_sql(str(row["sql"])) for row in table_rows}
        actual_triggers = {str(row["name"]): _normalize_sql(str(row["sql"])) for row in trigger_rows}
        expected_tables = {name: _normalize_sql(sql) for name, sql in _TABLE_DDL.items()}
        expected_triggers = {name: _normalize_sql(sql) for name, sql in _TRIGGER_DDL.items()}
        if actual_tables != expected_tables:
            raise SchemaMismatch("table schema differs from M2-A v1")
        if actual_triggers != expected_triggers:
            raise SchemaMismatch("append-only trigger schema differs from M2-A v1")
        metadata = dict(connection.execute("SELECT key,value FROM metadata"))
        if metadata != {
            "authority": SHADOW_AUTHORITY,
            "event_schema_version": EventEnvelope.__dataclass_fields__["schema_version"].default,
            "snapshot_schema_version": SNAPSHOT_SCHEMA_VERSION,
            "store_schema_version": STORE_SCHEMA_VERSION,
        }:
            raise SchemaMismatch("metadata differs from M2-A v1")
        rows = connection.execute("SELECT * FROM migrations ORDER BY ordinal").fetchall()
        expected_migration = (1, "m2-a.initial.v1", MIGRATION_SCHEMA_VERSION, _MIGRATION, _MIGRATION_DIGEST)
        if len(rows) != 1 or tuple(rows[0]) != expected_migration:
            raise SchemaMismatch("migration history differs from M2-A v1")

    @staticmethod
    def _event_from_row(row: sqlite3.Row) -> EventEnvelope:
        try:
            envelope = _event_from_json(str(row["event_json"]), str(row["envelope_digest"]))
            actual = (str(row["event_id"]), str(row["stream_id"]), int(row["sequence"]))
            expected = (envelope.event_id, envelope.stream_id, envelope.sequence)
            if actual != expected:
                raise PersistedEventCorruption("event index columns disagree with canonical envelope")
            if int(row["event_bytes"]) != len(str(row["event_json"]).encode("utf-8")):
                raise PersistedEventCorruption("event byte count mismatch")
            return envelope
        except (KeyError, TypeError, ValueError) as exc:
            raise PersistedEventCorruption("persisted event row is malformed") from exc

    def _event_state(
        self,
        connection: sqlite3.Connection,
    ) -> tuple[int, int, str, set[str], dict[str, int]]:
        rows = connection.execute("SELECT * FROM events ORDER BY ordinal").fetchall()
        chain = GENESIS_DIGEST
        known: set[str] = set()
        stream_sequences: dict[str, int] = {}
        total_bytes = 0
        for expected_ordinal, row in enumerate(rows, 1):
            if int(row["ordinal"]) != expected_ordinal or str(row["previous_chain_digest"]) != chain:
                raise PersistedEventCorruption("ordinal or previous chain mismatch")
            envelope = self._event_from_row(row)
            expected_sequence = stream_sequences.get(envelope.stream_id, 0) + 1
            if envelope.sequence != expected_sequence:
                raise PersistedEventCorruption("stream sequence mismatch")
            if envelope.causation_id is not None and envelope.causation_id not in known:
                raise PersistedEventCorruption("causation points outside prior history")
            expected_chain = _digest(
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

    def append(self, envelope: EventEnvelope) -> AppendReceipt:
        return self.append_many((envelope,))[0]

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
                materials: list[tuple[EventEnvelope, str, int]] = []
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
                    text = _canon(_event_material(envelope), "event_material")
                    materials.append((envelope, text, len(text.encode("utf-8"))))
                    batch_ids.add(envelope.event_id)
                    last_seq[envelope.stream_id] = envelope.sequence
                total_bytes = sum(item[2] for item in materials)
                if (
                    before_count + len(materials) > self._policy.max_event_count
                    or before_bytes + total_bytes > self._policy.max_event_bytes
                ):
                    raise StoragePolicyExceeded("event append exceeds bounded storage policy")

                receipts: list[AppendReceipt] = []
                count = before_count
                for envelope, text, event_bytes in materials:
                    ordinal = count + 1
                    next_chain = _digest(
                        {
                            "envelope_digest": envelope.digest,
                            "ordinal": ordinal,
                            "previous_chain_digest": chain,
                        },
                        "event_chain",
                    )
                    connection.execute(
                        "INSERT INTO events(ordinal,event_id,stream_id,sequence,event_json,envelope_digest,event_bytes,previous_chain_digest,chain_digest) "
                        "VALUES(?,?,?,?,?,?,?,?,?)",
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
                    if row is None or self._event_from_row(row) != envelope:
                        raise PersistedEventCorruption("event readback failed before commit")
                    if str(row["previous_chain_digest"]) != chain or str(row["chain_digest"]) != next_chain:
                        raise PersistedEventCorruption("event chain readback failed before commit")
                    transition = _digest(
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
                    "SELECT * FROM events WHERE stream_id=? AND sequence>? ORDER BY sequence",
                    (stream_id, after_sequence),
                ).fetchall()
            return tuple(self._event_from_row(row) for row in rows)

    def snapshot_due(self, stream_id: str) -> bool:
        with self._connect() as connection:
            self._event_state(connection)
            head = connection.execute(
                "SELECT COALESCE(MAX(sequence),0) FROM events WHERE stream_id=?", (stream_id,)
            ).fetchone()[0]
            snap = connection.execute(
                "SELECT COALESCE(MAX(through_sequence),0) FROM snapshots WHERE stream_id=?", (stream_id,)
            ).fetchone()[0]
        return int(head) - int(snap) >= self._policy.snapshot_interval_events

    def write_snapshot(
        self,
        *,
        snapshot_id: str,
        stream_id: str,
        through_sequence: int,
        state: Mapping[str, Any],
        state_schema_version: str,
    ) -> SnapshotReceipt:
        for field, value in (
            ("snapshot_id", snapshot_id),
            ("stream_id", stream_id),
            ("state_schema_version", state_schema_version),
        ):
            if not isinstance(value, str) or not value.strip():
                raise SnapshotCorruption(f"{field} must be a non-empty string")
        if isinstance(through_sequence, bool) or not isinstance(through_sequence, int) or through_sequence < 0:
            raise SnapshotCorruption("through_sequence must be a non-negative integer")
        if not isinstance(state, Mapping):
            raise SnapshotCorruption("state must be a mapping")
        try:
            state_json = _canon(state, "snapshot_state")
        except (TypeError, ValueError, InvalidEventEnvelope) as exc:
            raise SnapshotCorruption("state must be canonical JSON-compatible mapping") from exc
        state_digest = _sha(state_json)
        state_object = json.loads(state_json)
        manifest_json = _canon(
            {
                "canonical_bytes": len(state_json.encode("utf-8")),
                "hash_algorithm": "sha256",
                "serialization_schema": STATE_ENCODING_VERSION,
                "state_schema_version": state_schema_version,
                "top_level_key_count": len(state_object),
                "top_level_keys": sorted(state_object),
            },
            "snapshot_manifest",
        )
        manifest_digest = _sha(manifest_json)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                self._event_state(connection)
                count = int(connection.execute("SELECT COUNT(*) FROM snapshots").fetchone()[0])
                if count >= self._policy.max_snapshot_count:
                    raise StoragePolicyExceeded("snapshot count exceeds bounded policy")
                head = connection.execute(
                    "SELECT * FROM events WHERE stream_id=? ORDER BY sequence DESC LIMIT 1", (stream_id,)
                ).fetchone()
                if head is None:
                    if through_sequence != 0:
                        raise SnapshotCorruption("empty stream snapshot must bind sequence zero")
                    event_id = event_digest = None
                else:
                    envelope = self._event_from_row(head)
                    if through_sequence != envelope.sequence:
                        raise SnapshotCorruption("snapshot must bind the current stream head")
                    event_id, event_digest = envelope.event_id, envelope.digest
                snapshot_digest = _digest(
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
                connection.execute(
                    "INSERT INTO snapshots(ordinal,snapshot_id,stream_id,through_sequence,through_event_id,through_event_digest,state_schema_version,state_json,state_digest,manifest_json,manifest_digest,snapshot_digest) "
                    "VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
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
                row = connection.execute("SELECT * FROM snapshots WHERE ordinal=?", (ordinal,)).fetchone()
                persisted = self._snapshot_from_row(connection, row)
                transition = _digest(
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

    def latest_valid_snapshot(self, stream_id: str) -> SnapshotSelection:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM snapshots WHERE stream_id=? ORDER BY ordinal DESC", (stream_id,)
            ).fetchall()
            rejected: list[str] = []
            selected = None
            for row in rows:
                try:
                    selected = self._snapshot_from_row(connection, row)
                    break
                except SnapshotCorruption:
                    rejected.append(str(row["snapshot_id"]))
            material = {
                "rejected_snapshot_ids": rejected,
                "selected_snapshot_id": None if selected is None else selected.snapshot_id,
            }
            return SnapshotSelection(
                selected,
                tuple(rejected),
                bool(rejected and selected is not None),
                _digest(material, "snapshot_selection"),
            )

    def _snapshot_from_row(self, connection: sqlite3.Connection, row: sqlite3.Row | None) -> Snapshot:
        if row is None:
            raise SnapshotCorruption("snapshot row is missing")
        try:
            snap = Snapshot(
                int(row["ordinal"]),
                str(row["snapshot_id"]),
                str(row["stream_id"]),
                int(row["through_sequence"]),
                None if row["through_event_id"] is None else str(row["through_event_id"]),
                None if row["through_event_digest"] is None else str(row["through_event_digest"]),
                str(row["state_schema_version"]),
                str(row["state_json"]),
                str(row["state_digest"]),
                str(row["manifest_json"]),
                str(row["manifest_digest"]),
                str(row["snapshot_digest"]),
            )
            state = json.loads(snap.state_json)
            manifest = json.loads(snap.manifest_json)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise SnapshotCorruption("snapshot row is malformed") from exc
        if not isinstance(state, dict) or not isinstance(manifest, dict):
            raise SnapshotCorruption("snapshot JSON must encode objects")
        if _canon(state, "snapshot_state") != snap.state_json or _sha(snap.state_json) != snap.state_digest:
            raise SnapshotCorruption("snapshot state is noncanonical or corrupt")
        expected_manifest = {
            "canonical_bytes": len(snap.state_json.encode("utf-8")),
            "hash_algorithm": "sha256",
            "serialization_schema": STATE_ENCODING_VERSION,
            "state_schema_version": snap.state_schema_version,
            "top_level_key_count": len(state),
            "top_level_keys": sorted(state),
        }
        if (
            _canon(expected_manifest, "snapshot_manifest") != snap.manifest_json
            or _sha(snap.manifest_json) != snap.manifest_digest
        ):
            raise SnapshotCorruption("snapshot manifest is corrupt")
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
            try:
                envelope = self._event_from_row(event_row)
            except PersistedEventCorruption as exc:
                raise SnapshotCorruption("snapshot boundary event is corrupt") from exc
            if (snap.through_event_id, snap.through_event_digest) != (envelope.event_id, envelope.digest):
                raise SnapshotCorruption("snapshot boundary event mismatch")
        expected_digest = _digest(
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
        selection = self.latest_valid_snapshot(stream_id)
        if selection.selected is None:
            start_mapping = state_to_mapping(initial_state)
            after, snapshot_id = 0, None
        else:
            start_mapping = selection.selected.state
            after = selection.selected.through_sequence
            snapshot_id = selection.selected.snapshot_id
        try:
            start_json = _canon(start_mapping, "restore_start_state")
        except (TypeError, ValueError, InvalidEventEnvelope) as exc:
            raise RestoreVerificationError("restore start state must be a canonical mapping") from exc
        events = self.events(stream_id=stream_id, after_sequence=after)

        def replay() -> tuple[StateT, str]:
            decoded = json.loads(start_json)
            if not isinstance(decoded, dict):
                raise RestoreVerificationError("restore start state must be an object")
            state = state_from_mapping(decoded)
            for event in events:
                state = reducer(state, event)
                if state is None:
                    raise RestoreVerificationError("reducer returned None")
            try:
                digest = _sha(_canon(state_to_mapping(state), "restored_state"))
            except (TypeError, ValueError, InvalidEventEnvelope) as exc:
                raise RestoreVerificationError("restored state must be a canonical mapping") from exc
            return state, digest

        left, left_digest = replay()
        _right, right_digest = replay()
        if left_digest != right_digest:
            raise RestoreVerificationError("repeated replay produced different state")
        transition = _digest(
            {
                "final_state_digest": left_digest,
                "replayed_event_count": len(events),
                "selection_digest": selection.selection_digest,
                "stream_id": stream_id,
            },
            "restore_transition",
        )
        return RestoreReport(
            left,
            left_digest,
            right_digest,
            len(events),
            snapshot_id,
            selection.rejected_snapshot_ids,
            transition,
            True,
        )

    @classmethod
    def _integrity_report_for_connection(
        cls,
        connection: sqlite3.Connection,
        policy: ShadowStoragePolicy,
    ) -> IntegrityReport:
        errors: list[str] = []
        event_count = snapshot_count = 0
        chain = GENESIS_DIGEST
        result = str(connection.execute("PRAGMA integrity_check").fetchone()[0])
        if result != "ok":
            errors.append(f"sqlite_integrity:{result}")
        try:
            cls._validate_schema(connection)
        except SchemaMismatch as exc:
            errors.append(f"schema:{exc}")
        if not errors:
            helper = object.__new__(cls)
            helper._policy = policy
            try:
                event_count, total_bytes, chain, _known, _streams = helper._event_state(connection)
            except PersistedEventCorruption as exc:
                errors.append(f"event:{exc}")
                event_count = int(connection.execute("SELECT COUNT(*) FROM events").fetchone()[0])
                total_bytes = 0
            if event_count > policy.max_event_count or total_bytes > policy.max_event_bytes:
                errors.append("storage_policy:event_limit")
            snapshots = connection.execute("SELECT * FROM snapshots ORDER BY ordinal").fetchall()
            snapshot_count = len(snapshots)
            if snapshot_count > policy.max_snapshot_count:
                errors.append("storage_policy:snapshot_limit")
            for expected_ordinal, row in enumerate(snapshots, 1):
                try:
                    if int(row["ordinal"]) != expected_ordinal:
                        raise SnapshotCorruption("snapshot ordinal mismatch")
                    helper._snapshot_from_row(connection, row)
                except SnapshotCorruption as exc:
                    errors.append(f"snapshot:{row['snapshot_id']}:{exc}")
        material = {
            "chain_head_digest": chain,
            "errors": errors,
            "event_count": event_count,
            "schema_version": INTEGRITY_SCHEMA_VERSION,
            "snapshot_count": snapshot_count,
        }
        return IntegrityReport(
            not errors,
            tuple(errors),
            event_count,
            snapshot_count,
            chain,
            _digest(material, "integrity_report"),
        )

    def integrity_check(self) -> IntegrityReport:
        try:
            with self._connect(validate_schema=False) as connection:
                return self._integrity_report_for_connection(connection, self._policy)
        except (sqlite3.DatabaseError, ShadowPersistenceError) as exc:
            errors = (f"sqlite:{type(exc).__name__}:{exc}",)
            material = {
                "chain_head_digest": GENESIS_DIGEST,
                "errors": list(errors),
                "event_count": 0,
                "schema_version": INTEGRITY_SCHEMA_VERSION,
                "snapshot_count": 0,
            }
            return IntegrityReport(False, errors, 0, 0, GENESIS_DIGEST, _digest(material, "integrity_report"))

    @classmethod
    def _verify_database_path(cls, path: Path, policy: ShadowStoragePolicy) -> IntegrityReport:
        uri = path.resolve().as_uri() + "?mode=ro"
        connection = sqlite3.connect(uri, uri=True, isolation_level=None, timeout=5.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only=ON")
        connection.execute("PRAGMA busy_timeout=5000")
        try:
            return cls._integrity_report_for_connection(connection, policy)
        finally:
            connection.close()

    @staticmethod
    def _remove_sqlite_sidecars(path: Path) -> None:
        for suffix in ("-wal", "-shm", "-journal"):
            Path(f"{path}{suffix}").unlink(missing_ok=True)

    def create_backup(self, backup_directory: str | Path, *, backup_ordinal: int) -> BackupReceipt:
        if not self._initialized:
            raise StoreNotInitialized("store requires explicit initialize()")
        if (
            isinstance(backup_ordinal, bool)
            or not isinstance(backup_ordinal, int)
            or not 1 <= backup_ordinal <= 99_999_999
        ):
            raise BackupPolicyError("backup_ordinal must be 1..99999999")
        directory = Path(backup_directory)
        directory.mkdir(parents=True, exist_ok=True)
        existing = sorted(
            (int(match.group(1)), path)
            for path in directory.iterdir()
            if path.is_file() and (match := _BACKUP.fullmatch(path.name)) is not None
        )
        if existing and backup_ordinal <= existing[-1][0]:
            raise BackupPolicyError("backup_ordinal must increase monotonically")
        target = directory / f"shadow-backup-{backup_ordinal:08d}.sqlite3"
        temporary = directory / f".{target.name}.partial"
        if target.exists() or temporary.exists():
            raise BackupPolicyError("backup target already exists")
        try:
            with self._connect() as source:
                source_report = self._integrity_report_for_connection(source, self._policy)
                if not source_report.valid:
                    raise BackupPolicyError("source logical integrity verification failed")
                destination = sqlite3.connect(temporary, isolation_level=None)
                try:
                    destination.execute("PRAGMA synchronous=FULL")
                    source.backup(destination)
                    destination.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                    destination.execute("PRAGMA journal_mode=DELETE")
                finally:
                    destination.close()
            backup_report = self._verify_database_path(temporary, self._policy)
            self._remove_sqlite_sidecars(temporary)
            if not backup_report.valid:
                raise BackupPolicyError("backup logical integrity verification failed")
            temporary.replace(target)
        except Exception:
            temporary.unlink(missing_ok=True)
            self._remove_sqlite_sidecars(temporary)
            raise
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        candidates = sorted(
            (int(match.group(1)), path)
            for path in directory.iterdir()
            if path.is_file() and (match := _BACKUP.fullmatch(path.name)) is not None
        )
        pruned: list[str] = []
        while len(candidates) > self._policy.max_backups:
            _ordinal, old = candidates.pop(0)
            old.unlink()
            pruned.append(old.name)
        return BackupReceipt(str(target), backup_ordinal, digest, tuple(pruned), True)
