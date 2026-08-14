"""Fail-closed authoritative persistence for accepted EVE event history.

This store is deliberately separate from ``SQLiteShadowStore``.  It uses a
different environment variable, file, schema, tables, and authority label.
The existing ``EventEnvelope`` remains unchanged so replay keeps the event
kernel's validation and reducer semantics.

Durability protocol
-------------------

1. A canonical candidate row is committed with ``synchronous=FULL``.  It is
   not accepted history and may be removed after a restart only when its
   content, ordinal, predecessor, stream sequence, and causation are proven.
2. In a second ``BEGIN IMMEDIATE`` transaction the candidate is copied into
   the append-only authoritative log, an append-only accepted-tail record is
   written, and the candidate is removed.  The event and tail therefore
   become durable together or not at all.
3. Startup verifies the complete accepted event and tail chains before any
   residue cleanup or read is allowed.  Accepted corruption and ambiguity are
   never repaired or truncated; callers must terminate with exit code 86.

SQLite WAL is requested and verified.  A verified ``DELETE`` rollback journal
is the only controlled fallback.  Every connection verifies
``PRAGMA synchronous=FULL``.  An adjacent OS advisory lock permits exactly one
authoritative writer process at a time.  SQLite extended result codes and the
exact Windows sharing/lock-violation codes distinguish approved transient I/O
from integrity or unprovable failures.  Approved transient operations use a
bounded exponential retry; exhaustion still fails closed with exit code 86.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, TypeVar

from core.event_kernel import (
    EventEnvelope,
    InMemoryEventKernel,
    InvalidEventEnvelope,
    SHADOW_AUTHORITY,
    canonical_json_object,
)

AUTHORITY_FAILURE_EXIT_CODE = 86
AUTHORITY_PATH_ENV = "EVE_AUTHORITY_PATH"
SHADOW_PATH_ENV = "EVE_SQLITE_SHADOW_PATH"
AUTHORITY_LABEL = "authoritative_accepted_history"
STORE_SCHEMA_VERSION = "eve.authoritative-store.v1"
EVENT_SERIALIZATION_VERSION = "eve.authoritative-event.v1"
TAIL_SCHEMA_VERSION = "eve.accepted-tail.v1"
MIGRATION_SCHEMA_VERSION = "eve.authoritative-migration.v1"
APPEND_RECEIPT_SCHEMA_VERSION = "eve.authoritative-append-receipt.v1"
VERIFICATION_SCHEMA_VERSION = "eve.authoritative-verification.v1"
STARTUP_SCHEMA_VERSION = "eve.authoritative-startup.v1"
GENESIS_HASH = "0" * 64
DEFAULT_TRANSIENT_MAX_ATTEMPTS = 4
DEFAULT_TRANSIENT_BACKOFF_INITIAL_SECONDS = 0.25
DEFAULT_TRANSIENT_BACKOFF_MAX_SECONDS = 2.0

_LOGGER = logging.getLogger(__name__)
_SQLITE_PRIMARY_MASK = 0xFF
_WINDOWS_TRANSIENT_ERROR_CODES = frozenset({32, 33})


def _sqlite_ioerr_extended(subcode: int) -> int:
    """Build SQLite's stable ``SQLITE_IOERR | (subcode << 8)`` value."""

    return int(sqlite3.SQLITE_IOERR) | (subcode << 8)


_SQLITE_TRANSIENT_IOERR_CODES = frozenset(
    _sqlite_ioerr_extended(subcode)
    for subcode in (
        1,  # READ
        3,  # WRITE
        4,  # FSYNC
        5,  # DIR_FSYNC
        6,  # TRUNCATE
        7,  # FSTAT
        8,  # UNLOCK
        9,  # RDLOCK
        10,  # DELETE
        11,  # BLOCKED
        13,  # ACCESS
        14,  # CHECKRESERVEDLOCK
        15,  # LOCK
        16,  # CLOSE
        17,  # DIR_CLOSE
        18,  # SHMOPEN
        19,  # SHMSIZE
        20,  # SHMLOCK
        21,  # SHMMAP
        22,  # SEEK
        23,  # DELETE_NOENT
        24,  # MMAP
        25,  # GETTEMPPATH
        26,  # CONVPATH
    )
)
_SQLITE_INTEGRITY_IOERR_CODES = frozenset(
    {
        _sqlite_ioerr_extended(2),  # SHORT_READ
        _sqlite_ioerr_extended(32),  # DATA
        _sqlite_ioerr_extended(33),  # CORRUPTFS
    }
)

FAULT_BEFORE_EVENT_APPEND = "before_event_append"
FAULT_AFTER_EVENT_ROW_WRITE_BEFORE_COMMIT = "after_event_row_write_before_commit"
FAULT_AFTER_EVENT_TRANSACTION_COMMIT = "after_event_transaction_commit"
FAULT_BEFORE_ACCEPTED_TAIL_UPDATE = "before_accepted_tail_update"
FAULT_DURING_ACCEPTED_TAIL_UPDATE = "during_accepted_tail_update"
FAULT_AFTER_ACCEPTED_TAIL_UPDATE = "after_accepted_tail_update"
FAULT_POINTS = (
    FAULT_BEFORE_EVENT_APPEND,
    FAULT_AFTER_EVENT_ROW_WRITE_BEFORE_COMMIT,
    FAULT_AFTER_EVENT_TRANSACTION_COMMIT,
    FAULT_BEFORE_ACCEPTED_TAIL_UPDATE,
    FAULT_DURING_ACCEPTED_TAIL_UPDATE,
    FAULT_AFTER_ACCEPTED_TAIL_UPDATE,
)

StateT = TypeVar("StateT")
FaultInjector = Callable[[str], None]
WalProbe = Callable[[sqlite3.Connection], str]
RetryObserver = Callable[[Mapping[str, Any]], None]
RetrySleeper = Callable[[float], None]


class AuthorityPersistenceError(RuntimeError):
    """Base class for authoritative persistence failures."""


class AuthorityUnprovable(AuthorityPersistenceError):
    """Accepted history cannot be proven and the process must exit 86."""


class AuthorityAmbiguity(AuthorityUnprovable):
    """More than one defensible authority state is possible."""


class AuthorityBusy(AuthorityUnprovable):
    """Another authoritative writer holds the exclusive process lock."""


class AuthorityNotOpen(AuthorityPersistenceError):
    """The explicit startup/open protocol has not completed."""


class AuthorityAppendRejected(AuthorityPersistenceError):
    """A proposed append violates the accepted-history contract."""


class InjectedAuthorityFault(RuntimeError):
    """Deterministic test-only interruption raised by a fault injector."""


@dataclass(frozen=True, slots=True)
class AuthorityFailureClassification:
    """Stable evidence explaining whether an authority failure may be retried."""

    classification: str
    source: str
    reason: str
    sqlite_extended_errcode: int | None = None
    sqlite_primary_errcode: int | None = None
    sqlite_errorname: str | None = None
    windows_error_code: int | None = None

    def record(self) -> dict[str, Any]:
        return {
            "classification": self.classification,
            "source": self.source,
            "reason": self.reason,
            "sqlite_extended_errcode": self.sqlite_extended_errcode,
            "sqlite_primary_errcode": self.sqlite_primary_errcode,
            "sqlite_errorname": self.sqlite_errorname,
            "windows_error_code": self.windows_error_code,
        }


def classify_authority_failure(exc: BaseException) -> AuthorityFailureClassification:
    """Classify using SQLite's extended code or the exact Windows sharing code."""

    if isinstance(exc, AuthorityBusy):
        cause = exc.__cause__
        windows_error = getattr(cause, "winerror", None)
        return AuthorityFailureClassification(
            "TRANSIENT",
            "writer_lock",
            "the single-writer advisory lock is currently held",
            windows_error_code=windows_error if isinstance(windows_error, int) else None,
        )
    if isinstance(exc, AuthorityUnprovable):
        return AuthorityFailureClassification(
            "INTEGRITY",
            "authority_verification",
            str(exc),
        )
    if isinstance(exc, sqlite3.Error):
        code = getattr(exc, "sqlite_errorcode", None)
        extended = int(code) if isinstance(code, int) else None
        primary = extended & _SQLITE_PRIMARY_MASK if extended is not None else None
        name_value = getattr(exc, "sqlite_errorname", None)
        name = str(name_value) if name_value else None
        if primary in {sqlite3.SQLITE_CORRUPT, sqlite3.SQLITE_NOTADB}:
            classification = "INTEGRITY"
            reason = "SQLite reports corrupt or non-database content"
        elif extended in _SQLITE_INTEGRITY_IOERR_CODES:
            classification = "INTEGRITY"
            reason = (
                "SQLite extended I/O code reports truncated, checksum-invalid, "
                "or corrupt storage"
            )
        elif primary in {sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED}:
            classification = "TRANSIENT"
            reason = "SQLite reports a temporary busy or locked condition"
        elif extended in _SQLITE_TRANSIENT_IOERR_CODES:
            classification = "TRANSIENT"
            reason = "SQLite extended I/O code identifies a retryable operating-system operation"
        elif primary == sqlite3.SQLITE_IOERR and extended == sqlite3.SQLITE_IOERR:
            classification = "UNPROVABLE"
            reason = (
                "SQLite supplied only primary SQLITE_IOERR; the required extended "
                "subtype is unavailable"
            )
        elif primary == sqlite3.SQLITE_IOERR:
            classification = "UNPROVABLE"
            reason = "SQLite extended I/O subtype is not on the approved transient whitelist"
        else:
            classification = "UNPROVABLE"
            reason = (
                "SQLite error is neither an approved transient code nor a proven "
                "integrity code"
            )
        return AuthorityFailureClassification(
            classification,
            "sqlite",
            reason,
            sqlite_extended_errcode=extended,
            sqlite_primary_errcode=primary,
            sqlite_errorname=name,
        )
    if isinstance(exc, OSError):
        windows_error = getattr(exc, "winerror", None)
        if isinstance(windows_error, int) and windows_error in _WINDOWS_TRANSIENT_ERROR_CODES:
            return AuthorityFailureClassification(
                "TRANSIENT",
                "windows",
                "Windows reports a temporary sharing or lock violation",
                windows_error_code=windows_error,
            )
        return AuthorityFailureClassification(
            "UNPROVABLE",
            "os",
            "operating-system failure is not an approved Windows transient code",
            windows_error_code=windows_error if isinstance(windows_error, int) else None,
        )
    return AuthorityFailureClassification(
        "UNPROVABLE",
        "python",
        "failure type has no approved transient classification",
    )


def _handle_authority_retry(
    exc: AuthorityBusy | AuthorityUnprovable | sqlite3.Error | OSError,
    *,
    operation_name: str,
    attempt: int,
    max_attempts: int,
    backoff_initial: float,
    backoff_max: float,
    observer: RetryObserver | None = None,
    sleeper: RetrySleeper = time.sleep,
) -> None:
    """Log one classification, sleep only for an approved pending retry, or raise."""

    if max_attempts < 1:
        raise ValueError("transient max attempts must be at least one")
    if backoff_initial <= 0 or backoff_max < backoff_initial:
        raise ValueError("transient backoff bounds are invalid")
    classification = classify_authority_failure(exc)
    transient = classification.classification == "TRANSIENT"
    exhausted = transient and attempt >= max_attempts
    delay = min(backoff_initial * (2 ** (attempt - 1)), backoff_max)
    record = {
        "event": "authority_failure_classified",
        "operation": operation_name,
        "attempt": attempt,
        "max_attempts": max_attempts,
        "retry_scheduled": transient and not exhausted,
        "retry_exhausted": exhausted,
        "next_backoff_seconds": delay if transient and not exhausted else None,
        **classification.record(),
    }
    _LOGGER.warning(
        "authority_failure_classified %s",
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
    )
    if observer is not None:
        observer(record)
    if not transient:
        if isinstance(exc, AuthorityUnprovable):
            raise exc
        raise AuthorityUnprovable(
            f"{operation_name} failed closed: {classification.reason}"
        ) from exc
    if exhausted:
        message = (
            f"{operation_name} transient retry limit exhausted after {attempt} attempts; "
            f"source={classification.source}; "
            f"sqlite_extended_errcode={classification.sqlite_extended_errcode}; "
            f"windows_error_code={classification.windows_error_code}"
        )
        if isinstance(exc, AuthorityBusy):
            raise AuthorityBusy(message) from exc
        raise AuthorityUnprovable(message) from exc
    sleeper(delay)


def retry_authority_operation(
    operation: Callable[[int], StateT],
    *,
    operation_name: str,
    max_attempts: int = DEFAULT_TRANSIENT_MAX_ATTEMPTS,
    backoff_initial: float = DEFAULT_TRANSIENT_BACKOFF_INITIAL_SECONDS,
    backoff_max: float = DEFAULT_TRANSIENT_BACKOFF_MAX_SECONDS,
    observer: RetryObserver | None = None,
    sleeper: RetrySleeper = time.sleep,
) -> StateT:
    """Run a retry-safe operation with bounded, evidence-producing backoff."""

    for attempt in range(1, max_attempts + 1):
        try:
            return operation(attempt)
        except (AuthorityBusy, AuthorityUnprovable, sqlite3.Error, OSError) as exc:
            _handle_authority_retry(
                exc,
                operation_name=operation_name,
                attempt=attempt,
                max_attempts=max_attempts,
                backoff_initial=backoff_initial,
                backoff_max=backoff_max,
                observer=observer,
                sleeper=sleeper,
            )
    raise AssertionError("bounded authority retry loop did not terminate")


def _canonical(value: Mapping[str, Any], *, field: str) -> str:
    return canonical_json_object(value, field=field)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _digest(value: Mapping[str, Any], *, field: str) -> str:
    return _sha256(_canonical(value, field=field).encode("utf-8"))


def _canonical_sort_key(value: str) -> str:
    return " ".join(value.strip().rstrip(";").split())


def _type_exact_equal(left: Path, right: Path) -> bool:
    return os.path.normcase(os.path.abspath(left)) == os.path.normcase(os.path.abspath(right))


def load(environ: Mapping[str, str] | None = None) -> Path:
    """Resolve the explicit authority path and reject the M2 shadow path."""

    source = os.environ if environ is None else environ
    raw = source.get(AUTHORITY_PATH_ENV, "").strip()
    if not raw:
        raise AuthorityPersistenceError(f"{AUTHORITY_PATH_ENV} is required")
    authority = Path(raw).expanduser()
    if str(authority) == ":memory:" or authority.name == "":
        raise AuthorityPersistenceError("authority requires a concrete SQLite file")
    shadow_raw = source.get(SHADOW_PATH_ENV, "").strip()
    if shadow_raw and _type_exact_equal(authority, Path(shadow_raw).expanduser()):
        raise AuthorityPersistenceError("authority path must differ from the shadow path")
    return authority


def _base_payload(envelope: EventEnvelope) -> dict[str, Any]:
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


def canonical_record(envelope: EventEnvelope) -> bytes:
    value = {
        "envelope": _base_payload(envelope),
        "schema": EVENT_SERIALIZATION_VERSION,
    }
    return _canonical(value, field="authoritative_event").encode("utf-8")


def digest(*, content_hash: str, ordinal: int, prev_hash: str) -> str:
    return _digest(
        {
            "content_hash": content_hash,
            "ordinal": ordinal,
            "prev_hash": prev_hash,
            "schema": EVENT_SERIALIZATION_VERSION,
        },
        field="authoritative_event_hash",
    )


def receipt_digest(
    *, revision: int, accepted_ordinal: int, accepted_event_hash: str, previous_tail_hash: str
) -> str:
    return _digest(
        {
            "accepted_event_hash": accepted_event_hash,
            "accepted_ordinal": accepted_ordinal,
            "previous_tail_hash": previous_tail_hash,
            "revision": revision,
            "schema": TAIL_SCHEMA_VERSION,
        },
        field="accepted_tail_hash",
    )


def from_mapping(raw: bytes) -> EventEnvelope:
    try:
        text = raw.decode("utf-8")
        value = json.loads(text)
        if not isinstance(value, dict):
            raise AuthorityUnprovable("event bytes do not encode an object")
        if _canonical(value, field="persisted_authoritative_event") != text:
            raise AuthorityUnprovable("event bytes are not canonical")
        if value.get("schema") != EVENT_SERIALIZATION_VERSION:
            raise AuthorityUnprovable("event serialization version differs")
        material = value.get("envelope")
        if not isinstance(material, dict):
            raise AuthorityUnprovable("event envelope material is absent")
        envelope = EventEnvelope(**material)
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError, InvalidEventEnvelope) as exc:
        raise AuthorityUnprovable("accepted event bytes are malformed") from exc
    if canonical_record(envelope) != raw:
        raise AuthorityUnprovable("accepted event reserialization differs")
    return envelope


authority_path_from_environment = load


@dataclass(frozen=True, slots=True)
class AcceptedTail:
    revision: int
    accepted_ordinal: int
    accepted_event_hash: str
    previous_tail_hash: str
    tail_hash: str


@dataclass(frozen=True, slots=True)
class VerificationReport:
    accepted_event_count: int
    event_chain_head: str
    accepted_tail_hash: str
    candidate_count: int
    journal_mode: str
    synchronous: str
    valid: bool = True
    schema_version: str = VERIFICATION_SCHEMA_VERSION
    authority: str = AUTHORITY_LABEL


@dataclass(frozen=True, slots=True)
class StartupReport:
    database_path: str
    journal_mode: str
    wal_enabled: bool
    rollback_fallback: bool
    synchronous: str
    accepted_event_count: int
    event_chain_head: str
    accepted_tail_hash: str
    recovered_candidate_count: int
    schema_version: str = STARTUP_SCHEMA_VERSION
    authority: str = AUTHORITY_LABEL


@dataclass(frozen=True, slots=True)
class AuthorityAppendReceipt:
    ordinal: int
    event_id: str
    stream_id: str
    sequence: int
    content_hash: str
    prev_hash: str
    event_hash: str
    tail_hash: str
    candidate_commit_durable: bool
    accepted_commit_durable: bool
    readback_verified: bool
    schema_version: str = APPEND_RECEIPT_SCHEMA_VERSION
    authority: str = AUTHORITY_LABEL


@dataclass(frozen=True, slots=True)
class _AcceptedState:
    events: tuple[EventEnvelope, ...]
    event_chain_head: str
    tail_hash: str
    known_event_ids: frozenset[str]
    stream_sequences: Mapping[str, int]


_TABLE_DDL = {
    "authority_meta": (
        "CREATE TABLE authority_meta(key TEXT PRIMARY KEY,value TEXT NOT NULL)"
    ),
    "authority_migrations": (
        "CREATE TABLE authority_migrations(ordinal INTEGER PRIMARY KEY,"
        "migration_id TEXT UNIQUE NOT NULL,schema_version TEXT NOT NULL,"
        "manifest_json TEXT NOT NULL,migration_hash TEXT NOT NULL)"
    ),
    "authority_events": (
        "CREATE TABLE authority_events(ordinal INTEGER PRIMARY KEY,event_id TEXT UNIQUE NOT NULL,"
        "stream_id TEXT NOT NULL,sequence INTEGER NOT NULL,event_bytes BLOB NOT NULL,"
        "byte_length INTEGER NOT NULL,content_hash TEXT NOT NULL,prev_hash TEXT NOT NULL,"
        "event_hash TEXT UNIQUE NOT NULL,UNIQUE(stream_id,sequence))"
    ),
    "accepted_tail": (
        "CREATE TABLE accepted_tail(revision INTEGER PRIMARY KEY,accepted_ordinal INTEGER UNIQUE NOT NULL,"
        "accepted_event_hash TEXT UNIQUE NOT NULL,previous_tail_hash TEXT NOT NULL,tail_hash TEXT UNIQUE NOT NULL)"
    ),
    "event_candidate": (
        "CREATE TABLE event_candidate(slot INTEGER PRIMARY KEY CHECK(slot=1),expected_ordinal INTEGER NOT NULL,"
        "event_id TEXT NOT NULL,stream_id TEXT NOT NULL,sequence INTEGER NOT NULL,event_bytes BLOB NOT NULL,"
        "byte_length INTEGER NOT NULL,content_hash TEXT NOT NULL,prev_hash TEXT NOT NULL,event_hash TEXT NOT NULL)"
    ),
}

_TRIGGER_DDL = {
    "authority_meta_no_update": (
        "CREATE TRIGGER authority_meta_no_update BEFORE UPDATE ON authority_meta "
        "BEGIN SELECT RAISE(ABORT,'append-only authority_meta'); END"
    ),
    "authority_meta_no_delete": (
        "CREATE TRIGGER authority_meta_no_delete BEFORE DELETE ON authority_meta "
        "BEGIN SELECT RAISE(ABORT,'append-only authority_meta'); END"
    ),
    "authority_migrations_no_update": (
        "CREATE TRIGGER authority_migrations_no_update BEFORE UPDATE ON authority_migrations "
        "BEGIN SELECT RAISE(ABORT,'append-only authority_migrations'); END"
    ),
    "authority_migrations_no_delete": (
        "CREATE TRIGGER authority_migrations_no_delete BEFORE DELETE ON authority_migrations "
        "BEGIN SELECT RAISE(ABORT,'append-only authority_migrations'); END"
    ),
    "authority_events_no_update": (
        "CREATE TRIGGER authority_events_no_update BEFORE UPDATE ON authority_events "
        "BEGIN SELECT RAISE(ABORT,'append-only authority_events'); END"
    ),
    "authority_events_no_delete": (
        "CREATE TRIGGER authority_events_no_delete BEFORE DELETE ON authority_events "
        "BEGIN SELECT RAISE(ABORT,'append-only authority_events'); END"
    ),
    "accepted_tail_no_update": (
        "CREATE TRIGGER accepted_tail_no_update BEFORE UPDATE ON accepted_tail "
        "BEGIN SELECT RAISE(ABORT,'append-only accepted_tail'); END"
    ),
    "accepted_tail_no_delete": (
        "CREATE TRIGGER accepted_tail_no_delete BEFORE DELETE ON accepted_tail "
        "BEGIN SELECT RAISE(ABORT,'append-only accepted_tail'); END"
    ),
    "event_candidate_no_update": (
        "CREATE TRIGGER event_candidate_no_update BEFORE UPDATE ON event_candidate "
        "BEGIN SELECT RAISE(ABORT,'immutable event_candidate'); END"
    ),
}

_MIGRATION_MANIFEST = _canonical(
    {
        "append_only_tables": [
            "accepted_tail",
            "authority_events",
            "authority_meta",
            "authority_migrations",
        ],
        "candidate_table": "event_candidate",
        "from_version": None,
        "migration_id": "b2.authority.initial.v1",
        "to_version": STORE_SCHEMA_VERSION,
    },
    field="authority_migration",
)
_MIGRATION_HASH = _sha256(_MIGRATION_MANIFEST.encode("utf-8"))


class _WriterLock:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._handle: Any | None = None

    def run(self) -> None:
        handle = self._path.open("a+b")
        try:
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"\0")
                handle.flush()
                os.fsync(handle.fileno())
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, IOError) as exc:
            handle.close()
            raise AuthorityBusy("another authoritative writer is active") from exc
        self._handle = handle

    acquire = run

    def stop(self) -> None:
        handle = self._handle
        if handle is None:
            return
        try:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()
            self._handle = None

    release = stop


class AuthoritativeStore:
    """Single-writer, append-only authoritative SQLite event store."""

    def __init__(
        self,
        database_path: str | Path,
        *,
        shadow_path: str | Path | None = None,
        allow_rollback_fallback: bool = True,
        fault_injector: FaultInjector | None = None,
        wal_probe: WalProbe | None = None,
        transient_max_attempts: int = DEFAULT_TRANSIENT_MAX_ATTEMPTS,
        transient_backoff_initial: float = DEFAULT_TRANSIENT_BACKOFF_INITIAL_SECONDS,
        transient_backoff_max: float = DEFAULT_TRANSIENT_BACKOFF_MAX_SECONDS,
        retry_observer: RetryObserver | None = None,
        retry_sleeper: RetrySleeper = time.sleep,
    ) -> None:
        self._path = Path(database_path)
        if str(database_path) == ":memory:" or self._path.name == "":
            raise ValueError("authority requires a concrete SQLite file path")
        effective_shadow = shadow_path
        if effective_shadow is None:
            configured_shadow = os.environ.get(SHADOW_PATH_ENV, "").strip()
            effective_shadow = configured_shadow or None
        if effective_shadow is not None and _type_exact_equal(
            self._path, Path(effective_shadow).expanduser()
        ):
            raise ValueError("authority path must differ from the shadow path")
        self._allow_rollback_fallback = allow_rollback_fallback
        self._fault_injector = fault_injector
        self._wal_probe = wal_probe
        if transient_max_attempts < 1:
            raise ValueError("transient max attempts must be at least one")
        if transient_backoff_initial <= 0 or transient_backoff_max < transient_backoff_initial:
            raise ValueError("transient backoff bounds are invalid")
        self._transient_max_attempts = transient_max_attempts
        self._transient_backoff_initial = transient_backoff_initial
        self._transient_backoff_max = transient_backoff_max
        self._retry_observer = retry_observer
        self._retry_sleeper = retry_sleeper
        self._writer_lock = _WriterLock(Path(f"{self._path}.writer.lock"))
        self._opened = False
        self._journal_mode = ""

    @property
    def database_path(self) -> Path:
        return self._path

    @property
    def status(self) -> str:
        self._validate_boundary()
        return self._journal_mode

    journal_mode = status

    def _reject(self, point: str) -> None:
        if point not in FAULT_POINTS:
            raise ValueError(f"unknown fault point: {point}")
        if self._fault_injector is not None:
            self._fault_injector(point)

    def _validate_boundary(self) -> None:
        if not self._opened:
            raise AuthorityNotOpen("authoritative store requires explicit open()")

    def _retry_failure(
        self,
        operation_name: str,
        attempt: int,
        exc: AuthorityBusy | AuthorityUnprovable | sqlite3.Error | OSError,
    ) -> None:
        _handle_authority_retry(
            exc,
            operation_name=operation_name,
            attempt=attempt,
            max_attempts=self._transient_max_attempts,
            backoff_initial=self._transient_backoff_initial,
            backoff_max=self._transient_backoff_max,
            observer=self._retry_observer,
            sleeper=self._retry_sleeper,
        )

    @contextmanager
    def _resolve_store(self, *, validate_schema: bool = True) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self._path, isolation_level=None, timeout=5.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=5000")
        connection.execute("PRAGMA synchronous=FULL")
        synchronous = int(connection.execute("PRAGMA synchronous").fetchone()[0])
        if synchronous != 2:
            connection.close()
            raise AuthorityUnprovable("SQLite synchronous mode is not FULL")
        try:
            if validate_schema:
                self._validate_schema(connection)
            yield connection
        finally:
            connection.close()

    def _policy_flags(self, connection: sqlite3.Connection) -> str:
        if self._wal_probe is None:
            selected = str(connection.execute("PRAGMA journal_mode=WAL").fetchone()[0]).lower()
        else:
            selected = str(self._wal_probe(connection)).lower()
        if selected == "wal":
            actual = str(connection.execute("PRAGMA journal_mode").fetchone()[0]).lower()
            if actual != "wal":
                raise AuthorityUnprovable("SQLite WAL probe did not persist")
            connection.execute("PRAGMA wal_autocheckpoint=1000")
            return actual
        if not self._allow_rollback_fallback:
            raise AuthorityUnprovable("SQLite WAL is unavailable and fallback is disabled")
        fallback = str(connection.execute("PRAGMA journal_mode=DELETE").fetchone()[0]).lower()
        if fallback != "delete":
            raise AuthorityUnprovable("controlled DELETE-journal fallback is unavailable")
        return fallback

    def _plan(self, connection: sqlite3.Connection) -> None:
        connection.execute("BEGIN IMMEDIATE")
        try:
            for statement in _TABLE_DDL.values():
                connection.execute(statement)
            for statement in _TRIGGER_DDL.values():
                connection.execute(statement)
            metadata = {
                "authority": AUTHORITY_LABEL,
                "event_envelope_authority": SHADOW_AUTHORITY,
                "event_serialization_version": EVENT_SERIALIZATION_VERSION,
                "store_schema_version": STORE_SCHEMA_VERSION,
                "tail_schema_version": TAIL_SCHEMA_VERSION,
            }
            connection.executemany(
                "INSERT INTO authority_meta(key,value) VALUES(?,?)", sorted(metadata.items())
            )
            connection.execute(
                "INSERT INTO authority_migrations VALUES(1,?,?,?,?)",
                (
                    "b2.authority.initial.v1",
                    MIGRATION_SCHEMA_VERSION,
                    _MIGRATION_MANIFEST,
                    _MIGRATION_HASH,
                ),
            )
            connection.commit()
        except Exception:
            connection.rollback()
            raise

    @staticmethod
    def _validate_schema(connection: sqlite3.Connection) -> None:
        tables = connection.execute(
            "SELECT name,sql FROM sqlite_schema WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
        triggers = connection.execute(
            "SELECT name,sql FROM sqlite_schema WHERE type='trigger'"
        ).fetchall()
        actual_tables = {str(row["name"]): _canonical_sort_key(str(row["sql"])) for row in tables}
        actual_triggers = {str(row["name"]): _canonical_sort_key(str(row["sql"])) for row in triggers}
        expected_tables = {name: _canonical_sort_key(sql) for name, sql in _TABLE_DDL.items()}
        expected_triggers = {name: _canonical_sort_key(sql) for name, sql in _TRIGGER_DDL.items()}
        if actual_tables != expected_tables:
            raise AuthorityUnprovable("authoritative table schema differs")
        if actual_triggers != expected_triggers:
            raise AuthorityUnprovable("authoritative trigger schema differs")
        metadata = dict(connection.execute("SELECT key,value FROM authority_meta"))
        if metadata != {
            "authority": AUTHORITY_LABEL,
            "event_envelope_authority": SHADOW_AUTHORITY,
            "event_serialization_version": EVENT_SERIALIZATION_VERSION,
            "store_schema_version": STORE_SCHEMA_VERSION,
            "tail_schema_version": TAIL_SCHEMA_VERSION,
        }:
            raise AuthorityUnprovable("authoritative metadata differs")
        migrations = connection.execute(
            "SELECT * FROM authority_migrations ORDER BY ordinal"
        ).fetchall()
        expected = (
            1,
            "b2.authority.initial.v1",
            MIGRATION_SCHEMA_VERSION,
            _MIGRATION_MANIFEST,
            _MIGRATION_HASH,
        )
        if len(migrations) != 1 or tuple(migrations[0]) != expected:
            raise AuthorityUnprovable("authoritative migration history differs")

    @staticmethod
    def _record(row: sqlite3.Row, *, expected_ordinal: int) -> EventEnvelope:
        try:
            raw = bytes(row["event_bytes"])
            if int(row["ordinal"] if "ordinal" in row.keys() else row["expected_ordinal"]) != expected_ordinal:
                raise AuthorityUnprovable("event ordinal is not contiguous")
            if int(row["byte_length"]) != len(raw):
                raise AuthorityUnprovable("event byte length differs")
            content_hash = _sha256(raw)
            if str(row["content_hash"]) != content_hash:
                raise AuthorityUnprovable("event content hash differs")
            envelope = from_mapping(raw)
            indexes = (str(row["event_id"]), str(row["stream_id"]), int(row["sequence"]))
            if indexes != (envelope.event_id, envelope.stream_id, envelope.sequence):
                raise AuthorityUnprovable("event index columns differ from canonical bytes")
            return envelope
        except (KeyError, TypeError, ValueError, sqlite3.Error) as exc:
            raise AuthorityUnprovable("accepted event row is malformed") from exc

    def _state_record(self, connection: sqlite3.Connection) -> _AcceptedState:
        sqlite_result = str(connection.execute("PRAGMA integrity_check").fetchone()[0])
        if sqlite_result != "ok":
            raise AuthorityUnprovable(f"SQLite integrity check failed: {sqlite_result}")
        rows = connection.execute("SELECT * FROM authority_events ORDER BY ordinal").fetchall()
        event_hash = GENESIS_HASH
        known: set[str] = set()
        streams: dict[str, int] = {}
        events: list[EventEnvelope] = []
        for ordinal, row in enumerate(rows, 1):
            envelope = self._record(row, expected_ordinal=ordinal)
            if str(row["prev_hash"]) != event_hash:
                raise AuthorityUnprovable("accepted event predecessor differs")
            expected_hash = digest(
                content_hash=str(row["content_hash"]), ordinal=ordinal, prev_hash=event_hash
            )
            if str(row["event_hash"]) != expected_hash:
                raise AuthorityUnprovable("accepted event hash-chain differs")
            expected_sequence = streams.get(envelope.stream_id, 0) + 1
            if envelope.sequence != expected_sequence:
                raise AuthorityUnprovable("accepted stream sequence is not contiguous")
            if envelope.event_id in known:
                raise AuthorityUnprovable("accepted event id is duplicated")
            if envelope.causation_id is not None and envelope.causation_id not in known:
                raise AuthorityUnprovable("accepted event causation is outside prior history")
            event_hash = expected_hash
            known.add(envelope.event_id)
            streams[envelope.stream_id] = envelope.sequence
            events.append(envelope)

        tails = connection.execute("SELECT * FROM accepted_tail ORDER BY revision").fetchall()
        if len(tails) != len(rows):
            raise AuthorityUnprovable("accepted-tail length differs from accepted history")
        previous_tail = GENESIS_HASH
        for revision, (tail, event_row) in enumerate(zip(tails, rows), 1):
            if int(tail["revision"]) != revision or int(tail["accepted_ordinal"]) != revision:
                raise AuthorityUnprovable("accepted-tail revision is not contiguous")
            if str(tail["accepted_event_hash"]) != str(event_row["event_hash"]):
                raise AuthorityUnprovable("accepted-tail event hash differs")
            if str(tail["previous_tail_hash"]) != previous_tail:
                raise AuthorityUnprovable("accepted-tail predecessor differs")
            expected_tail = receipt_digest(
                revision=revision,
                accepted_ordinal=revision,
                accepted_event_hash=str(tail["accepted_event_hash"]),
                previous_tail_hash=previous_tail,
            )
            if str(tail["tail_hash"]) != expected_tail:
                raise AuthorityUnprovable("accepted-tail hash-chain differs")
            previous_tail = expected_tail
        return _AcceptedState(
            tuple(events), event_hash, previous_tail, frozenset(known), dict(streams)
        )

    def _entry(
        self, row: sqlite3.Row, state: _AcceptedState
    ) -> EventEnvelope:
        expected_ordinal = len(state.events) + 1
        if int(row["slot"]) != 1 or int(row["expected_ordinal"]) != expected_ordinal:
            raise AuthorityAmbiguity("candidate ordinal cannot be proven as unaccepted suffix")
        envelope = self._record(row, expected_ordinal=expected_ordinal)
        if str(row["prev_hash"]) != state.event_chain_head:
            raise AuthorityAmbiguity("candidate predecessor cannot be proven")
        expected_hash = digest(
            content_hash=str(row["content_hash"]),
            ordinal=expected_ordinal,
            prev_hash=state.event_chain_head,
        )
        if str(row["event_hash"]) != expected_hash:
            raise AuthorityAmbiguity("candidate event hash cannot be proven")
        if envelope.event_id in state.known_event_ids:
            raise AuthorityAmbiguity("candidate duplicates accepted history")
        expected_sequence = state.stream_sequences.get(envelope.stream_id, 0) + 1
        if envelope.sequence != expected_sequence:
            raise AuthorityAmbiguity("candidate stream sequence cannot be proven")
        if envelope.causation_id is not None and envelope.causation_id not in state.known_event_ids:
            raise AuthorityAmbiguity("candidate causation cannot be proven")
        return envelope

    def _remember(self, connection: sqlite3.Connection, state: _AcceptedState) -> int:
        candidates = connection.execute("SELECT * FROM event_candidate").fetchall()
        if not candidates:
            return 0
        if len(candidates) != 1:
            raise AuthorityAmbiguity("multiple unaccepted candidates exist")
        self._entry(candidates[0], state)
        connection.execute("BEGIN IMMEDIATE")
        try:
            connection.execute("DELETE FROM event_candidate WHERE slot=1")
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        return 1

    def run(self) -> StartupReport:
        """Acquire the writer lock, verify accepted history, and clean proven residue."""

        if self._opened:
            raise AuthorityPersistenceError("authoritative store is already open")

        for attempt in range(1, self._transient_max_attempts + 1):
            try:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                self._writer_lock.acquire()
                with self._resolve_store(validate_schema=False) as connection:
                    self._journal_mode = self._policy_flags(connection)
                    objects = int(
                        connection.execute(
                            "SELECT COUNT(*) FROM sqlite_schema WHERE name NOT LIKE 'sqlite_%'"
                        ).fetchone()[0]
                    )
                    if objects == 0:
                        self._plan(connection)
                    self._validate_schema(connection)
                    state = self._state_record(connection)
                    recovered = self._remember(connection, state)
                    if recovered:
                        state = self._state_record(connection)
                    candidate_count = int(
                        connection.execute("SELECT COUNT(*) FROM event_candidate").fetchone()[0]
                    )
                    if candidate_count:
                        raise AuthorityAmbiguity("unaccepted residue remains after recovery")
                self._opened = True
                return StartupReport(
                    str(self._path),
                    self._journal_mode,
                    self._journal_mode == "wal",
                    self._journal_mode == "delete",
                    "FULL",
                    len(state.events),
                    state.event_chain_head,
                    state.tail_hash,
                    recovered,
                )
            except (AuthorityBusy, AuthorityUnprovable, sqlite3.Error, OSError) as exc:
                self._writer_lock.release()
                self._journal_mode = ""
                self._retry_failure("authoritative_startup", attempt, exc)
            except AuthorityPersistenceError:
                self._writer_lock.release()
                self._journal_mode = ""
                raise
            except Exception as exc:
                self._writer_lock.release()
                self._journal_mode = ""
                raise AuthorityUnprovable(
                    "authoritative startup verification failed"
                ) from exc
        raise AssertionError("bounded authoritative startup retry did not terminate")

    open = run

    def stop(self) -> None:
        self._opened = False
        self._journal_mode = ""
        self._writer_lock.release()

    close = stop

    def create(self) -> "AuthoritativeStore":
        self.open()
        return self

    __enter__ = create

    def update(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    __exit__ = update

    @staticmethod
    def _accepted_append_receipt(
        connection: sqlite3.Connection,
        envelope: EventEnvelope,
        raw: bytes,
        state: _AcceptedState,
    ) -> AuthorityAppendReceipt:
        row = connection.execute(
            "SELECT * FROM authority_events WHERE event_id=?", (envelope.event_id,)
        ).fetchone()
        if row is None:
            raise AuthorityAmbiguity("retried append is absent from accepted history")
        ordinal = int(row["ordinal"])
        if ordinal != len(state.events) or bytes(row["event_bytes"]) != raw:
            raise AuthorityAmbiguity("retried append differs from accepted history")
        if state.events[-1] != envelope:
            raise AuthorityAmbiguity("retried append is not the accepted chain head")
        tail = connection.execute(
            "SELECT * FROM accepted_tail WHERE accepted_ordinal=?", (ordinal,)
        ).fetchone()
        if tail is None or int(tail["revision"]) != ordinal:
            raise AuthorityAmbiguity("retried append tail receipt is absent")
        content_hash = str(row["content_hash"])
        previous_event_hash = str(row["prev_hash"])
        event_hash = str(row["event_hash"])
        if content_hash != _sha256(raw) or event_hash != digest(
            content_hash=content_hash,
            ordinal=ordinal,
            prev_hash=previous_event_hash,
        ):
            raise AuthorityAmbiguity("retried append event proof differs")
        previous_tail_hash = str(tail["previous_tail_hash"])
        tail_hash = str(tail["tail_hash"])
        if str(tail["accepted_event_hash"]) != event_hash or tail_hash != receipt_digest(
            revision=ordinal,
            accepted_ordinal=ordinal,
            accepted_event_hash=event_hash,
            previous_tail_hash=previous_tail_hash,
        ):
            raise AuthorityAmbiguity("retried append tail proof differs")
        return AuthorityAppendReceipt(
            ordinal,
            envelope.event_id,
            envelope.stream_id,
            envelope.sequence,
            content_hash,
            previous_event_hash,
            event_hash,
            tail_hash,
            True,
            True,
            True,
        )

    def append(self, envelope: EventEnvelope) -> AuthorityAppendReceipt:
        """Durably accept one event using the candidate-to-tail protocol."""

        self._validate_boundary()
        if not isinstance(envelope, EventEnvelope):
            raise AuthorityAppendRejected("authority accepts EventEnvelope only")
        raw = canonical_record(envelope)

        for attempt in range(1, self._transient_max_attempts + 1):
            try:
                self._reject(FAULT_BEFORE_EVENT_APPEND)
                with self._resolve_store() as connection:
                    state = self._state_record(connection)
                    if envelope.event_id in state.known_event_ids:
                        if attempt == 1:
                            raise AuthorityAppendRejected(
                                f"duplicate event id: {envelope.event_id}"
                            )
                        return self._accepted_append_receipt(
                            connection, envelope, raw, state
                        )

                    candidate = connection.execute(
                        "SELECT * FROM event_candidate WHERE slot=1"
                    ).fetchone()
                    if candidate is not None:
                        if attempt == 1:
                            raise AuthorityAmbiguity(
                                "unaccepted residue requires restart verification"
                            )
                        recovered = self._entry(candidate, state)
                        if recovered != envelope or bytes(candidate["event_bytes"]) != raw:
                            raise AuthorityAmbiguity("retried append candidate differs")
                        ordinal = int(candidate["expected_ordinal"])
                        content_hash = str(candidate["content_hash"])
                        event_hash = str(candidate["event_hash"])
                    else:
                        expected_sequence = (
                            state.stream_sequences.get(envelope.stream_id, 0) + 1
                        )
                        if envelope.sequence != expected_sequence:
                            raise AuthorityAppendRejected(
                                f"expected sequence {expected_sequence} for {envelope.stream_id}"
                            )
                        if (
                            envelope.causation_id is not None
                            and envelope.causation_id not in state.known_event_ids
                        ):
                            raise AuthorityAppendRejected(
                                f"unknown causation: {envelope.causation_id}"
                            )
                        ordinal = len(state.events) + 1
                        content_hash = _sha256(raw)
                        event_hash = digest(
                            content_hash=content_hash,
                            ordinal=ordinal,
                            prev_hash=state.event_chain_head,
                        )
                        connection.execute("BEGIN IMMEDIATE")
                        try:
                            connection.execute(
                                "INSERT INTO event_candidate VALUES(1,?,?,?,?,?,?,?,?,?)",
                                (
                                    ordinal,
                                    envelope.event_id,
                                    envelope.stream_id,
                                    envelope.sequence,
                                    raw,
                                    len(raw),
                                    content_hash,
                                    state.event_chain_head,
                                    event_hash,
                                ),
                            )
                            self._reject(FAULT_AFTER_EVENT_ROW_WRITE_BEFORE_COMMIT)
                            connection.commit()
                        except sqlite3.Error:
                            raise
                        except Exception:
                            connection.rollback()
                            raise

                        self._reject(FAULT_AFTER_EVENT_TRANSACTION_COMMIT)

                    self._reject(FAULT_BEFORE_ACCEPTED_TAIL_UPDATE)
                    connection.execute("BEGIN IMMEDIATE")
                    try:
                        current = self._state_record(connection)
                        candidate = connection.execute(
                            "SELECT * FROM event_candidate WHERE slot=1"
                        ).fetchone()
                        if candidate is None:
                            raise AuthorityAmbiguity(
                                "candidate disappeared before acceptance"
                            )
                        accepted_candidate = self._entry(candidate, current)
                        if (
                            accepted_candidate != envelope
                            or bytes(candidate["event_bytes"]) != raw
                        ):
                            raise AuthorityAmbiguity("candidate differs before acceptance")
                        connection.execute(
                            "INSERT INTO authority_events VALUES(?,?,?,?,?,?,?,?,?)",
                            (
                                ordinal,
                                envelope.event_id,
                                envelope.stream_id,
                                envelope.sequence,
                                raw,
                                len(raw),
                                content_hash,
                                current.event_chain_head,
                                event_hash,
                            ),
                        )
                        next_tail_hash = receipt_digest(
                            revision=ordinal,
                            accepted_ordinal=ordinal,
                            accepted_event_hash=event_hash,
                            previous_tail_hash=current.tail_hash,
                        )
                        connection.execute(
                            "INSERT INTO accepted_tail VALUES(?,?,?,?,?)",
                            (
                                ordinal,
                                ordinal,
                                event_hash,
                                current.tail_hash,
                                next_tail_hash,
                            ),
                        )
                        self._reject(FAULT_DURING_ACCEPTED_TAIL_UPDATE)
                        connection.execute("DELETE FROM event_candidate WHERE slot=1")
                        connection.commit()
                    except sqlite3.Error:
                        raise
                    except Exception:
                        connection.rollback()
                        raise

                    self._reject(FAULT_AFTER_ACCEPTED_TAIL_UPDATE)
                    verified = self._state_record(connection)
                    if (
                        len(verified.events) != ordinal
                        or verified.event_chain_head != event_hash
                        or verified.tail_hash != next_tail_hash
                    ):
                        raise AuthorityUnprovable("accepted append readback differs")
                    return AuthorityAppendReceipt(
                        ordinal,
                        envelope.event_id,
                        envelope.stream_id,
                        envelope.sequence,
                        content_hash,
                        state.event_chain_head,
                        event_hash,
                        next_tail_hash,
                        True,
                        True,
                        True,
                    )
            except (AuthorityBusy, AuthorityUnprovable, sqlite3.Error, OSError) as exc:
                self._retry_failure("authoritative_append", attempt, exc)
        raise AssertionError("bounded authoritative append retry did not terminate")

    def report(self) -> VerificationReport:
        self._validate_boundary()

        for attempt in range(1, self._transient_max_attempts + 1):
            try:
                with self._resolve_store() as connection:
                    state = self._state_record(connection)
                    candidates = int(
                        connection.execute(
                            "SELECT COUNT(*) FROM event_candidate"
                        ).fetchone()[0]
                    )
                    if candidates:
                        raise AuthorityAmbiguity(
                            "unaccepted residue requires restart verification"
                        )
                    mode = str(
                        connection.execute("PRAGMA journal_mode").fetchone()[0]
                    ).lower()
                    synchronous = int(
                        connection.execute("PRAGMA synchronous").fetchone()[0]
                    )
                    if mode != self._journal_mode or synchronous != 2:
                        raise AuthorityUnprovable(
                            "journal or synchronous mode drifted"
                        )
                    return VerificationReport(
                        len(state.events),
                        state.event_chain_head,
                        state.tail_hash,
                        candidates,
                        mode,
                        "FULL",
                    )
            except (AuthorityBusy, AuthorityUnprovable, sqlite3.Error, OSError) as exc:
                self._retry_failure("authoritative_verify", attempt, exc)
        raise AssertionError("bounded authoritative verification retry did not terminate")

    verify = report

    def events(self, *, stream_id: str | None = None) -> tuple[EventEnvelope, ...]:
        self._validate_boundary()

        for attempt in range(1, self._transient_max_attempts + 1):
            try:
                with self._resolve_store() as connection:
                    events = self._state_record(connection).events
                break
            except (AuthorityBusy, AuthorityUnprovable, sqlite3.Error, OSError) as exc:
                self._retry_failure("authoritative_events", attempt, exc)
        else:
            raise AssertionError("bounded authoritative events retry did not terminate")
        if stream_id is None:
            return events
        return tuple(event for event in events if event.stream_id == stream_id)

    def replay(
        self,
        initial_state: StateT,
        reducer: Callable[[StateT, EventEnvelope], StateT],
        *,
        stream_id: str | None = None,
    ) -> StateT:
        """Replay accepted events through the unchanged in-memory kernel contract."""

        kernel: InMemoryEventKernel[StateT] = InMemoryEventKernel()
        for envelope in self.events():
            kernel.append(envelope)
        return kernel.replay(initial_state, reducer, stream_id=stream_id)


def status(exc: BaseException) -> int:
    """Map an unprovable authority state to the mandatory process exit code."""

    if isinstance(exc, AuthorityUnprovable):
        return AUTHORITY_FAILURE_EXIT_CODE
    return 1


exit_code_for_authority_failure = status
