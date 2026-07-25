"""Minimal EVE v4.1 event-kernel contract for M1-A.

This module is deliberately disconnected from the pre-kernel legacy runtime.
It creates no files or databases, starts no threads, changes no production
defaults, and grants no persistence or behavioral authority.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Callable, Generic, Iterable, Mapping, TypeVar

EVENT_SCHEMA_VERSION = "eve.event-envelope.v1"
SHADOW_AUTHORITY = "shadow_only"
MAX_CANONICAL_JSON_BYTES = 65_536
MAX_JSON_DEPTH = 32

_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")
_EVENT_TYPE_PATTERN = re.compile(r"^[a-z][a-z0-9_.-]{2,127}$")
_VERSION_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,63}$")

StateT = TypeVar("StateT")


class EventContractError(ValueError):
    """Base error for an invalid M1-A event contract."""


class InvalidEventEnvelope(EventContractError):
    """Raised when an event envelope is malformed or claims authority."""


class DuplicateEventId(EventContractError):
    """Raised when an event identifier is appended more than once."""


class StreamSequenceError(EventContractError):
    """Raised when a stream sequence is not contiguous and one-based."""


class UnknownCausation(EventContractError):
    """Raised when an event points to a cause not already in the kernel."""


class ReducerContractError(EventContractError):
    """Raised when replay violates the explicit reducer boundary."""


def _require_identifier(value: str, *, field: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER_PATTERN.fullmatch(value):
        raise InvalidEventEnvelope(f"{field} is not a canonical identifier")
    return value


def _require_event_type(value: str) -> str:
    if not isinstance(value, str) or not _EVENT_TYPE_PATTERN.fullmatch(value):
        raise InvalidEventEnvelope("event_type is not canonical")
    return value


def _require_version(value: str, *, field: str) -> str:
    if not isinstance(value, str) or not _VERSION_PATTERN.fullmatch(value):
        raise InvalidEventEnvelope(f"{field} is not a canonical version")
    return value


def _validate_json_value(value: Any, *, depth: int = 0) -> None:
    if depth > MAX_JSON_DEPTH:
        raise InvalidEventEnvelope("JSON value exceeds maximum nesting depth")
    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, int) and not isinstance(value, bool):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise InvalidEventEnvelope("JSON numbers must be finite")
        return
    if isinstance(value, list):
        for item in value:
            _validate_json_value(item, depth=depth + 1)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise InvalidEventEnvelope("JSON object keys must be strings")
            _validate_json_value(item, depth=depth + 1)
        return
    raise InvalidEventEnvelope(
        f"unsupported JSON value type: {type(value).__name__}"
    )


def canonical_json_object(value: Mapping[str, Any], *, field: str) -> str:
    """Validate and encode a bounded JSON object into a canonical string."""

    if not isinstance(value, Mapping):
        raise InvalidEventEnvelope(f"{field} must be a mapping")
    plain = dict(value)
    _validate_json_value(plain)
    encoded = json.dumps(
        plain,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    if len(encoded.encode("utf-8")) > MAX_CANONICAL_JSON_BYTES:
        raise InvalidEventEnvelope(f"{field} exceeds canonical size limit")
    return encoded


def _validate_canonical_json_object(value: str, *, field: str) -> None:
    if not isinstance(value, str):
        raise InvalidEventEnvelope(f"{field} must be canonical JSON text")
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError as exc:
        raise InvalidEventEnvelope(f"{field} is not valid JSON") from exc
    if not isinstance(decoded, dict):
        raise InvalidEventEnvelope(f"{field} must encode a JSON object")
    canonical = canonical_json_object(decoded, field=field)
    if canonical != value:
        raise InvalidEventEnvelope(f"{field} is not canonical JSON")


@dataclass(frozen=True, slots=True)
class EventEnvelope:
    """Immutable, canonical, shadow-only event envelope.

    All ordering and causal identifiers are caller supplied. M1-A generates no
    timestamps, UUIDs, randomness, or persistence artifacts.
    """

    event_id: str
    event_type: str
    stream_id: str
    sequence: int
    producer: str
    producer_version: str
    correlation_id: str
    causation_id: str | None
    payload_json: str
    causal_context_json: str
    schema_version: str = EVENT_SCHEMA_VERSION
    authority: str = SHADOW_AUTHORITY

    def __post_init__(self) -> None:
        if self.schema_version != EVENT_SCHEMA_VERSION:
            raise InvalidEventEnvelope("unsupported event schema version")
        if self.authority != SHADOW_AUTHORITY:
            raise InvalidEventEnvelope(
                "M1-A envelopes cannot claim authoritative runtime status"
            )
        _require_identifier(self.event_id, field="event_id")
        _require_event_type(self.event_type)
        _require_identifier(self.stream_id, field="stream_id")
        if (
            isinstance(self.sequence, bool)
            or not isinstance(self.sequence, int)
            or self.sequence < 1
        ):
            raise InvalidEventEnvelope("sequence must be a positive integer")
        _require_identifier(self.producer, field="producer")
        _require_version(self.producer_version, field="producer_version")
        _require_identifier(self.correlation_id, field="correlation_id")
        if self.causation_id is not None:
            _require_identifier(self.causation_id, field="causation_id")
            if self.causation_id == self.event_id:
                raise InvalidEventEnvelope("an event cannot cause itself")
        _validate_canonical_json_object(self.payload_json, field="payload_json")
        _validate_canonical_json_object(
            self.causal_context_json,
            field="causal_context_json",
        )

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
        """Create an envelope from bounded JSON-compatible mappings."""

        return cls(
            event_id=event_id,
            event_type=event_type,
            stream_id=stream_id,
            sequence=sequence,
            producer=producer,
            producer_version=producer_version,
            correlation_id=correlation_id,
            causation_id=causation_id,
            payload_json=canonical_json_object(payload, field="payload"),
            causal_context_json=canonical_json_object(
                causal_context,
                field="causal_context",
            ),
            authority=authority,
        )

    @property
    def payload(self) -> dict[str, Any]:
        """Return a detached mutable copy; the envelope remains immutable."""

        return json.loads(self.payload_json)

    @property
    def causal_context(self) -> dict[str, Any]:
        """Return a detached causal-context copy."""

        return json.loads(self.causal_context_json)

    @property
    def digest(self) -> str:
        """Return a deterministic digest of the complete envelope."""

        canonical = json.dumps(
            {
                "authority": self.authority,
                "causal_context": json.loads(self.causal_context_json),
                "causation_id": self.causation_id,
                "correlation_id": self.correlation_id,
                "event_id": self.event_id,
                "event_type": self.event_type,
                "payload": json.loads(self.payload_json),
                "producer": self.producer,
                "producer_version": self.producer_version,
                "schema_version": self.schema_version,
                "sequence": self.sequence,
                "stream_id": self.stream_id,
            },
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class AppendReceipt:
    """Immutable evidence that an envelope was accepted in memory."""

    index: int
    event_id: str
    stream_id: str
    sequence: int
    envelope_digest: str
    authority: str = SHADOW_AUTHORITY


class InMemoryEventKernel(Generic[StateT]):
    """Append-only, non-persistent M1-A event kernel.

    The class exposes no update, delete, save, load, file, database, network,
    clock, thread, or legacy-runtime integration surface.
    """

    def __init__(self) -> None:
        self._events: list[EventEnvelope] = []
        self._events_by_id: dict[str, EventEnvelope] = {}
        self._last_sequence_by_stream: dict[str, int] = {}

    def append(self, envelope: EventEnvelope) -> AppendReceipt:
        """Validate and append one event after all fail-closed checks pass."""

        if not isinstance(envelope, EventEnvelope):
            raise InvalidEventEnvelope("kernel accepts EventEnvelope only")
        if envelope.event_id in self._events_by_id:
            raise DuplicateEventId(envelope.event_id)
        expected_sequence = self._last_sequence_by_stream.get(
            envelope.stream_id,
            0,
        ) + 1
        if envelope.sequence != expected_sequence:
            raise StreamSequenceError(
                f"expected sequence {expected_sequence} for {envelope.stream_id}"
            )
        if (
            envelope.causation_id is not None
            and envelope.causation_id not in self._events_by_id
        ):
            raise UnknownCausation(envelope.causation_id)

        index = len(self._events)
        self._events.append(envelope)
        self._events_by_id[envelope.event_id] = envelope
        self._last_sequence_by_stream[envelope.stream_id] = envelope.sequence
        return AppendReceipt(
            index=index,
            event_id=envelope.event_id,
            stream_id=envelope.stream_id,
            sequence=envelope.sequence,
            envelope_digest=envelope.digest,
        )

    def events(self) -> tuple[EventEnvelope, ...]:
        """Return all accepted events as an immutable tuple."""

        return tuple(self._events)

    def stream(self, stream_id: str) -> tuple[EventEnvelope, ...]:
        """Return one stream without exposing internal collections."""

        _require_identifier(stream_id, field="stream_id")
        return tuple(
            event for event in self._events if event.stream_id == stream_id
        )

    def get(self, event_id: str) -> EventEnvelope | None:
        """Read one event by identifier."""

        _require_identifier(event_id, field="event_id")
        return self._events_by_id.get(event_id)

    def __len__(self) -> int:
        return len(self._events)

    def replay(
        self,
        initial_state: StateT,
        reducer: Callable[[StateT, EventEnvelope], StateT],
        *,
        stream_id: str | None = None,
    ) -> StateT:
        """Replay through an explicit pure reducer boundary.

        Reducer exceptions propagate. Returning ``None`` is rejected so a
        missing state transition cannot be silently accepted.
        """

        if not callable(reducer):
            raise ReducerContractError("reducer must be callable")
        source: Iterable[EventEnvelope]
        if stream_id is None:
            source = self.events()
        else:
            source = self.stream(stream_id)
        state = initial_state
        for envelope in source:
            next_state = reducer(state, envelope)
            if next_state is None:
                raise ReducerContractError("reducer returned None")
            state = next_state
        return state
