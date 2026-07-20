from __future__ import annotations

import ast
import dataclasses
import math
from pathlib import Path

import pytest

from core.event_kernel import (
    EVENT_SCHEMA_VERSION,
    SHADOW_AUTHORITY,
    DuplicateEventId,
    EventEnvelope,
    InMemoryEventKernel,
    InvalidEventEnvelope,
    ReducerContractError,
    StreamSequenceError,
    UnknownCausation,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "core/event_kernel.py"


def _event(
    event_id: str = "evt:1",
    *,
    stream_id: str = "stream:alpha",
    sequence: int = 1,
    causation_id: str | None = None,
    payload: dict | None = None,
    causal_context: dict | None = None,
    authority: str = SHADOW_AUTHORITY,
) -> EventEnvelope:
    return EventEnvelope.create(
        event_id=event_id,
        event_type="kernel.test_event",
        stream_id=stream_id,
        sequence=sequence,
        producer="tests.m1_a",
        producer_version="1.0.0",
        correlation_id="corr:m1-a",
        causation_id=causation_id,
        payload=payload or {"value": sequence},
        causal_context=causal_context
        or {
            "input_digest": f"sha256:{sequence}",
            "model_version": None,
            "parameters": {},
            "seed": None,
        },
        authority=authority,
    )


def test_envelope_is_frozen_canonical_and_shadow_only():
    envelope = _event(
        payload={"z": [3, 2, 1], "a": {"한국어": True}},
        causal_context={"seed": 7, "parameters": {"b": 2, "a": 1}},
    )

    assert envelope.schema_version == EVENT_SCHEMA_VERSION
    assert envelope.authority == SHADOW_AUTHORITY
    assert envelope.payload_json == '{"a":{"한국어":true},"z":[3,2,1]}'
    assert envelope.causal_context_json == (
        '{"parameters":{"a":1,"b":2},"seed":7}'
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        envelope.sequence = 2  # type: ignore[misc]


def test_payload_and_context_accessors_are_detached_copies():
    envelope = _event(payload={"nested": {"value": 1}})

    payload = envelope.payload
    payload["nested"]["value"] = 99
    context = envelope.causal_context
    context["parameters"]["new"] = True

    assert envelope.payload == {"nested": {"value": 1}}
    assert "new" not in envelope.causal_context["parameters"]


def test_digest_is_deterministic_and_covers_causal_metadata():
    first = _event(
        payload={"b": 2, "a": 1},
        causal_context={"seed": 1, "parameters": {}},
    )
    reordered = _event(
        payload={"a": 1, "b": 2},
        causal_context={"parameters": {}, "seed": 1},
    )
    changed_context = _event(
        payload={"a": 1, "b": 2},
        causal_context={"parameters": {}, "seed": 2},
    )

    assert first.digest == reordered.digest
    assert first.digest != changed_context.digest
    assert len(first.digest) == 64


@pytest.mark.parametrize(
    "field,value",
    [
        ("event_id", ""),
        ("event_type", "Invalid Type"),
        ("stream_id", "bad stream"),
        ("producer", "bad producer!"),
        ("producer_version", "bad version!"),
        ("correlation_id", "bad correlation!"),
    ],
)
def test_identifier_and_version_fields_fail_closed(field: str, value: str):
    kwargs = {
        "event_id": "evt:1",
        "event_type": "kernel.test_event",
        "stream_id": "stream:alpha",
        "sequence": 1,
        "producer": "tests.m1_a",
        "producer_version": "1.0.0",
        "correlation_id": "corr:m1-a",
        "payload": {},
        "causal_context": {},
    }
    kwargs[field] = value

    with pytest.raises(InvalidEventEnvelope):
        EventEnvelope.create(**kwargs)


@pytest.mark.parametrize("sequence", [0, -1, True, 1.5])
def test_invalid_sequence_fails_closed(sequence):
    with pytest.raises(InvalidEventEnvelope):
        _event(sequence=sequence)


def test_authority_claim_and_self_causation_are_rejected():
    with pytest.raises(InvalidEventEnvelope, match="authoritative"):
        _event(authority="authoritative")
    with pytest.raises(InvalidEventEnvelope, match="cause itself"):
        _event(causation_id="evt:1")


def test_noncanonical_direct_constructor_is_rejected():
    with pytest.raises(InvalidEventEnvelope, match="not canonical"):
        EventEnvelope(
            event_id="evt:1",
            event_type="kernel.test_event",
            stream_id="stream:alpha",
            sequence=1,
            producer="tests.m1_a",
            producer_version="1.0.0",
            correlation_id="corr:m1-a",
            causation_id=None,
            payload_json='{ "b": 2, "a": 1 }',
            causal_context_json="{}",
        )


@pytest.mark.parametrize(
    "payload",
    [
        {"not_finite": math.nan},
        {"not_finite": math.inf},
        {"unsupported": {1, 2}},
        {1: "non-string-key"},
        {"tuple": (1, 2)},
    ],
)
def test_non_json_or_ambiguous_payloads_are_rejected(payload):
    with pytest.raises(InvalidEventEnvelope):
        _event(payload=payload)


def test_payload_size_and_depth_are_bounded():
    with pytest.raises(InvalidEventEnvelope, match="size limit"):
        _event(payload={"text": "x" * 70_000})

    value: object = "leaf"
    for _ in range(35):
        value = [value]
    with pytest.raises(InvalidEventEnvelope, match="nesting depth"):
        _event(payload={"deep": value})


def test_append_is_one_based_contiguous_and_returns_immutable_receipt():
    kernel: InMemoryEventKernel[dict] = InMemoryEventKernel()
    first = _event()
    second = _event("evt:2", sequence=2, causation_id="evt:1")

    first_receipt = kernel.append(first)
    second_receipt = kernel.append(second)

    assert first_receipt.index == 0
    assert first_receipt.envelope_digest == first.digest
    assert second_receipt.index == 1
    assert second_receipt.authority == SHADOW_AUTHORITY
    assert kernel.events() == (first, second)
    assert kernel.stream("stream:alpha") == (first, second)
    assert kernel.get("evt:2") is second
    assert len(kernel) == 2
    with pytest.raises(dataclasses.FrozenInstanceError):
        first_receipt.index = 99  # type: ignore[misc]


def test_duplicate_id_sequence_gap_and_unknown_cause_fail_before_append():
    kernel: InMemoryEventKernel[dict] = InMemoryEventKernel()
    first = _event()
    kernel.append(first)

    with pytest.raises(DuplicateEventId):
        kernel.append(first)
    with pytest.raises(StreamSequenceError):
        kernel.append(_event("evt:3", sequence=3))
    with pytest.raises(UnknownCausation):
        kernel.append(_event("evt:2", sequence=2, causation_id="evt:missing"))

    assert kernel.events() == (first,)


def test_stream_sequences_are_independent_but_causation_is_global():
    kernel: InMemoryEventKernel[dict] = InMemoryEventKernel()
    alpha = _event()
    beta = _event(
        "evt:beta:1",
        stream_id="stream:beta",
        sequence=1,
        causation_id="evt:1",
    )

    kernel.append(alpha)
    kernel.append(beta)

    assert kernel.stream("stream:alpha") == (alpha,)
    assert kernel.stream("stream:beta") == (beta,)


def test_kernel_rejects_non_envelope_values():
    kernel: InMemoryEventKernel[dict] = InMemoryEventKernel()

    with pytest.raises(InvalidEventEnvelope):
        kernel.append({"event_id": "evt:1"})  # type: ignore[arg-type]
    assert len(kernel) == 0


def test_replay_uses_explicit_reducer_without_mutating_kernel():
    kernel: InMemoryEventKernel[dict[str, int]] = InMemoryEventKernel()
    kernel.append(_event(payload={"delta": 2}))
    kernel.append(
        _event(
            "evt:2",
            sequence=2,
            causation_id="evt:1",
            payload={"delta": 3},
        )
    )

    def reducer(state: dict[str, int], envelope: EventEnvelope) -> dict[str, int]:
        return {"total": state["total"] + envelope.payload["delta"]}

    result = kernel.replay({"total": 0}, reducer)

    assert result == {"total": 5}
    assert len(kernel) == 2
    assert kernel.events()[0].payload == {"delta": 2}


def test_replay_propagates_reducer_failure_and_rejects_none():
    kernel: InMemoryEventKernel[dict] = InMemoryEventKernel()
    kernel.append(_event())

    with pytest.raises(ReducerContractError, match="callable"):
        kernel.replay({}, None)  # type: ignore[arg-type]
    with pytest.raises(ReducerContractError, match="returned None"):
        kernel.replay({}, lambda _state, _event: None)  # type: ignore[arg-type]

    def broken(_state, _event):
        raise RuntimeError("reducer defect")

    with pytest.raises(RuntimeError, match="reducer defect"):
        kernel.replay({}, broken)


def test_module_has_no_persistence_clock_thread_random_or_runtime_integration():
    source = MODULE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_roots = set()
    called_names = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".")[0])
        elif isinstance(node, ast.Call):
            target = node.func
            if isinstance(target, ast.Name):
                called_names.add(target.id)
            elif isinstance(target, ast.Attribute):
                called_names.add(target.attr)

    assert not imported_roots & {
        "asyncio",
        "datetime",
        "pathlib",
        "pickle",
        "random",
        "secrets",
        "sqlite3",
        "threading",
        "time",
        "uuid",
    }
    assert not called_names & {
        "open",
        "save",
        "load",
        "connect",
        "start",
        "sleep",
        "write_text",
        "write_bytes",
    }
    assert "main" not in imported_roots
    assert "language" not in imported_roots
    assert "adapters" not in imported_roots
