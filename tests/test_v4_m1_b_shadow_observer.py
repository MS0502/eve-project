from __future__ import annotations

import ast
from pathlib import Path

import pytest

from adapters.activation_adapter import ActivationAdapter
from core.event_kernel import InMemoryEventKernel
from core.shadow_observer import (
    ACTIVATION_LEARN_PAIR_TARGET,
    FAILURE_EVENT_TYPE,
    SUCCESS_EVENT_TYPE,
    LegacyFunnelShadowObserver,
    ShadowObserverContractError,
    UnknownShadowTarget,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
OBSERVER_PATH = REPO_ROOT / "core/shadow_observer.py"
TARGET_PATH = REPO_ROOT / "adapters/activation_adapter.py"


class FakeSpreadingActivation:
    def __init__(
        self,
        *,
        failure: Exception | None = None,
        trace: list[str] | None = None,
    ) -> None:
        self.learned: list[tuple[str, str, float]] = []
        self.calls: list[tuple[str, tuple]] = []
        self.failure = failure
        self.trace = trace

    def learn_pair(self, a: str, b: str, *, strength: float) -> None:
        if self.trace is not None:
            self.trace.append("legacy")
        self.calls.append(("learn_pair", (a, b, strength)))
        if self.failure is not None:
            raise self.failure
        self.learned.append((a, b, strength))


class FakeWorkingMemory:
    pass


def _adapter(
    *,
    failure: Exception | None = None,
    trace: list[str] | None = None,
) -> ActivationAdapter:
    return ActivationAdapter(
        sa=FakeSpreadingActivation(failure=failure, trace=trace),
        wm=FakeWorkingMemory(),
    )


def _snapshot(adapter: ActivationAdapter, trace: list[str], phase: str):
    def capture():
        trace.append(phase)
        return {
            "calls": [list(item[1]) for item in adapter.sa.calls],
            "learned": [list(item) for item in adapter.sa.learned],
        }

    return capture


def _observe_learn_pair(
    observer: LegacyFunnelShadowObserver,
    adapter: ActivationAdapter,
    *,
    event_id: str,
    trace: list[str],
    causation_id: str | None = None,
):
    return observer.observe_call(
        ACTIVATION_LEARN_PAIR_TARGET.target_id,
        event_id=event_id,
        correlation_id="corr:m1-b",
        causation_id=causation_id,
        legacy_callable=adapter.learn_pair,
        before_snapshot=_snapshot(adapter, trace, "before"),
        after_snapshot=_snapshot(adapter, trace, "after"),
        args=("alpha", "beta"),
        kwargs={"strength": 0.4},
    )


def test_registered_target_matches_actual_legacy_callable_and_wrap_disposition():
    source = TARGET_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    methods = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
    }
    learn_pair = methods["learn_pair"]

    assert ACTIVATION_LEARN_PAIR_TARGET.module_path == (
        "adapters/activation_adapter.py"
    )
    assert ACTIVATION_LEARN_PAIR_TARGET.callable_name == (
        "ActivationAdapter.learn_pair"
    )
    assert ACTIVATION_LEARN_PAIR_TARGET.evidence_range == "103-105"
    assert ACTIVATION_LEARN_PAIR_TARGET.module_disposition == "WRAP"
    assert learn_pair.lineno == 103
    assert learn_pair.end_lineno == 105


def test_success_preserves_legacy_return_state_and_call_order():
    baseline = _adapter()
    trace: list[str] = []
    observed = _adapter(trace=trace)
    baseline_result = baseline.learn_pair("alpha", "beta", strength=0.4)
    kernel = InMemoryEventKernel()
    observer = LegacyFunnelShadowObserver(kernel)

    observed_result = _observe_learn_pair(
        observer,
        observed,
        event_id="shadow:activation:1",
        trace=trace,
    )

    assert observed_result is baseline_result is None
    assert observed.sa.calls == baseline.sa.calls
    assert observed.sa.learned == baseline.sa.learned
    assert trace == ["before", "legacy", "after"]
    assert observer.failures() == ()


def test_success_candidate_is_shadow_only_and_excludes_args_and_result():
    trace: list[str] = []
    adapter = _adapter(trace=trace)
    kernel = InMemoryEventKernel()
    observer = LegacyFunnelShadowObserver(kernel)

    _observe_learn_pair(
        observer,
        adapter,
        event_id="shadow:activation:1",
        trace=trace,
    )

    assert len(kernel) == 1
    event = kernel.events()[0]
    assert event.event_type == SUCCESS_EVENT_TYPE
    assert event.authority == "shadow_only"
    assert event.stream_id == ACTIVATION_LEARN_PAIR_TARGET.stream_id
    assert event.payload["before"]["learned"] == []
    assert event.payload["after"]["learned"] == [["alpha", "beta", 0.4]]
    assert event.payload["legacy_outcome"] == {
        "error_type": None,
        "succeeded": True,
    }
    assert event.causal_context["arguments_captured"] is False
    assert event.causal_context["legacy_result_captured"] is False
    assert "alpha" not in event.causal_context_json
    assert "beta" not in event.causal_context_json


def test_legacy_exception_is_re_raised_unchanged_after_failure_candidate():
    legacy_error = RuntimeError("legacy defect")
    trace: list[str] = []
    adapter = _adapter(failure=legacy_error, trace=trace)
    kernel = InMemoryEventKernel()
    observer = LegacyFunnelShadowObserver(kernel)

    with pytest.raises(RuntimeError) as captured:
        _observe_learn_pair(
            observer,
            adapter,
            event_id="shadow:activation:failure:1",
            trace=trace,
        )

    assert captured.value is legacy_error
    assert trace == ["before", "legacy", "after"]
    assert len(adapter.sa.calls) == 1
    assert adapter.sa.learned == []
    assert observer.failures() == ()
    assert len(kernel) == 1
    event = kernel.events()[0]
    assert event.event_type == FAILURE_EVENT_TYPE
    assert event.payload["legacy_outcome"] == {
        "error_type": "RuntimeError",
        "succeeded": False,
    }
    assert "legacy defect" not in event.payload_json


def test_before_snapshot_failure_is_visible_but_legacy_still_succeeds():
    trace: list[str] = []
    adapter = _adapter(trace=trace)
    kernel = InMemoryEventKernel()
    observer = LegacyFunnelShadowObserver(kernel)

    def broken_before():
        trace.append("before")
        raise RuntimeError("snapshot secret")

    result = observer.observe_call(
        ACTIVATION_LEARN_PAIR_TARGET.target_id,
        event_id="shadow:activation:1",
        correlation_id="corr:m1-b",
        legacy_callable=adapter.learn_pair,
        before_snapshot=broken_before,
        after_snapshot=_snapshot(adapter, trace, "after"),
        args=("alpha", "beta"),
        kwargs={"strength": 0.4},
    )

    assert result is None
    assert adapter.sa.learned == [("alpha", "beta", 0.4)]
    assert trace == ["before", "legacy", "after"]
    assert len(kernel) == 0
    failure = observer.failures()[0]
    assert failure.stage == "before_snapshot"
    assert failure.error_type == "RuntimeError"
    assert failure.legacy_succeeded is None
    assert len(failure.error_message_digest) == 64
    assert "snapshot secret" not in repr(failure)


def test_after_snapshot_failure_does_not_replace_legacy_return():
    adapter = _adapter()
    kernel = InMemoryEventKernel()
    observer = LegacyFunnelShadowObserver(kernel)

    def broken_after():
        raise ValueError("after failure")

    result = observer.observe_call(
        ACTIVATION_LEARN_PAIR_TARGET.target_id,
        event_id="shadow:activation:1",
        correlation_id="corr:m1-b",
        legacy_callable=adapter.learn_pair,
        before_snapshot=lambda: {"learned": []},
        after_snapshot=broken_after,
        args=("alpha", "beta"),
        kwargs={"strength": 0.4},
    )

    assert result is None
    assert adapter.sa.learned == [("alpha", "beta", 0.4)]
    assert len(kernel) == 0
    failure = observer.failures()[0]
    assert failure.stage == "after_snapshot"
    assert failure.legacy_succeeded is True


def test_event_append_failure_is_visible_after_legacy_mutation():
    adapter = _adapter()
    kernel = InMemoryEventKernel()
    observer = LegacyFunnelShadowObserver(kernel)

    _observe_learn_pair(
        observer,
        adapter,
        event_id="shadow:activation:duplicate",
        trace=[],
    )
    result = _observe_learn_pair(
        observer,
        adapter,
        event_id="shadow:activation:duplicate",
        trace=[],
    )

    assert result is None
    assert adapter.sa.learned == [
        ("alpha", "beta", 0.4),
        ("alpha", "beta", 0.4),
    ]
    assert len(kernel) == 1
    failure = observer.failures()[0]
    assert failure.stage == "event_append"
    assert failure.error_type == "DuplicateEventId"
    assert failure.legacy_succeeded is True


def test_causation_and_sequence_advance_only_for_successful_candidates():
    adapter = _adapter()
    kernel = InMemoryEventKernel()
    observer = LegacyFunnelShadowObserver(kernel)

    _observe_learn_pair(
        observer,
        adapter,
        event_id="shadow:activation:1",
        trace=[],
    )
    _observe_learn_pair(
        observer,
        adapter,
        event_id="shadow:activation:2",
        causation_id="shadow:activation:1",
        trace=[],
    )

    first, second = kernel.events()
    assert first.sequence == 1
    assert second.sequence == 2
    assert second.causation_id == first.event_id


def test_unknown_target_invalid_inputs_and_callable_mismatch_fail_before_call():
    adapter = _adapter()
    kernel = InMemoryEventKernel()
    observer = LegacyFunnelShadowObserver(kernel)

    with pytest.raises(UnknownShadowTarget):
        observer.observe_call(
            "legacy.unknown.target",
            event_id="shadow:unknown:1",
            correlation_id="corr:m1-b",
            legacy_callable=adapter.learn_pair,
            before_snapshot=lambda: {},
            after_snapshot=lambda: {},
            args=("alpha", "beta"),
            kwargs={"strength": 0.4},
        )
    with pytest.raises(ShadowObserverContractError):
        observer.observe_call(
            ACTIVATION_LEARN_PAIR_TARGET.target_id,
            event_id="shadow:activation:1",
            correlation_id="corr:m1-b",
            legacy_callable=None,  # type: ignore[arg-type]
            before_snapshot=lambda: {},
            after_snapshot=lambda: {},
        )
    with pytest.raises(ShadowObserverContractError, match="registered bound method"):
        observer.observe_call(
            ACTIVATION_LEARN_PAIR_TARGET.target_id,
            event_id="shadow:activation:2",
            correlation_id="corr:m1-b",
            legacy_callable=lambda: None,
            before_snapshot=lambda: {},
            after_snapshot=lambda: {},
        )
    with pytest.raises(ShadowObserverContractError, match="registered bound method"):
        observer.observe_call(
            ACTIVATION_LEARN_PAIR_TARGET.target_id,
            event_id="shadow:activation:3",
            correlation_id="corr:m1-b",
            legacy_callable=adapter.top_active,
            before_snapshot=lambda: {},
            after_snapshot=lambda: {},
        )

    assert adapter.sa.calls == []
    assert len(kernel) == 0
    assert observer.failures() == ()


def test_target_registry_and_failure_views_are_read_only():
    kernel = InMemoryEventKernel()
    observer = LegacyFunnelShadowObserver(kernel)

    assert observer.target(ACTIVATION_LEARN_PAIR_TARGET.target_id) is (
        ACTIVATION_LEARN_PAIR_TARGET
    )
    assert observer.failures() == ()
    assert isinstance(observer.failures(), tuple)


def test_observer_module_has_no_legacy_import_persistence_clock_or_thread_surface():
    source = OBSERVER_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_roots = set()
    called_names = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".")[0])
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                called_names.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                called_names.add(node.func.attr)

    assert not imported_roots & {
        "adapters",
        "asyncio",
        "datetime",
        "language",
        "main",
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
        "connect",
        "load",
        "open",
        "save",
        "sleep",
        "start",
        "write_bytes",
        "write_text",
    }


def test_observer_exception_handlers_are_never_silent():
    source = OBSERVER_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    handlers = [node for node in ast.walk(tree) if isinstance(node, ast.ExceptHandler)]

    assert handlers
    for handler in handlers:
        assert any(
            isinstance(node, ast.Raise)
            or (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "_record_failure"
            )
            for node in ast.walk(handler)
        )
