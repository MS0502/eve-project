#!/usr/bin/env python3
"""Expanded, disconnected M1 mechanism observation campaign.

The campaign proves the shadow-observer mechanism across reviewed WRAP and
REWRITE call paths. It never installs an observer into the production runtime,
never changes a default, and confines the one direct-write probe to temporary
storage that is removed before the result is returned.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import adapters.persistence_adapter as persistence_module
from adapters.activation_adapter import ActivationAdapter
from adapters.live_loop import LiveLoop
from adapters.persistence_adapter import PersistenceAdapter
from core.event_kernel import InMemoryEventKernel, SHADOW_AUTHORITY, canonical_json_object
from core.shadow_observer import (
    ACTIVATION_LEARN_PAIR_TARGET,
    FAILURE_EVENT_TYPE,
    SUCCESS_EVENT_TYPE,
    LegacyFunnelShadowObserver,
    ShadowTarget,
)
from core.shadow_projection import (
    ActivationLearnPairShadowState,
    ShadowProjectionError,
    reduce_activation_learn_pair,
)
from legacy.eve_modules.spreading_activation import SpreadingActivation

CAMPAIGN_SCHEMA_VERSION = "eve.m1-extended-controlled-observation.v1"
CAMPAIGN_ID = "m1:extended-controlled-observation:mechanism:v1"
CORRELATION_ID = "corr:m1-extended-controlled-observation"
BASELINE_SHA = "847621bcd61634958ce505108ade491c50ced0d4"

LIVE_LOOP_DRAIN_TARGET = ShadowTarget(
    target_id="legacy.live_loop.drain_user_inputs",
    module_path="adapters/live_loop.py",
    callable_name="LiveLoop._drain_user_inputs",
    evidence_range="68-77",
    module_disposition="REWRITE",
    stream_id="shadow:legacy.live_loop.drain_user_inputs",
)
PERSISTENCE_SAVE_TARGET = ShadowTarget(
    target_id="legacy.persistence.save",
    module_path="adapters/persistence_adapter.py",
    callable_name="PersistenceAdapter.save",
    evidence_range="54-80",
    module_disposition="REWRITE",
    stream_id="shadow:legacy.persistence.save",
)
EXTENDED_TARGETS = (
    ACTIVATION_LEARN_PAIR_TARGET,
    LIVE_LOOP_DRAIN_TARGET,
    PERSISTENCE_SAVE_TARGET,
)
REQUIRED_MUTATION_FORMS = (
    "attribute_assignment",
    "subscript_assignment",
    "augmented_assignment",
    "mutating_method_call",
    "direct_write",
)
STANDALONE_TICK_STEPS = 4


class ExtendedCampaignError(RuntimeError):
    """Raised when a bounded evidence invariant is not met."""


def _canonical(value: Mapping[str, Any], field: str) -> str:
    return canonical_json_object(value, field=field)


def _sha(value: Mapping[str, Any], field: str) -> str:
    return hashlib.sha256(_canonical(value, field).encode("utf-8")).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _event_record(event: Any) -> dict[str, Any]:
    return {
        "authority": event.authority,
        "causal_context": event.causal_context,
        "causation_id": event.causation_id,
        "correlation_id": event.correlation_id,
        "digest": event.digest,
        "event_id": event.event_id,
        "event_type": event.event_type,
        "payload": event.payload,
        "producer": event.producer,
        "producer_version": event.producer_version,
        "schema_version": event.schema_version,
        "sequence": event.sequence,
        "stream_id": event.stream_id,
    }


def _directory_snapshot(root: Path) -> dict[str, Any]:
    files = []
    if root.exists():
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            files.append(
                {
                    "relative_path": path.relative_to(root).as_posix(),
                    "sha256": _file_sha(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    return {"files": files}


class _NoopWorkingMemory:
    def decay(self, dt: float) -> None:
        return None


class _DelegatingObservedSpreadingActivation:
    """Deterministic ledger around the retained spreading-activation class."""

    def __init__(self, *, fail_on_call: int, failure: RuntimeError) -> None:
        self.inner = SpreadingActivation()
        self.calls: list[tuple[str, str, float]] = []
        self.learned: list[tuple[str, str, float]] = []
        self.fail_on_call = fail_on_call
        self.failure = failure
        self.trace: list[str] = []

    def learn_pair(self, a: str, b: str, *, strength: float) -> None:
        record = (a, b, float(strength))
        self.calls.append(record)
        self.trace.append(f"legacy:{len(self.calls)}")
        if len(self.calls) == self.fail_on_call:
            raise self.failure
        self.inner.learn_pair(a, b, strength=strength)
        self.learned.append(record)

    def decay(self, dt: float) -> None:
        self.inner.decay(dt)

    def snapshot(self) -> dict[str, Any]:
        return {
            "calls": [list(item) for item in self.calls],
            "learned": [list(item) for item in self.learned],
        }

    def actual_state(self) -> dict[str, Any]:
        return {
            "time": float(self.inner.time),
            "weights": [
                [left, right, float(weight)]
                for (left, right), weight in sorted(self.inner.weights.items())
            ],
            "neighbors": [
                [category, sorted(values)]
                for category, values in sorted(self.inner.neighbors.items())
            ],
        }


class _DialogueEngine:
    def __init__(self, trace: list[str]) -> None:
        self.trace = trace
        self.teaching_adapter = None
        self.autonomy_adapter = None

    def chat_stream(self, text: str) -> Iterable[str]:
        self.trace.append(f"chat:{text}")
        return ("controlled", "reply")


class _PersistenceEngine:
    def __init__(self) -> None:
        self.hormone_adapter = None
        self.activation_adapter = None
        self.memory_adapter = None
        self.nl_adapter = None
        self.sd_adapter = None
        self.dmn_adapter = None
        self.vsa_adapter = None
        self.ai_adapter = None
        self.goal_adapter = None
        self.norm_adapter = None
        self.history: list[Any] = []
        self.task_solver = None


class _BlockingHormoneSystem:
    def __init__(
        self,
        entered: threading.Event,
        release: threading.Event,
        trace: list[str],
    ) -> None:
        self.entered = entered
        self.release = release
        self.trace = trace

    def update(self, dt: float) -> None:
        self.trace.append("tick:entered")
        self.entered.set()
        if not self.release.wait(timeout=3.0):
            raise RuntimeError("controlled tick release timeout")
        self.trace.append("tick:released")


@dataclass(frozen=True, slots=True)
class _ThreadHormoneAdapter:
    hs: _BlockingHormoneSystem


class _NoopAutonomousLoop:
    def step(self, emit: bool = True) -> dict[str, Any]:
        return {"emitted": False, "response": None}


class _ThreadEngine:
    def __init__(self, hs: _BlockingHormoneSystem) -> None:
        self.hormone_adapter = _ThreadHormoneAdapter(hs)
        self.salience_adapter = None
        self.autonomy_adapter = None
        self.autonomous_loop = _NoopAutonomousLoop()
        self.urge_adapter = None
        self.safety_adapter = None
        self.teaching_adapter = None

    def proactive_stream(self, force: bool = False) -> Iterable[str]:
        return ()

    def chat_stream(self, text: str) -> Iterable[str]:
        return ()


def _live_snapshot(loop: LiveLoop, emissions: list[str]) -> dict[str, Any]:
    return {
        "emissions": list(emissions),
        "processed_input_count": int(loop.processed_input_count),
        "queue_size": int(loop._user_input_queue.qsize()),
    }


def _run_live_loop_drain(
    observer: LegacyFunnelShadowObserver,
    kernel: InMemoryEventKernel[Any],
    *,
    causation_id: str | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    observed_trace: list[str] = []
    baseline_trace: list[str] = []
    observed_emissions: list[str] = []
    baseline_emissions: list[str] = []
    observed = LiveLoop(
        _DialogueEngine(observed_trace),
        interval=60.0,
        emit_callback=observed_emissions.append,
    )
    baseline = LiveLoop(
        _DialogueEngine(baseline_trace),
        interval=60.0,
        emit_callback=baseline_emissions.append,
    )
    observed.push_user_input("controlled input")
    baseline.push_user_input("controlled input")
    initial = _live_snapshot(observed, observed_emissions)
    event_id = "m1-extended:event:live-drain:001"
    before_count = len(kernel.events())

    def before() -> Mapping[str, Any]:
        observed_trace.append("snapshot:before")
        return _live_snapshot(observed, observed_emissions)

    def after() -> Mapping[str, Any]:
        observed_trace.append("snapshot:after")
        return _live_snapshot(observed, observed_emissions)

    observed_result = observer.observe_call(
        LIVE_LOOP_DRAIN_TARGET.target_id,
        event_id=event_id,
        correlation_id=CORRELATION_ID,
        causation_id=causation_id,
        legacy_callable=observed._drain_user_inputs,
        before_snapshot=before,
        after_snapshot=after,
    )
    baseline_result = baseline._drain_user_inputs()
    final = _live_snapshot(observed, observed_emissions)
    baseline_final = _live_snapshot(baseline, baseline_emissions)
    event_delta = len(kernel.events()) - before_count
    if event_delta != 1:
        raise ExtendedCampaignError("live-loop drain did not emit exactly one candidate")
    if observed_result is not baseline_result or final != baseline_final:
        raise ExtendedCampaignError("live-loop observer changed retained behavior")
    return initial, {
        "baseline_snapshot": baseline_final,
        "event_id": event_id,
        "event_delta": event_delta,
        "final_snapshot": final,
        "observed_trace": observed_trace,
        "state_matches_unobserved": final == baseline_final,
    }


def _run_persistence_write(
    observer: LegacyFunnelShadowObserver,
    kernel: InMemoryEventKernel[Any],
    *,
    causation_id: str | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    trace: list[str] = []
    original_descriptor = persistence_module.Persistence.__dict__["save"]

    def controlled_v40_save(
        _mock: Any,
        path: str,
        compress: bool = True,
    ) -> dict[str, Any]:
        trace.append("legacy:persistence-save")
        return {"path": path, "controlled": True, "compress": bool(compress)}

    setattr(
        persistence_module.Persistence,
        "save",
        staticmethod(controlled_v40_save),
    )
    observed_root_path: Path | None = None
    baseline_root_path: Path | None = None
    try:
        with tempfile.TemporaryDirectory(prefix="eve-m1-observed-") as observed_dir, tempfile.TemporaryDirectory(
            prefix="eve-m1-baseline-"
        ) as baseline_dir:
            observed_root_path = Path(observed_dir)
            baseline_root_path = Path(baseline_dir)
            observed = PersistenceAdapter(_PersistenceEngine())
            baseline = PersistenceAdapter(_PersistenceEngine())
            initial = _directory_snapshot(observed_root_path)
            event_id = "m1-extended:event:persistence-save:001"
            before_count = len(kernel.events())

            def before() -> Mapping[str, Any]:
                trace.append("snapshot:before")
                return _directory_snapshot(observed_root_path)

            def after() -> Mapping[str, Any]:
                trace.append("snapshot:after")
                return _directory_snapshot(observed_root_path)

            observer.observe_call(
                PERSISTENCE_SAVE_TARGET.target_id,
                event_id=event_id,
                correlation_id=CORRELATION_ID,
                causation_id=causation_id,
                legacy_callable=observed.save,
                before_snapshot=before,
                after_snapshot=after,
                args=(str(observed_root_path / "state"),),
                kwargs={"compress": False},
            )
            baseline.save(str(baseline_root_path / "state"), compress=False)
            final = _directory_snapshot(observed_root_path)
            baseline_final = _directory_snapshot(baseline_root_path)
            event_delta = len(kernel.events()) - before_count
            if event_delta != 1:
                raise ExtendedCampaignError(
                    "persistence write did not emit exactly one candidate"
                )
            if final != baseline_final:
                raise ExtendedCampaignError(
                    "persistence observer changed controlled direct-write result"
                )
            if [row["relative_path"] for row in final["files"]] != [
                "state.v41sidecar"
            ]:
                raise ExtendedCampaignError("direct-write probe escaped bounded sidecar")
            scenario = {
                "baseline_snapshot": baseline_final,
                "controlled_legacy_save_replaced": True,
                "event_delta": event_delta,
                "event_id": event_id,
                "final_snapshot": final,
                "state_matches_unobserved": final == baseline_final,
                "trace": list(trace),
            }
    finally:
        setattr(persistence_module.Persistence, "save", original_descriptor)
    cleanup_verified = bool(
        observed_root_path is not None
        and baseline_root_path is not None
        and not observed_root_path.exists()
        and not baseline_root_path.exists()
    )
    scenario["temporary_roots_removed"] = cleanup_verified
    if not cleanup_verified:
        raise ExtendedCampaignError("temporary direct-write roots were not removed")
    return initial, scenario


def _activation_snapshot(
    ledger: _DelegatingObservedSpreadingActivation,
    phase: str,
) -> Mapping[str, Any]:
    ledger.trace.append(f"snapshot:{phase}")
    return ledger.snapshot()


def _run_concurrent_activation(
    observer: LegacyFunnelShadowObserver,
    kernel: InMemoryEventKernel[Any],
    *,
    causation_id: str | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    observed_error = RuntimeError("controlled activation failure")
    baseline_error = RuntimeError("controlled activation baseline failure")
    observed_ledger = _DelegatingObservedSpreadingActivation(
        fail_on_call=2,
        failure=observed_error,
    )
    baseline_ledger = _DelegatingObservedSpreadingActivation(
        fail_on_call=2,
        failure=baseline_error,
    )
    observed = ActivationAdapter(sa=observed_ledger, wm=_NoopWorkingMemory())
    baseline = ActivationAdapter(sa=baseline_ledger, wm=_NoopWorkingMemory())
    initial = observed_ledger.snapshot()

    entered = threading.Event()
    release = threading.Event()
    thread_trace: list[str] = []
    loop = LiveLoop(
        _ThreadEngine(_BlockingHormoneSystem(entered, release, thread_trace)),
        interval=60.0,
        emit_callback=lambda _text: None,
    )
    event_id_success = "m1-extended:event:activation:001"
    event_id_failure = "m1-extended:event:activation:002"
    events_before_start = len(kernel.events())
    started = loop.start()
    try:
        barrier_reached = entered.wait(timeout=2.0)
        thread = loop._thread
        thread_alive_before = bool(thread is not None and thread.is_alive())
        tick_count_at_barrier = int(loop.tick_count)
        events_at_barrier = len(kernel.events())
        if not started or not barrier_reached or not thread_alive_before:
            raise ExtendedCampaignError("live tick thread did not reach controlled barrier")
        if tick_count_at_barrier != 1:
            raise ExtendedCampaignError("controlled tick barrier was not the first tick")
        observer.observe_call(
            ACTIVATION_LEARN_PAIR_TARGET.target_id,
            event_id=event_id_success,
            correlation_id=CORRELATION_ID,
            causation_id=causation_id,
            legacy_callable=observed.learn_pair,
            before_snapshot=lambda: _activation_snapshot(observed_ledger, "before:1"),
            after_snapshot=lambda: _activation_snapshot(observed_ledger, "after:1"),
            args=("alpha", "beta"),
            kwargs={"strength": 0.25},
        )
        baseline.learn_pair("alpha", "beta", strength=0.25)
        thread_alive_after = bool(thread is not None and thread.is_alive())
        events_after_mutation = len(kernel.events())
    finally:
        release.set()
        loop.stop()
    thread_stopped = bool(loop._thread is not None and not loop._thread.is_alive())

    propagated_identity = False
    observed_outcome = "failure"
    try:
        observer.observe_call(
            ACTIVATION_LEARN_PAIR_TARGET.target_id,
            event_id=event_id_failure,
            correlation_id=CORRELATION_ID,
            causation_id=event_id_success,
            legacy_callable=observed.learn_pair,
            before_snapshot=lambda: _activation_snapshot(observed_ledger, "before:2"),
            after_snapshot=lambda: _activation_snapshot(observed_ledger, "after:2"),
            args=("beta", "gamma"),
            kwargs={"strength": 0.5},
        )
        observed_outcome = "success"
    except RuntimeError as exc:
        propagated_identity = exc is observed_error
    baseline_outcome = "failure"
    try:
        baseline.learn_pair("beta", "gamma", strength=0.5)
        baseline_outcome = "success"
    except RuntimeError as exc:
        if exc is not baseline_error:
            raise
    if observed_outcome != baseline_outcome:
        raise ExtendedCampaignError("activation failure outcome diverged from baseline")

    events_before_ticks = len(kernel.events())
    for _ in range(STANDALONE_TICK_STEPS):
        observed.tick(dt=1.0)
        baseline.tick(dt=1.0)
    standalone_tick_event_delta = len(kernel.events()) - events_before_ticks
    final = observed_ledger.snapshot()
    baseline_final = baseline_ledger.snapshot()
    actual_state_matches = observed_ledger.actual_state() == baseline_ledger.actual_state()
    if final != baseline_final or not actual_state_matches:
        raise ExtendedCampaignError("activation state diverged from unobserved baseline")
    if standalone_tick_event_delta != 0:
        raise ExtendedCampaignError("standalone continuous ticks emitted candidates")

    return initial, {
        "actual_state_matches_unobserved": actual_state_matches,
        "event_ids": [event_id_success, event_id_failure],
        "exception_identity_preserved": propagated_identity,
        "final_snapshot": final,
        "legacy_snapshot_matches_unobserved": final == baseline_final,
        "live_tick_event_delta": events_at_barrier - events_before_start,
        "mutation_event_delta_while_thread_alive": (
            events_after_mutation - events_at_barrier
        ),
        "standalone_tick_event_delta": standalone_tick_event_delta,
        "standalone_tick_steps": STANDALONE_TICK_STEPS,
        "thread_alive_after_mutation": thread_alive_after,
        "thread_alive_before_mutation": thread_alive_before,
        "thread_barrier_reached": barrier_reached,
        "thread_started": started,
        "thread_stopped": thread_stopped,
        "thread_trace": thread_trace,
        "tick_count_at_barrier": tick_count_at_barrier,
        "trace": list(observed_ledger.trace),
    }


def _run_observer_failure_probe() -> dict[str, Any]:
    observed_trace: list[str] = []
    baseline_trace: list[str] = []
    observed_emissions: list[str] = []
    baseline_emissions: list[str] = []
    observed = LiveLoop(
        _DialogueEngine(observed_trace),
        interval=60.0,
        emit_callback=observed_emissions.append,
    )
    baseline = LiveLoop(
        _DialogueEngine(baseline_trace),
        interval=60.0,
        emit_callback=baseline_emissions.append,
    )
    observed.push_user_input("observer failure probe")
    baseline.push_user_input("observer failure probe")
    kernel: InMemoryEventKernel[Any] = InMemoryEventKernel()
    observer = LegacyFunnelShadowObserver(kernel, targets=EXTENDED_TARGETS)

    def broken_before() -> Mapping[str, Any]:
        raise RuntimeError("controlled extended observer snapshot failure")

    observed_result = observer.observe_call(
        LIVE_LOOP_DRAIN_TARGET.target_id,
        event_id="m1-extended:observer-failure:001",
        correlation_id=CORRELATION_ID,
        legacy_callable=observed._drain_user_inputs,
        before_snapshot=broken_before,
        after_snapshot=lambda: _live_snapshot(observed, observed_emissions),
    )
    baseline_result = baseline._drain_user_inputs()
    failures = observer.failures()
    if len(failures) != 1 or kernel.events():
        raise ExtendedCampaignError("observer failure visibility contract changed")
    failure = failures[0]
    observed_final = _live_snapshot(observed, observed_emissions)
    baseline_final = _live_snapshot(baseline, baseline_emissions)
    return {
        "error_message_digest": failure.error_message_digest,
        "error_type": failure.error_type,
        "event_count": len(kernel.events()),
        "event_id": failure.event_id,
        "legacy_state_preserved": observed_final == baseline_final,
        "return_value_preserved": observed_result is baseline_result,
        "stage": failure.stage,
        "target_id": failure.target_id,
    }


def _generic_replay_event(
    state: Mapping[str, Any],
    event: Any,
    target: ShadowTarget,
    expected_sequence: int,
) -> tuple[dict[str, Any], list[str]]:
    mismatches: list[str] = []
    payload = event.payload
    expected_target = {
        "callable": target.callable_name,
        "disposition": target.module_disposition,
        "module_path": target.module_path,
        "target_id": target.target_id,
    }
    if event.stream_id != target.stream_id:
        mismatches.append("stream_id")
    if event.sequence != expected_sequence:
        mismatches.append("sequence")
    if payload.get("target") != expected_target:
        mismatches.append("target_metadata")
    if payload.get("before") != dict(state):
        mismatches.append("before_snapshot")
    after = payload.get("after")
    if not isinstance(after, Mapping):
        mismatches.append("after_snapshot")
        return dict(state), mismatches
    return dict(after), mismatches


def _replay_all(
    events: tuple[Any, ...],
    initial_by_target: Mapping[str, Mapping[str, Any]],
    final_by_target: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    target_by_id = {target.target_id: target for target in EXTENDED_TARGETS}
    target_by_stream = {target.stream_id: target for target in EXTENDED_TARGETS}
    generic_states = {
        target_id: dict(snapshot)
        for target_id, snapshot in initial_by_target.items()
        if target_id != ACTIVATION_LEARN_PAIR_TARGET.target_id
    }
    activation_state = ActivationLearnPairShadowState.from_initial_snapshot(
        initial_by_target[ACTIVATION_LEARN_PAIR_TARGET.target_id]
    )
    generic_sequences = {
        target_id: 0 for target_id in generic_states
    }
    rows: list[dict[str, Any]] = []
    divergences: list[dict[str, Any]] = []
    for event in events:
        target = target_by_stream.get(event.stream_id)
        mismatch_codes: list[str] = []
        if target is None:
            mismatch_codes.append("unknown_stream")
            target_id = "unknown"
        else:
            target_id = target.target_id
            if target_id == ACTIVATION_LEARN_PAIR_TARGET.target_id:
                try:
                    activation_state = reduce_activation_learn_pair(
                        activation_state,
                        event,
                    )
                    if activation_state.snapshot != event.payload["after"]:
                        mismatch_codes.append("projected_after")
                except ShadowProjectionError as exc:
                    mismatch_codes.append(f"reducer:{type(exc).__name__}")
            else:
                generic_sequences[target_id] += 1
                generic_states[target_id], mismatch_codes = _generic_replay_event(
                    generic_states[target_id],
                    event,
                    target,
                    generic_sequences[target_id],
                )
        row = {
            "event_digest": event.digest,
            "event_id": event.event_id,
            "matches": not mismatch_codes,
            "mismatch_codes": mismatch_codes,
            "sequence": event.sequence,
            "stream_id": event.stream_id,
            "target_id": target_id,
        }
        rows.append(row)
        if mismatch_codes:
            divergences.append(row)

    final_states: dict[str, Mapping[str, Any]] = {
        ACTIVATION_LEARN_PAIR_TARGET.target_id: activation_state.snapshot,
        **generic_states,
    }
    final_rows = []
    for target_id in sorted(target_by_id):
        actual = dict(final_states[target_id])
        expected = dict(final_by_target[target_id])
        final_rows.append(
            {
                "actual_digest": _sha(actual, f"replay-final:{target_id}:actual"),
                "expected_digest": _sha(
                    expected,
                    f"replay-final:{target_id}:expected",
                ),
                "matches": actual == expected,
                "target_id": target_id,
            }
        )
    matching = sum(1 for row in rows if row["matches"])
    return {
        "compared_events": len(rows),
        "divergence_count": len(divergences),
        "divergences": divergences,
        "final_equivalence": final_rows,
        "match_rate": {
            "denominator": len(rows),
            "numerator": matching,
            "value": matching / len(rows),
        },
        "rows": rows,
    }


def _mutation_form_rows(replay: Mapping[str, Any]) -> list[dict[str, Any]]:
    replay_matches = {
        row["event_id"]: bool(row["matches"])
        for row in replay["rows"]
    }
    rows = [
        {
            "form": "attribute_assignment",
            "path": "adapters/live_loop.py",
            "line_range": "101-105",
            "call_path": "LiveLoop._drain_user_inputs -> LiveLoop._handle_user_input",
            "target_id": LIVE_LOOP_DRAIN_TARGET.target_id,
            "event_ids": ["m1-extended:event:live-drain:001"],
        },
        {
            "form": "subscript_assignment",
            "path": "legacy/eve_modules/spreading_activation.py",
            "line_range": "239-243",
            "call_path": "ActivationAdapter.learn_pair -> SpreadingActivation.learn_pair",
            "target_id": ACTIVATION_LEARN_PAIR_TARGET.target_id,
            "event_ids": ["m1-extended:event:activation:001"],
        },
        {
            "form": "augmented_assignment",
            "path": "adapters/live_loop.py",
            "line_range": "68-77",
            "call_path": "LiveLoop._drain_user_inputs",
            "target_id": LIVE_LOOP_DRAIN_TARGET.target_id,
            "event_ids": ["m1-extended:event:live-drain:001"],
        },
        {
            "form": "mutating_method_call",
            "path": "legacy/eve_modules/spreading_activation.py",
            "line_range": "241-243",
            "call_path": "ActivationAdapter.learn_pair -> SpreadingActivation.learn_pair",
            "target_id": ACTIVATION_LEARN_PAIR_TARGET.target_id,
            "event_ids": ["m1-extended:event:activation:001"],
        },
        {
            "form": "direct_write",
            "path": "adapters/persistence_adapter.py",
            "line_range": "65-74",
            "call_path": "PersistenceAdapter.save",
            "target_id": PERSISTENCE_SAVE_TARGET.target_id,
            "event_ids": ["m1-extended:event:persistence-save:001"],
        },
    ]
    for row in rows:
        row["observed"] = True
        row["replay_matches"] = all(
            replay_matches.get(event_id, False) for event_id in row["event_ids"]
        )
    return rows


def run_extended_controlled_observation_campaign() -> dict[str, Any]:
    kernel: InMemoryEventKernel[Any] = InMemoryEventKernel()
    observer = LegacyFunnelShadowObserver(kernel, targets=EXTENDED_TARGETS)

    live_initial, live = _run_live_loop_drain(
        observer,
        kernel,
        causation_id=None,
    )
    persistence_initial, persistence = _run_persistence_write(
        observer,
        kernel,
        causation_id=live["event_id"],
    )
    activation_initial, activation = _run_concurrent_activation(
        observer,
        kernel,
        causation_id=persistence["event_id"],
    )
    observer_failure = _run_observer_failure_probe()

    events = kernel.events()
    initial_by_target = {
        ACTIVATION_LEARN_PAIR_TARGET.target_id: activation_initial,
        LIVE_LOOP_DRAIN_TARGET.target_id: live_initial,
        PERSISTENCE_SAVE_TARGET.target_id: persistence_initial,
    }
    final_by_target = {
        ACTIVATION_LEARN_PAIR_TARGET.target_id: activation["final_snapshot"],
        LIVE_LOOP_DRAIN_TARGET.target_id: live["final_snapshot"],
        PERSISTENCE_SAVE_TARGET.target_id: persistence["final_snapshot"],
    }
    replay = _replay_all(events, initial_by_target, final_by_target)
    mutation_forms = _mutation_form_rows(replay)
    success_count = sum(
        1 for event in events if event.event_type == SUCCESS_EVENT_TYPE
    )
    failure_count = sum(
        1 for event in events if event.event_type == FAILURE_EVENT_TYPE
    )
    target_rows = [
        {
            "callable": target.callable_name,
            "disposition": target.module_disposition,
            "evidence_range": target.evidence_range,
            "module_path": target.module_path,
            "stream_id": target.stream_id,
            "target_id": target.target_id,
        }
        for target in EXTENDED_TARGETS
    ]
    event_ids = [event.event_id for event in events]
    event_counts_by_id = {
        event_id: event_ids.count(event_id) for event_id in sorted(set(event_ids))
    }
    max_events_per_observed_call = max(event_counts_by_id.values())
    all_mutation_forms = (
        tuple(sorted(row["form"] for row in mutation_forms))
        == tuple(sorted(REQUIRED_MUTATION_FORMS))
        and all(row["observed"] and row["replay_matches"] for row in mutation_forms)
    )
    final_replay_matches = all(
        row["matches"] for row in replay["final_equivalence"]
    )
    machine_passed = all(
        (
            all_mutation_forms,
            len(target_rows) == 3,
            {row["disposition"] for row in target_rows} == {"WRAP", "REWRITE"},
            replay["divergence_count"] == 0,
            replay["match_rate"]["numerator"] == replay["match_rate"]["denominator"],
            final_replay_matches,
            failure_count == 1,
            observer_failure["event_count"] == 0,
            observer_failure["legacy_state_preserved"],
            observer_failure["return_value_preserved"],
            activation["thread_alive_before_mutation"],
            activation["thread_alive_after_mutation"],
            activation["mutation_event_delta_while_thread_alive"] == 1,
            activation["live_tick_event_delta"] == 0,
            activation["standalone_tick_event_delta"] == 0,
            activation["exception_identity_preserved"],
            activation["legacy_snapshot_matches_unobserved"],
            activation["actual_state_matches_unobserved"],
            live["state_matches_unobserved"],
            persistence["state_matches_unobserved"],
            persistence["temporary_roots_removed"],
            max_events_per_observed_call == 1,
            observer.failures() == (),
        )
    )
    source_record = {
        "activation": activation,
        "events": [_event_record(event) for event in events],
        "live_loop": live,
        "mutation_forms": mutation_forms,
        "observer_failure": observer_failure,
        "persistence": persistence,
        "replay": replay,
        "targets": target_rows,
    }
    result = {
        "authority": SHADOW_AUTHORITY,
        "baseline_sha": BASELINE_SHA,
        "campaign_id": CAMPAIGN_ID,
        "campaign_schema_version": CAMPAIGN_SCHEMA_VERSION,
        "events": [_event_record(event) for event in events],
        "failure_visibility": {
            "legacy_failure_event_count": failure_count,
            "legacy_failure_visible": failure_count == 1,
            "observer_failure": observer_failure,
        },
        "granularity": {
            "candidate_events": len(events),
            "discrete_observed_calls": len(events),
            "events_during_live_tick_before_mutation": activation[
                "live_tick_event_delta"
            ],
            "events_during_standalone_tick_steps": activation[
                "standalone_tick_event_delta"
            ],
            "max_events_per_observed_call": max_events_per_observed_call,
            "standalone_tick_steps": activation["standalone_tick_steps"],
        },
        "human_gate": {
            "eligible_for_human_review": machine_passed,
            "human_accepted": False,
            "human_review_status": "required_not_performed",
            "v4_2_eligible": False,
        },
        "legacy_preservation": {
            "activation_actual_state_matches_unobserved": activation[
                "actual_state_matches_unobserved"
            ],
            "activation_exception_identity_preserved": activation[
                "exception_identity_preserved"
            ],
            "activation_snapshot_matches_unobserved": activation[
                "legacy_snapshot_matches_unobserved"
            ],
            "live_loop_state_matches_unobserved": live[
                "state_matches_unobserved"
            ],
            "persistence_state_matches_unobserved": persistence[
                "state_matches_unobserved"
            ],
        },
        "machine_gate": {
            "machine_passed": machine_passed,
            "status": (
                "extended_mechanism_evidence_complete"
                if machine_passed
                else "extended_mechanism_evidence_failed"
            ),
        },
        "mutation_classification": {
            "required_forms": list(REQUIRED_MUTATION_FORMS),
            "rows": mutation_forms,
        },
        "observation_window": {
            "adapter_call_paths": target_rows,
            "adapter_count": len(target_rows),
            "controlled_direct_write": True,
            "controlled_direct_write_cleanup_verified": persistence[
                "temporary_roots_removed"
            ],
            "production_observer_installed": False,
            "production_persistence_enabled": False,
            "runtime_integrated": False,
            "scenario_count": 4,
            "scenarios": [
                "live_loop_discrete_drain_transition",
                "persistence_ephemeral_direct_write_transition",
                "activation_success_and_failure_during_live_tick_thread",
                "observer_failure_isolation_on_rewrite_target",
            ],
        },
        "raw_observations": {
            "activation": activation,
            "live_loop": live,
            "persistence": persistence,
        },
        "replay_equivalence": replay,
        "source_evidence_sha256": _sha(
            source_record,
            "m1_extended_controlled_observation_source",
        ),
        "success_event_count": success_count,
        "unauthorized_effects": {
            "defaults_changed": False,
            "external_effects_outside_temporary_roots": False,
            "legacy_authority_changed": False,
            "production_persistence_changed": False,
        },
    }
    if not machine_passed:
        raise ExtendedCampaignError("expanded controlled campaign did not pass")
    return result


def canonical_raw_text(result: Mapping[str, Any]) -> str:
    return json.dumps(
        dict(result),
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ) + "\n"


def render_evidence_markdown(
    result: Mapping[str, Any],
    raw_artifact_sha256: str,
) -> str:
    replay = result["replay_equivalence"]
    concurrency = result["raw_observations"]["activation"]
    granularity = result["granularity"]
    forms = result["mutation_classification"]["rows"]
    targets = result["observation_window"]["adapter_call_paths"]
    form_lines = "\n".join(
        f"| `{row['form']}` | `{row['path']}:{row['line_range']}` | "
        f"`{row['target_id']}` | `{str(row['replay_matches']).lower()}` |"
        for row in forms
    )
    target_lines = "\n".join(
        f"| `{row['callable']}` | `{row['disposition']}` | "
        f"`{row['module_path']}:{row['evidence_range']}` |"
        for row in targets
    )
    return f"""# M1 Extended Controlled Observation Evidence

Campaign schema: `{result['campaign_schema_version']}`

Campaign ID: `{result['campaign_id']}`

Baseline: `main` at `{result['baseline_sha']}`

Raw observation artifact SHA-256: `{raw_artifact_sha256}`

Status: **extended mechanism machine evidence complete; M1 human acceptance not performed**

## Boundary

This is a disconnected controlled window. It uses the existing after-the-fact
observer with a campaign-local reviewed target registry. No observer is installed
into production, no legacy authority changes, no default changes, and no
production persistence is enabled. The direct-write probe is confined to two
temporary roots and both roots are removed before the campaign returns.

## Mechanism acceptance criteria

### Mutation forms

| M0-A form | Executed source | Observed target | Replay match |
|---|---|---|---|
{form_lines}

All five required forms were executed at least once and tied to raw before/after
events. The rows are mechanism evidence, not a claim that all historical mutation
sites are covered or safe.

### Multiple adapter dispositions

| Bound call path | M0-D disposition | Evidence location |
|---|---|---|
{target_lines}

Observed adapter call paths: `{result['observation_window']['adapter_count']}`.
The window includes both `WRAP` and `REWRITE` dispositions.

### Concurrency

```text
live tick thread reached barrier: {str(concurrency['thread_barrier_reached']).lower()}
thread alive before mutation: {str(concurrency['thread_alive_before_mutation']).lower()}
thread alive after mutation: {str(concurrency['thread_alive_after_mutation']).lower()}
tick count at barrier: {concurrency['tick_count_at_barrier']}
mutation candidates while thread alive: {concurrency['mutation_event_delta_while_thread_alive']}
tick candidates before mutation: {concurrency['live_tick_event_delta']}
thread stopped and joined: {str(concurrency['thread_stopped']).lower()}
```

### Replay and failure visibility

```text
compared events: {replay['compared_events']}
matching events: {replay['match_rate']['numerator']}
match rate: {replay['match_rate']['numerator']} / {replay['match_rate']['denominator']} = {replay['match_rate']['value']}
divergence count: {replay['divergence_count']}
complete divergence list: {json.dumps(replay['divergences'], ensure_ascii=False, sort_keys=True)}
legacy failure events: {result['failure_visibility']['legacy_failure_event_count']}
observer failure records: 1
observer failure emitted candidate: {result['failure_visibility']['observer_failure']['event_count']}
```

The controlled legacy exception remains visible as a failure event and the exact
exception object is re-raised. A separate observer-snapshot failure remains
visible in the observer failure ledger while the retained call still completes
and produces no candidate.

### Granularity and no amplification

```text
discrete observed calls: {granularity['discrete_observed_calls']}
candidate events: {granularity['candidate_events']}
maximum events per observed call: {granularity['max_events_per_observed_call']}
standalone tick steps: {granularity['standalone_tick_steps']}
events during standalone tick steps: {granularity['events_during_standalone_tick_steps']}
events from live tick before discrete mutation: {granularity['events_during_live_tick_before_mutation']}
```

The measured policy remains one candidate per discrete observed call and zero
candidates for continuous tick/decay alone. The live-loop drain contains several
low-level mutations but produces one call-boundary candidate.

## Raw-data sufficiency

`docs/audit/M1_EXTENDED_CONTROLLED_OBSERVATION_RAW.json` contains every event
envelope, before/after state, per-event replay row, complete divergence ledger,
mutation-form source row, thread-barrier observation, temporary-file name/size/hash,
legacy-failure record, observer-failure record, and final equivalence digest used
above. Every claimed metric in this report can be independently recalculated from
that artifact.

## Scope ruling carried into human review

This window tests the observer/event/replay/failure mechanism. It does not use
`5 / 532` or any other historical-site fraction as an M1 acceptance metric.
Repository-wide coverage remains an A2/M2 dual-read and cutover obligation. Any
unobserved historical site remains tracked debt and is not represented as safe.

## Gate state

```text
machine_status: {result['machine_gate']['status']}
machine_passed: {str(result['machine_gate']['machine_passed']).lower()}
eligible_for_human_review: {str(result['human_gate']['eligible_for_human_review']).lower()}
human_review_status: {result['human_gate']['human_review_status']}
human_accepted: {str(result['human_gate']['human_accepted']).lower()}
v4_2_eligible: {str(result['human_gate']['v4_2_eligible']).lower()}
authority: {result['authority']}
production observer installed: false
production persistence enabled: false
```

This PR supplies the extended-window evidence only. A separate approval-record PR
must perform the human acceptance and artifact-hash pinning before M1 can close.
"""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-output", type=Path)
    parser.add_argument("--evidence-output", type=Path)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    result = run_extended_controlled_observation_campaign()
    raw_text = canonical_raw_text(result)
    raw_sha = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
    evidence_text = render_evidence_markdown(result, raw_sha)
    if args.raw_output is None and args.evidence_output is None:
        print(raw_text, end="")
        return 0
    if args.raw_output is not None:
        args.raw_output.parent.mkdir(parents=True, exist_ok=True)
        args.raw_output.write_text(raw_text, encoding="utf-8")
    if args.evidence_output is not None:
        args.evidence_output.parent.mkdir(parents=True, exist_ok=True)
        args.evidence_output.write_text(evidence_text, encoding="utf-8")
    return 0


__all__ = [
    "BASELINE_SHA",
    "CAMPAIGN_ID",
    "CAMPAIGN_SCHEMA_VERSION",
    "EXTENDED_TARGETS",
    "ExtendedCampaignError",
    "LIVE_LOOP_DRAIN_TARGET",
    "PERSISTENCE_SAVE_TARGET",
    "REQUIRED_MUTATION_FORMS",
    "STANDALONE_TICK_STEPS",
    "canonical_raw_text",
    "render_evidence_markdown",
    "run_extended_controlled_observation_campaign",
]


if __name__ == "__main__":
    raise SystemExit(main())
