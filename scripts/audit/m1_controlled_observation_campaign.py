"""Deterministic, disconnected controlled evidence for the M1 review gate."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping

from adapters.activation_adapter import ActivationAdapter
from core.event_kernel import InMemoryEventKernel, SHADOW_AUTHORITY, canonical_json_object
from core.shadow_acceptance import (
    LegacyPreservationEvidence,
    ObservationWindowSpec,
    evaluate_m1_shadow_window,
)
from core.shadow_observer import (
    ACTIVATION_LEARN_PAIR_TARGET,
    FAILURE_EVENT_TYPE,
    SUCCESS_EVENT_TYPE,
    LegacyFunnelShadowObserver,
)
from core.shadow_projection import (
    ActivationLearnPairShadowState,
    ShadowProjectionError,
    compare_activation_learn_pair_equivalence,
    reduce_activation_learn_pair,
)
from legacy.eve_modules.spreading_activation import SpreadingActivation
from utils.types import Meaning

CAMPAIGN_SCHEMA_VERSION = "eve.m1-controlled-observation-evidence.v1"
CAMPAIGN_ID = "m1:controlled-observation:activation-learn-pair:v1"
CORRELATION_ID = "corr:m1-controlled-observation"
STATIC_SILENT_BROAD_FROZEN = 525
STATIC_SILENT_BROAD_INTEGRATED = 532
LEARN_STEPS: tuple[tuple[str, str, float], ...] = (
    ("alpha", "beta", 0.10),
    ("beta", "gamma", 0.20),
    ("gamma", "delta", 0.15),
    ("alpha", "beta", 0.10),
    ("delta", "epsilon", 0.25),
    ("epsilon", "zeta", 0.30),
    ("zeta", "eta", 0.40),
    ("eta", "theta", 0.35),
    ("theta", "iota", 0.10),
    ("iota", "kappa", 0.20),
    ("alpha", "beta", 0.15),
    ("kappa", "lambda", 0.50),
)
FAIL_ON_CALL = 7
TICK_AFTER_CALLS = (2, 4, 6, 8, 10, 12)


class ControlledCampaignError(RuntimeError):
    pass


def _canonical(value: Mapping[str, Any], field: str) -> str:
    return canonical_json_object(value, field=field)


def _sha(value: Mapping[str, Any], field: str) -> str:
    return hashlib.sha256(_canonical(value, field).encode("utf-8")).hexdigest()


def _json_bytes(value: Mapping[str, Any]) -> int:
    return len(
        json.dumps(
            dict(value),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )


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


class DelegatingObservedSpreadingActivation:
    """Isolated ledger that delegates successful calls to retained legacy SA."""

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
            "legacy_class": "legacy.eve_modules.spreading_activation.SpreadingActivation",
            "time": float(self.inner.time),
            "weights": [
                [left, right, float(weight)]
                for (left, right), weight in sorted(self.inner.weights.items())
            ],
        }


@dataclass(frozen=True, slots=True)
class _ProbeHormone:
    hs: Mapping[str, float]


class _ProbeSA:
    def __init__(self, fail_method: str | None = None) -> None:
        self.fail_method = fail_method

    def _fail(self, method: str) -> None:
        if self.fail_method == method:
            raise RuntimeError(f"controlled silent probe:{method}")

    def decay(self, dt: float) -> None:
        self._fail("sa.decay")

    def apply_hormone_modulation(self, state: Mapping[str, float]) -> None:
        self._fail("sa.apply_hormone_modulation")

    def activate(self, category: str, strength: float = 0.5) -> None:
        return None

    def spread(self, steps: int = 1) -> None:
        return None

    def get_top_active(self, n: int = 5) -> list[tuple[str, float]]:
        return []


class _ProbeWM:
    def __init__(self, fail_method: str | None = None) -> None:
        self.fail_method = fail_method

    def _fail(self, method: str) -> None:
        if self.fail_method == method:
            raise RuntimeError(f"controlled silent probe:{method}")

    def decay(self, dt: float) -> None:
        self._fail("wm.decay")

    def apply_hormone_state(self, state: Mapping[str, float]) -> None:
        self._fail("wm.apply_hormone_state")

    def add(self, category: str, salience: float = 0.5) -> None:
        return None

    def get_focus(self) -> str | None:
        self._fail("wm.get_focus")
        return "focus"

    def get_focus_set(self) -> set[str]:
        self._fail("wm.get_focus_set")
        return {"focus"}


def _silent_candidate(
    *,
    scenario_id: str,
    line_range: str,
    callable_name: str,
    stage: str,
    invoke: Any,
    expected_fallback: Any,
) -> dict[str, Any]:
    error_text = f"controlled silent probe:{stage}"
    result: Any = None
    outward_error: str | None = None
    try:
        result = invoke()
    except RuntimeError as exc:
        outward_error = type(exc).__name__
    if isinstance(result, set):
        result = sorted(result)
    return {
        "callable": callable_name,
        "candidate_type": "silent_failure_observed_candidate",
        "error_message_digest": hashlib.sha256(error_text.encode("utf-8")).hexdigest(),
        "error_type": "RuntimeError",
        "fallback": result,
        "line_range": line_range,
        "observed_silent": outward_error is None and result == expected_fallback,
        "outward_error_type": outward_error,
        "path": "adapters/activation_adapter.py",
        "scenario_id": scenario_id,
        "stage": stage,
    }


def _run_silent_failure_probes() -> tuple[dict[str, Any], ...]:
    empty = Meaning()
    rows = (
        _silent_candidate(
            scenario_id="silent:activation-ingest:sa-decay",
            line_range="40-43",
            callable_name="ActivationAdapter.ingest",
            stage="sa.decay",
            invoke=lambda: ActivationAdapter(
                sa=_ProbeSA("sa.decay"), wm=_ProbeWM()
            ).ingest(empty),
            expected_fallback=None,
        ),
        _silent_candidate(
            scenario_id="silent:activation-ingest:wm-decay",
            line_range="44-47",
            callable_name="ActivationAdapter.ingest",
            stage="wm.decay",
            invoke=lambda: ActivationAdapter(
                sa=_ProbeSA(), wm=_ProbeWM("wm.decay")
            ).ingest(empty),
            expected_fallback=None,
        ),
        _silent_candidate(
            scenario_id="silent:activation-ingest:hormone-modulation",
            line_range="50-55",
            callable_name="ActivationAdapter.ingest",
            stage="sa.apply_hormone_modulation",
            invoke=lambda: ActivationAdapter(
                sa=_ProbeSA("sa.apply_hormone_modulation"),
                wm=_ProbeWM(),
                hormone_adapter=_ProbeHormone({}),
            ).ingest(empty),
            expected_fallback=None,
        ),
        _silent_candidate(
            scenario_id="silent:activation-focus-category",
            line_range="88-91",
            callable_name="ActivationAdapter.focus_category",
            stage="wm.get_focus",
            invoke=lambda: ActivationAdapter(
                sa=_ProbeSA(), wm=_ProbeWM("wm.get_focus")
            ).focus_category(),
            expected_fallback=None,
        ),
        _silent_candidate(
            scenario_id="silent:activation-focus-set",
            line_range="94-97",
            callable_name="ActivationAdapter.focus_set",
            stage="wm.get_focus_set",
            invoke=lambda: ActivationAdapter(
                sa=_ProbeSA(), wm=_ProbeWM("wm.get_focus_set")
            ).focus_set(),
            expected_fallback=[],
        ),
    )
    if not all(row["observed_silent"] for row in rows):
        raise ControlledCampaignError("silent probe escaped or fallback changed")
    return rows


def _run_observer_failure_probe() -> tuple[Any, dict[str, Any]]:
    adapter = ActivationAdapter()
    baseline = ActivationAdapter()
    kernel = InMemoryEventKernel()
    observer = LegacyFunnelShadowObserver(kernel)

    def broken_before() -> Mapping[str, Any]:
        raise RuntimeError("controlled observer snapshot failure")

    result = observer.observe_call(
        ACTIVATION_LEARN_PAIR_TARGET.target_id,
        event_id="m1-controlled:observer-failure:001",
        correlation_id=CORRELATION_ID,
        legacy_callable=adapter.learn_pair,
        before_snapshot=broken_before,
        after_snapshot=lambda: {"calls": [], "learned": []},
        args=("probe", "pair"),
        kwargs={"strength": 0.2},
    )
    baseline_result = baseline.learn_pair("probe", "pair", strength=0.2)
    failures = observer.failures()
    if len(failures) != 1 or kernel.events():
        raise ControlledCampaignError("observer-failure probe contract changed")
    failure = failures[0]
    return failure, {
        "baseline_return_preserved": result is baseline_result is None,
        "event_id": failure.event_id,
        "error_message_digest": failure.error_message_digest,
        "error_type": failure.error_type,
        "legacy_state_preserved": (
            adapter.sa.get_weight("probe", "pair")
            == baseline.sa.get_weight("probe", "pair")
        ),
        "stage": failure.stage,
        "target_id": failure.target_id,
    }


def _run_learn_window() -> tuple[
    DelegatingObservedSpreadingActivation,
    DelegatingObservedSpreadingActivation,
    tuple[Any, ...],
    list[dict[str, Any]],
    bool,
]:
    observed_error = RuntimeError("controlled legacy failure")
    baseline_error = RuntimeError("controlled baseline failure")
    observed_sa = DelegatingObservedSpreadingActivation(
        fail_on_call=FAIL_ON_CALL, failure=observed_error
    )
    baseline_sa = DelegatingObservedSpreadingActivation(
        fail_on_call=FAIL_ON_CALL, failure=baseline_error
    )
    observed = ActivationAdapter(sa=observed_sa)
    baseline = ActivationAdapter(sa=baseline_sa)
    kernel = InMemoryEventKernel()
    observer = LegacyFunnelShadowObserver(kernel)
    steps: list[dict[str, Any]] = []
    propagated_identity = False
    previous_event_id: str | None = None
    logical_step = 0

    for call_index, (left, right, strength) in enumerate(LEARN_STEPS, start=1):
        logical_step += 1
        event_id = f"m1-controlled:event:{call_index:03d}"
        before_count = len(kernel.events())

        def before_snapshot(event_id: str = event_id) -> Mapping[str, Any]:
            observed_sa.trace.append(f"before:{event_id}")
            return observed_sa.snapshot()

        def after_snapshot(event_id: str = event_id) -> Mapping[str, Any]:
            observed_sa.trace.append(f"after:{event_id}")
            return observed_sa.snapshot()

        observed_outcome = "success"
        try:
            observer.observe_call(
                ACTIVATION_LEARN_PAIR_TARGET.target_id,
                event_id=event_id,
                correlation_id=CORRELATION_ID,
                causation_id=previous_event_id,
                legacy_callable=observed.learn_pair,
                before_snapshot=before_snapshot,
                after_snapshot=after_snapshot,
                args=(left, right),
                kwargs={"strength": strength},
            )
        except RuntimeError as exc:
            if call_index != FAIL_ON_CALL:
                raise
            propagated_identity = exc is observed_error
            observed_outcome = "failure"

        baseline_outcome = "success"
        try:
            baseline.learn_pair(left, right, strength=strength)
        except RuntimeError as exc:
            if call_index != FAIL_ON_CALL or exc is not baseline_error:
                raise
            baseline_outcome = "failure"
        if observed_outcome != baseline_outcome:
            raise ControlledCampaignError("observed and baseline outcomes differ")

        event_delta = len(kernel.events()) - before_count
        if event_delta != 1:
            raise ControlledCampaignError("observed call did not emit exactly one event")
        event = kernel.events()[-1]
        previous_event_id = event.event_id
        steps.append(
            {
                "actual_state_digest": _sha(observed_sa.actual_state(), "actual_state"),
                "event_digest": event.digest,
                "event_id": event.event_id,
                "event_type": event.event_type,
                "events_emitted": 1,
                "legacy_call_index": call_index,
                "logical_step": logical_step,
                "observed_outcome": observed_outcome,
                "operation": "learn_pair",
            }
        )
        if call_index in TICK_AFTER_CALLS:
            logical_step += 1
            before_tick = len(kernel.events())
            observed.tick(dt=1.0)
            baseline.tick(dt=1.0)
            if len(kernel.events()) != before_tick:
                raise ControlledCampaignError("tick emitted a candidate")
            steps.append(
                {
                    "actual_state_digest": _sha(observed_sa.actual_state(), "actual_state"),
                    "events_emitted": 0,
                    "logical_step": logical_step,
                    "operation": "tick",
                    "tick_dt": 1.0,
                }
            )
    return observed_sa, baseline_sa, kernel.events(), steps, propagated_identity


def run_controlled_observation_campaign() -> dict[str, Any]:
    observed_sa, baseline_sa, events, steps, propagated_identity = _run_learn_window()
    state = ActivationLearnPairShadowState(calls=(), learned=())
    replay_rows: list[dict[str, Any]] = []
    divergences: list[dict[str, Any]] = []
    for event in events:
        mismatch_codes: list[str] = []
        try:
            state = reduce_activation_learn_pair(state, event)
            if state.snapshot != event.payload["after"]:
                mismatch_codes.append("projected_after_mismatch")
        except ShadowProjectionError as exc:
            mismatch_codes.append(f"reducer_error:{type(exc).__name__}")
        row = {
            "event_digest": event.digest,
            "event_id": event.event_id,
            "matches": not mismatch_codes,
            "mismatch_codes": mismatch_codes,
            "sequence": event.sequence,
        }
        replay_rows.append(row)
        if mismatch_codes:
            divergences.append(row)

    final_equivalence = compare_activation_learn_pair_equivalence(
        state, observed_sa.snapshot()
    )
    observer_failure, observer_failure_evidence = _run_observer_failure_probe()
    silent_rows = _run_silent_failure_probes()
    legacy_state_preserved = (
        observed_sa.snapshot() == baseline_sa.snapshot()
        and observed_sa.actual_state() == baseline_sa.actual_state()
    )
    expected_trace: list[str] = []
    for index in range(1, len(LEARN_STEPS) + 1):
        event_id = f"m1-controlled:event:{index:03d}"
        expected_trace.extend(
            [f"before:{event_id}", f"legacy:{index}", f"after:{event_id}"]
        )
    call_order_preserved = observed_sa.trace == expected_trace
    source_record = {
        "baseline_actual_state": baseline_sa.actual_state(),
        "baseline_snapshot": baseline_sa.snapshot(),
        "call_order_preserved": call_order_preserved,
        "observed_actual_state": observed_sa.actual_state(),
        "observed_snapshot": observed_sa.snapshot(),
        "observer_failure": observer_failure_evidence,
        "propagated_exception_identity": propagated_identity,
        "silent_candidates": list(silent_rows),
        "steps": steps,
    }
    source_digest = _sha(source_record, "controlled_campaign_source")
    evidence = LegacyPreservationEvidence(
        evidence_id="m1-controlled:legacy-preservation:v1",
        case_ids=tuple(event.event_id for event in events) + (observer_failure.event_id,),
        return_value_preserved=observer_failure_evidence["baseline_return_preserved"],
        exception_identity_preserved=propagated_identity,
        call_order_preserved=call_order_preserved,
        legacy_state_matches_unobserved=legacy_state_preserved,
        persistence_behavior_unchanged=True,
        defaults_unchanged=True,
        external_effects_unchanged=True,
        source_evidence_digest=source_digest,
    )
    success_count = sum(1 for event in events if event.event_type == SUCCESS_EVENT_TYPE)
    failure_count = sum(1 for event in events if event.event_type == FAILURE_EVENT_TYPE)
    spec = ObservationWindowSpec(
        window_id=CAMPAIGN_ID,
        expected_event_count=len(events),
        expected_success_count=success_count,
        expected_failure_count=failure_count,
        expected_observer_failure_count=1,
        initial_checkpoint_id="m1-controlled:checkpoint:initial",
        final_checkpoint_id="m1-controlled:checkpoint:final",
    )
    packet = evaluate_m1_shadow_window(
        spec,
        initial_state=ActivationLearnPairShadowState(calls=(), learned=()),
        events=events,
        expected_final_snapshot=observed_sa.snapshot(),
        observer_failures=(observer_failure,),
        legacy_evidence=evidence,
    )
    event_records = [_event_record(event) for event in events]
    event_bytes = [_json_bytes(record) for record in event_records]
    matching = sum(1 for row in replay_rows if row["matches"])
    event_count = len(events)
    logical_steps = len(steps)
    tick_steps = sum(1 for row in steps if row["operation"] == "tick")
    result = {
        "authority": SHADOW_AUTHORITY,
        "campaign_id": CAMPAIGN_ID,
        "campaign_schema_version": CAMPAIGN_SCHEMA_VERSION,
        "event_rate": {
            "candidate_events": event_count,
            "events_during_tick_steps": 0,
            "events_per_legacy_call": {
                "denominator": len(LEARN_STEPS),
                "numerator": event_count,
                "value": event_count / len(LEARN_STEPS),
            },
            "events_per_logical_step": {
                "denominator": logical_steps,
                "numerator": event_count,
                "value": event_count / logical_steps,
            },
            "logical_steps": logical_steps,
            "max_events_in_one_step": max(row["events_emitted"] for row in steps),
            "serialized_event_bytes_max": max(event_bytes),
            "serialized_event_bytes_total": sum(event_bytes),
            "serialized_packet_bytes": _json_bytes(packet.canonical_record),
            "tick_steps": tick_steps,
        },
        "events": event_records,
        "legacy_preservation": {
            "call_order_preserved": call_order_preserved,
            "exception_identity_preserved": propagated_identity,
            "legacy_state_matches_unobserved": legacy_state_preserved,
            "source_evidence_digest": source_digest,
        },
        "observation_window": {
            "decay_cycles": tick_steps,
            "duration_kind": "logical_steps_only",
            "legacy_calls": len(LEARN_STEPS),
            "logical_steps": logical_steps,
            "production_runtime_integrated": False,
            "scenario_count": 3,
            "scenarios": [
                "delegated_legacy_learn_pair_success_and_failure",
                "observer_snapshot_failure_isolation",
                "selected_active_silent_handler_probes",
            ],
            "tick_dt_total": float(tick_steps),
            "tick_steps": tick_steps,
            "wall_clock_duration": None,
        },
        "packet": {"canonical_record": packet.canonical_record, "digest": packet.digest},
        "persistence_mode": "none",
        "replay_equivalence": {
            "compared_events": len(replay_rows),
            "divergence_count": len(divergences),
            "divergences": divergences,
            "final_equivalence_matches": final_equivalence.matches,
            "final_mismatches": list(final_equivalence.mismatches),
            "match_rate": {
                "denominator": len(replay_rows),
                "numerator": matching,
                "value": matching / len(replay_rows),
            },
            "rows": replay_rows,
        },
        "runtime_integrated": False,
        "silent_failure_observation": {
            "candidates": list(silent_rows),
            "integrated_static_denominator": STATIC_SILENT_BROAD_INTEGRATED,
            "integrated_unobserved_remainder": STATIC_SILENT_BROAD_INTEGRATED - len(silent_rows),
            "observed_candidate_count": len(silent_rows),
            "selected_occurrence_count": len(silent_rows),
            "selected_occurrences_observed": len(silent_rows),
            "frozen_static_denominator": STATIC_SILENT_BROAD_FROZEN,
            "frozen_unobserved_remainder": STATIC_SILENT_BROAD_FROZEN - len(silent_rows),
        },
        "step_rows": steps,
    }
    if not packet.machine_passed or packet.human_accepted or packet.v4_2_eligible:
        raise ControlledCampaignError("campaign crossed authority boundary")
    if divergences or not final_equivalence.matches:
        raise ControlledCampaignError("controlled replay diverged")
    return result


__all__ = [
    "CAMPAIGN_ID",
    "CAMPAIGN_SCHEMA_VERSION",
    "ControlledCampaignError",
    "FAIL_ON_CALL",
    "LEARN_STEPS",
    "STATIC_SILENT_BROAD_FROZEN",
    "STATIC_SILENT_BROAD_INTEGRATED",
    "TICK_AFTER_CALLS",
    "run_controlled_observation_campaign",
]
