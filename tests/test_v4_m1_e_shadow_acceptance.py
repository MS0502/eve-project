from __future__ import annotations

import ast
import hashlib
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from adapters.activation_adapter import ActivationAdapter
from core.event_kernel import InMemoryEventKernel
from core.shadow_acceptance import (
    HUMAN_REVIEW_REQUIRED,
    MACHINE_COMPLETE_STATUS,
    MACHINE_INCOMPLETE_STATUS,
    PACKET_SCHEMA_VERSION,
    LegacyPreservationEvidence,
    ObservationWindowSpec,
    ShadowAcceptanceContractError,
    evaluate_m1_shadow_window,
)
from core.shadow_lifecycle import DEFAULT_BRIDGE_REGISTRY
from core.shadow_observer import (
    ACTIVATION_LEARN_PAIR_TARGET,
    LegacyFunnelShadowObserver,
)
from core.shadow_projection import ActivationLearnPairShadowState


REPO_ROOT = Path(__file__).resolve().parents[1]
ACCEPTANCE_PATH = REPO_ROOT / "core/shadow_acceptance.py"


class FakeSpreadingActivation:
    def __init__(
        self,
        *,
        fail_on_call: int | None = None,
        error: Exception | None = None,
        trace: list[str] | None = None,
    ) -> None:
        self.calls: list[tuple[str, tuple[str, str, float]]] = []
        self.learned: list[tuple[str, str, float]] = []
        self.fail_on_call = fail_on_call
        self.error = error
        self.trace = trace

    def learn_pair(self, a: str, b: str, *, strength: float) -> None:
        if self.trace is not None:
            self.trace.append("legacy")
        record = (a, b, float(strength))
        self.calls.append(("learn_pair", record))
        if self.fail_on_call == len(self.calls):
            assert self.error is not None
            raise self.error
        self.learned.append(record)


class FakeWorkingMemory:
    pass


def _adapter(
    *,
    fail_on_call: int | None = None,
    error: Exception | None = None,
    trace: list[str] | None = None,
) -> ActivationAdapter:
    return ActivationAdapter(
        sa=FakeSpreadingActivation(
            fail_on_call=fail_on_call,
            error=error,
            trace=trace,
        ),
        wm=FakeWorkingMemory(),
    )


def _snapshot(adapter: ActivationAdapter, trace: list[str], label: str):
    def capture():
        trace.append(label)
        return {
            "calls": [list(item[1]) for item in adapter.sa.calls],
            "learned": [list(item) for item in adapter.sa.learned],
        }

    return capture


def _observe(
    observer: LegacyFunnelShadowObserver,
    adapter: ActivationAdapter,
    *,
    event_id: str,
    trace: list[str],
    left: str,
    right: str,
    strength: float,
    causation_id: str | None = None,
):
    return observer.observe_call(
        ACTIVATION_LEARN_PAIR_TARGET.target_id,
        event_id=event_id,
        correlation_id="corr:m1-e",
        causation_id=causation_id,
        legacy_callable=adapter.learn_pair,
        before_snapshot=_snapshot(adapter, trace, f"before:{event_id}"),
        after_snapshot=_snapshot(adapter, trace, f"after:{event_id}"),
        args=(left, right),
        kwargs={"strength": strength},
    )


def _window_fixture():
    error = RuntimeError("legacy failure secret")
    trace: list[str] = []
    adapter = _adapter(fail_on_call=2, error=error, trace=trace)
    kernel = InMemoryEventKernel()
    observer = LegacyFunnelShadowObserver(kernel)

    success_result = _observe(
        observer,
        adapter,
        event_id="m1e:event:1",
        trace=trace,
        left="alpha",
        right="beta",
        strength=0.4,
    )
    with pytest.raises(RuntimeError) as captured:
        _observe(
            observer,
            adapter,
            event_id="m1e:event:2",
            causation_id="m1e:event:1",
            trace=trace,
            left="gamma",
            right="delta",
            strength=0.6,
        )

    expected_snapshot = {
        "calls": [list(item[1]) for item in adapter.sa.calls],
        "learned": [list(item) for item in adapter.sa.learned],
    }

    baseline_error = RuntimeError("baseline failure")
    baseline = _adapter(fail_on_call=2, error=baseline_error)
    baseline_result = baseline.learn_pair("alpha", "beta", strength=0.4)
    with pytest.raises(RuntimeError):
        baseline.learn_pair("gamma", "delta", strength=0.6)

    probe_adapter = _adapter()
    probe_kernel = InMemoryEventKernel()
    probe_observer = LegacyFunnelShadowObserver(probe_kernel)

    def broken_before():
        raise ValueError("observer snapshot secret")

    probe_result = probe_observer.observe_call(
        ACTIVATION_LEARN_PAIR_TARGET.target_id,
        event_id="m1e:observer-failure:1",
        correlation_id="corr:m1-e-probe",
        legacy_callable=probe_adapter.learn_pair,
        before_snapshot=broken_before,
        after_snapshot=lambda: {
            "calls": [list(item[1]) for item in probe_adapter.sa.calls],
            "learned": [list(item) for item in probe_adapter.sa.learned],
        },
        args=("probe", "pair"),
        kwargs={"strength": 0.2},
    )

    source_digest = hashlib.sha256(
        repr(
            (
                trace,
                adapter.sa.calls,
                adapter.sa.learned,
                baseline.sa.calls,
                baseline.sa.learned,
            )
        ).encode("utf-8")
    ).hexdigest()
    evidence = LegacyPreservationEvidence(
        evidence_id="m1e:legacy-preservation",
        case_ids=(
            "m1e:event:1",
            "m1e:event:2",
            "m1e:observer-failure:1",
        ),
        return_value_preserved=(success_result is baseline_result is probe_result is None),
        exception_identity_preserved=(captured.value is error),
        call_order_preserved=trace
        == [
            "before:m1e:event:1",
            "legacy",
            "after:m1e:event:1",
            "before:m1e:event:2",
            "legacy",
            "after:m1e:event:2",
        ],
        legacy_state_matches_unobserved=(
            adapter.sa.calls == baseline.sa.calls
            and adapter.sa.learned == baseline.sa.learned
        ),
        persistence_behavior_unchanged=True,
        defaults_unchanged=True,
        external_effects_unchanged=True,
        source_evidence_digest=source_digest,
    )
    spec = ObservationWindowSpec(
        window_id="m1e:window:activation-learn-pair",
        expected_event_count=2,
        expected_success_count=1,
        expected_failure_count=1,
        expected_observer_failure_count=1,
        initial_checkpoint_id="m1e:checkpoint:initial",
        final_checkpoint_id="m1e:checkpoint:final",
    )
    return {
        "spec": spec,
        "initial_state": ActivationLearnPairShadowState(calls=(), learned=()),
        "events": kernel.events(),
        "expected_snapshot": expected_snapshot,
        "observer_failures": probe_observer.failures(),
        "evidence": evidence,
    }


def _evaluate(data, **overrides):
    values = {
        "spec": data["spec"],
        "initial_state": data["initial_state"],
        "events": data["events"],
        "expected_final_snapshot": data["expected_snapshot"],
        "observer_failures": data["observer_failures"],
        "legacy_evidence": data["evidence"],
        "lifecycle_registry": DEFAULT_BRIDGE_REGISTRY,
    }
    values.update(overrides)
    return evaluate_m1_shadow_window(**values)


def test_complete_real_observation_window_is_machine_ready_but_not_human_accepted():
    packet = _evaluate(_window_fixture())

    assert packet.schema_version == PACKET_SCHEMA_VERSION
    assert packet.machine_status == MACHINE_COMPLETE_STATUS
    assert packet.machine_passed is True
    assert packet.eligible_for_human_review is True
    assert packet.human_review_status == HUMAN_REVIEW_REQUIRED
    assert packet.human_accepted is False
    assert packet.v4_2_eligible is False
    assert packet.runtime_integrated is False
    assert packet.persistence_mode == "none"
    assert packet.unauthorized_effects_detected is False
    assert packet.event_count == 2
    assert packet.success_count == 1
    assert packet.failure_count == 1
    assert packet.observer_failure_count == 1
    assert packet.first_sequence == 1
    assert packet.last_sequence == 2
    assert packet.lifecycle_domains == ("activity", "chat", "goal", "memory")
    assert all(item.passed for item in packet.criteria)
    assert "legacy failure secret" not in repr(packet)
    assert "observer snapshot secret" not in repr(packet)


def test_same_window_is_deterministic_and_packet_record_is_detached():
    data = _window_fixture()
    first = _evaluate(data)
    second = _evaluate(data)

    assert first == second
    assert first.digest == second.digest
    record = first.canonical_record
    record["criteria"][0]["passed"] = False
    record["lifecycle_domains"].append("spoof")
    assert first.canonical_record["criteria"][0]["passed"] is True
    assert "spoof" not in first.lifecycle_domains

    with pytest.raises(FrozenInstanceError):
        first.machine_passed = False  # type: ignore[misc]


def test_event_count_mismatch_returns_incomplete_human_review_packet():
    data = _window_fixture()
    spec = replace(
        data["spec"],
        expected_event_count=3,
        expected_success_count=2,
    )
    packet = _evaluate(data, spec=spec)

    assert packet.machine_status == MACHINE_INCOMPLETE_STATUS
    assert packet.machine_passed is False
    assert packet.eligible_for_human_review is False
    assert packet.criterion("event_count_exact").passed is False
    assert packet.criterion("success_failure_visible").passed is False
    assert packet.human_accepted is False
    assert packet.v4_2_eligible is False


def test_replay_equivalence_mismatch_is_visible_without_authority():
    data = _window_fixture()
    expected = {
        "calls": data["expected_snapshot"]["calls"],
        "learned": [],
    }
    packet = _evaluate(data, expected_final_snapshot=expected)

    assert packet.machine_passed is False
    assert packet.criterion("replay_equivalent").passed is False
    assert packet.human_accepted is False
    assert packet.v4_2_eligible is False


def test_failed_legacy_preservation_evidence_blocks_machine_completion():
    data = _window_fixture()
    evidence = replace(data["evidence"], defaults_unchanged=False)
    packet = _evaluate(data, legacy_evidence=evidence)

    assert evidence.passes is False
    assert packet.machine_passed is False
    assert packet.criterion("legacy_behavior_preserved").passed is False
    assert packet.criterion("zero_unauthorized_effects").passed is False
    assert packet.eligible_for_human_review is False


def test_observer_failure_count_mismatch_is_visible_and_redacted():
    data = _window_fixture()
    spec = replace(data["spec"], expected_observer_failure_count=2)
    packet = _evaluate(data, spec=spec)

    assert packet.machine_passed is False
    assert packet.criterion("observer_failure_visible").passed is False
    failure = data["observer_failures"][0]
    assert failure.stage == "before_snapshot"
    assert failure.legacy_succeeded is None
    assert len(failure.error_message_digest) == 64
    assert "observer snapshot secret" not in repr(failure)


def test_malformed_observer_failure_and_noncontiguous_event_fail_closed():
    data = _window_fixture()
    malformed = replace(
        data["observer_failures"][0],
        error_message_digest="not-a-digest",
    )
    with pytest.raises(ShadowAcceptanceContractError, match="SHA-256"):
        _evaluate(data, observer_failures=(malformed,))

    bad_second = replace(data["events"][1], sequence=3)
    with pytest.raises(ShadowAcceptanceContractError, match="contiguous"):
        _evaluate(data, events=(data["events"][0], bad_second))


def test_wrong_lifecycle_registry_type_and_malformed_inputs_fail_closed():
    data = _window_fixture()
    with pytest.raises(ShadowAcceptanceContractError):
        _evaluate(data, lifecycle_registry=None)
    with pytest.raises(ShadowAcceptanceContractError):
        _evaluate(data, events=[])
    with pytest.raises(ShadowAcceptanceContractError):
        _evaluate(data, initial_state=None)


def test_spec_and_legacy_evidence_contracts_are_strict():
    data = _window_fixture()
    with pytest.raises(ShadowAcceptanceContractError):
        replace(data["spec"], expected_event_count=3)
    with pytest.raises(ShadowAcceptanceContractError):
        replace(data["spec"], initial_checkpoint_id=data["spec"].final_checkpoint_id)
    with pytest.raises(ShadowAcceptanceContractError):
        replace(data["evidence"], case_ids=("duplicate", "duplicate"))
    with pytest.raises(ShadowAcceptanceContractError):
        replace(data["evidence"], source_evidence_digest="bad")

    spoofed = replace(
        data["evidence"],
        case_ids=("unrelated:1", "unrelated:2", "unrelated:3"),
    )
    packet = _evaluate(data, legacy_evidence=spoofed)
    assert packet.criterion("legacy_behavior_preserved").passed is False
    assert packet.machine_passed is False


def test_checkpoint_restore_rollback_and_lifecycle_criteria_are_explicit():
    packet = _evaluate(_window_fixture())

    assert packet.criterion("checkpoint_restore_verified").passed is True
    assert packet.criterion("rollback_verified").passed is True
    assert packet.criterion("lifecycle_registry_complete").passed is True
    assert packet.lifecycle_registry_digest == DEFAULT_BRIDGE_REGISTRY.digest
    assert packet.criterion("zero_unauthorized_effects").passed is True


def test_packet_cannot_be_replaced_into_acceptance_or_runtime_authority():
    packet = _evaluate(_window_fixture())

    with pytest.raises(ShadowAcceptanceContractError):
        replace(packet, human_accepted=True)
    with pytest.raises(ShadowAcceptanceContractError):
        replace(packet, v4_2_eligible=True)
    with pytest.raises(ShadowAcceptanceContractError):
        replace(packet, runtime_integrated=True)
    with pytest.raises(ShadowAcceptanceContractError):
        replace(packet, persistence_mode="sqlite")
    with pytest.raises(ShadowAcceptanceContractError):
        replace(packet, unauthorized_effects_detected=True)


def test_acceptance_module_has_no_legacy_import_io_clock_thread_or_auto_acceptance():
    source = ACCEPTANCE_PATH.read_text(encoding="utf-8")
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
        "accept",
        "connect",
        "emit",
        "load",
        "observe_call",
        "open",
        "save",
        "sleep",
        "start",
        "write_bytes",
        "write_text",
    }
    assert "human_accepted: bool = False" in source
    assert "v4_2_eligible: bool = False" in source
    assert not [node for node in ast.walk(tree) if isinstance(node, ast.ExceptHandler)]
