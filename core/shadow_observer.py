"""M1-B after-the-fact shadow observer for one registered legacy funnel.

This module does not patch, import, or activate the legacy runtime. A caller may
explicitly pass the registered bound callable into ``observe_call``. The legacy
return value or exception remains authoritative; observation output is
``shadow_only`` in-memory evidence.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Callable, Mapping, TypeVar

from core.event_kernel import (
    EventEnvelope,
    InMemoryEventKernel,
    canonical_json_object,
)

ResultT = TypeVar("ResultT")

SUCCESS_EVENT_TYPE = "shadow.legacy_mutation_observed_candidate"
FAILURE_EVENT_TYPE = "shadow.legacy_mutation_failed_candidate"
OBSERVER_PRODUCER = "core.shadow_observer"
OBSERVER_VERSION = "1.0.0"


class ShadowObserverContractError(ValueError):
    """Raised for caller-side misuse before any legacy callable is invoked."""


class UnknownShadowTarget(ShadowObserverContractError):
    """Raised when a target is not in the reviewed M1-B registry."""


@dataclass(frozen=True, slots=True)
class ShadowTarget:
    """Reviewed metadata for one bounded legacy mutation funnel."""

    target_id: str
    module_path: str
    callable_name: str
    evidence_range: str
    module_disposition: str
    stream_id: str


@dataclass(frozen=True, slots=True)
class ShadowObservationFailure:
    """Visible in-memory evidence that only the observer path failed."""

    target_id: str
    event_id: str
    stage: str
    error_type: str
    error_message_digest: str
    legacy_succeeded: bool | None


ACTIVATION_LEARN_PAIR_TARGET = ShadowTarget(
    target_id="legacy.activation.learn_pair",
    module_path="adapters/activation_adapter.py",
    callable_name="ActivationAdapter.learn_pair",
    evidence_range="105-107",
    module_disposition="WRAP",
    stream_id="shadow:legacy.activation.learn_pair",
)

REVIEWED_SHADOW_TARGETS: tuple[ShadowTarget, ...] = (
    ACTIVATION_LEARN_PAIR_TARGET,
)


class LegacyFunnelShadowObserver:
    """Observe a reviewed legacy call without gaining legacy authority.

    The observer captures detached before/after snapshots, calls the supplied
    legacy callable exactly once, and then attempts to append one shadow-only
    candidate. Observer defects are retained in ``failures()`` and never replace
    a successful legacy return or suppress a legacy exception.
    """

    def __init__(
        self,
        kernel: InMemoryEventKernel[Any],
        *,
        targets: tuple[ShadowTarget, ...] = REVIEWED_SHADOW_TARGETS,
    ) -> None:
        if not isinstance(kernel, InMemoryEventKernel):
            raise ShadowObserverContractError(
                "observer requires an InMemoryEventKernel"
            )
        if not targets:
            raise ShadowObserverContractError("target registry cannot be empty")
        by_id: dict[str, ShadowTarget] = {}
        for target in targets:
            if target.target_id in by_id:
                raise ShadowObserverContractError(
                    f"duplicate shadow target: {target.target_id}"
                )
            by_id[target.target_id] = target
        self._kernel = kernel
        self._targets_by_id = by_id
        self._failures: list[ShadowObservationFailure] = []

    def target(self, target_id: str) -> ShadowTarget:
        """Resolve one reviewed target before invoking legacy behavior."""

        try:
            return self._targets_by_id[target_id]
        except KeyError as exc:
            raise UnknownShadowTarget(target_id) from exc

    def failures(self) -> tuple[ShadowObservationFailure, ...]:
        """Return immutable observer-failure evidence."""

        return tuple(self._failures)

    def observe_call(
        self,
        target_id: str,
        *,
        event_id: str,
        correlation_id: str,
        legacy_callable: Callable[..., ResultT],
        before_snapshot: Callable[[], Mapping[str, Any]],
        after_snapshot: Callable[[], Mapping[str, Any]],
        causation_id: str | None = None,
        args: tuple[Any, ...] = (),
        kwargs: Mapping[str, Any] | None = None,
    ) -> ResultT:
        """Call one registered funnel and emit after-the-fact shadow evidence.

        Unknown targets and non-callable inputs fail before the legacy callable.
        Once the legacy callable begins, observer errors are isolated and exposed
        through ``failures()``. Legacy exceptions are re-raised unchanged.
        """

        target = self.target(target_id)
        if not callable(legacy_callable):
            raise ShadowObserverContractError("legacy_callable must be callable")
        if not callable(before_snapshot) or not callable(after_snapshot):
            raise ShadowObserverContractError("snapshot providers must be callable")
        if not isinstance(args, tuple):
            raise ShadowObserverContractError("args must be a tuple")
        if kwargs is None:
            call_kwargs: dict[str, Any] = {}
        elif isinstance(kwargs, Mapping):
            call_kwargs = dict(kwargs)
        else:
            raise ShadowObserverContractError("kwargs must be a mapping")

        before = self._capture_snapshot(
            target,
            event_id,
            stage="before_snapshot",
            provider=before_snapshot,
            legacy_succeeded=None,
        )
        try:
            result = legacy_callable(*args, **call_kwargs)
        except Exception as legacy_error:
            after = self._capture_snapshot(
                target,
                event_id,
                stage="after_snapshot",
                provider=after_snapshot,
                legacy_succeeded=False,
            )
            self._emit_candidate(
                target,
                event_id=event_id,
                correlation_id=correlation_id,
                causation_id=causation_id,
                before=before,
                after=after,
                legacy_succeeded=False,
                legacy_error_type=type(legacy_error).__name__,
            )
            raise

        after = self._capture_snapshot(
            target,
            event_id,
            stage="after_snapshot",
            provider=after_snapshot,
            legacy_succeeded=True,
        )
        self._emit_candidate(
            target,
            event_id=event_id,
            correlation_id=correlation_id,
            causation_id=causation_id,
            before=before,
            after=after,
            legacy_succeeded=True,
            legacy_error_type=None,
        )
        return result

    def _capture_snapshot(
        self,
        target: ShadowTarget,
        event_id: str,
        *,
        stage: str,
        provider: Callable[[], Mapping[str, Any]],
        legacy_succeeded: bool | None,
    ) -> dict[str, Any] | None:
        try:
            value = provider()
            encoded = canonical_json_object(value, field=stage)
            return EventEnvelope.create(
                event_id="snapshot:validation",
                event_type="shadow.snapshot_validation",
                stream_id="shadow:snapshot_validation",
                sequence=1,
                producer=OBSERVER_PRODUCER,
                producer_version=OBSERVER_VERSION,
                correlation_id="snapshot:validation",
                payload={"snapshot": value},
                causal_context={},
            ).payload["snapshot"]
        except Exception as observer_error:
            self._record_failure(
                target,
                event_id=event_id,
                stage=stage,
                error=observer_error,
                legacy_succeeded=legacy_succeeded,
            )
            return None

    def _emit_candidate(
        self,
        target: ShadowTarget,
        *,
        event_id: str,
        correlation_id: str,
        causation_id: str | None,
        before: dict[str, Any] | None,
        after: dict[str, Any] | None,
        legacy_succeeded: bool,
        legacy_error_type: str | None,
    ) -> None:
        if before is None or after is None:
            return
        try:
            sequence = len(self._kernel.stream(target.stream_id)) + 1
            envelope = EventEnvelope.create(
                event_id=event_id,
                event_type=(
                    SUCCESS_EVENT_TYPE if legacy_succeeded else FAILURE_EVENT_TYPE
                ),
                stream_id=target.stream_id,
                sequence=sequence,
                producer=OBSERVER_PRODUCER,
                producer_version=OBSERVER_VERSION,
                correlation_id=correlation_id,
                causation_id=causation_id,
                payload={
                    "after": after,
                    "before": before,
                    "legacy_outcome": {
                        "error_type": legacy_error_type,
                        "succeeded": legacy_succeeded,
                    },
                    "target": {
                        "callable": target.callable_name,
                        "disposition": target.module_disposition,
                        "module_path": target.module_path,
                        "target_id": target.target_id,
                    },
                },
                causal_context={
                    "arguments_captured": False,
                    "legacy_result_captured": False,
                    "observation_phase": "after_the_fact",
                    "source_evidence_range": target.evidence_range,
                },
            )
            self._kernel.append(envelope)
        except Exception as observer_error:
            self._record_failure(
                target,
                event_id=event_id,
                stage="event_append",
                error=observer_error,
                legacy_succeeded=legacy_succeeded,
            )

    def _record_failure(
        self,
        target: ShadowTarget,
        *,
        event_id: str,
        stage: str,
        error: Exception,
        legacy_succeeded: bool | None,
    ) -> None:
        message_digest = hashlib.sha256(
            str(error).encode("utf-8", errors="replace")
        ).hexdigest()
        self._failures.append(
            ShadowObservationFailure(
                target_id=target.target_id,
                event_id=event_id,
                stage=stage,
                error_type=type(error).__name__,
                error_message_digest=message_digest,
                legacy_succeeded=legacy_succeeded,
            )
        )
