"""Candidate-only M3-C substrate binding and in-memory rollback preflight.

This module binds M3-C-D candidates to the canonical v4 EventEnvelope and
rehearses append/replay/rollback only in an isolated InMemoryEventKernel. It
performs no SQLite/file I/O, installs no live writer, changes no production
configuration, transfers no legacy goal authority, and does not open M3-E.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from core.event_kernel import EventEnvelope, InMemoryEventKernel, SHADOW_AUTHORITY, canonical_json_object
from core.m2_e_cutover_activation import CutoverAuthorityState, EVENT_STORE_ACTIVE_ROLE, active_cutover_authority
from core.m3_c_c_goal_lifecycle_kernel import GoalLifecycleTransitionCandidate
from core.m3_c_d_goal_lifecycle_event_preflight import (
    EVENT_STREAM,
    EVENT_TYPE,
    GoalLifecycleEventEnvelopeCandidate,
    GoalLifecycleReducerSnapshot,
    apply_event_candidate_in_memory,
    build_event_envelope_candidate,
    replay_event_candidates_in_memory,
)

BINDING_PREFLIGHT_VERSION = "eve.m3-c-e.goal-lifecycle-substrate-binding-preflight.v1"
BINDING_CANDIDATE_VERSION = "eve.m3-c-e.goal-lifecycle-substrate-binding-candidate.v1"
ROLLBACK_REHEARSAL_VERSION = "eve.m3-c-e.goal-lifecycle-substrate-rollback-rehearsal.v1"
BINDING_AUTHORITY = "candidate_only"
PRODUCER_ID = "m3c.goal-lifecycle-binding"
PRODUCER_VERSION = "v1"


class M3CGoalLifecycleBindingError(ValueError):
    """Fail-closed substrate-binding preflight error."""


def _digest(value: Mapping[str, Any]) -> str:
    text = canonical_json_object(value, field="m3_c_e_binding_material")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256(value: str, field: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise M3CGoalLifecycleBindingError(f"{field} must be lowercase SHA-256")


def _authority_digest(state: CutoverAuthorityState) -> str:
    if not isinstance(state, CutoverAuthorityState):
        raise M3CGoalLifecycleBindingError("authority_state must be CutoverAuthorityState")
    if (
        not state.cutover_authorized
        or not state.m3_authority_open
        or state.operational_rollback_active
        or state.event_store_role != EVENT_STORE_ACTIVE_ROLE
        or state.legacy_domain_authority_transfer_authorized
        or state.m3_e_affect_cutover_authorized
        or state.legacy_persistence_path_changed
    ):
        raise M3CGoalLifecycleBindingError("v4-native substrate authority is not active within M3-C scope")
    return _digest(state.canonical_record)


@dataclass(frozen=True, slots=True)
class GoalLifecycleSubstrateBindingCandidate:
    source: GoalLifecycleEventEnvelopeCandidate
    sequence: int
    causation_event_id: str | None
    authority_state_digest: str
    schema_version: str = BINDING_CANDIDATE_VERSION
    authority: str = BINDING_AUTHORITY
    target_event_store_role: str = EVENT_STORE_ACTIVE_ROLE
    authoritative_append_authorized: bool = False
    authoritative_append_performed: bool = False
    sqlite_write_performed: bool = False
    live_writer_installed: bool = False
    production_integration_performed: bool = False
    action_authorized: bool = False
    scheduler_authorized: bool = False
    speech_authorized: bool = False
    legacy_goal_authority_transferred: bool = False
    m3_e_authority_open: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.source, GoalLifecycleEventEnvelopeCandidate):
            raise M3CGoalLifecycleBindingError("source must be GoalLifecycleEventEnvelopeCandidate")
        if isinstance(self.sequence, bool) or not isinstance(self.sequence, int) or self.sequence < 1:
            raise M3CGoalLifecycleBindingError("sequence must be positive")
        if (self.sequence == 1) != (self.causation_event_id is None):
            raise M3CGoalLifecycleBindingError("causation must be absent only for sequence one")
        _sha256(self.authority_state_digest, "authority_state_digest")
        if (self.schema_version, self.authority, self.target_event_store_role) != (
            BINDING_CANDIDATE_VERSION,
            BINDING_AUTHORITY,
            EVENT_STORE_ACTIVE_ROLE,
        ):
            raise M3CGoalLifecycleBindingError("binding metadata mismatch")
        if any((
            self.authoritative_append_authorized,
            self.authoritative_append_performed,
            self.sqlite_write_performed,
            self.live_writer_installed,
            self.production_integration_performed,
            self.action_authorized,
            self.scheduler_authorized,
            self.speech_authorized,
            self.legacy_goal_authority_transferred,
            self.m3_e_authority_open,
        )):
            raise M3CGoalLifecycleBindingError("binding candidate cannot claim writer, effects, or authority")
        if self.event_envelope.authority != SHADOW_AUTHORITY:
            raise M3CGoalLifecycleBindingError("preflight envelope must remain shadow_only")

    @property
    def event_envelope(self) -> EventEnvelope:
        transition = self.source.transition
        return EventEnvelope.create(
            event_id=self.source.event_id,
            event_type=EVENT_TYPE,
            stream_id=EVENT_STREAM,
            sequence=self.sequence,
            producer=PRODUCER_ID,
            producer_version=PRODUCER_VERSION,
            correlation_id=f"m3c:goal:{transition.candidate_id}",
            causation_id=self.causation_event_id,
            payload={
                "source_envelope_digest": self.source.envelope_digest,
                "source_payload_digest": self.source.payload_digest,
                "source_schema_version": self.source.schema_version,
                "transition": transition.to_mapping(),
                "transition_id": transition.transition_id,
            },
            causal_context={
                "authority_state_digest": self.authority_state_digest,
                "binding_authority": BINDING_AUTHORITY,
                "legacy_goal_authority_transferred": False,
                "live_writer_installed": False,
                "m3_e_authority_open": False,
                "target_event_store_role": EVENT_STORE_ACTIVE_ROLE,
            },
        )

    def to_mapping(self) -> dict[str, Any]:
        envelope = self.event_envelope
        return {
            "action_authorized": self.action_authorized,
            "authoritative_append_authorized": self.authoritative_append_authorized,
            "authoritative_append_performed": self.authoritative_append_performed,
            "authority": self.authority,
            "authority_state_digest": self.authority_state_digest,
            "binding_envelope_digest": envelope.digest,
            "causation_event_id": self.causation_event_id,
            "event_id": envelope.event_id,
            "legacy_goal_authority_transferred": self.legacy_goal_authority_transferred,
            "live_writer_installed": self.live_writer_installed,
            "m3_e_authority_open": self.m3_e_authority_open,
            "production_integration_performed": self.production_integration_performed,
            "scheduler_authorized": self.scheduler_authorized,
            "schema_version": self.schema_version,
            "sequence": self.sequence,
            "source_envelope_digest": self.source.envelope_digest,
            "speech_authorized": self.speech_authorized,
            "sqlite_write_performed": self.sqlite_write_performed,
            "target_event_store_role": self.target_event_store_role,
            "transition_id": self.source.transition.transition_id,
        }

    @property
    def binding_digest(self) -> str:
        return _digest(self.to_mapping())


@dataclass(frozen=True, slots=True)
class GoalLifecycleSubstrateRollbackRehearsal:
    binding_digests: tuple[str, ...]
    event_envelope_digests: tuple[str, ...]
    checkpoint_count: int
    checkpoint_snapshot_digest: str
    forward_snapshot_digest: str
    direct_snapshot_digest: str
    restored_snapshot_digest: str
    resumed_snapshot_digest: str
    isolated_kernel_append_count: int
    authority_state_digest: str
    rollback_verified: bool
    substrate_replay_equivalent: bool
    checkpoint_resume_equivalent: bool
    schema_version: str = ROLLBACK_REHEARSAL_VERSION
    authority: str = BINDING_AUTHORITY
    isolated_in_memory_only: bool = True
    authoritative_append_performed: bool = False
    sqlite_write_performed: bool = False
    live_writer_installed: bool = False
    production_integration_performed: bool = False
    legacy_goal_authority_transferred: bool = False
    m3_e_authority_open: bool = False

    def __post_init__(self) -> None:
        digests = (*self.binding_digests, *self.event_envelope_digests, self.checkpoint_snapshot_digest,
                   self.forward_snapshot_digest, self.direct_snapshot_digest, self.restored_snapshot_digest,
                   self.resumed_snapshot_digest, self.authority_state_digest)
        for value in digests:
            _sha256(value, "rehearsal digest")
        size = len(self.binding_digests)
        if not size or size != len(self.event_envelope_digests):
            raise M3CGoalLifecycleBindingError("rehearsal digest counts disagree")
        if isinstance(self.checkpoint_count, bool) or not 0 <= self.checkpoint_count < size:
            raise M3CGoalLifecycleBindingError("checkpoint_count must leave a forward replay suffix")
        if self.isolated_kernel_append_count != size:
            raise M3CGoalLifecycleBindingError("isolated append count disagrees with binding count")
        expected = (
            self.restored_snapshot_digest == self.checkpoint_snapshot_digest,
            self.forward_snapshot_digest == self.direct_snapshot_digest,
            self.resumed_snapshot_digest == self.forward_snapshot_digest,
        )
        actual = (self.rollback_verified, self.substrate_replay_equivalent, self.checkpoint_resume_equivalent)
        if (
            actual != expected
            or not all(actual)
            or self.schema_version != ROLLBACK_REHEARSAL_VERSION
            or self.authority != BINDING_AUTHORITY
            or not self.isolated_in_memory_only
            or any((self.authoritative_append_performed, self.sqlite_write_performed, self.live_writer_installed,
                    self.production_integration_performed, self.legacy_goal_authority_transferred, self.m3_e_authority_open))
        ):
            raise M3CGoalLifecycleBindingError("rehearsal evidence or authority boundary mismatch")

    def to_mapping(self) -> dict[str, Any]:
        return {name: (list(value) if isinstance(value, tuple) else value) for name, value in (
            ("authority", self.authority), ("authority_state_digest", self.authority_state_digest),
            ("authoritative_append_performed", self.authoritative_append_performed),
            ("binding_digests", self.binding_digests), ("checkpoint_count", self.checkpoint_count),
            ("checkpoint_resume_equivalent", self.checkpoint_resume_equivalent),
            ("checkpoint_snapshot_digest", self.checkpoint_snapshot_digest),
            ("direct_snapshot_digest", self.direct_snapshot_digest),
            ("event_envelope_digests", self.event_envelope_digests),
            ("forward_snapshot_digest", self.forward_snapshot_digest),
            ("isolated_in_memory_only", self.isolated_in_memory_only),
            ("isolated_kernel_append_count", self.isolated_kernel_append_count),
            ("legacy_goal_authority_transferred", self.legacy_goal_authority_transferred),
            ("live_writer_installed", self.live_writer_installed), ("m3_e_authority_open", self.m3_e_authority_open),
            ("production_integration_performed", self.production_integration_performed),
            ("restored_snapshot_digest", self.restored_snapshot_digest),
            ("resumed_snapshot_digest", self.resumed_snapshot_digest), ("rollback_verified", self.rollback_verified),
            ("schema_version", self.schema_version), ("sqlite_write_performed", self.sqlite_write_performed),
            ("substrate_replay_equivalent", self.substrate_replay_equivalent),
        )}

    @property
    def rehearsal_digest(self) -> str:
        return _digest(self.to_mapping())


def build_substrate_binding_candidates(
    sources: Sequence[GoalLifecycleEventEnvelopeCandidate],
    *,
    authority_state: CutoverAuthorityState | None = None,
) -> tuple[GoalLifecycleSubstrateBindingCandidate, ...]:
    values = tuple(sources)
    if not values:
        raise M3CGoalLifecycleBindingError("at least one source candidate is required")
    if any(not isinstance(item, GoalLifecycleEventEnvelopeCandidate) for item in values):
        raise M3CGoalLifecycleBindingError("all sources must be GoalLifecycleEventEnvelopeCandidate")
    try:
        replay_event_candidates_in_memory(values)
    except ValueError as exc:
        raise M3CGoalLifecycleBindingError("source candidate chain is not replay-valid") from exc
    authority_digest = _authority_digest(authority_state or active_cutover_authority())
    result = []
    prior = None
    for sequence, source in enumerate(values, 1):
        item = GoalLifecycleSubstrateBindingCandidate(source, sequence, prior, authority_digest)
        result.append(item)
        prior = item.event_envelope.event_id
    return tuple(result)


def source_from_bound_envelope(
    envelope: EventEnvelope,
    *,
    authority_state_digest: str,
) -> GoalLifecycleEventEnvelopeCandidate:
    if not isinstance(envelope, EventEnvelope):
        raise M3CGoalLifecycleBindingError("reducer requires EventEnvelope")
    if (envelope.event_type, envelope.stream_id, envelope.producer, envelope.producer_version, envelope.authority) != (
        EVENT_TYPE, EVENT_STREAM, PRODUCER_ID, PRODUCER_VERSION, SHADOW_AUTHORITY
    ):
        raise M3CGoalLifecycleBindingError("bound event metadata mismatch")
    expected_context = {
        "authority_state_digest": authority_state_digest,
        "binding_authority": BINDING_AUTHORITY,
        "legacy_goal_authority_transferred": False,
        "live_writer_installed": False,
        "m3_e_authority_open": False,
        "target_event_store_role": EVENT_STORE_ACTIVE_ROLE,
    }
    if envelope.causal_context != expected_context:
        raise M3CGoalLifecycleBindingError("bound causal context mismatch")
    payload = envelope.payload
    transition_mapping = payload.get("transition")
    if not isinstance(transition_mapping, dict):
        raise M3CGoalLifecycleBindingError("bound transition payload is missing")
    try:
        transition = GoalLifecycleTransitionCandidate(**transition_mapping)
    except (TypeError, ValueError) as exc:
        raise M3CGoalLifecycleBindingError("bound transition payload is invalid") from exc
    source = build_event_envelope_candidate(transition)
    expected_payload = {
        "source_envelope_digest": source.envelope_digest,
        "source_payload_digest": source.payload_digest,
        "source_schema_version": source.schema_version,
        "transition": transition.to_mapping(),
        "transition_id": transition.transition_id,
    }
    if payload != expected_payload or envelope.event_id != source.event_id:
        raise M3CGoalLifecycleBindingError("bound event payload or identity does not match source candidate")
    return source


def run_substrate_binding_rollback_rehearsal(
    sources: Sequence[GoalLifecycleEventEnvelopeCandidate],
    *,
    checkpoint_count: int,
    authority_state: CutoverAuthorityState | None = None,
) -> GoalLifecycleSubstrateRollbackRehearsal:
    bindings = build_substrate_binding_candidates(sources, authority_state=authority_state)
    if isinstance(checkpoint_count, bool) or not isinstance(checkpoint_count, int) or not 0 <= checkpoint_count < len(bindings):
        raise M3CGoalLifecycleBindingError("checkpoint_count must leave a forward replay suffix")
    authority_digest = bindings[0].authority_state_digest

    def reducer(state: GoalLifecycleReducerSnapshot, envelope: EventEnvelope) -> GoalLifecycleReducerSnapshot:
        source = source_from_bound_envelope(envelope, authority_state_digest=authority_digest)
        return apply_event_candidate_in_memory(state, source)[0]

    kernel: InMemoryEventKernel[GoalLifecycleReducerSnapshot] = InMemoryEventKernel()
    for item in bindings:
        kernel.append(item.event_envelope)
    forward = kernel.replay(GoalLifecycleReducerSnapshot.empty(), reducer)
    direct = replay_event_candidates_in_memory(tuple(sources))[0]

    checkpoint_kernel: InMemoryEventKernel[GoalLifecycleReducerSnapshot] = InMemoryEventKernel()
    for item in bindings[:checkpoint_count]:
        checkpoint_kernel.append(item.event_envelope)
    checkpoint = checkpoint_kernel.replay(GoalLifecycleReducerSnapshot.empty(), reducer)

    restored_kernel: InMemoryEventKernel[GoalLifecycleReducerSnapshot] = InMemoryEventKernel()
    for item in bindings[:checkpoint_count]:
        restored_kernel.append(item.event_envelope)
    restored = restored_kernel.replay(GoalLifecycleReducerSnapshot.empty(), reducer)

    resumed = checkpoint
    for envelope in kernel.events()[checkpoint_count:]:
        resumed = reducer(resumed, envelope)

    return GoalLifecycleSubstrateRollbackRehearsal(
        tuple(item.binding_digest for item in bindings),
        tuple(item.event_envelope.digest for item in bindings),
        checkpoint_count,
        checkpoint.snapshot_digest,
        forward.snapshot_digest,
        direct.snapshot_digest,
        restored.snapshot_digest,
        resumed.snapshot_digest,
        len(kernel),
        authority_digest,
        restored.snapshot_digest == checkpoint.snapshot_digest,
        forward.snapshot_digest == direct.snapshot_digest,
        resumed.snapshot_digest == forward.snapshot_digest,
    )
