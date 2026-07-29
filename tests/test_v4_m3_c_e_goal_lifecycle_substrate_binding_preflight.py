from __future__ import annotations

import ast
import hashlib
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from core.event_kernel import EventEnvelope, SHADOW_AUTHORITY
from core.m2_e_cutover_activation import (
    EVENT_STORE_ACTIVE_ROLE,
    build_operational_rollback_record,
    resolve_cutover_authority,
)
from core.m3_c_b_goal_selection_kernel import (
    ALLOWED_DRIVES,
    DriveSample,
    GoalCandidate,
    select_goal_proposal,
)
from core.m3_c_c_goal_lifecycle_kernel import (
    GoalLifecycleState,
    LifecycleEvidence,
    evaluate_lifecycle_transition,
)
from core.m3_c_d_goal_lifecycle_event_preflight import (
    build_event_envelope_candidate,
)
from core.m3_c_e_goal_lifecycle_substrate_binding_preflight import (
    BINDING_AUTHORITY,
    M3CGoalLifecycleBindingError,
    build_substrate_binding_candidates,
    run_substrate_binding_rollback_rehearsal,
    source_from_bound_envelope,
)

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "core/m3_c_e_goal_lifecycle_substrate_binding_preflight.py"


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _candidate() -> GoalCandidate:
    return GoalCandidate(
        semantic_goal_id="recover_operating_margin",
        decision_epoch=0,
        evidence_digest=_digest("recovery-evidence"),
        base_value=0.30,
        expected_value=0.0,
        urgency=0.0,
        continuity=0.0,
        cost=0.0,
        risk=0.0,
        drive_alignment={
            drive: {
                "energy": -0.90,
                "safety": -0.80,
                "curiosity": -0.10,
            }.get(drive, 0.0)
            for drive in ALLOWED_DRIVES
        },
        drive_confidence={drive: 1.0 for drive in ALLOWED_DRIVES},
    )


def _samples() -> dict[str, DriveSample]:
    values = {"energy": -0.70, "safety": -0.80, "curiosity": -0.20}
    return {
        drive: DriveSample(
            drive=drive,
            value=values.get(drive, 0.0),
            lower_bound=-1.0,
            upper_bound=1.0,
            sample_digest=_digest(f"sample:{drive}"),
            replay_elapsed_seconds=0.0,
        )
        for drive in ALLOWED_DRIVES
    }


def _sources():
    candidate = _candidate()
    selection = select_goal_proposal([candidate], _samples())
    score = selection.scored_candidates[0]
    state = GoalLifecycleState(
        candidate_id=candidate.candidate_id,
        semantic_goal_id=candidate.semantic_goal_id,
        decision_epoch=candidate.decision_epoch,
        evidence_digest=candidate.evidence_digest,
    )
    transitions = []
    for evidence in (
        LifecycleEvidence(candidate_score=score, logical_step=1),
        LifecycleEvidence(
            candidate_score=score,
            logical_step=2,
            validation_status="passed",
        ),
        LifecycleEvidence(candidate_score=score, logical_step=3),
        LifecycleEvidence(
            candidate_score=score,
            logical_step=4,
            selection_receipt=selection,
        ),
    ):
        decision = evaluate_lifecycle_transition(state, evidence)
        assert decision.transition is not None
        transitions.append(decision.transition)
        state = decision.transition.next_state()
    return tuple(build_event_envelope_candidate(item) for item in transitions)


def test_binding_targets_active_substrate_but_remains_candidate_only():
    binding = build_substrate_binding_candidates(_sources())[0]
    envelope = binding.event_envelope
    assert binding.authority == BINDING_AUTHORITY == "candidate_only"
    assert binding.target_event_store_role == EVENT_STORE_ACTIVE_ROLE
    assert envelope.authority == SHADOW_AUTHORITY
    assert binding.authoritative_append_authorized is False
    assert binding.authoritative_append_performed is False
    assert binding.sqlite_write_performed is False
    assert binding.live_writer_installed is False
    assert binding.production_integration_performed is False
    assert binding.legacy_goal_authority_transferred is False
    assert binding.m3_e_authority_open is False


def test_binding_is_deterministic_and_immutable():
    first = build_substrate_binding_candidates(_sources())
    second = build_substrate_binding_candidates(_sources())
    assert [item.binding_digest for item in first] == [
        item.binding_digest for item in second
    ]
    assert [item.event_envelope.digest for item in first] == [
        item.event_envelope.digest for item in second
    ]
    with pytest.raises(FrozenInstanceError):
        first[0].authority = "authoritative"


def test_sequence_and_causation_chain_are_contiguous():
    bindings = build_substrate_binding_candidates(_sources())
    assert [item.sequence for item in bindings] == [1, 2, 3, 4]
    assert bindings[0].causation_event_id is None
    assert [item.causation_event_id for item in bindings[1:]] == [
        item.event_envelope.event_id for item in bindings[:-1]
    ]


def test_bound_payload_round_trips_to_source_candidate():
    binding = build_substrate_binding_candidates(_sources())[0]
    recovered = source_from_bound_envelope(
        binding.event_envelope,
        authority_state_digest=binding.authority_state_digest,
    )
    assert recovered == binding.source
    assert recovered.envelope_digest == binding.source.envelope_digest


def test_substrate_replay_rollback_and_checkpoint_resume_are_verified():
    result = run_substrate_binding_rollback_rehearsal(
        _sources(), checkpoint_count=2
    )
    assert result.substrate_replay_equivalent is True
    assert result.rollback_verified is True
    assert result.checkpoint_resume_equivalent is True
    assert result.forward_snapshot_digest == result.direct_snapshot_digest
    assert result.restored_snapshot_digest == result.checkpoint_snapshot_digest
    assert result.resumed_snapshot_digest == result.forward_snapshot_digest
    assert result.isolated_kernel_append_count == 4
    assert result.authoritative_append_performed is False
    assert result.sqlite_write_performed is False
    assert result.live_writer_installed is False
    assert result.production_integration_performed is False
    assert result.legacy_goal_authority_transferred is False
    assert result.m3_e_authority_open is False


def test_rehearsal_is_deterministic():
    first = run_substrate_binding_rollback_rehearsal(
        _sources(), checkpoint_count=2
    )
    second = run_substrate_binding_rollback_rehearsal(
        _sources(), checkpoint_count=2
    )
    assert first == second
    assert first.rehearsal_digest == second.rehearsal_digest


def test_operationally_rolled_back_substrate_fails_closed():
    rollback = build_operational_rollback_record(
        requested_by="m3-c-e-test", reason="bounded-test"
    )
    rolled_back = resolve_cutover_authority(rollback)
    with pytest.raises(M3CGoalLifecycleBindingError, match="not active"):
        build_substrate_binding_candidates(
            _sources(), authority_state=rolled_back
        )


def test_tampered_transition_payload_fails_closed():
    binding = build_substrate_binding_candidates(_sources())[0]
    envelope = binding.event_envelope
    payload = envelope.payload
    payload["transition_id"] = _digest("tampered")
    tampered = EventEnvelope.create(
        event_id=envelope.event_id,
        event_type=envelope.event_type,
        stream_id=envelope.stream_id,
        sequence=envelope.sequence,
        producer=envelope.producer,
        producer_version=envelope.producer_version,
        correlation_id=envelope.correlation_id,
        causation_id=envelope.causation_id,
        payload=payload,
        causal_context=envelope.causal_context,
    )
    with pytest.raises(M3CGoalLifecycleBindingError, match="payload or identity"):
        source_from_bound_envelope(
            tampered,
            authority_state_digest=binding.authority_state_digest,
        )


def test_tampered_authority_context_fails_closed():
    binding = build_substrate_binding_candidates(_sources())[0]
    envelope = binding.event_envelope
    context = envelope.causal_context
    context["m3_e_authority_open"] = True
    tampered = EventEnvelope.create(
        event_id=envelope.event_id,
        event_type=envelope.event_type,
        stream_id=envelope.stream_id,
        sequence=envelope.sequence,
        producer=envelope.producer,
        producer_version=envelope.producer_version,
        correlation_id=envelope.correlation_id,
        causation_id=envelope.causation_id,
        payload=envelope.payload,
        causal_context=context,
    )
    with pytest.raises(M3CGoalLifecycleBindingError, match="causal context"):
        source_from_bound_envelope(
            tampered,
            authority_state_digest=binding.authority_state_digest,
        )


def test_binding_rejects_authoritative_or_writer_claims():
    binding = build_substrate_binding_candidates(_sources())[0]
    with pytest.raises(M3CGoalLifecycleBindingError, match="writer, effects"):
        replace(binding, authoritative_append_authorized=True)
    with pytest.raises(M3CGoalLifecycleBindingError, match="writer, effects"):
        replace(binding, sqlite_write_performed=True)


def test_empty_or_wrong_source_inputs_fail_closed():
    with pytest.raises(M3CGoalLifecycleBindingError, match="at least one"):
        build_substrate_binding_candidates(())
    with pytest.raises(M3CGoalLifecycleBindingError, match="all sources"):
        build_substrate_binding_candidates((object(),))


def test_out_of_order_source_chain_fails_before_binding():
    sources = _sources()
    with pytest.raises(M3CGoalLifecycleBindingError, match="not replay-valid"):
        build_substrate_binding_candidates((sources[1], sources[0]))


def test_checkpoint_must_leave_forward_suffix():
    sources = _sources()
    with pytest.raises(M3CGoalLifecycleBindingError, match="forward replay suffix"):
        run_substrate_binding_rollback_rehearsal(
            sources, checkpoint_count=len(sources)
        )


def test_module_has_no_sqlite_file_network_thread_or_live_writer_surface():
    tree = ast.parse(MODULE.read_text(encoding="utf-8"), filename=str(MODULE))
    imported = set()
    calls = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".", 1)[0])
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                calls.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                calls.add(node.func.attr)
    assert not imported & {
        "os",
        "pathlib",
        "shutil",
        "socket",
        "sqlite3",
        "subprocess",
        "threading",
        "time",
    }
    assert not calls & {
        "SQLiteShadowStore",
        "connect",
        "create_backup",
        "initialize",
        "mkdir",
        "open",
        "write_snapshot",
        "write_text",
    }
