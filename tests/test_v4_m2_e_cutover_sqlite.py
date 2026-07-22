from __future__ import annotations

from pathlib import Path

from core.m2_c_migration import StateEvidence
from core.m2_d_rehearsal import _restore
from core.m2_e_cutover import (
    HumanCutoverDecision,
    StoreObservation,
    authorize_cutover,
    evaluate_cutover_candidate,
    record_post_cutover_observation,
)
from core.shadow_observer import ACTIVATION_LEARN_PAIR_TARGET
from core.shadow_projection import PROJECTION_SCHEMA_VERSION
from core.sqlite_shadow_store import SQLiteShadowStore, ShadowStoragePolicy
from scripts.audit.m2_d_rehearsal import (
    AFTER_ONE,
    AFTER_TWO,
    EMPTY,
    generate_packet,
    observed_event,
)

HEAD = "1" * 40
WORKFLOW = 123456789
ARTIFACT_SHA = "2" * 64


def test_disposable_sqlite_window_replays_and_rolls_back(tmp_path: Path) -> None:
    prerequisite = generate_packet(tmp_path / "m2-d-prerequisite")
    candidate = evaluate_cutover_candidate(
        prerequisite.canonical_record,
        candidate_head=HEAD,
        workflow=WORKFLOW,
    )
    decision = HumanCutoverDecision.create(
        candidate=candidate,
        decision_id="m2-e:sqlite-window:test",
        artifact_sha256=ARTIFACT_SHA,
        reviewer="project-owner",
        accepted=True,
    )
    authorization = authorize_cutover(candidate, decision)

    policy = ShadowStoragePolicy(snapshot_interval_events=1, max_backups=2)
    store = SQLiteShadowStore(tmp_path / "bounded.sqlite3", policy=policy)
    store.initialize()
    first = observed_event(
        1,
        before=EMPTY,
        after=AFTER_ONE,
        event_id="m2e:event:before-cutover",
    )
    second = observed_event(
        2,
        before=AFTER_ONE,
        after=AFTER_TWO,
        event_id="m2e:event:post-cutover",
    )
    store.append(first)
    store.write_snapshot(
        snapshot_id="m2e:snapshot:before-cutover",
        stream_id=ACTIVATION_LEARN_PAIR_TARGET.stream_id,
        through_sequence=1,
        state=AFTER_ONE,
        state_schema_version=PROJECTION_SCHEMA_VERSION,
    )
    backup = store.create_backup(tmp_path / "backups", backup_ordinal=1)
    store_before = StoreObservation.from_integrity_report(store.integrity_check())

    store.append(second)
    store.write_snapshot(
        snapshot_id="m2e:snapshot:post-cutover",
        stream_id=ACTIVATION_LEARN_PAIR_TARGET.stream_id,
        through_sequence=2,
        state=AFTER_TWO,
        state_schema_version=PROJECTION_SCHEMA_VERSION,
    )
    store_after = StoreObservation.from_integrity_report(store.integrity_check())
    restored = _restore(store, EMPTY)

    rollback_store = SQLiteShadowStore(Path(backup.backup_path), policy=policy)
    rollback_store.initialize()
    rollback = _restore(rollback_store, EMPTY)

    evidence = record_post_cutover_observation(
        observation_id="m2-e:sqlite-window:test",
        authorization=authorization,
        store_before=store_before,
        store_after=store_after,
        before_state=StateEvidence.from_snapshot(rollback.state.snapshot),
        authoritative_state=StateEvidence.from_snapshot(AFTER_TWO),
        replay_state=StateEvidence.from_snapshot(restored.state.snapshot),
        rollback_state=StateEvidence.from_snapshot(rollback.state.snapshot),
    )

    assert all(evidence.checks.values())
    assert evidence.event_count_delta == 1
    assert evidence.store_before["event_count"] == 1
    assert evidence.store_after["event_count"] == 2
    assert restored.state.snapshot == AFTER_TWO
    assert rollback.state.snapshot == AFTER_ONE
