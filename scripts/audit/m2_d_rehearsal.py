#!/usr/bin/env python3
"""Generate deterministic M2-D recovery-rehearsal evidence in a disposable workspace."""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any, Mapping

from core.event_kernel import EventEnvelope
from core.m2_c_migration import StateEvidence
from core.m2_d_rehearsal import ObservationWindowSpec, run_recovery_rehearsal
from core.shadow_observer import (
    ACTIVATION_LEARN_PAIR_TARGET,
    OBSERVER_PRODUCER,
    OBSERVER_VERSION,
    SUCCESS_EVENT_TYPE,
)

EMPTY = {"calls": [], "learned": []}
PAIR_ONE = ["alpha", "beta", 0.4]
PAIR_TWO = ["gamma", "delta", 0.6]
PAIR_PROBE = ["epsilon", "zeta", 0.8]
AFTER_ONE = {"calls": [PAIR_ONE], "learned": [PAIR_ONE]}
AFTER_TWO = {"calls": [PAIR_ONE, PAIR_TWO], "learned": [PAIR_ONE, PAIR_TWO]}
AFTER_PROBE = {
    "calls": [PAIR_ONE, PAIR_TWO, PAIR_PROBE],
    "learned": [PAIR_ONE, PAIR_TWO, PAIR_PROBE],
}


def observed_event(
    sequence: int,
    *,
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    event_id: str,
) -> EventEnvelope:
    return EventEnvelope.create(
        event_id=event_id,
        event_type=SUCCESS_EVENT_TYPE,
        stream_id=ACTIVATION_LEARN_PAIR_TARGET.stream_id,
        sequence=sequence,
        producer=OBSERVER_PRODUCER,
        producer_version=OBSERVER_VERSION,
        correlation_id="m2d:rehearsal:activation-learn-pair",
        causation_id=None,
        payload={
            "after": dict(after),
            "before": dict(before),
            "legacy_outcome": {"error_type": None, "succeeded": True},
            "target": {
                "callable": ACTIVATION_LEARN_PAIR_TARGET.callable_name,
                "disposition": ACTIVATION_LEARN_PAIR_TARGET.module_disposition,
                "module_path": ACTIVATION_LEARN_PAIR_TARGET.module_path,
                "target_id": ACTIVATION_LEARN_PAIR_TARGET.target_id,
            },
        },
        causal_context={
            "arguments_captured": False,
            "legacy_result_captured": False,
            "observation_phase": "after_the_fact",
            "source_evidence_range": ACTIVATION_LEARN_PAIR_TARGET.evidence_range,
        },
    )


def generate_packet(workspace: Path):
    events = (
        observed_event(
            1,
            before=EMPTY,
            after=AFTER_ONE,
            event_id="m2d:event:1",
        ),
        observed_event(
            2,
            before=AFTER_ONE,
            after=AFTER_TWO,
            event_id="m2d:event:2",
        ),
    )
    probe = observed_event(
        3,
        before=AFTER_TWO,
        after=AFTER_PROBE,
        event_id="m2d:event:rollback-probe",
    )
    window = ObservationWindowSpec(
        window_id="m2-d:activation-learn-pair:v1",
        baseline_event_count=2,
        snapshot_sequence=1,
        expected_final_state=StateEvidence.from_snapshot(AFTER_TWO),
    )
    return run_recovery_rehearsal(
        workspace=workspace,
        window=window,
        baseline_events=events,
        rollback_probe_event=probe,
        initial_snapshot=EMPTY,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="eve-m2d-") as temporary:
        packet = generate_packet(Path(temporary) / "rehearsal")
    text = json.dumps(
        packet.canonical_record,
        ensure_ascii=False,
        sort_keys=True,
        indent=2 if args.pretty else None,
        separators=None if args.pretty else (",", ":"),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text + "\n", encoding="utf-8")
    return 0 if packet.machine_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
