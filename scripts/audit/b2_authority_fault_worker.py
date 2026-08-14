"""Subprocess worker used to prove B2 interruption behavior."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.authoritative_store import FAULT_POINTS, AuthoritativeStore  # noqa: E402
from core.event_kernel import EventEnvelope  # noqa: E402

INTERRUPTION_EXIT_CODE = 93


def _event(sequence: int, *, physical_gate: bool = False) -> EventEnvelope:
    if physical_gate:
        return EventEnvelope.create(
            event_id=f"b2:physical:{sequence}",
            event_type="authority.physical-gate",
            stream_id="authority:physical-gate",
            sequence=sequence,
            producer="scripts.operator.b2_authority_physical_gate",
            producer_version="1.0.0",
            correlation_id="corr:b2-physical-gate",
            causation_id=None if sequence == 1 else f"b2:physical:{sequence - 1}",
            payload={"delta": sequence},
            causal_context={"gate": "ryzen-7-8840u"},
        )
    return EventEnvelope.create(
        event_id=f"b2:worker:{sequence}",
        event_type="authority.worker",
        stream_id="authority:worker",
        sequence=sequence,
        producer="scripts.audit.b2_authority_fault_worker",
        producer_version="1.0.0",
        correlation_id="corr:b2-worker",
        causation_id=None if sequence == 1 else f"b2:worker:{sequence - 1}",
        payload={"sequence": sequence},
        causal_context={"source": "deterministic-fault-worker"},
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--sequence", type=int, required=True)
    parser.add_argument("--fault-point", choices=FAULT_POINTS, required=True)
    parser.add_argument("--physical-gate", action="store_true")
    args = parser.parse_args(argv)

    def interrupt(point: str) -> None:
        if point == args.fault_point:
            os._exit(INTERRUPTION_EXIT_CODE)

    store = AuthoritativeStore(args.database, fault_injector=interrupt)
    try:
        store.open()
        store.append(_event(args.sequence, physical_gate=args.physical_gate))
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
