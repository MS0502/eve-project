#!/usr/bin/env python3
"""Generate exact-pin M2-E technical-candidate evidence from M2-D evidence."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.m2_e_cutover import evaluate_cutover_candidate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m2-d-packet", required=True, type=Path)
    parser.add_argument("--candidate-head", required=True)
    parser.add_argument("--workflow", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args()

    source = json.loads(args.m2_d_packet.read_text(encoding="utf-8"))
    packet = evaluate_cutover_candidate(
        source,
        candidate_head=args.candidate_head,
        workflow=args.workflow,
    )
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
