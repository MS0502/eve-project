#!/usr/bin/env python3
"""Temporary independent probe for tainted calls the M2-B extractor cannot resolve."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from scripts.audit import m2_b_read_capability_manifest as manifest


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    original_resolve = manifest.resolve
    observations: list[dict[str, Any]] = []

    def recording_resolve(info, target, functions, leaf_index):
        resolved = original_resolve(info, target, functions, leaf_index)
        if resolved is None and target:
            observations.append({
                "path": info.path,
                "symbol": info.qualname,
                "target": target,
                "external_sink": manifest.external_sink(target),
                "sink_named_function": manifest.sink_name(info.qualname),
            })
        return resolved

    manifest.resolve = recording_resolve
    try:
        report = manifest.extract_candidates(root)
    finally:
        manifest.resolve = original_resolve

    unresolved = [item for item in observations if not item["external_sink"]]
    unique = {
        (item["path"], item["symbol"], item["target"]): item
        for item in unresolved
    }
    payload = {
        "candidate_report_digest": report["report_digest"],
        "candidate_edge_count": report["summary"]["candidate_edge_count"],
        "extractor_reported_unresolved_count": report["summary"]["unresolved_boundary_call_count"],
        "observed_unresolved_call_count": len(unresolved),
        "unique_unresolved_context_count": len(unique),
        "target_counts": dict(sorted(Counter(item["target"] for item in unresolved).items())),
        "contexts": [unique[key] for key in sorted(unique)],
    }
    output = root / "m2b-unresolved-probe.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in (
        "candidate_edge_count",
        "extractor_reported_unresolved_count",
        "observed_unresolved_call_count",
        "unique_unresolved_context_count",
    )}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
