#!/usr/bin/env python3
"""Register the four exact PR #158 forward-gate findings, then verify them."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "docs/audit/FORWARD_ADDITIONS_MANIFEST.json"
TEST_PATH = "tests/test_v4_m1_human_acceptance_record.py"
TARGET_PR = 158
BASE_SHA = "7c4573e628e5ac51d0d64ad1040078741f3630e0"
FINGERPRINTS = {
    "09ea2c859612abe2c16337858309d88475c5c36b4660ff2eb533cf604d4dffdc": 1,
    "7f5c886779c29bbd1ab48087a5ad04193079f5240fec411748ea5e1648427c45": 1,
    "92e219d1817ad4b1afae787e4a111438428c59c3020db35a77e4484a72605a5d": 1,
    "d7bbf9c8beeb64cbe5d1d8a85f9f7843a8cb648b51fc90a32359a110d42daeff": 1,
}
SYMBOLS = [
    "_transition_sha",
    "test_acceptance_record_is_canonical_pinned_and_explicit",
    "test_evidence_pins_match_exact_committed_artifacts",
    "test_markdown_and_status_pin_the_decision_without_overclaiming_coverage",
]


def run(args: list[str]) -> None:
    subprocess.run(args, cwd=REPO_ROOT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-sha", required=True)
    args = parser.parse_args()
    if args.base_sha != BASE_SHA:
        raise SystemExit(f"unexpected base SHA: {args.base_sha}")

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    groups = manifest["registered_addition_groups"]
    groups[:] = [
        group
        for group in groups
        if not (
            group.get("introduced_by_pr") == TARGET_PR
            and group.get("path") == TEST_PATH
        )
    ]
    groups.append(
        {
            "categories": ["adaptive_numeric"],
            "disposition": "TEST_EVIDENCE",
            "fingerprints": FINGERPRINTS,
            "introduced_by_pr": TARGET_PR,
            "owner": "M1 human acceptance verification",
            "path": TEST_PATH,
            "rationale": (
                "Independent fail-closed recalculation of the external M1 human-acceptance "
                "decision from committed raw observations. The four encode calls are hashing "
                "operations in test evidence and do not mutate runtime state or grant authority."
            ),
            "symbols": SYMBOLS,
        }
    )
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n",
        encoding="utf-8",
    )

    Path(__file__).unlink()

    run(
        [
            sys.executable,
            "scripts/audit/forward_regression_gate.py",
            "--current-pr",
            str(TARGET_PR),
            "--base-sha",
            args.base_sha,
            "--pretty",
            "--output",
            ".m1-human-acceptance-forward-registered.json",
        ]
    )
    report = json.loads(
        (REPO_ROOT / ".m1-human-acceptance-forward-registered.json").read_text(
            encoding="utf-8"
        )
    )
    (REPO_ROOT / ".m1-human-acceptance-forward-registered.json").unlink()
    if report["pass"] is not True:
        raise RuntimeError("forward gate did not pass after exact registration")
    if report["unregistered_additions"] != []:
        raise RuntimeError("unregistered additions remain after exact registration")
    if report["stale_registrations"] != []:
        raise RuntimeError("stale registrations remain after exact registration")
    if report["same_pr_registration_errors"] != []:
        raise RuntimeError("same-PR registration errors remain")

    run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "--tb=short",
            TEST_PATH,
        ]
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
