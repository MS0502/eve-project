from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PATHS = {
    "core/sqlite_shadow_store.py",
    "tests/test_v4_m2_a_sqlite_shadow_store.py",
}


def test_emit_exact_m2a_manifest_replacement(tmp_path: Path) -> None:
    manifest_path = ROOT / "docs/audit/FORWARD_ADDITIONS_MANIFEST.json"
    suggested_path = tmp_path / "suggested.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/audit/forward_regression_gate.py"),
            "--suggest-manifest-for-pr",
            "161",
            "--output",
            str(suggested_path),
        ],
        cwd=ROOT,
        check=True,
    )
    current = json.loads(manifest_path.read_text(encoding="utf-8"))
    suggested = json.loads(suggested_path.read_text(encoding="utf-8"))
    current_by_path = {
        group["path"]: group
        for group in current["registered_addition_groups"]
        if group.get("introduced_by_pr") == 161 and group.get("path") in PATHS
    }
    suggested_by_path = {
        group["path"]: group
        for group in suggested["registered_addition_groups"]
        if group.get("path") in PATHS
    }
    replacements = []
    for path in sorted(PATHS):
        replacement = dict(suggested_by_path[path])
        existing = current_by_path[path]
        replacement["rationale"] = existing["rationale"]
        replacement["owner"] = existing["owner"]
        replacement["disposition"] = existing["disposition"]
        replacements.append(replacement)
    expected = dict(current)
    expected["registered_addition_groups"] = sorted(
        [
            group
            for group in current["registered_addition_groups"]
            if not (group.get("introduced_by_pr") == 161 and group.get("path") in PATHS)
        ]
        + replacements,
        key=lambda group: (group["introduced_by_pr"], group["path"]),
    )
    compact = json.dumps(
        expected,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    pytest.fail(f"M2A_EXPECTED_MANIFEST_BEGIN{compact}M2A_EXPECTED_MANIFEST_END")
