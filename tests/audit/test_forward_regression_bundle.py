from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from scripts.audit.forward_regression_bundle import (
    FRAGMENT_SCHEMA_VERSION,
    load_current_fragments,
    load_fragments_at_sha,
    merge_manifest,
)


def fragment(groups):
    return {
        "schema_version": FRAGMENT_SCHEMA_VERSION,
        "registered_addition_groups": groups,
    }


def test_current_fragments_are_loaded_in_path_order(tmp_path: Path):
    directory = tmp_path / "fragments"
    directory.mkdir()
    (directory / "b.json").write_text(
        json.dumps(fragment([{"path": "b.py"}])), encoding="utf-8"
    )
    (directory / "a.json").write_text(
        json.dumps(fragment([{"path": "a.py"}])), encoding="utf-8"
    )
    groups, names = load_current_fragments(directory)
    assert [group["path"] for group in groups] == ["a.py", "b.py"]
    assert [Path(name).name for name in names] == ["a.json", "b.json"]


def test_invalid_fragment_fails_closed(tmp_path: Path):
    directory = tmp_path / "fragments"
    directory.mkdir()
    (directory / "bad.json").write_text(
        '{"schema_version":"wrong","registered_addition_groups":[]}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="schema_version mismatch"):
        load_current_fragments(directory)


def test_merge_preserves_primary_and_appends_without_mutating_input():
    primary = {
        "registered_addition_groups": [{"path": "base.py"}],
        "baseline": {"x": 1},
    }
    merged = merge_manifest(primary, [{"path": "new.py"}])
    assert [
        group["path"] for group in merged["registered_addition_groups"]
    ] == ["base.py", "new.py"]
    assert primary["registered_addition_groups"] == [{"path": "base.py"}]


def test_fragment_history_is_loaded_from_exact_git_sha(tmp_path: Path):
    subprocess.check_call(["git", "init", "-q", str(tmp_path)])
    subprocess.check_call(
        ["git", "-C", str(tmp_path), "config", "user.email", "test@example.com"]
    )
    subprocess.check_call(
        ["git", "-C", str(tmp_path), "config", "user.name", "Test"]
    )
    directory = tmp_path / "docs/audit/forward_additions"
    directory.mkdir(parents=True)
    path = directory / "pr-1.json"
    path.write_text(
        json.dumps(fragment([{"path": "base.py"}])), encoding="utf-8"
    )
    subprocess.check_call(["git", "-C", str(tmp_path), "add", "."])
    subprocess.check_call(
        ["git", "-C", str(tmp_path), "commit", "-qm", "base"]
    )
    sha = subprocess.check_output(
        ["git", "-C", str(tmp_path), "rev-parse", "HEAD"], text=True
    ).strip()
    path.write_text(
        json.dumps(fragment([{"path": "head.py"}])), encoding="utf-8"
    )
    groups, names = load_fragments_at_sha(tmp_path, sha, directory)
    assert groups == [{"path": "base.py"}]
    assert names == ["docs/audit/forward_additions/pr-1.json"]
