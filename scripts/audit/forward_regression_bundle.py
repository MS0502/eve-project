#!/usr/bin/env python3
"""Run the forward regression gate with append-only manifest fragments.

The frozen baseline contract remains in the primary manifest. Reviewed additions
may live in deterministic per-PR fragments so governance updates do not require
rewriting the increasingly large historical registry.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable, Mapping

from scripts.audit import forward_regression_gate as gate

FRAGMENT_SCHEMA_VERSION = "1.0.0-forward-additions-fragment"
DEFAULT_FRAGMENT_DIR = Path("docs/audit/forward_additions")


def _validate_fragment(payload: Any, label: str) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        raise ValueError(f"{label} root must be an object")
    if payload.get("schema_version") != FRAGMENT_SCHEMA_VERSION:
        raise ValueError(f"{label} schema_version mismatch")
    groups = payload.get("registered_addition_groups")
    if not isinstance(groups, list):
        raise ValueError(f"{label} registered_addition_groups must be a list")
    if not all(isinstance(group, dict) for group in groups):
        raise ValueError(f"{label} groups must be objects")
    return [dict(group) for group in groups]


def _fragment_paths(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(path for path in directory.glob("*.json") if path.is_file())


def load_current_fragments(directory: Path) -> tuple[list[dict[str, Any]], list[str]]:
    groups: list[dict[str, Any]] = []
    names: list[str] = []
    for path in _fragment_paths(directory):
        payload = json.loads(path.read_text(encoding="utf-8"))
        groups.extend(_validate_fragment(payload, path.as_posix()))
        names.append(path.as_posix())
    return groups, names


def _git_paths(root: Path, sha: str, directory: Path) -> list[str]:
    relative = directory.relative_to(root).as_posix()
    try:
        raw = subprocess.check_output(
            ["git", "-C", str(root), "ls-tree", "-r", "--name-only", sha, "--", relative],
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return []
    return sorted(
        line for line in raw.decode("utf-8").splitlines() if line.endswith(".json")
    )


def load_fragments_at_sha(
    root: Path, sha: str, directory: Path
) -> tuple[list[dict[str, Any]], list[str]]:
    groups: list[dict[str, Any]] = []
    names = _git_paths(root, sha, directory)
    for relative in names:
        raw = subprocess.check_output(
            ["git", "-C", str(root), "show", f"{sha}:{relative}"],
            stderr=subprocess.DEVNULL,
        )
        payload = json.loads(raw.decode("utf-8"))
        groups.extend(_validate_fragment(payload, relative))
    return groups, names


def merge_manifest(
    primary: Mapping[str, Any], groups: Iterable[Mapping[str, Any]]
) -> dict[str, Any]:
    merged = deepcopy(dict(primary))
    existing = merged.get("registered_addition_groups")
    if not isinstance(existing, list):
        raise ValueError("primary registered_addition_groups must be a list")
    merged["registered_addition_groups"] = [
        *existing,
        *(dict(group) for group in groups),
    ]
    return merged


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--fragment-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--pretty", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--current-pr", type=int)
    parser.add_argument("--base-sha")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    if (args.current_pr is None) != (args.base_sha is None):
        raise SystemExit("--current-pr and --base-sha must be supplied together")
    root = args.root.resolve()
    manifest_path = args.manifest or (root / gate.DEFAULT_MANIFEST)
    fragment_dir = args.fragment_dir or (root / DEFAULT_FRAGMENT_DIR)

    primary = gate._load_manifest(manifest_path)
    current_groups, current_names = load_current_fragments(fragment_dir)
    manifest = merge_manifest(primary, current_groups)
    base_manifest = None
    base_names: list[str] = []
    if args.base_sha is not None:
        base_primary = gate._load_manifest_at_sha(root, args.base_sha, manifest_path)
        base_groups, base_names = load_fragments_at_sha(root, args.base_sha, fragment_dir)
        base_manifest = merge_manifest(base_primary, base_groups)

    baseline_scan = gate.scan_sources(
        root, gate._snapshot_sources(root, gate.CONSTITUTION_BASELINE_SHA)
    )
    current_scan = gate.scan_sources(root, gate._current_sources(root))
    result = gate.evaluate(
        baseline_scan,
        current_scan,
        manifest,
        base_manifest=base_manifest,
        current_pr=args.current_pr,
    )
    payload = {
        "schema_version": gate.SCHEMA_VERSION,
        "baseline_sha": gate.CONSTITUTION_BASELINE_SHA,
        "manifest": manifest_path.relative_to(root).as_posix(),
        "manifest_fragments": [
            Path(name).relative_to(root).as_posix()
            if Path(name).is_absolute()
            else name
            for name in current_names
        ],
        "base_manifest_fragments": base_names,
        **result,
    }
    text = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        indent=2 if args.pretty else None,
        separators=None if args.pretty else (",", ":"),
    ) + "\n"
    if args.output is None:
        sys.stdout.write(text)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    return 0 if result["pass"] or args.report_only else 1


if __name__ == "__main__":
    raise SystemExit(main())
