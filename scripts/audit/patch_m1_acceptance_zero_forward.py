#!/usr/bin/env python3
"""One-shot correction for an acceptance test with zero scanner findings."""
from pathlib import Path

TARGET = Path(__file__).with_name("apply_m1_human_acceptance_record.py")
text = TARGET.read_text(encoding="utf-8")
old = '''    actual_paths = {row["path"] for row in rows}
    if not rows or actual_paths != {expected_path}:
        raise RuntimeError(
            f"unexpected unregistered paths: expected={[expected_path]} actual={sorted(actual_paths)}"
        )
    grouped: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
'''
new = '''    actual_paths = {row["path"] for row in rows}
    if not rows:
        return
    if actual_paths != {expected_path}:
        raise RuntimeError(
            f"unexpected unregistered paths: expected={[expected_path]} actual={sorted(actual_paths)}"
        )
    grouped: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
'''
if text.count(old) != 1:
    raise SystemExit(f"expected one forward-registration block, found {text.count(old)}")
TARGET.write_text(text.replace(old, new, 1), encoding="utf-8")
Path(__file__).unlink()
