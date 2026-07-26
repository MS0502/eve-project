#!/usr/bin/env python3
"""Durably retain the exact reviewed C2 phone energy-budget observation once."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.m3_b_c2_energy_budget_retention_activation import (
    append_reviewed_energy_budget_observation,
)
from core.sqlite_shadow_store import SQLiteShadowStore

DATABASE_FILENAME = "retained_real_observations.sqlite3"
PUBLIC_RECEIPT_FILENAME = "c2_energy_budget_retention_public_receipt.json"


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _outside_repository(path: Path, field: str) -> Path:
    resolved = path.expanduser().resolve()
    try:
        resolved.relative_to(ROOT.resolve())
    except ValueError:
        return resolved
    raise SystemExit(f"{field} must remain outside the repository")


def _repository_head(expected_head: str) -> str:
    actual = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    if actual != expected_head:
        raise SystemExit(
            f"repository head mismatch: expected {expected_head}, got {actual}"
        )
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    )
    if dirty.strip():
        raise SystemExit(
            "energy-budget C2 retention requires a clean exact repository checkout"
        )
    return actual


def _load_public_review(path: str) -> Mapping[str, Any]:
    review_path = _outside_repository(Path(path), "public review file")
    try:
        value = json.loads(review_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit("public review file is missing or malformed") from exc
    if not isinstance(value, dict):
        raise SystemExit("public review file must contain one JSON object")
    return value


def _prepare_private_root(path: str) -> Path:
    root = _outside_repository(Path(path), "private retention root")
    root.mkdir(parents=True, exist_ok=True)
    os.chmod(root, 0o700)
    return root


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.partial")
    temporary.write_text(_canonical(value) + "\n", encoding="utf-8")
    os.chmod(temporary, 0o600)
    temporary.replace(path)
    os.chmod(path, 0o600)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-review-file", required=True)
    parser.add_argument("--private-root", required=True)
    parser.add_argument("--expected-head", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repository_head = _repository_head(args.expected_head)
    public_review = _load_public_review(args.public_review_file)
    private_root = _prepare_private_root(args.private_root)

    store = SQLiteShadowStore(private_root / DATABASE_FILENAME)
    store.initialize()
    receipt = append_reviewed_energy_budget_observation(store, public_review)
    public = {
        "activation_repository_head_sha": repository_head,
        "database_location": "operator_private_companion_only",
        "receipt": receipt.to_mapping(),
        "receipt_digest": receipt.receipt_digest,
        "retained_real_observation_count_after_append": 2,
        "schema_version": "eve.m3-b.c2-energy-budget-retention-public-review.v1",
    }
    _atomic_json(private_root / PUBLIC_RECEIPT_FILENAME, public)
    print(_canonical(public))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
