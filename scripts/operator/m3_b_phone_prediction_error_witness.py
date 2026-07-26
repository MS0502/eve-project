#!/usr/bin/env python3
"""Run two real full-engine interactions and emit a digest-only M3-B C2 review witness.

The operator-private nonce and raw prediction/error trace stay outside the repository.
This command does not register the attestation, register a verifier, retain an
observation, start the M3-B observation window, or authorize any cutover.
"""
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

from core.m3_b_phone_prediction_error_witness import (
    DEFAULT_SOURCE_INSTANCE_ID,
    ENTRYPOINT_ID,
    build_phone_prediction_error_witness,
)
from core.m3_b_prediction_error_runtime_source_bridge import (
    read_prediction_error_runtime_source,
)
from main import build_full_engine

PRIVATE_FILENAME = "prediction_error_witness_private.json"
PUBLIC_FILENAME = "prediction_error_witness_public_review.json"


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


def _private_nonce(path: str) -> bytes:
    nonce_path = _outside_repository(Path(path), "private nonce file")
    stat_result = nonce_path.stat()
    if os.name != "nt" and stat_result.st_mode & 0o077:
        raise SystemExit("private nonce file must not grant group/other permissions")
    nonce = nonce_path.read_bytes()
    if len(nonce) < 32:
        raise SystemExit("private nonce file must contain at least 32 bytes")
    return nonce


def _prepare_private_root(path: str) -> Path:
    root = _outside_repository(Path(path), "private witness root")
    root.mkdir(parents=True, exist_ok=True)
    os.chmod(root, 0o700)
    return root


def _repository_head(expected_head: str) -> str:
    actual = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()
    if actual != expected_head:
        raise SystemExit(f"repository head mismatch: expected {expected_head}, got {actual}")
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"],
        cwd=ROOT,
        text=True,
    )
    if dirty.strip():
        raise SystemExit("operator witness requires a clean exact repository checkout")
    return actual


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.partial")
    temporary.write_text(_canonical(value) + "\n", encoding="utf-8")
    os.chmod(temporary, 0o600)
    temporary.replace(path)
    os.chmod(path, 0o600)


def _run_interaction(engine: Any, text: str) -> None:
    for chunk in engine.chat_stream(text):
        print(chunk, end="", file=sys.stderr, flush=True)
    print(file=sys.stderr, flush=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nonce-file", required=True)
    parser.add_argument("--private-root", required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--runtime-instance-id", required=True)
    parser.add_argument("--launch-attestation-id", required=True)
    parser.add_argument("--source-instance-id", default=DEFAULT_SOURCE_INSTANCE_ID)
    parser.add_argument(
        "--input",
        action="append",
        dest="inputs",
        required=True,
        help="Real operator interaction text. Supply exactly twice.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if len(args.inputs) != 2:
        raise SystemExit("phone prediction-error witness requires exactly two --input values")

    repository_head = _repository_head(args.expected_head)
    private_root = _prepare_private_root(args.private_root)
    private_nonce = _private_nonce(args.nonce_file)

    engine = build_full_engine()
    ai_adapter = getattr(engine, "ai_adapter", None)
    if ai_adapter is None:
        raise SystemExit("full engine does not expose ai_adapter")

    snapshots = []
    for text in args.inputs:
        if not isinstance(text, str) or not text.strip():
            raise SystemExit("witness input must be non-empty text")
        _run_interaction(engine, text)
        snapshot = read_prediction_error_runtime_source(
            ai_adapter,
            source_instance_id=args.source_instance_id,
            fixture_only=False,
        )
        if snapshot is None:
            raise SystemExit("full engine did not produce a completed prediction-error trace")
        snapshots.append(snapshot)

    witness = build_phone_prediction_error_witness(
        private_nonce=private_nonce,
        runtime_instance_id=args.runtime_instance_id,
        source_instance_id=args.source_instance_id,
        repository_head_sha=repository_head,
        launch_attestation_id=args.launch_attestation_id,
        snapshots=snapshots,
        launch_logical_tick=0,
        entrypoint_id=ENTRYPOINT_ID,
    )

    private_mapping = witness.private_mapping()
    public_mapping = witness.public_review_mapping()
    _atomic_json(private_root / PRIVATE_FILENAME, private_mapping)
    _atomic_json(private_root / PUBLIC_FILENAME, public_mapping)

    # Only this final canonical object is intended to leave the private companion.
    print(_canonical(public_mapping))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
