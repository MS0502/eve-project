#!/usr/bin/env python3
"""Operator-only C1 helper for digest-only launch attestation evidence.

The nonce is read from an operator-private file and is never printed. The command
emits only public attestation material or a digest-only local verification summary.
It does not register provenance, retain observations, or touch EVE runtime state.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.m3_b_operator_attestation_trust_root import (
    OperatorLaunchBinding,
    OperatorPublicLaunchAttestation,
    build_operator_public_launch_attestation,
    verify_operator_private_binding,
)


def _canonical(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _private_nonce(path: str) -> bytes:
    nonce_path = Path(path).expanduser().resolve()
    try:
        nonce_path.relative_to(ROOT.resolve())
    except ValueError:
        pass
    else:
        raise SystemExit("private nonce file must remain outside the repository")
    stat_result = nonce_path.stat()
    if os.name != "nt" and stat_result.st_mode & 0o077:
        raise SystemExit("private nonce file must not grant group/other permissions")
    return nonce_path.read_bytes()


def _read_public(path: str) -> OperatorPublicLaunchAttestation:
    value = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    return OperatorPublicLaunchAttestation.from_mapping(value)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    attest = subparsers.add_parser("attest")
    attest.add_argument("--nonce-file", required=True)
    attest.add_argument("--runtime-instance-id", required=True)
    attest.add_argument("--source-instance-id", required=True)
    attest.add_argument("--repository-head-sha", required=True)
    attest.add_argument("--entrypoint-id", required=True)
    attest.add_argument("--launch-attestation-id", required=True)
    attest.add_argument("--logical-tick", required=True, type=int)
    attest.add_argument("--fixture-only", action="store_true")

    verify = subparsers.add_parser("verify-local")
    verify.add_argument("--nonce-file", required=True)
    verify.add_argument("--attestation-file", required=True)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "attest":
        binding = OperatorLaunchBinding(
            runtime_instance_id=args.runtime_instance_id,
            source_instance_id=args.source_instance_id,
            repository_head_sha=args.repository_head_sha,
            entrypoint_id=args.entrypoint_id,
            launch_attestation_id=args.launch_attestation_id,
            logical_tick=args.logical_tick,
            fixture_only=args.fixture_only,
        )
        attestation = build_operator_public_launch_attestation(
            binding,
            _private_nonce(args.nonce_file),
        )
        print(_canonical(attestation.to_mapping()))
        return 0

    attestation = _read_public(args.attestation_file)
    verification_trace_digest = verify_operator_private_binding(
        attestation,
        _private_nonce(args.nonce_file),
    )
    print(
        _canonical(
            {
                "attestation_digest": attestation.attestation_digest,
                "fixture_only": attestation.fixture_only,
                "launch_attestation_id": attestation.launch_attestation_id,
                "local_verification_trace_digest": verification_trace_digest,
                "private_nonce_commitment_digest": attestation.private_nonce_commitment_digest,
                "repository_head_sha": attestation.repository_head_sha,
                "runtime_instance_id": attestation.runtime_instance_id,
                "source_instance_id": attestation.source_instance_id,
                "trust_domain": attestation.trust_domain,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
