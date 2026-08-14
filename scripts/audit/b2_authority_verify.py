"""Fail-closed startup verifier for the B2 authoritative event store."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.authoritative_store import (  # noqa: E402
    AUTHORITY_FAILURE_EXIT_CODE,
    AuthorityPersistenceError,
    AuthorityUnprovable,
    AuthoritativeStore,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--create", action="store_true")
    args = parser.parse_args(argv)
    if not args.database.exists() and not args.create:
        print("authoritative database is absent", file=sys.stderr)
        return AUTHORITY_FAILURE_EXIT_CODE
    store = AuthoritativeStore(args.database)
    try:
        startup = store.open()
        verification = store.verify()
        packet = {
            "schema": "eve.b2-authority-startup-verification.v1",
            "startup": asdict(startup),
            "verification": asdict(verification),
        }
        print(json.dumps(packet, sort_keys=True, separators=(",", ":")))
        return 0
    except (AuthorityUnprovable, AuthorityPersistenceError, OSError) as exc:
        print(f"authority unprovable: {exc}", file=sys.stderr)
        return AUTHORITY_FAILURE_EXIT_CODE
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
