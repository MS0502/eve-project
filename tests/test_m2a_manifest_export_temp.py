from __future__ import annotations

import base64
from pathlib import Path


def test_export_current_forward_manifest_for_branch_local_update() -> None:
    manifest = Path("docs/audit/FORWARD_ADDITIONS_MANIFEST.json").read_bytes()
    encoded = base64.b64encode(manifest).decode("ascii")
    with Path("/tmp/focused-paths.txt").open("a", encoding="utf-8") as handle:
        handle.write(f"\nM2A_MANIFEST_B64={encoded}\n")
