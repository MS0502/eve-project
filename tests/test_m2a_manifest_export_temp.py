from __future__ import annotations

import base64
from pathlib import Path


def test_export_current_governance_files_for_branch_local_update() -> None:
    manifest = base64.b64encode(
        Path("docs/audit/FORWARD_ADDITIONS_MANIFEST.json").read_bytes()
    ).decode("ascii")
    status = base64.b64encode(
        Path("docs/EVE_IMPLEMENTATION_STATUS_v4.md").read_bytes()
    ).decode("ascii")
    with Path("/tmp/focused-paths.txt").open("a", encoding="utf-8") as handle:
        handle.write(f"\nM2A_MANIFEST_B64={manifest}\n")
        handle.write(f"M2A_STATUS_B64={status}\n")
