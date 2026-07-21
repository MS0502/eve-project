from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE_SHA = "dcd3ca93b559becb1831b7ab04c2c8164c1bd3f9"
BRANCH = "codex/m2-a-sqlite-shadow-persistence"
MANIFEST = ROOT / "docs/audit/FORWARD_ADDITIONS_MANIFEST.json"
WORKFLOW = ROOT / ".github/workflows/exact-head-validation.yml"
SELF = Path(__file__)
CORE_FP = "566b34b7f5861c71e9dcbb6731c7feff45d0fc27cbc573b6f5d82446754a2f65"
TEST_ADDITIONS = {
    "66984fb7424b09372cb312085056739088191da27321c09e97423d310006f469": 1,
    "f539a6cb74f2b2b3e6de46427e82c4d525b7074e2dd1725f41003f20a2fd2ecb": 1,
}


def run(*args: str) -> None:
    subprocess.run(args, cwd=ROOT, check=True)


def main() -> None:
    document = json.loads(MANIFEST.read_text(encoding="utf-8"))
    groups = {group["path"]: group for group in document["registered_addition_groups"] if group.get("introduced_by_pr") == 161}
    if set(groups) != {"core/sqlite_shadow_store.py", "tests/test_v4_m2_a_sqlite_shadow_store.py"}:
        raise RuntimeError(f"unexpected PR #161 groups: {sorted(groups)}")
    core = groups["core/sqlite_shadow_store.py"]["fingerprints"]
    if core.get(CORE_FP) != 2:
        raise RuntimeError(f"unexpected existing _connect count: {core.get(CORE_FP)!r}")
    core[CORE_FP] = 4
    tests = groups["tests/test_v4_m2_a_sqlite_shadow_store.py"]["fingerprints"]
    overlap = set(tests) & set(TEST_ADDITIONS)
    if overlap:
        raise RuntimeError(f"hardening test fingerprints already registered: {sorted(overlap)}")
    tests.update(TEST_ADDITIONS)
    groups["tests/test_v4_m2_a_sqlite_shadow_store.py"]["symbols"] = sorted(
        set(groups["tests/test_v4_m2_a_sqlite_shadow_store.py"]["symbols"])
        | {
            "test_restore_uses_fresh_canonical_start_state_for_each_replay",
            "test_restore_uses_fresh_canonical_start_state_for_each_replay.mutating_reducer",
        }
    )
    MANIFEST.write_text(json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    run("git", "checkout", BASE_SHA, "--", str(WORKFLOW.relative_to(ROOT)))
    SELF.unlink()
    run("git", "add", "-A")
    run("git", "config", "user.name", "github-actions[bot]")
    run("git", "config", "user.email", "41898282+github-actions[bot]@users.noreply.github.com")
    run("git", "commit", "-m", "Register M2-A hardening evidence")
    run("git", "push", "origin", f"HEAD:{BRANCH}")


if __name__ == "__main__":
    main()
