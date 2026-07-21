from __future__ import annotations

import base64
import io
import json
import os
import subprocess
import tarfile
import urllib.parse
import urllib.request
import zlib
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BRANCH = "codex/m2-a-sqlite-shadow-persistence"
BASE_SHA = "dcd3ca93b559becb1831b7ab04c2c8164c1bd3f9"
PAYLOAD = ROOT / ".github/m2a_payload.b85"
WORKFLOW = ROOT / ".github/workflows/m2a-materialize.yml"
BOOTSTRAP = ROOT / "docs/audit/M2_A_BOOTSTRAP.md"
STATUS = ROOT / "docs/EVE_IMPLEMENTATION_STATUS_v4.md"
MANIFEST = ROOT / "docs/audit/FORWARD_ADDITIONS_MANIFEST.json"


def run(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=ROOT, check=check, text=True)


def resolve_pr_number() -> int:
    repo = os.environ["GITHUB_REPOSITORY"]
    owner = repo.split("/", 1)[0]
    query = urllib.parse.urlencode({"state": "open", "head": f"{owner}:{BRANCH}"})
    request = urllib.request.Request(
        f"https://api.github.com/repos/{repo}/pulls?{query}",
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.load(response)
    if not isinstance(payload, list) or len(payload) != 1:
        raise RuntimeError(f"expected exactly one open PR for {BRANCH}, got {len(payload)}")
    return int(payload[0]["number"])


def extract_payload() -> None:
    encoded = "".join(PAYLOAD.read_text(encoding="ascii").splitlines())
    archive = zlib.decompress(base64.b85decode(encoded.encode("ascii")))
    allowed = {
        "core/sqlite_shadow_store.py",
        "tests/test_v4_m2_a_sqlite_shadow_store.py",
    }
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as handle:
        names = {member.name for member in handle.getmembers() if member.isfile()}
        if names != allowed:
            raise RuntimeError(f"payload paths differ from reviewed set: {sorted(names)}")
        for member in handle.getmembers():
            if not member.isfile():
                continue
            target = ROOT / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            source = handle.extractfile(member)
            if source is None:
                raise RuntimeError(f"cannot extract {member.name}")
            target.write_bytes(source.read())


def remove_bootstrap_helpers() -> None:
    for path in (Path(__file__), PAYLOAD, WORKFLOW, BOOTSTRAP):
        path.unlink(missing_ok=True)


def update_status(pr_number: int) -> None:
    text = STATUS.read_text(encoding="utf-8")
    text = text.replace(
        "M2-A status: **not started; eligible to be scoped next only as separate work after v4.2, with no implementation or authority granted by this amendment**",
        f"M2-A status: **implemented as an explicit, disconnected SQLite shadow-persistence candidate by PR #{pr_number}; this head starts M2 only within the bounded M2-A scope, while legacy runtime and persistence authority remain unchanged pending review and merge**",
    )
    text = text.replace(
        "Current next step: **M2-A append-only SQLite shadow-persistence work may be separately scoped; no M2 pre-design or implementation is included here**",
        f"Current next step: **review and validate the M2-A candidate in PR #{pr_number}; M2-B remains blocked until the M2-A schemas and restore evidence are accepted**",
    )
    marker = (
        "None of M1-A through M1-E is connected to `main.py`, `language/streaming.py`, live/autonomous loops, production composition, persistence adapters, or default startup paths. M1-D names legacy source modules as evidence only. M1-E imports or calls no legacy module and installs no observer or bridge. No SQLite database, file event store, durable snapshot, checkpoint artifact, sidecar, WAL, backup, migration, model/vector activation, scheduler, external effect, cutover, or production authority is introduced by v4.2.\n"
    )
    insertion = marker + (
        f"\nPR #{pr_number} adds the separately constructed `core/sqlite_shadow_store.py` M2-A candidate. Import and construction perform no I/O; a caller must explicitly initialize a concrete file path and explicitly append `shadow_only` envelopes or validated snapshots. The module is not imported by `main.py`, legacy persistence, live loops, composition, observers, bridges, or defaults. It grants no dual-read, recovery authority, production integration, or cutover.\n"
    )
    if marker not in text:
        raise RuntimeError("current-authority marker not found")
    text = text.replace(marker, insertion, 1)
    section_marker = "## Merged source-of-truth evidence\n"
    section = f"""## M2-A implementation candidate — PR #{pr_number}\n\n`core/sqlite_shadow_store.py` defines the bounded durable schemas:\n\n```text\neve.sqlite-shadow-store.v1\neve.sqlite-shadow-migration.v1\neve.sqlite-shadow-snapshot.v1\neve.sqlite-shadow-append-receipt.v1\neve.sqlite-shadow-snapshot-receipt.v1\neve.sqlite-shadow-integrity-report.v1\neve.sqlite-shadow-restore-report.v1\neve.sqlite-shadow-backup-receipt.v1\n```\n\nThe candidate provides explicit SQLite initialization, WAL request with reported fallback, `FULL` synchronous mode, explicit transactions, immutable schema-migration history, append-only event and snapshot triggers, canonical event digests, a chained durable event digest, computed before/after `state_changed` evidence, readback verification before commit, bounded event/byte/snapshot/backup policy, periodic snapshot eligibility, newest-valid-snapshot selection with corrupt-snapshot fallback, repeated deterministic restore verification, SQLite and logical integrity reports, and verified bounded backups. Historical events are never pruned. Storage-limit exhaustion rejects the new write rather than deleting history.\n\nThe implementation remains limited to the single M1 event-envelope contract and caller-supplied pure reducer/state codecs. It does not install the M1-B observer, connect a lifecycle bridge, read legacy sidecars, perform dual-read comparison, make recovery authoritative, change defaults, or transfer persistence authority. Those boundaries remain assigned to later M2 milestones and explicit cutover review.\n\n`tests/test_v4_m2_a_sqlite_shadow_store.py` supplies focused evidence for explicit creation, WAL/schema/migration contracts, append ordering and hash-chain fidelity, transaction rollback, append-only triggers, storage bounds, snapshots, corrupt-snapshot fallback, repeated replay, reopen/forced-termination resilience, integrity failure visibility, bounded backups, and absence of production integration.\n\n"""
    if section_marker not in text:
        raise RuntimeError("source-of-truth section marker not found")
    text = text.replace(section_marker, section + section_marker, 1)
    text = text.replace(
        "This does not start M2 or activate any runtime capability.",
        f"PR #{pr_number} starts only the bounded M2-A implementation candidate. It does not activate a runtime bridge, dual-read path, recovery authority, cutover, or any production capability.",
        1,
    )
    text = text.replace(
        "M2-A is next and must be a separate, tightly scoped task. Until a later M2 change is independently reviewed and accepted:\n\n1. `m2_started` remains false;",
        f"M2-A is implemented as a separate candidate in PR #{pr_number} and must pass exact-head validation plus human review before merge. On this candidate head, `m2_started=true` means only that bounded M2-A work has begun; it grants no runtime or persistence authority. Until this M2-A change is independently reviewed and accepted:\n\n1. M2-A remains shadow-only and non-authoritative;",
    )
    STATUS.write_text(text, encoding="utf-8")


def append_forward_registrations(pr_number: int) -> None:
    report_path = Path("/tmp/m2a-forward-before.json")
    result = run(
        "python",
        "scripts/audit/forward_regression_gate.py",
        "--output",
        str(report_path),
        "--pretty",
        "--current-pr",
        str(pr_number),
        "--base-sha",
        BASE_SHA,
        check=False,
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    additions = report.get("unregistered_additions", [])
    if result.returncode == 0 or not additions:
        raise RuntimeError("expected unregistered M2-A additions before registration")
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for item in additions:
        grouped[str(item["path"])].append(item)
    expected_paths = {
        "core/sqlite_shadow_store.py",
        "tests/test_v4_m2_a_sqlite_shadow_store.py",
    }
    if set(grouped) != expected_paths:
        raise RuntimeError(f"unexpected forward-addition paths: {sorted(grouped)}")

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    groups = manifest["registered_addition_groups"]
    for path in sorted(grouped):
        items = grouped[path]
        is_test = path.startswith("tests/")
        groups.append(
            {
                "path": path,
                "categories": sorted({str(item["category"]) for item in items}),
                "symbols": sorted({str(item["symbol"]) for item in items}),
                "rationale": (
                    "Focused fail-closed verification for M2-A explicit SQLite initialization, append-only transactions, hash-chain and before/after fidelity, snapshots, corrupt-snapshot fallback, repeated restore verification, integrity visibility, storage bounds, backups, forced-termination resilience, and absence of runtime activation."
                    if is_test
                    else "Bounded M2-A append-only SQLite shadow persistence with explicit initialization, versioned migrations, WAL/transaction/integrity contracts, canonical event-chain evidence, validated snapshots, deterministic restore verification, and bounded backup/storage policy; legacy authority, default startup, bridges, dual-read, and cutover remain unchanged."
                ),
                "owner": (
                    "M2-A SQLite shadow persistence verification"
                    if is_test
                    else "M2-A SQLite shadow persistence"
                ),
                "disposition": (
                    "TEST_EVIDENCE" if is_test else "M2_A_SQLITE_SHADOW_PERSISTENCE"
                ),
                "introduced_by_pr": pr_number,
                "fingerprints": {
                    str(item["fingerprint"]): int(item["count"])
                    for item in sorted(items, key=lambda value: str(value["fingerprint"]))
                },
            }
        )
    MANIFEST.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    run(
        "python",
        "scripts/audit/forward_regression_gate.py",
        "--output",
        "/tmp/m2a-forward-after.json",
        "--pretty",
        "--current-pr",
        str(pr_number),
        "--base-sha",
        BASE_SHA,
    )


def main() -> None:
    pr_number = resolve_pr_number()
    extract_payload()
    remove_bootstrap_helpers()
    update_status(pr_number)
    append_forward_registrations(pr_number)
    run("python", "-m", "compileall", "-q", "core/sqlite_shadow_store.py", "tests/test_v4_m2_a_sqlite_shadow_store.py")
    run("python", "-m", "pytest", "-q", "--tb=short", "tests/test_v4_m2_a_sqlite_shadow_store.py")
    run("git", "add", "-A")
    run("git", "config", "user.name", "github-actions[bot]")
    run("git", "config", "user.email", "41898282+github-actions[bot]@users.noreply.github.com")
    run("git", "commit", "-m", "Implement M2-A SQLite shadow persistence")
    run("git", "push", "origin", f"HEAD:{BRANCH}")


if __name__ == "__main__":
    main()
