from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE_SHA = "dcd3ca93b559becb1831b7ab04c2c8164c1bd3f9"
BRANCH = "codex/m2-a-sqlite-shadow-persistence"
MANIFEST = ROOT / "docs/audit/FORWARD_ADDITIONS_MANIFEST.json"
STATUS = ROOT / "docs/EVE_IMPLEMENTATION_STATUS_v4.md"
WORKFLOW = ROOT / ".github/workflows/exact-head-validation.yml"
SELF = Path(__file__)
GROUPS = [
  {
    "path": "core/sqlite_shadow_store.py",
    "categories": ["adaptive_numeric", "direct_write", "mutation"],
    "symbols": ["SQLiteShadowStore.__init__", "SQLiteShadowStore._connect", "SQLiteShadowStore._counts", "SQLiteShadowStore._snapshot_from_row", "SQLiteShadowStore._validate_schema", "SQLiteShadowStore.append_many", "SQLiteShadowStore.create_backup", "SQLiteShadowStore.events", "SQLiteShadowStore.initialize", "SQLiteShadowStore.integrity_check", "SQLiteShadowStore.latest_valid_snapshot", "SQLiteShadowStore.snapshot_due", "SQLiteShadowStore.write_snapshot", "_sha"],
    "rationale": "Bounded M2-A append-only SQLite shadow persistence with explicit initialization, versioned migration history, WAL request, explicit transactions, canonical event hash-chain evidence, validated snapshots, deterministic restore verification, integrity reporting, and bounded backups; legacy authority, defaults, bridges, dual-read, recovery cutover, and production integration remain unchanged.",
    "owner": "M2-A SQLite shadow persistence",
    "disposition": "M2_A_SQLITE_SHADOW_PERSISTENCE",
    "introduced_by_pr": 161,
    "fingerprints": {
      "0424903630208a4363e3c0c84369b2f60d146b9b9b181cf01e6be86c04224b93": 1,
      "04b1e5b9065e8e50ffdf50c54ddcba610f5a5b7b6ecc1ec70f8cb8e29743ec6c": 1,
      "14604bd5f8e81020f73ec5446adde513fc9c8d35b29b390037f892e554d9be76": 1,
      "18555558350f49444fd4f0a1bf1c2ae497056f529fb71cb7b5366a4f519b077f": 5,
      "1d2d415323979bed2c672b4e058e1807fa42253a48ca99a8773f50604237b4f9": 2,
      "251320a3d2a517839a2295a25488846c4d67dfeaef4580b62876836d4ac19ceb": 1,
      "26528af09680a49214a08116194042f59331177d14fab63f3b4c5e1405d97ee7": 7,
      "2c3b2b43a30b46ceabf3e1f132708b36a58d23b123d23d826f81bfb1a57e2439": 1,
      "2cdfe9594406046890cd2d69b239deb722a925ff91d669ab631785f7eead6684": 1,
      "2cf0646557e5f29888ed551efc4671184f9fad690d1bc364ad77e9b174614426": 2,
      "343b7d5ff966794b58b7e6d9deb3ccb377bf90e2cb2c690c3dc5bd562a4393c1": 1,
      "3cc79ae5f283dc333c3de4e9720a992ebccdd02797ffcb69c631f03121613074": 3,
      "566b34b7f5861c71e9dcbb6731c7feff45d0fc27cbc573b6f5d82446754a2f65": 2,
      "570364d1587c9aca145a2a6de114d6274662d7d40bfd5e59a9c4b6b1a29641d0": 5,
      "6075a37c6b736997033e29f16cc95727b581e52cc54055dc918d1e7a5c0b8aa3": 1,
      "6ac0a6b5ff774d113c7810923bdb565eda38f3aa6512d6858330c21436a92935": 1,
      "70a675b9b2d447ed96fd86a59365f30e4638bf4608db22f269023ab1b690583a": 1,
      "7e2350dba19ee55e2ddf4d1784ee8dcc34d2f5fc149663ef5a5756a0c63cd873": 1,
      "8067bdc3783e0fa129d630e4f8e2933aeccfc8014d88ebf786273c6d7e38581f": 1,
      "863dad8834dbd21e02224cc1506c87dd8dc2a91bd95e5888203d117b1ca5bfb2": 1,
      "8b5521f5f0698a2e8a18b2e9d6fafab8e42acc2dc05215e1a7003384470ebbb5": 1,
      "948277dc54d08ba6a3388772cd74abc52139895ba791c27f7bcc19937175190c": 1,
      "99fd60a5695cc3c5db409a5401dc240a2b4a48db50f778e4512f2742c40f55ee": 1,
      "9b4ae850f143c13acc680149d211c0c03727477f5280c02bfceba1120edcfcf2": 1,
      "9f53e83028d4a47c0aa23459e5a71b0a208b210486de097ef7db6afbe2bfb12f": 3,
      "a56d91435bc984e20598af767a4698bdca27be96d9192b86de264fb474d44430": 1,
      "ac043feced46228903a238bbfed2b72833788b49ad5c3ef56c207befd67ab1d7": 1,
      "ac2c6633e54762ec485bc04a8389896fb29bada64d1093c457d806706e04e0a3": 1,
      "ae4b162c5f9878d4df2ed3cabdc3b00c17e47c117d41adb7d83b7d6baea3bcc9": 1,
      "afdc9698ea8b39901d2b05fc9b1dbd7dd59823cb5445258e080fe0fb10098827": 1,
      "b14492c8d0bfac5011c5b87c3066a12b40aad2403056741e76f27bc9b5e4d16d": 1,
      "b604a31af288ceaceb05b858d31c2a697dd148c2c9fd13de7363e4a17b0a07e3": 1,
      "d5f1931233049230cbd0c0ac3888475aa2db1cbc6de83319698be26a20ff1d5a": 2,
      "dc1d5dbad2c10e308adaa21a453ee74749523e3fa1f408174aeeddc7c4de2303": 1,
      "dee2dbb0117838c05e9546ba93de04b7ff0b1ce279f7666c7386e33fbfb2f7b8": 1,
      "df9e5acc027c7f77a6f28fa20f771e5ba7095522cbe68a85dd0a908890d6680a": 1,
      "e32e61a62a7fbe6c00186e535378de6aee08f0d1062b0cb77f8136b220bd6adf": 1,
      "ec1b532d8003e813744d1bb319fa9fc259ae334b64fdea120678e88b17b58379": 5,
      "f35ed3287e9fa8bb10297358434f91c50c9965951d80b244f21c976d707f3c66": 1,
      "f6381050991dfbade5015b66907b22c1ea67797acd7f163149d2fb13f15d35a8": 1
    }
  },
  {
    "path": "tests/test_v4_m2_a_sqlite_shadow_store.py",
    "categories": ["adaptive_numeric", "direct_write", "mutation"],
    "symbols": ["test_corrupt_newest_snapshot_falls_back_to_previous_valid_snapshot", "test_duplicate_gap_unknown_cause_and_non_envelope_fail_without_partial_write", "test_event_corruption_is_visible_to_reads_and_integrity_report", "test_module_has_no_default_activation_legacy_bridge_thread_clock_random_or_pickle_surface", "test_reopen_after_uncommitted_external_transaction_preserves_committed_history", "test_restore_replays_twice_from_valid_snapshot_and_is_reproducible", "test_schema_migration_and_append_only_triggers_are_durable", "test_storage_policy_rejects_new_history_and_never_prunes_old_events", "test_verified_backups_are_bounded_without_touching_event_history"],
    "rationale": "Focused fail-closed verification for M2-A explicit creation, schema and migration contracts, append-only transactions, hash-chain and before/after fidelity, storage bounds, validated snapshots, corrupt-snapshot fallback, repeated replay, uncommitted-write recovery, integrity visibility, bounded backups, and absence of runtime activation.",
    "owner": "M2-A SQLite shadow persistence verification",
    "disposition": "TEST_EVIDENCE",
    "introduced_by_pr": 161,
    "fingerprints": {
      "2ce785529fa4473a3459e1dcfa616d6d4f19e1d1bc24f30aafa44a51f4f5d511": 1,
      "392db1a7cd0d5c8d94de35eaffc01144eec8cd83a417c1a87e59e40f8cb01e89": 1,
      "44d6e6b4b9642c1c768a3d787c6cedecbe956af4130ca202c4d680957ff1c0a2": 2,
      "4baebd08070bf13d1fbe484baa3d887eba1f5bbf84bb6947151a5a91f1b74c49": 2,
      "4e2cdfd11ca671e2a5d0ad128046d74ec77e3cd0cc5e892370564d1a96b459c9": 2,
      "6a7e47701805e568d5b2209e48d38b61d6ec8c623c14c4c576e148a3c8e53a4a": 2,
      "6e6a6efa1d220b6ed5cda4e0ee2d3cef7e09fe72ef3295bcbfc21d9726fac04f": 2,
      "9d35ffaadbf6b84fc4e81a947f15aba6fe139d26a07a42ed5c45b95b62bc8c0b": 1,
      "9f7197828c16f728cb53be26502bb8d50eedc340f5f288aec7986b4df6abe339": 1,
      "a4d04f61fa4e5d794f6c554425d094136c74c6c8a9df0a61531d1d76a28e8233": 2,
      "afdb9bdebecb8c9cd264329e917414f8dc0e47fd3719989d20b7266614fb15d7": 2,
      "b83e91f53914eb9296bdfde0dc97174aa7d03bbf33c898b1893bc832b8a0b703": 1,
      "badfc29e79cd03b93563145dddd0e03ef0ed93c3e338a841a85964940939c92a": 1,
      "cdf90966eb656a3a09dc0b2f334f26f544efed79a70de2230729c5898c914133": 3,
      "d1c30167e7c1bf285d92be84a0dd7b0a30a12b685d65bca8a4a477a71cf39dd3": 3,
      "e3a0ebdfafd7f113096d055c9b4a3da63368155518c04e0b47d08846b9c9d395": 1,
      "ea164c1450eebf55dac4a8f1207af76ffc05a171e2520eb0e02a20bfd630b940": 1,
      "fb1c63d182cc6e9c1eb525b0a22853c61e567134585b3ad2f0a43ede84e865eb": 1
    }
  }
]


def run(*args: str) -> None:
    subprocess.run(args, cwd=ROOT, check=True)


def update_manifest() -> None:
    document = json.loads(MANIFEST.read_text(encoding="utf-8"))
    paths = {group["path"] for group in document["registered_addition_groups"]}
    expected = {"core/sqlite_shadow_store.py", "tests/test_v4_m2_a_sqlite_shadow_store.py"}
    if paths & expected:
        raise RuntimeError("M2-A forward groups already exist")
    document["registered_addition_groups"].extend(GROUPS)
    MANIFEST.write_text(json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")


def replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise RuntimeError(f"expected one STATUS match, found {text.count(old)}: {old[:80]}")
    return text.replace(old, new, 1)


def update_status() -> None:
    text = STATUS.read_text(encoding="utf-8")
    text = replace_once(text, "M2-A status: **not started; eligible to be scoped next only as separate work after v4.2, with no implementation or authority granted by this amendment**", "M2-A status: **implemented as a bounded, explicit, disconnected SQLite shadow-persistence candidate in PR #161; not yet human-accepted or merged, with no runtime, recovery, cutover, or legacy-authority change**")
    text = replace_once(text, "Current next step: **M2-A append-only SQLite shadow-persistence work may be separately scoped; no M2 pre-design or implementation is included here**", "Current next step: **complete exact-head validation and human review of PR #161; M2-B remains blocked until M2-A is accepted with its schemas and restore evidence stable**")
    marker = "None of M1-A through M1-E is connected to `main.py`, `language/streaming.py`, live/autonomous loops, production composition, persistence adapters, or default startup paths. M1-D names legacy source modules as evidence only. M1-E imports or calls no legacy module and installs no observer or bridge. No SQLite database, file event store, durable snapshot, checkpoint artifact, sidecar, WAL, backup, migration, model/vector activation, scheduler, external effect, cutover, or production authority is introduced by v4.2.\n"
    text = replace_once(text, marker, marker + "\nPR #161 separately introduces `core/sqlite_shadow_store.py` as the M2-A candidate. Import and construction perform no I/O. A caller must explicitly initialize a concrete SQLite path and explicitly append immutable `shadow_only` envelopes or validated snapshots. The module is not imported by `main.py`, legacy persistence, live/autonomous loops, production composition, M1 observers, or lifecycle bridges. It grants no dual-read, authoritative recovery, migration cutover, scheduler, model/vector activation, or production persistence authority.\n")
    section = """## M2-A implementation candidate — PR #161

`core/sqlite_shadow_store.py` defines the bounded durable contracts:

```text
eve.sqlite-shadow-store.v1
eve.sqlite-shadow-migration.v1
eve.sqlite-shadow-snapshot.v1
eve.sqlite-shadow-append-receipt.v1
eve.sqlite-shadow-snapshot-receipt.v1
eve.sqlite-shadow-integrity-report.v1
eve.sqlite-shadow-restore-report.v1
eve.sqlite-shadow-backup-receipt.v1
```

The candidate provides explicit file initialization, a WAL request with visible fallback reporting, explicit SQLite transactions, immutable migration history, update/delete rejection triggers for durable tables, canonical envelope digests, a chained durable event digest, computed before/after count and chain evidence, readback verification before commit, bounded event/byte/snapshot/backup policy, periodic snapshot eligibility, snapshots bound to the current stream head, newest-valid-snapshot selection with corrupt-snapshot fallback, repeated deterministic restore verification, SQLite plus logical integrity reports, and verified bounded backups. Historical events are never pruned; storage-limit exhaustion rejects the new append instead of deleting prior history.

The candidate remains limited to the accepted M1 event-envelope contract and caller-supplied pure reducer/state codecs. It does not install the M1-B observer, connect an M1-D bridge, read legacy sidecars, compare dual reads, become the recovery authority, alter defaults, or perform cutover. Those boundaries remain assigned to later M2 milestones and separate human-reviewed decisions.

`tests/test_v4_m2_a_sqlite_shadow_store.py` provides focused evidence for explicit creation, WAL/schema/migration contracts, append ordering and hash-chain fidelity, atomic rollback, append-only enforcement, bounded storage, validated snapshots, corrupt-snapshot fallback, repeated replay, reopen after an uncommitted write, integrity-failure visibility, bounded backups, and absence of production integration.

"""
    text = replace_once(text, "## Merged source-of-truth evidence\n", section + "## Merged source-of-truth evidence\n")
    text = replace_once(text, "Reviewed additions are registered by introducing PR: #145 forward scanner; #146 M1-A; #147 M1-B; #148 M1-C; #149 M1-D; #150 M1-E; #151 evidence-gap documentation; #152 controlled evidence; #153 corrected expanded evidence; and #158 external human acceptance. Registration is review evidence, not automatic runtime authority.", "Reviewed additions are registered by introducing PR: #145 forward scanner; #146 M1-A; #147 M1-B; #148 M1-C; #149 M1-D; #150 M1-E; #151 evidence-gap documentation; #152 controlled evidence; #153 corrected expanded evidence; #158 external human acceptance; and candidate PR #161 M2-A SQLite shadow persistence. Registration is review evidence, not automatic runtime authority.")
    text = replace_once(text, "This does not start M2 or activate any runtime capability.", "PR #161 starts only the bounded M2-A implementation candidate. It does not activate a runtime bridge, dual-read path, recovery authority, cutover, or any production capability.")
    old = """M2-A is next and must be a separate, tightly scoped task. Until a later M2 change is independently reviewed and accepted:

1. `m2_started` remains false;
2. no bridge, persistence path, scheduler, recovery behavior, cutover, or production hook may be activated;
3. the pre-kernel legacy runtime remains authoritative;
4. the 527 unobserved historical sites remain tracked debt, not safe coverage;
5. A9-A12 bind all future M2 evidence and decision artifacts.
"""
    new = """M2-A is now implemented only as the separate candidate in PR #161. Until that exact head is independently validated, human-reviewed, and merged:

1. M2-A remains `shadow_only`, disconnected, and non-authoritative;
2. no observer, bridge, default persistence path, scheduler, recovery behavior, dual-read, cutover, or production hook may be activated;
3. the pre-kernel legacy runtime and legacy persistence remain authoritative;
4. M2-B and later M2 work remain blocked;
5. the 527 unobserved historical sites remain tracked debt, not safe coverage;
6. A9-A12 bind all M2 evidence, acceptance, supersession, and revocation artifacts.
"""
    text = replace_once(text, old, new)
    STATUS.write_text(text, encoding="utf-8")


def main() -> None:
    update_manifest()
    update_status()
    run("git", "checkout", BASE_SHA, "--", str(WORKFLOW.relative_to(ROOT)))
    SELF.unlink()
    run("git", "add", "-A")
    run("git", "config", "user.name", "github-actions[bot]")
    run("git", "config", "user.email", "41898282+github-actions[bot]@users.noreply.github.com")
    run("git", "commit", "-m", "Register and document M2-A SQLite shadow persistence")
    run("git", "push", "origin", f"HEAD:{BRANCH}")


if __name__ == "__main__":
    main()
