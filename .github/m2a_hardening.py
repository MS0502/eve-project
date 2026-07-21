from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE_SHA = "dcd3ca93b559becb1831b7ab04c2c8164c1bd3f9"
BRANCH = "codex/m2-a-sqlite-shadow-persistence"
CODE = ROOT / "core/sqlite_shadow_store.py"
TEST = ROOT / "tests/test_v4_m2_a_sqlite_shadow_store.py"
WORKFLOW = ROOT / ".github/workflows/exact-head-validation.yml"
SELF = Path(__file__)


def replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise RuntimeError(f"expected exactly one match, got {text.count(old)}: {old[:100]!r}")
    return text.replace(old, new, 1)


def run(*args: str) -> None:
    subprocess.run(args, cwd=ROOT, check=True)


def harden_code() -> None:
    text = CODE.read_text(encoding="utf-8")
    text = replace_once(
        text,
        '''        connection.row_factory = sqlite3.Row\n        connection.execute("PRAGMA foreign_keys=ON")\n        connection.execute("PRAGMA busy_timeout=5000")\n''',
        '''        connection.row_factory = sqlite3.Row\n        connection.execute("PRAGMA foreign_keys=ON")\n        connection.execute("PRAGMA synchronous=FULL")\n        connection.execute("PRAGMA wal_autocheckpoint=1000")\n        connection.execute("PRAGMA busy_timeout=5000")\n''',
    )
    text = replace_once(
        text,
        '''        if isinstance(after_sequence, bool) or not isinstance(after_sequence, int) or after_sequence < 0:\n            raise ValueError("after_sequence must be non-negative")\n        with self._connect() as connection:\n''',
        '''        if isinstance(after_sequence, bool) or not isinstance(after_sequence, int) or after_sequence < 0:\n            raise ValueError("after_sequence must be non-negative")\n        if stream_id is None and after_sequence:\n            raise ValueError("after_sequence requires an explicit stream_id")\n        with self._connect() as connection:\n''',
    )
    text = replace_once(
        text,
        '''    def write_snapshot(self, *, snapshot_id: str, stream_id: str, through_sequence: int,\n                       state: Mapping[str, Any], state_schema_version: str) -> SnapshotReceipt:\n        state_json = _canon(state, "snapshot_state")\n''',
        '''    def write_snapshot(self, *, snapshot_id: str, stream_id: str, through_sequence: int,\n                       state: Mapping[str, Any], state_schema_version: str) -> SnapshotReceipt:\n        for field, value in (("snapshot_id", snapshot_id), ("stream_id", stream_id),\n                             ("state_schema_version", state_schema_version)):\n            if not isinstance(value, str) or not value.strip():\n                raise SnapshotCorruption(f"{field} must be a non-empty string")\n        if isinstance(through_sequence, bool) or not isinstance(through_sequence, int) or through_sequence < 0:\n            raise SnapshotCorruption("through_sequence must be a non-negative integer")\n        if not isinstance(state, Mapping):\n            raise SnapshotCorruption("state must be a mapping")\n        state_json = _canon(state, "snapshot_state")\n''',
    )
    text = replace_once(
        text,
        '''        selection = self.latest_valid_snapshot(stream_id)\n        if selection.selected is None:\n            start, after, snapshot_id = initial_state, 0, None\n        else:\n            start = state_from_mapping(selection.selected.state)\n            after = selection.selected.through_sequence\n            snapshot_id = selection.selected.snapshot_id\n        events = self.events(stream_id=stream_id, after_sequence=after)\n\n        def replay() -> tuple[StateT, str]:\n            state = start\n            for event in events:\n                state = reducer(state, event)\n                if state is None:\n                    raise RestoreVerificationError("reducer returned None")\n            return state, _sha(_canon(state_to_mapping(state), "restored_state"))\n''',
        '''        selection = self.latest_valid_snapshot(stream_id)\n        if selection.selected is None:\n            start_mapping = state_to_mapping(initial_state)\n            after, snapshot_id = 0, None\n        else:\n            start_mapping = selection.selected.state\n            after = selection.selected.through_sequence\n            snapshot_id = selection.selected.snapshot_id\n        start_json = _canon(start_mapping, "restore_start_state")\n        events = self.events(stream_id=stream_id, after_sequence=after)\n\n        def replay() -> tuple[StateT, str]:\n            decoded = json.loads(start_json)\n            if not isinstance(decoded, dict):\n                raise RestoreVerificationError("restore start state must be an object")\n            state = state_from_mapping(decoded)\n            for event in events:\n                state = reducer(state, event)\n                if state is None:\n                    raise RestoreVerificationError("reducer returned None")\n            return state, _sha(_canon(state_to_mapping(state), "restored_state"))\n''',
    )
    text = replace_once(
        text,
        '''    def create_backup(self, backup_directory: str | Path, *, backup_ordinal: int) -> BackupReceipt:\n        if isinstance(backup_ordinal, bool) or not isinstance(backup_ordinal, int) or not 1 <= backup_ordinal <= 99_999_999:\n            raise BackupPolicyError("backup_ordinal must be 1..99999999")\n        directory = Path(backup_directory)\n        directory.mkdir(parents=True, exist_ok=True)\n        target = directory / f"shadow-backup-{backup_ordinal:08d}.sqlite3"\n''',
        '''    def create_backup(self, backup_directory: str | Path, *, backup_ordinal: int) -> BackupReceipt:\n        if not self._initialized:\n            raise StoreNotInitialized("store requires explicit initialize()")\n        if isinstance(backup_ordinal, bool) or not isinstance(backup_ordinal, int) or not 1 <= backup_ordinal <= 99_999_999:\n            raise BackupPolicyError("backup_ordinal must be 1..99999999")\n        directory = Path(backup_directory)\n        directory.mkdir(parents=True, exist_ok=True)\n        existing = sorted(\n            (int(match.group(1)), path)\n            for path in directory.iterdir()\n            if path.is_file() and (match := _BACKUP.fullmatch(path.name)) is not None\n        )\n        if existing and backup_ordinal <= existing[-1][0]:\n            raise BackupPolicyError("backup_ordinal must increase monotonically")\n        target = directory / f"shadow-backup-{backup_ordinal:08d}.sqlite3"\n''',
    )
    CODE.write_text(text, encoding="utf-8")


def harden_tests() -> None:
    text = TEST.read_text(encoding="utf-8")
    text = replace_once(
        text,
        '''    with pytest.raises(StoreNotInitialized):\n        store.events()\n    report = store.initialize()\n''',
        '''    with pytest.raises(StoreNotInitialized):\n        store.events()\n    with pytest.raises(StoreNotInitialized):\n        store.create_backup(tmp_path / "backups", backup_ordinal=1)\n    assert not (tmp_path / "backups").exists()\n    report = store.initialize()\n''',
    )
    text = replace_once(
        text,
        '''    assert store.events() == (first, second)\n    assert store.integrity_check().valid is True\n''',
        '''    assert store.events() == (first, second)\n    with pytest.raises(ValueError):\n        store.events(after_sequence=1)\n    assert store.integrity_check().valid is True\n''',
    )
    text = replace_once(
        text,
        '''    assert selection.selected is not None\n    assert selection.selected.state == {"sum": 3}\n    with pytest.raises(SnapshotCorruption):\n        store.write_snapshot(\n            snapshot_id="snapshot:wrong",\n''',
        '''    assert selection.selected is not None\n    assert selection.selected.state == {"sum": 3}\n    with pytest.raises(SnapshotCorruption):\n        store.write_snapshot(\n            snapshot_id="",\n            stream_id="shadow:test",\n            through_sequence=2,\n            state={"sum": 3},\n            state_schema_version="test.state.v1",\n        )\n    with pytest.raises(SnapshotCorruption):\n        store.write_snapshot(\n            snapshot_id="snapshot:wrong",\n''',
    )
    marker = '''def test_reopen_after_uncommitted_external_transaction_preserves_committed_history(tmp_path: Path):\n'''
    inserted = '''def test_restore_uses_fresh_canonical_start_state_for_each_replay(tmp_path: Path):\n    store = SQLiteShadowStore(tmp_path / "shadow.sqlite3")\n    store.initialize()\n    store.append(event(1))\n    initial = {"total": 0}\n\n    def mutating_reducer(state: dict[str, int], envelope: EventEnvelope) -> dict[str, int]:\n        state["total"] += int(envelope.payload["delta"])\n        return state\n\n    result = store.restore_verified(\n        stream_id="shadow:test",\n        initial_state=initial,\n        reducer=mutating_reducer,\n        state_to_mapping=lambda state: state,\n        state_from_mapping=lambda value: {"total": int(value["total"])},\n    )\n    assert result.state == {"total": 1}\n    assert result.state_digest == result.repeated_state_digest\n    assert initial == {"total": 0}\n\n\n'''
    if text.count(marker) != 1 or "test_restore_uses_fresh_canonical_start_state_for_each_replay" in text:
        raise RuntimeError("restore hardening test marker mismatch")
    text = text.replace(marker, inserted + marker, 1)
    text = replace_once(
        text,
        '''    with pytest.raises(BackupPolicyError):\n        store.create_backup(backup_dir, backup_ordinal=3)\n\n\ndef test_module_has_no_default_activation_legacy_bridge_thread_clock_random_or_pickle_surface():\n''',
        '''    with pytest.raises(BackupPolicyError):\n        store.create_backup(backup_dir, backup_ordinal=3)\n    with pytest.raises(BackupPolicyError):\n        store.create_backup(backup_dir, backup_ordinal=1)\n    assert not (backup_dir / "shadow-backup-00000001.sqlite3").exists()\n\n\ndef test_module_has_no_default_activation_legacy_bridge_thread_clock_random_or_pickle_surface():\n''',
    )
    TEST.write_text(text, encoding="utf-8")


def main() -> None:
    harden_code()
    harden_tests()
    run("python", "-m", "compileall", "-q", str(CODE.relative_to(ROOT)), str(TEST.relative_to(ROOT)))
    run("python", "-m", "pytest", "-q", "--tb=short", str(TEST.relative_to(ROOT)))
    run("git", "checkout", BASE_SHA, "--", str(WORKFLOW.relative_to(ROOT)))
    SELF.unlink()
    run("git", "add", "-A")
    run("git", "config", "user.name", "github-actions[bot]")
    run("git", "config", "user.email", "41898282+github-actions[bot]@users.noreply.github.com")
    run("git", "commit", "-m", "Harden M2-A restore and backup contracts")
    run("git", "push", "origin", f"HEAD:{BRANCH}")


if __name__ == "__main__":
    main()
