from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts/audit/m0_c_persistence_state_inventory.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "m0_c_persistence_state_inventory", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_fixture(root: Path) -> None:
    (root / "pkg").mkdir()
    (root / "tests").mkdir()
    (root / "pkg/__init__.py").write_text("", encoding="utf-8")
    (root / "pkg/state.py").write_text(
        """import json
import pickle
import sqlite3
from pathlib import Path

class StateStore:
    def __init__(self):
        self.episodic_memory = []
        self.semantic_memory = {}
        self.self_model = {"identity": "eve"}
        self.relationships = {}
        self.hormone_state = {"cortisol": 0.1}
        self.needs = {"social": 0.2}
        self.goals = []
        self.learned_parameters = {}
        self.vector_store = {}
        self.vocabulary = {}
        self.autosave_path = Path("state/eve.ckpt")
        self.checkpoint_path = Path("state/manual.checkpoint")
        self.debug_export_path = "state/debug.jsonl"

    def save(self):
        with open("state/eve.pkl", "wb") as handle:
            pickle.dump(self.episodic_memory, handle)
        Path("state/meta.json").write_text(json.dumps(self.semantic_memory))
        sqlite3.connect("state/eve.db")

    def load(self):
        with open("state/eve.pkl", "rb") as handle:
            self.episodic_memory = pickle.load(handle)

    def migrate_hormones_to_drives(self):
        return self.hormone_state, self.needs
""",
        encoding="utf-8",
    )
    (root / "tests/test_keep.py").write_text(
        """def test_keep():
    assert True
""",
        encoding="utf-8",
    )
    (root / "tests/test_legacy_pickle.py").write_text(
        """import pickle

def test_pickle_round_trip(tmp_path):
    target = tmp_path / "legacy.pkl"
    target.write_bytes(pickle.dumps({"ok": True}))
    assert pickle.loads(target.read_bytes()) == {"ok": True}
""",
        encoding="utf-8",
    )


def test_fixture_detects_persistence_formats_and_operations(tmp_path: Path):
    module = _load_module()
    _write_fixture(tmp_path)

    report = module.audit_repository(tmp_path)
    entries = report["entries"]
    formats = {
        entry.get("details", {}).get("format")
        for entry in entries
        if entry["category"] in {"persistence_io", "artifact_path"}
    }
    operations = {
        entry.get("details", {}).get("operation")
        for entry in entries
        if entry["category"] == "persistence_io"
    }

    assert {"pickle", "json", "sqlite", "checkpoint", "jsonl"} <= formats
    assert {"read", "write", "read_write"} <= operations
    assert any(
        entry["mechanical_evidence"] == "persistence_call=pickle.dump"
        for entry in entries
    )
    assert any(
        entry["mechanical_evidence"] == "path_method=Path.write_text"
        for entry in entries
    )


def test_fixture_detects_required_state_domains(tmp_path: Path):
    module = _load_module()
    _write_fixture(tmp_path)

    report = module.audit_repository(tmp_path)
    domains = {
        entry.get("details", {}).get("domain")
        for entry in report["entries"]
        if entry["category"] == "state_domain"
    }

    assert {
        "episodic_memory",
        "semantic_memory",
        "self_model",
        "relationships",
        "affect_hormones",
        "goals",
        "learned_parameters",
        "vectors",
        "vocabularies",
        "checkpoints",
        "autosave",
        "debug_exports",
    } <= domains


def test_fixture_detects_hormone_drive_migration_candidates(tmp_path: Path):
    module = _load_module()
    _write_fixture(tmp_path)

    report = module.audit_repository(tmp_path)
    entries = report["entries"]

    assert any(entry["category"] == "hormone_state" for entry in entries)
    assert any(entry["category"] == "drive_state" for entry in entries)
    assert any(
        entry["category"] == "hormone_drive_bridge"
        and entry["callable"] == "StateStore.migrate_hormones_to_drives"
        for entry in entries
    )


def test_test_policy_preserves_legacy_format_evidence(tmp_path: Path):
    module = _load_module()
    _write_fixture(tmp_path)

    report = module.audit_repository(tmp_path)
    classified = {
        entry["path"]: entry
        for entry in report["entries"]
        if entry["category"] == "test_classification"
    }

    assert classified["tests/test_keep.py"]["manual_classification"] == "KEEP"
    legacy = classified["tests/test_legacy_pickle.py"]
    assert legacy["manual_classification"] == "KEEP"
    assert legacy["confidence"] == "medium"
    assert legacy["unresolved"] is True
    assert legacy["line_start"] > 1
    assert "migration evidence" in legacy["manual_reason"]


def test_report_is_deterministic_complete_and_read_only(tmp_path: Path):
    module = _load_module()
    _write_fixture(tmp_path)

    first = module.audit_repository(tmp_path)
    second = module.audit_repository(tmp_path)

    assert first == second
    required = {
        "category",
        "path",
        "line_start",
        "line_end",
        "callable",
        "mechanical_evidence",
        "detector",
        "manual_classification",
        "confidence",
        "unresolved",
        "manual_only",
    }
    assert first["entries"]
    assert all(required <= set(entry) for entry in first["entries"])
    assert first["scope"] == {
        "tracked_python_only_when_git_available": True,
        "generated_json_committed": False,
        "runtime_activation_performed": False,
        "persistence_activation_performed": False,
        "source_mutation_performed": False,
        "test_behavior_changed": False,
        "hormone_drive_migration_performed": False,
    }


def test_cli_writes_only_to_explicit_output(tmp_path: Path):
    _write_fixture(tmp_path)

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--root",
            str(tmp_path),
            "--summary-only",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    summary = json.loads(completed.stdout)
    assert summary["python_files_scanned"] == 4
    assert not list(tmp_path.glob("*.json"))

    output = tmp_path / "artifacts/m0-c.json"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--root",
            str(tmp_path),
            "--output",
            str(output),
        ],
        check=True,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "1.0.0-m0-c"


def test_repository_smoke_covers_known_persistence_paths():
    module = _load_module()
    report = module.audit_repository(REPO_ROOT)
    entries = report["entries"]

    assert report["summary"]["python_files_scanned"] > 100
    assert report["summary"]["category_counts"]["persistence_io"] > 0
    assert report["summary"]["category_counts"]["state_domain"] > 0
    assert any(
        entry["path"] == "adapters/persistence_adapter.py"
        and entry["category"] == "persistence_io"
        for entry in entries
    )
    assert any(
        entry["path"] == "adapters/live_loop.py"
        and entry.get("details", {}).get("domain") == "autosave"
        for entry in entries
    )
    assert any(
        entry["path"] == "main.py"
        and entry["category"] == "artifact_path"
        and "eve.ckpt" in entry["mechanical_evidence"]
        for entry in entries
    )


def test_path_like_rejects_labels_commands_and_root():
    module = _load_module()
    assert module._path_like("state/eve.ckpt") is True
    assert module._path_like("state/debug.jsonl") is True
    assert module._path_like("checkpoint_created") is False
    assert module._path_like("python scripts/operator_report.py") is False
    assert module._path_like("/") is False


def test_path_like_requires_persistence_directory_markers():
    module = _load_module()
    assert module._path_like("_operator_artifacts/run") is True
    assert module._path_like("seeds/subsets") is True
    assert module._path_like("state/checkpoints/current") is True
    assert module._path_like("/save") is False
    assert module._path_like("language/streaming") is False
    assert module._path_like("tests/test_file.py::test_case") is False
    assert module._path_like(r"^foo\s+/bar$") is False
    assert module._path_like("/nonexistent.wav") is False
    assert module._path_like("adapters/self_embedding_adapter.py") is False
    assert module._path_like("docs/STATE_REPORT.md") is False


def test_repository_policy_does_not_flag_embedded_fixture_source():
    module = _load_module()
    report = module.audit_repository(REPO_ROOT)
    own = next(
        entry
        for entry in report["entries"]
        if entry["category"] == "test_classification"
        and entry["path"] == "tests/audit/test_m0_c_persistence_state_inventory.py"
    )
    assert own["manual_classification"] == "KEEP"
    assert own["confidence"] == "high"
    assert own["unresolved"] is False
