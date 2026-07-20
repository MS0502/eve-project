from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts/audit/m0_a_runtime_inventory.py"
LEGACY_MARKER = "docs/" + "EVE_DESIGN_" + "v3_1.md"


def _load_module():
    spec = importlib.util.spec_from_file_location("m0_a_runtime_inventory", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_fixture(root: Path) -> None:
    (root / "pkg").mkdir()
    (root / "tests").mkdir()
    (root / "pkg/__init__.py").write_text("", encoding="utf-8")
    (root / "pkg/runtime.py").write_text(
        """from pathlib import Path

class Engine:
    def __init__(self):
        self.state = {}

    def start(self):
        self.state[\"running\"] = True
        Path(\"state.txt\").write_text(\"ready\", encoding=\"utf-8\")

def main():
    engine = Engine()
    with open(\"audit.log\", \"a\", encoding=\"utf-8\") as handle:
        handle.write(\"started\\n\")
    return engine

if __name__ == \"__main__\":
    main()
""",
        encoding="utf-8",
    )
    (root / "tests/test_runtime.py").write_text(
        """def test_runtime_contract():
    assert True
""",
        encoding="utf-8",
    )
    (root / "tests/test_legacy_authority.py").write_text(
        f"""LEGACY = \"{LEGACY_MARKER}\"

def test_legacy_authority_literal():
    assert LEGACY.endswith(\".md\")
""",
        encoding="utf-8",
    )


def test_fixture_inventory_detects_required_categories(tmp_path: Path):
    module = _load_module()
    _write_fixture(tmp_path)

    report = module.audit_repository(tmp_path)
    entries = report["entries"]
    categories = {entry["category"] for entry in entries}

    assert {
        "entrypoint",
        "import",
        "dependency_construction",
        "mutation",
        "direct_write",
        "test_classification",
    } <= categories
    assert any(
        entry["mechanical_evidence"] == "module_main_guard"
        and entry["manual_classification"] == "ACTIVE_MODULE_ENTRYPOINT"
        for entry in entries
    )
    assert any(
        entry["category"] == "direct_write"
        and "open_write_mode=a" in entry["mechanical_evidence"]
        for entry in entries
    )
    assert any(
        entry["category"] == "mutation"
        and entry["callable"] == "Engine.start"
        for entry in entries
    )


def test_test_classification_is_conservative_and_evidenced(tmp_path: Path):
    module = _load_module()
    _write_fixture(tmp_path)

    report = module.audit_repository(tmp_path)
    classified = {
        entry["path"]: entry
        for entry in report["entries"]
        if entry["category"] == "test_classification"
    }

    assert classified["tests/test_runtime.py"]["manual_classification"] == "KEEP"
    assert classified["tests/test_runtime.py"]["line_start"] == 1
    legacy = classified["tests/test_legacy_authority.py"]
    assert legacy["manual_classification"] == "REWRITE"
    assert legacy["line_start"] == 1
    assert LEGACY_MARKER in legacy["mechanical_evidence"]
    assert legacy["manual_reason"]


def test_report_is_deterministic_and_has_complete_evidence_fields(tmp_path: Path):
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
        "source_mutation_performed": False,
    }


def test_cli_writes_only_when_output_is_explicit(tmp_path: Path):
    _write_fixture(tmp_path)
    completed = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--root", str(tmp_path), "--summary-only"],
        check=True,
        capture_output=True,
        text=True,
    )
    summary = json.loads(completed.stdout)
    assert summary["python_files_scanned"] == 4
    assert not list(tmp_path.glob("*.json"))

    output = tmp_path / "artifacts/m0-a.json"
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
    assert payload["schema_version"] == "1.0.0-m0-a"


def test_repository_smoke_inventory_covers_known_runtime_paths():
    module = _load_module()
    report = module.audit_repository(REPO_ROOT)
    entries = report["entries"]

    assert report["summary"]["python_files_scanned"] > 100
    assert report["summary"]["test_files_classified"] > 100
    assert any(
        entry["path"] == "main.py"
        and entry["mechanical_evidence"] == "module_main_guard"
        for entry in entries
    )
    assert any(
        entry["path"] == "main.py"
        and entry["callable"] == "build_full_engine"
        and entry["manual_classification"] == "ACTIVE_RUNTIME_COMPOSITION_ROOT"
        for entry in entries
    )
    assert any(
        entry["path"] == "adapters/live_loop.py"
        and entry["callable"] == "LiveLoop.start"
        and entry["category"] == "execution_boundary"
        for entry in entries
    )
    assert any(
        entry["path"] == "adapters/persistence_adapter.py"
        and entry["callable"] == "PersistenceAdapter.save"
        and entry["category"] == "direct_write"
        for entry in entries
    )
    assert any(
        entry["path"] == "core/autonomous.py"
        and entry["callable"] == "AutonomousLoop.step"
        and entry["category"] == "entrypoint"
        for entry in entries
    )
    classified = {
        entry["path"]: entry["manual_classification"]
        for entry in entries
        if entry["category"] == "test_classification"
    }
    assert classified["tests/audit/test_m0_a_runtime_inventory.py"] == "KEEP"


def test_audit_snapshot_freezes_paths_and_source_content(tmp_path):
    module = _load_module()
    subprocess.check_call(["git", "init", "-q"], cwd=tmp_path)
    subprocess.check_call(
        ["git", "config", "user.email", "audit@example.invalid"], cwd=tmp_path
    )
    subprocess.check_call(["git", "config", "user.name", "Audit Test"], cwd=tmp_path)
    (tmp_path / "baseline.py").write_text(
        "VALUE = 'baseline'\n", encoding="utf-8"
    )
    subprocess.check_call(["git", "add", "baseline.py"], cwd=tmp_path)
    subprocess.check_call(["git", "commit", "-q", "-m", "baseline"], cwd=tmp_path)
    snapshot = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=tmp_path, text=True
    ).strip()
    (tmp_path / "baseline.py").write_text(
        "VALUE = 'changed'\n", encoding="utf-8"
    )
    (tmp_path / "later.py").write_text("VALUE = 'later'\n", encoding="utf-8")
    subprocess.check_call(["git", "add", "baseline.py", "later.py"], cwd=tmp_path)
    subprocess.check_call(["git", "commit", "-q", "-m", "later"], cwd=tmp_path)

    original = module.AUDIT_SNAPSHOT_SHA
    module.AUDIT_SNAPSHOT_SHA = snapshot
    try:
        relative = [
            path.relative_to(tmp_path).as_posix()
            for path in module._git_tracked_python_files(tmp_path)
        ]
        source = module._read_source(tmp_path, tmp_path / "baseline.py")
    finally:
        module.AUDIT_SNAPSHOT_SHA = original

    assert relative == ["baseline.py"]
    assert "'baseline'" in source
    assert "'changed'" not in source
