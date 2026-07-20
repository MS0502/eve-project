from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts/audit/m0_b_controlflow_concurrency_inventory.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "m0_b_controlflow_concurrency_inventory", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_fixture(root: Path) -> None:
    (root / "pkg").mkdir()
    (root / "tests").mkdir()
    (root / "pkg/__init__.py").write_text("", encoding="utf-8")
    (root / "pkg/runtime.py").write_text(
        """import queue
import random
import threading
import time

class Runtime:
    def __init__(self):
        self.enabled = False
        self.queue = queue.Queue()
        self.thread = None

    def start(self, force=False):
        if self.enabled or force:
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()

    def _run(self):
        try:
            time.sleep(0.01)
            self.queue.put(random.random())
            print(\"ready\")
        except Exception:
            pass
""",
        encoding="utf-8",
    )
    (root / "tests/test_keep.py").write_text(
        """def test_keep():
    assert True
""",
        encoding="utf-8",
    )
    (root / "tests/test_clock.py").write_text(
        """import time

def test_clock():
    time.sleep(0.001)
    assert True
""",
        encoding="utf-8",
    )


def test_fixture_detects_all_m0_b_categories(tmp_path: Path):
    module = _load_module()
    _write_fixture(tmp_path)

    report = module.audit_repository(tmp_path)
    categories = {entry["category"] for entry in report["entries"]}

    assert {
        "gate",
        "bypass",
        "output",
        "exception",
        "clock",
        "queue",
        "concurrency",
        "nondeterminism",
        "test_classification",
    } <= categories
    assert any(
        entry["manual_classification"] == "SILENT_BROAD_EXCEPTION_PATH"
        for entry in report["entries"]
    )


def test_test_policy_is_conservative_and_evidenced(tmp_path: Path):
    module = _load_module()
    _write_fixture(tmp_path)

    report = module.audit_repository(tmp_path)
    classified = {
        entry["path"]: entry
        for entry in report["entries"]
        if entry["category"] == "test_classification"
    }

    assert classified["tests/test_keep.py"]["manual_classification"] == "KEEP"
    assert classified["tests/test_clock.py"]["manual_classification"] == "REWRITE"
    assert classified["tests/test_clock.py"]["line_start"] == 4
    assert classified["tests/test_clock.py"]["manual_reason"]


def test_report_is_deterministic_and_complete(tmp_path: Path):
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
        "test_behavior_changed": False,
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

    output = tmp_path / "artifacts/m0-b.json"
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
    assert payload["schema_version"] == "1.0.0-m0-b"


def test_repository_smoke_covers_known_runtime_paths():
    module = _load_module()
    report = module.audit_repository(REPO_ROOT)
    entries = report["entries"]

    assert report["summary"]["python_files_scanned"] > 100
    assert report["summary"]["category_counts"]["exception"] > 0
    assert report["summary"]["category_counts"]["clock"] > 0
    assert any(
        entry["path"] == "adapters/live_loop.py"
        and entry["callable"] == "LiveLoop.start"
        and entry["category"] == "concurrency"
        for entry in entries
    )
    assert any(
        entry["path"] == "adapters/live_loop.py"
        and entry["callable"] == "LiveLoop._run"
        and entry["category"] == "clock"
        for entry in entries
    )
    assert any(
        entry["path"] == "main.py"
        and entry["callable"] == "repl"
        and entry["category"] == "output"
        for entry in entries
    )


def test_bypass_detection_uses_symbol_tokens(tmp_path: Path):
    module = _load_module()
    (tmp_path / "runtime.py").write_text(
        "def run(enforcement_rows, teacher, safety):\n"
        "    enforcement_rows.append(1)\n"
        "    teacher.reinforce_if_no_correction()\n"
        "    safety.force_alternative()\n",
        encoding="utf-8",
    )
    report = module.audit_repository(tmp_path)
    evidence = {
        entry["mechanical_evidence"]
        for entry in report["entries"]
        if entry["category"] == "bypass"
    }
    assert "bypass_call=safety.force_alternative" in evidence
    assert "bypass_call=enforcement_rows.append" not in evidence
    assert "bypass_call=teacher.reinforce_if_no_correction" not in evidence


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
