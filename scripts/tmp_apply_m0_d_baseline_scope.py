#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import textwrap

CONFIGS = (
    (
        "M0-A",
        Path("scripts/audit/m0_a_runtime_inventory.py"),
        Path("tests/audit/test_m0_a_runtime_inventory.py"),
        Path("docs/audit/M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md"),
        'SCHEMA_VERSION = "1.0.0-m0-a"',
        "78544d74af84afed450014d599b360c9b4af4f03",
    ),
    (
        "M0-B",
        Path("scripts/audit/m0_b_controlflow_concurrency_inventory.py"),
        Path("tests/audit/test_m0_b_controlflow_concurrency_inventory.py"),
        Path("docs/audit/M0_B_GATE_FAILURE_CLOCK_CONCURRENCY_MAP.md"),
        'SCHEMA_VERSION = "1.0.0-m0-b"',
        "eea70c286e947cbc180db9565bfa5ddc062d1ac3",
    ),
    (
        "M0-C",
        Path("scripts/audit/m0_c_persistence_state_inventory.py"),
        Path("tests/audit/test_m0_c_persistence_state_inventory.py"),
        Path("docs/audit/M0_C_PERSISTENCE_AND_STATE_MAP.md"),
        'SCHEMA_VERSION = "1.0.0-m0-c"',
        "fe10cd954bdf445400ea6aa9708dd214ed761114",
    ),
)

SNAPSHOT_BLOCK = textwrap.dedent(
    '''
    _SNAPSHOT_SOURCE_CACHE: dict[tuple[str, str], dict[str, str] | None] = {}


    def _git_snapshot_sources(root: Path) -> dict[str, str] | None:
        """Return Python source text from the completed audit snapshot."""
        key = (str(root.resolve()), AUDIT_SNAPSHOT_SHA)
        if key in _SNAPSHOT_SOURCE_CACHE:
            return _SNAPSHOT_SOURCE_CACHE[key]
        try:
            archive = subprocess.check_output(
                ["git", "-C", str(root), "archive", "--format=tar", AUDIT_SNAPSHOT_SHA],
                stderr=subprocess.DEVNULL,
            )
        except (OSError, subprocess.CalledProcessError):
            _SNAPSHOT_SOURCE_CACHE[key] = None
            return None
        sources: dict[str, str] = {}
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as handle:
            for member in handle.getmembers():
                if not member.isfile() or not member.name.endswith(".py"):
                    continue
                extracted = handle.extractfile(member)
                if extracted is not None:
                    sources[Path(member.name).as_posix()] = extracted.read().decode(
                        "utf-8", errors="replace"
                    )
        _SNAPSHOT_SOURCE_CACHE[key] = sources
        return sources


    def _git_tracked_python_files(root: Path) -> list[Path]:
        sources = _git_snapshot_sources(root)
        if sources is None:
            return []
        return [
            root / Path(value)
            for value in sorted(sources)
            if not any(part in EXCLUDED_PARTS for part in Path(value).parts)
        ]


    def _read_source(root: Path, path: Path) -> str:
        relative = path.relative_to(root).as_posix()
        sources = _git_snapshot_sources(root)
        if sources is not None and relative in sources:
            return sources[relative]
        try:
            return path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return path.read_text(encoding="utf-8", errors="replace")
    '''
).strip() + "\n\n"

TEST_BLOCK = textwrap.dedent(
    r'''


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
    '''
).rstrip() + "\n"

READ_BLOCK_A = (
    '        try:\n'
    '            source = path.read_text(encoding="utf-8")\n'
    '        except UnicodeDecodeError:\n'
    '            source = path.read_text(encoding="utf-8", errors="replace")\n'
)
READ_BLOCK_BC = (
    '    try:\n'
    '        source = path.read_text(encoding="utf-8")\n'
    '    except UnicodeDecodeError:\n'
    '        source = path.read_text(encoding="utf-8", errors="replace")\n'
)

for label, script_path, test_path, doc_path, schema_line, snapshot in CONFIGS:
    text = script_path.read_text(encoding="utf-8")
    assert "AUDIT_SNAPSHOT_SHA" not in text, script_path
    text = text.replace("import ast\n", "import ast\nimport io\n", 1)
    text = text.replace("import subprocess\n", "import subprocess\nimport tarfile\n", 1)
    text = text.replace(
        schema_line,
        schema_line + f'\nAUDIT_SNAPSHOT_SHA = "{snapshot}"',
        1,
    )
    start = text.index("def _git_tracked_python_files(root: Path) -> list[Path]:")
    end = text.index("def iter_python_files(root: Path) -> Iterator[Path]:", start)
    text = text[:start] + SNAPSHOT_BLOCK + text[end:]
    if label == "M0-A":
        assert READ_BLOCK_A in text, script_path
        text = text.replace(READ_BLOCK_A, "        source = _read_source(root, path)\n", 1)
    else:
        assert READ_BLOCK_BC in text, script_path
        text = text.replace(READ_BLOCK_BC, "    source = _read_source(root, path)\n", 1)
    script_path.write_text(text, encoding="utf-8")

    test = test_path.read_text(encoding="utf-8")
    assert "test_audit_snapshot_freezes_paths_and_source_content" not in test, test_path
    test_path.write_text(test.rstrip() + TEST_BLOCK, encoding="utf-8")

    doc = doc_path.read_text(encoding="utf-8")
    marker = "Generated JSON must not be committed."
    assert marker in doc, doc_path
    assert "source-content universe is frozen" not in doc, doc_path
    paragraph = (
        f"\n\nThe audit path and source-content universe is frozen to the merged {label} "
        f"snapshot `{snapshot}`. Later Python additions, deletions, or edits cannot "
        "retroactively change this completed audit's canonical output. Fixture "
        "repositories without that snapshot continue to use their local files for "
        "focused tests."
    )
    doc_path.write_text(doc.replace(marker, marker + paragraph, 1), encoding="utf-8")
