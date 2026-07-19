#!/usr/bin/env python3
from pathlib import Path

script_path = Path("scripts/audit/m0_d_component_inventory.py")
text = script_path.read_text(encoding="utf-8")
old = '''def _git_tracked_python_files(root: Path) -> list[Path]:
    try:
        raw = subprocess.check_output(["git", "-C", str(root), "ls-files", "-z", "--", "*.py"], stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        return []
    paths: list[Path] = []
    for value in raw.split(b"\\0"):
        if not value:
            continue
        relative = Path(os.fsdecode(value))
        if any(part in EXCLUDED_PARTS for part in relative.parts):
            continue
        paths.append(root / relative)
    return sorted(paths, key=lambda path: path.relative_to(root).as_posix())
'''
new = '''def _git_tracked_python_files(root: Path) -> list[Path]:
    """Return the Python path universe recorded at the M0-D baseline.

    M0-D is a merged historical audit. Later audit/support Python files must not
    retroactively alter its counts or module matrix. Current content is read only
    for paths that existed at BASELINE_SHA; missing baseline paths remain an
    explicit future compatibility concern rather than silently expanding scope.
    """
    try:
        raw = subprocess.check_output(
            [
                "git", "-C", str(root), "ls-tree", "-r", "-z", "--name-only",
                BASELINE_SHA, "--", "*.py",
            ],
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return []
    paths: list[Path] = []
    for value in raw.split(b"\\0"):
        if not value:
            continue
        relative = Path(os.fsdecode(value))
        if any(part in EXCLUDED_PARTS for part in relative.parts):
            continue
        path = root / relative
        if not path.is_file():
            continue
        paths.append(path)
    return sorted(paths, key=lambda path: path.relative_to(root).as_posix())
'''
assert old in text
script_path.write_text(text.replace(old, new, 1), encoding="utf-8")

test_path = Path("tests/audit/test_m0_d_component_inventory.py")
test = test_path.read_text(encoding="utf-8")
addition = '''


def test_python_path_universe_is_pinned_to_m0_d_baseline(tmp_path):
    module = _load_module()
    subprocess.check_call(["git", "init", "-q"], cwd=tmp_path)
    subprocess.check_call(["git", "config", "user.email", "audit@example.invalid"], cwd=tmp_path)
    subprocess.check_call(["git", "config", "user.name", "Audit Test"], cwd=tmp_path)
    (tmp_path / "baseline.py").write_text("VALUE = 1\\n", encoding="utf-8")
    subprocess.check_call(["git", "add", "baseline.py"], cwd=tmp_path)
    subprocess.check_call(["git", "commit", "-q", "-m", "baseline"], cwd=tmp_path)
    baseline = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=tmp_path, text=True).strip()
    (tmp_path / "post_baseline.py").write_text("VALUE = 2\\n", encoding="utf-8")
    subprocess.check_call(["git", "add", "post_baseline.py"], cwd=tmp_path)
    subprocess.check_call(["git", "commit", "-q", "-m", "post baseline"], cwd=tmp_path)

    original = module.BASELINE_SHA
    module.BASELINE_SHA = baseline
    try:
        relative = [path.relative_to(tmp_path).as_posix() for path in module._git_tracked_python_files(tmp_path)]
    finally:
        module.BASELINE_SHA = original

    assert relative == ["baseline.py"]
'''
assert "test_python_path_universe_is_pinned_to_m0_d_baseline" not in test
test_path.write_text(test + addition, encoding="utf-8")

doc_path = Path("docs/audit/M0_D_NEURAL_VECTOR_LIFELOOP_INVENTORY.md")
doc = doc_path.read_text(encoding="utf-8")
needle = "The scanner is stdlib-only, scans tracked Python source with `ast.parse`, cross-references the merged M0-A/B/C inventory scripts, and emits canonical JSON to stdout. Generated JSON must remain ephemeral. Two consecutive runs are byte-identical."
replacement = needle + "\n\nThe Python path universe is pinned to the M0-D baseline SHA through `git ls-tree`. Python audit or support files added after M0-D do not retroactively change its historical counts or module matrix."
assert needle in doc
assert "path universe is pinned" not in doc
doc_path.write_text(doc.replace(needle, replacement, 1), encoding="utf-8")
