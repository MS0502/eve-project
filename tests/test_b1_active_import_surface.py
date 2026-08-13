import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ACTIVE = (ROOT / 'main.py', ROOT / 'core', ROOT / 'adapters', ROOT / 'utils')
FORBIDDEN = ('brian2', 'brian2tools', 'scipy', 'h5py', 'faiss', 'matplotlib', 'seaborn', 'tqdm', 'yaml', 'jupyter')
PATTERN = re.compile(r'^\s*(?:from|import)\s+(' + '|'.join(FORBIDDEN) + r')(?:\.|\s|$)', re.MULTILINE)


def test_experimental_dependencies_are_absent_from_active_import_surface():
    violations = []
    for item in ACTIVE:
        files = [item] if item.is_file() else sorted(item.rglob('*.py')) if item.is_dir() else []
        for path in files:
            match = PATTERN.search(path.read_text(encoding='utf-8'))
            if match:
                violations.append(f'{path.relative_to(ROOT)}:{match.group(1)}')
    assert violations == []
