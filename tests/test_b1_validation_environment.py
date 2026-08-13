import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def requirement_names(path: Path) -> set[str]:
    result = set()
    for raw in path.read_text(encoding='utf-8').splitlines():
        line = raw.strip()
        if not line or line.startswith('#') or line.startswith('--'):
            continue
        name = line.split('=', 1)[0].split('<', 1)[0].split('>', 1)[0].strip().lower().replace('_', '-')
        if name:
            result.add(name)
    return result


def test_b1_dependency_surfaces_are_disjoint():
    contract = json.loads((ROOT / 'docs/audit/VALIDATION_CONTRACT.json').read_text(encoding='utf-8'))
    runtime = requirement_names(ROOT / contract['dependency_surfaces']['runtime'])
    development = requirement_names(ROOT / contract['dependency_surfaces']['development'])
    experimental = set(contract['experimental_legacy_distributions'])
    assert runtime == {'numpy'}
    assert development == {'pytest'}
    assert not runtime & experimental
    assert not development & experimental


def test_b1_validation_lock_is_hash_pinned_and_excludes_experimental():
    contract = json.loads((ROOT / 'docs/audit/VALIDATION_CONTRACT.json').read_text(encoding='utf-8'))
    lock = ROOT / contract['validation_lock_path']
    text = lock.read_text(encoding='utf-8')
    experimental = set(contract['experimental_legacy_distributions'])
    assert not requirement_names(lock) & experimental
    blocks = [block for block in text.split('\n') if block and not block.startswith('#')]
    assert '--hash=sha256:' in text
    assert 'numpy==' in text
    assert 'pytest==' in text
