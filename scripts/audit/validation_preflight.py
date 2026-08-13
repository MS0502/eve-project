import hashlib
import json
import os
import platform
import re
from pathlib import Path

root = Path(__file__).resolve().parents[2]
contract_path = root / 'docs' / 'audit' / 'VALIDATION_CONTRACT.json'
contract = json.loads(contract_path.read_text(encoding='utf-8'))
pin = (root / contract['python_pin_path']).read_text(encoding='utf-8').strip()
if platform.python_version() != pin:
    raise RuntimeError(f'expected Python {pin}, got {platform.python_version()}')

runtime = root / contract['dependency_surfaces']['runtime']
dev = root / contract['dependency_surfaces']['development']
validation_input = root / contract['validation_input_path']
lock = root / contract['validation_lock_path']
experimental = {name.lower().replace('_', '-') for name in contract['experimental_legacy_distributions']}

def names(path):
    result = set()
    for raw in path.read_text(encoding='utf-8').splitlines():
        line = raw.strip()
        if not line or line.startswith('#') or line.startswith('--'):
            continue
        match = re.match(r'([A-Za-z0-9_.-]+)', line)
        if match:
            result.add(match.group(1).lower().replace('_', '-'))
    return result

if names(runtime) != set(contract['required_runtime_distributions']):
    raise RuntimeError('runtime dependency surface drift')
if names(dev) != set(contract['required_development_distributions']):
    raise RuntimeError('development dependency surface drift')
for path in (runtime, dev, validation_input, lock):
    leaked = names(path) & experimental
    if leaked:
        raise RuntimeError(f'experimental dependency leaked into {path.name}: {sorted(leaked)}')

lock_text = lock.read_text(encoding='utf-8')
for block in re.split(r'\n(?=[A-Za-z0-9_.-]+==)', lock_text):
    if '==' in block and '--hash=sha256:' not in block:
        raise RuntimeError('unhashed requirement in validation lock')

commit_sha = os.environ.get('EVE_VALIDATION_COMMIT', '').strip()
tree_sha = os.environ.get('EVE_VALIDATION_TREE', '').strip()
if len(commit_sha) != 40 or len(tree_sha) != 40:
    raise RuntimeError('commit/tree identity was not supplied')
identity = {
    'commit_sha': commit_sha,
    'tree_sha': tree_sha,
    'python_pin': pin,
    'requirements_lock_sha256': hashlib.sha256(lock.read_bytes()).hexdigest(),
    'validation_contract_sha256': hashlib.sha256(contract_path.read_bytes()).hexdigest(),
}
canonical = json.dumps(identity, sort_keys=True, separators=(',', ':')).encode()
packet = {'schema': 'eve.validation.identity.v1', **identity, 'identity_sha256': hashlib.sha256(canonical).hexdigest()}
output = os.environ.get('EVE_VALIDATION_IDENTITY_OUTPUT', '').strip()
if output:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(packet, indent=2, sort_keys=True) + '\n', encoding='utf-8')
print(json.dumps(packet, sort_keys=True))
