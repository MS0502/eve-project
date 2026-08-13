import hashlib
import json
import platform
from pathlib import Path

root = Path(__file__).resolve().parents[2]
pin = (root / '.python-version').read_text().strip()
if platform.python_version() != pin:
    raise RuntimeError(f'expected Python {pin}, got {platform.python_version()}')
contract = root / 'docs' / 'audit' / 'VALIDATION_CONTRACT.json'
lock = root / 'requirements-lock.txt'
print(json.dumps({'python_pin': pin, 'requirements_lock_sha256': hashlib.sha256(lock.read_bytes()).hexdigest(), 'validation_contract_sha256': hashlib.sha256(contract.read_bytes()).hexdigest()}, sort_keys=True))
