#!/usr/bin/env python3
import hashlib
from pathlib import Path
import subprocess
import sys
import urllib.request

url = "https://files.pythonhosted.org/packages/72/d2/ef65d0f3c150bfc99f5c4a516ae57e7c3acddfaacc1196dd296b1299ea7f/markdown_strings-3.4.0.tar.gz"
expected = "7574de0606160d7291ac2e1933a8ed47d31f0b49b674f128da1f548930c8578b"
target = Path("/tmp/markdown_strings-3.4.0.tar.gz")
urllib.request.urlretrieve(url, target)
assert hashlib.sha256(target.read_bytes()).hexdigest() == expected
subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-deps", str(target)])
