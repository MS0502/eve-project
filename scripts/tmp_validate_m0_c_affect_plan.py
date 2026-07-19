#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.request

ROOT = Path.cwd()
BASE_SHA = "28ec113a8ee371fdc6ac13341c0d70e00db26ce4"
EXPECTED_SCOPE = sorted([
    "docs/EVE_IMPLEMENTATION_STATUS_v4.md",
    "docs/audit/M0_C_AFFECT_MIGRATION_PLAN.md",
    "scripts/audit/m0_c_affect_migration_check.py",
    "tests/audit/test_m0_c_affect_migration_check.py",
])
EXPECTED_UNCOMMITTED = sorted([
    "docs/EVE_IMPLEMENTATION_STATUS_v4.md",
    "docs/audit/M0_C_AFFECT_MIGRATION_PLAN.md",
])


def run(command: list[str], *, output: Path | None = None) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if output is not None:
        output.write_text(result.stdout, encoding="utf-8")
    sys.stdout.write(result.stdout)
    if result.returncode:
        raise SystemExit(result.returncode)
    return result


url = "https://files.pythonhosted.org/packages/72/d2/ef65d0f3c150bfc99f5c4a516ae57e7c3acddfaacc1196dd296b1299ea7f/markdown_strings-3.4.0.tar.gz"
expected = "7574de0606160d7291ac2e1933a8ed47d31f0b49b674f128da1f548930c8578b"
target = Path("/tmp/markdown_strings-3.4.0.tar.gz")
urllib.request.urlretrieve(url, target)
assert hashlib.sha256(target.read_bytes()).hexdigest() == expected
run([sys.executable, "-m", "pip", "install", "--no-deps", str(target)])
run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
run([sys.executable, "-m", "pip", "check"])
run([
    sys.executable, "-m", "compileall", "-q",
    "scripts/audit/m0_c_affect_migration_check.py",
    "tests/audit/test_m0_c_affect_migration_check.py",
])

first = Path("/tmp/affect-plan-1.json")
second = Path("/tmp/affect-plan-2.json")
run([sys.executable, "scripts/audit/m0_c_affect_migration_check.py", "--output", str(first)])
run([sys.executable, "scripts/audit/m0_c_affect_migration_check.py", "--output", str(second)])
assert first.read_bytes() == second.read_bytes()
summary_result = run([
    sys.executable,
    "scripts/audit/m0_c_affect_migration_check.py",
    "--fail-on-unresolved",
    "--summary-only",
])
Path("/tmp/summary.json").write_text(summary_result.stdout, encoding="utf-8")
summary = json.loads(summary_result.stdout)
assert summary["authoritative_found_axes"] == 63
assert summary["mapped"] == 59
assert summary["proposed_drop"] == 4
assert summary["unresolved"] == 0
assert summary["validation_errors"] == 0

run([
    sys.executable, "-m", "pytest", "-q",
    "tests/audit/test_m0_c_affect_migration_check.py",
], output=Path("/tmp/focused.txt"))
run([sys.executable, "-m", "pytest", "--collect-only", "-q"], output=Path("/tmp/collect.txt"))
run([sys.executable, "-m", "pytest", "-q"], output=Path("/tmp/full.txt"))

scope = subprocess.check_output(
    ["git", "diff", "--name-only", f"{BASE_SHA}...HEAD"],
    cwd=ROOT,
    text=True,
).splitlines()
assert sorted(scope) == EXPECTED_SCOPE, (sorted(scope), EXPECTED_SCOPE)
uncommitted = subprocess.check_output(
    ["git", "diff", "--name-only"],
    cwd=ROOT,
    text=True,
).splitlines()
assert sorted(uncommitted) == EXPECTED_UNCOMMITTED, (sorted(uncommitted), EXPECTED_UNCOMMITTED)
run(["git", "diff", "--check"])

Path("/tmp/validation-result.json").write_text(
    json.dumps(
        {
            "baseline_sha": BASE_SHA,
            "deterministic_double_run": True,
            "scope": EXPECTED_SCOPE,
            "summary": summary,
            "runtime_executed": False,
            "migration_executed": False,
        },
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
    ) + "\n",
    encoding="utf-8",
)
