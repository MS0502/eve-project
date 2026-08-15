"""Build and verify the hash-pinned environment used by B5 supervision.

The authoritative runtime path is deliberately separate from the broad
``requirements-runtime.txt`` compatibility declaration.  B5 accepts only an
environment installed from ``requirements-lock.txt`` with
``pip --require-hashes`` and records the exact interpreter and numpy version.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import venv
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
LOCK = ROOT / "requirements-lock.txt"
RUNTIME_RANGE = ROOT / "requirements-runtime.txt"
SCHEMA = "eve.b5-runtime-environment.v2"
INSTALL_COMMAND = [
    "-m",
    "pip",
    "install",
    "--require-hashes",
    "-r",
    "requirements-lock.txt",
]


class RuntimeEnvironmentError(RuntimeError):
    """The pinned runtime receipt or installed environment is unprovable."""


def _canonical(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_executable() -> Path:
    candidate = shutil.which("git")
    if candidate is None:
        raise RuntimeEnvironmentError("Git repository verifier is absent")
    executable = Path(candidate).resolve()
    if not executable.is_file():
        raise RuntimeEnvironmentError("Git repository verifier is not a file")
    return executable


def _git_argv(executable: Path, *args: str) -> list[str]:
    safe_directory = ROOT.resolve().as_posix()
    return [
        str(executable.resolve()),
        "-c",
        f"safe.directory={safe_directory}",
        *args,
    ]


def _git(*args: str, executable: Path | None = None) -> str:
    verifier = _git_executable() if executable is None else executable.resolve()
    return subprocess.check_output(
        _git_argv(verifier, *args), cwd=ROOT, text=True, encoding="utf-8"
    ).strip()


def _outside_repository(path: Path, field: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(ROOT.resolve())
    except ValueError:
        return resolved
    raise RuntimeEnvironmentError(f"{field} must remain outside the repository")


def _venv_python(environment: Path) -> Path:
    relative = Path("Scripts/python.exe") if os.name == "nt" else Path("bin/python")
    return (environment / relative).resolve()


def _installed_versions(python: Path) -> dict[str, str]:
    code = (
        "import importlib.metadata,json;"
        "names=('colorama','iniconfig','numpy','packaging','pluggy','pygments','pytest');"
        "print(json.dumps({n:importlib.metadata.version(n) for n in names},sort_keys=True))"
    )
    result = subprocess.run(
        [str(python), "-I", "-c", code],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if result.returncode != 0:
        raise RuntimeEnvironmentError(
            f"cannot inventory pinned environment: {result.stderr.strip()}"
        )
    data = json.loads(result.stdout)
    if not isinstance(data, dict) or data.get("numpy") != "2.5.2":
        raise RuntimeEnvironmentError("installed numpy does not match the lock")
    return {str(key): str(value) for key, value in data.items()}


def _python_version(python: Path) -> str:
    result = subprocess.run(
        [str(python), "-I", "-c", "import platform;print(platform.python_version())"],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if result.returncode != 0:
        raise RuntimeEnvironmentError("cannot read pinned Python version")
    return result.stdout.strip()


def _write_receipt(path: Path, packet: dict[str, Any]) -> None:
    if path.exists():
        raise RuntimeEnvironmentError("refusing to overwrite runtime receipt")
    payload = dict(packet)
    payload["receipt_sha256"] = hashlib.sha256(_canonical(packet)).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def install(environment: Path, output: Path) -> dict[str, Any]:
    environment = _outside_repository(environment, "runtime environment")
    output = _outside_repository(output, "runtime receipt")
    if environment.exists():
        raise RuntimeEnvironmentError("runtime environment must be a new path")
    git_executable = _git_executable()
    if _git("status", "--porcelain", executable=git_executable):
        raise RuntimeEnvironmentError("runtime installation requires a clean checkout")
    expected_python = (ROOT / ".python-version").read_text(encoding="utf-8").strip()
    if platform.python_version() != expected_python:
        raise RuntimeEnvironmentError(
            f"bootstrap Python must be {expected_python}, got {platform.python_version()}"
        )
    venv.EnvBuilder(with_pip=True, clear=False, symlinks=False).create(environment)
    python = _venv_python(environment)
    command = [str(python), *INSTALL_COMMAND]
    subprocess.run(command, cwd=ROOT, check=True)
    subprocess.run([str(python), "-m", "pip", "check"], cwd=ROOT, check=True)
    versions = _installed_versions(python)
    packet: dict[str, Any] = {
        "schema": SCHEMA,
        "authoritative_runtime": False,
        "t0_started": False,
        "repository": {
            "commit_sha": _git("rev-parse", "HEAD", executable=git_executable),
            "tree_sha": _git(
                "rev-parse", "HEAD^{tree}", executable=git_executable
            ),
            "clean_checkout": True,
        },
        "repository_verifier": {
            "kind": "git",
            "executable": str(git_executable),
            "sha256": _sha256(git_executable),
        },
        "python": {
            "bootstrap_version": platform.python_version(),
            "interpreter": str(python),
            "installed_version": _python_version(python),
        },
        "dependency_source": {
            "path": "requirements-lock.txt",
            "sha256": _sha256(LOCK),
            "require_hashes": True,
            "install_argv": INSTALL_COMMAND,
            "requirements_runtime_used": False,
            "requirements_runtime_sha256": _sha256(RUNTIME_RANGE),
        },
        "installed_distributions": versions,
        "numpy_version": versions["numpy"],
    }
    _write_receipt(output, packet)
    packet["receipt_sha256"] = hashlib.sha256(_canonical(packet)).hexdigest()
    return packet


def load_and_verify_receipt(
    path: Path,
    *,
    require_current_interpreter: bool = True,
    require_repository_identity: bool = True,
) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeEnvironmentError(f"cannot read runtime receipt: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeEnvironmentError("runtime receipt is not an object")
    receipt_hash = payload.pop("receipt_sha256", None)
    if receipt_hash != hashlib.sha256(_canonical(payload)).hexdigest():
        raise RuntimeEnvironmentError("runtime receipt digest differs")
    if payload.get("schema") != SCHEMA:
        raise RuntimeEnvironmentError("runtime receipt schema differs")
    verifier = payload.get("repository_verifier", {})
    git_executable = Path(str(verifier.get("executable", ""))).resolve()
    if (
        verifier.get("kind") != "git"
        or not git_executable.is_file()
        or verifier.get("sha256") != _sha256(git_executable)
    ):
        raise RuntimeEnvironmentError("runtime receipt repository verifier differs")
    source = payload.get("dependency_source", {})
    if (
        source.get("path") != "requirements-lock.txt"
        or source.get("sha256") != _sha256(LOCK)
        or source.get("require_hashes") is not True
        or source.get("install_argv") != INSTALL_COMMAND
        or source.get("requirements_runtime_used") is not False
    ):
        raise RuntimeEnvironmentError("runtime receipt is not bound to the hash-pinned lock")
    repository = payload.get("repository", {})
    if require_repository_identity and (
        repository.get("commit_sha")
        != _git("rev-parse", "HEAD", executable=git_executable)
        or repository.get("tree_sha")
        != _git("rev-parse", "HEAD^{tree}", executable=git_executable)
        or repository.get("clean_checkout") is not True
    ):
        raise RuntimeEnvironmentError("runtime receipt repository identity differs")
    interpreter = Path(str(payload.get("python", {}).get("interpreter", ""))).resolve()
    if not interpreter.is_file():
        raise RuntimeEnvironmentError("runtime receipt interpreter is absent")
    if require_current_interpreter and not Path(sys.executable).resolve().samefile(interpreter):
        raise RuntimeEnvironmentError("supervisor is not running in the pinned interpreter")
    if _python_version(interpreter) != payload.get("python", {}).get("installed_version"):
        raise RuntimeEnvironmentError("installed Python version differs from receipt")
    versions = _installed_versions(interpreter)
    if versions != payload.get("installed_distributions"):
        raise RuntimeEnvironmentError("installed distributions differ from receipt")
    if payload.get("numpy_version") != versions["numpy"]:
        raise RuntimeEnvironmentError("installed numpy version differs from receipt")
    payload["receipt_sha256"] = receipt_hash
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="action", required=True)
    install_parser = subparsers.add_parser("install")
    install_parser.add_argument("--environment", type=Path, required=True)
    install_parser.add_argument("--output", type=Path, required=True)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--receipt", type=Path, required=True)
    verify_parser.add_argument("--allow-different-interpreter", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.action == "install":
            packet = install(args.environment, args.output)
        else:
            packet = load_and_verify_receipt(
                args.receipt,
                require_current_interpreter=not args.allow_different_interpreter,
            )
        print(
            json.dumps(
                {
                    "valid": True,
                    "schema": packet["schema"],
                    "numpy_version": packet["numpy_version"],
                    "receipt_sha256": packet["receipt_sha256"],
                },
                sort_keys=True,
            )
        )
        return 0
    except (RuntimeEnvironmentError, OSError, subprocess.CalledProcessError) as exc:
        print(f"runtime environment unprovable: {exc}", file=sys.stderr)
        return 86


if __name__ == "__main__":
    raise SystemExit(main())
