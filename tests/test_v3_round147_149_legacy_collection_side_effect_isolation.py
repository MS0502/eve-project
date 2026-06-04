"""Round147-149 legacy root collection side-effect isolation checks."""

from __future__ import annotations

import importlib
import inspect
import subprocess
import sys


LEGACY_ROOT_MODULES = ("test_eve_main_ab", "test_eve_main_abc")


def test_round148_legacy_root_validation_modules_import_without_running_scripts():
    """Importing legacy root validations must be collection-safe and quiet."""
    code = "import test_eve_main_ab, test_eve_main_abc"
    proc = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == ""
    assert proc.stderr == ""


def test_round148_legacy_validation_entrypoints_are_preserved_behind_main_guards():
    """The historical script validations remain available as explicit entrypoints."""
    for module_name in LEGACY_ROOT_MODULES:
        module = importlib.import_module(module_name)
        entrypoint = getattr(module, "run_legacy_validation", None)

        assert callable(entrypoint)
        source = inspect.getsource(module)
        assert 'if __name__ == "__main__":' in source
        assert "run_legacy_validation()" in source
