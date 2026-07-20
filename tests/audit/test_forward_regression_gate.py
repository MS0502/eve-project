from __future__ import annotations

import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts/audit/forward_regression_gate.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("forward_regression_gate", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _manifest(module, baseline_scan, registrations=None):
    return {
        "schema_version": module.MANIFEST_SCHEMA_VERSION,
        "baseline_sha": module.CONSTITUTION_BASELINE_SHA,
        "baseline_fingerprints": dict(module._counter(baseline_scan["findings"])),
        "registered_additions": registrations or [],
    }


def test_unregistered_mutation_is_rejected():
    module = _load_module()
    baseline = module.scan_sources(
        REPO_ROOT, {"pkg/runtime.py": "class Engine:\n    pass\n"}
    )
    current = module.scan_sources(
        REPO_ROOT,
        {
            "pkg/runtime.py": (
                "class Engine:\n"
                "    def set_value(self):\n"
                "        self.state = {}\n"
            )
        },
    )

    result = module.evaluate(baseline, current, _manifest(module, baseline))

    assert not result["pass"]
    assert result["summary"]["unregistered_addition_occurrences"] > 0
    assert any(
        item["category"] == "mutation" for item in result["unregistered_additions"]
    )


def test_same_pr_registration_allows_exact_delta():
    module = _load_module()
    baseline = module.scan_sources(
        REPO_ROOT, {"pkg/runtime.py": "class Engine:\n    pass\n"}
    )
    current = module.scan_sources(
        REPO_ROOT,
        {
            "pkg/runtime.py": (
                "class Engine:\n"
                "    def set_value(self):\n"
                "        self.state = {}\n"
            )
        },
    )
    manifest = module.suggested_manifest(baseline, current, introduced_by_pr=145)

    result = module.evaluate(baseline, current, manifest)

    assert result["pass"]
    assert result["summary"]["registered_addition_occurrences"] > 0
    assert result["summary"]["unregistered_addition_occurrences"] == 0


def test_stale_registration_is_rejected():
    module = _load_module()
    baseline = module.scan_sources(
        REPO_ROOT, {"pkg/runtime.py": "class Engine:\n    pass\n"}
    )
    changed = module.scan_sources(
        REPO_ROOT,
        {
            "pkg/runtime.py": (
                "class Engine:\n"
                "    def set_value(self):\n"
                "        self.state = {}\n"
            )
        },
    )
    manifest = module.suggested_manifest(baseline, changed, introduced_by_pr=145)

    result = module.evaluate(baseline, baseline, manifest)

    assert not result["pass"]
    assert result["stale_registrations"]


def test_line_shift_does_not_create_false_delta():
    module = _load_module()
    baseline = module.scan_sources(
        REPO_ROOT,
        {
            "pkg/runtime.py": (
                "class Engine:\n"
                "    def set_value(self):\n"
                "        self.state = {}\n"
            )
        },
    )
    current = module.scan_sources(
        REPO_ROOT,
        {
            "pkg/runtime.py": (
                "\n\nclass Engine:\n"
                "    def set_value(self):\n"
                "        self.state = {}\n"
            )
        },
    )

    result = module.evaluate(baseline, current, _manifest(module, baseline))

    assert result["pass"]
    assert result["summary"]["unregistered_addition_occurrences"] == 0


def test_duplicate_occurrences_preserve_counts():
    module = _load_module()
    baseline = module.scan_sources(
        REPO_ROOT, {"pkg/runtime.py": "class Engine:\n    pass\n"}
    )
    current = module.scan_sources(
        REPO_ROOT,
        {
            "pkg/runtime.py": (
                "class Engine:\n"
                "    def add(self, value):\n"
                "        self.items.append(value)\n"
                "        self.items.append(value)\n"
            )
        },
    )
    manifest = module.suggested_manifest(baseline, current, introduced_by_pr=145)
    append_registrations = []
    for item in manifest["registered_additions"]:
        evidence = next(
            finding["evidence"]
            for finding in current["findings"]
            if finding["fingerprint"] == item["fingerprint"]
        )
        if "mutation_method=self.items.append" in evidence:
            append_registrations.append(item)

    assert append_registrations
    assert append_registrations[0]["count"] == 2
    assert module.evaluate(baseline, current, manifest)["pass"]


def test_silent_broad_adaptive_and_raw_capability_detectors_run():
    module = _load_module()
    report = module.scan_sources(
        REPO_ROOT,
        {
            "pkg/runtime.py": (
                "class VectorStore:\n"
                "    def update_vector(self, vector):\n"
                "        self.vector = vector\n"
                "    def generate_response(self, raw_text):\n"
                "        try:\n"
                "            return raw_text\n"
                "        except Exception:\n"
                "            pass\n"
            )
        },
    )
    categories = {finding["category"] for finding in report["findings"]}

    assert "silent_broad" in categories
    assert "adaptive_numeric" in categories
    assert "raw_capability" in categories


def test_manifest_requires_review_metadata():
    module = _load_module()
    baseline = module.scan_sources(
        REPO_ROOT, {"pkg/runtime.py": "class Engine:\n    pass\n"}
    )
    current = module.scan_sources(
        REPO_ROOT,
        {
            "pkg/runtime.py": (
                "class Engine:\n"
                "    def set_value(self):\n"
                "        self.state = {}\n"
            )
        },
    )
    manifest = module.suggested_manifest(baseline, current, introduced_by_pr=145)
    del manifest["registered_additions"][0]["rationale"]

    result = module.evaluate(baseline, current, manifest)

    assert not result["pass"]
    assert any("missing fields" in error for error in result["errors"])
