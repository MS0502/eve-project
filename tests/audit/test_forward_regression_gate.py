from __future__ import annotations

import copy
import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts/audit/forward_regression_gate.py"
BASE_ENGINE = (
    "class Engine:\n"
    "    def __init__(self):\n"
    "        self.baseline_state = {}\n"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("forward_regression_gate", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _manifest(module, baseline_scan, groups=None):
    return {
        "schema_version": module.MANIFEST_SCHEMA_VERSION,
        "baseline_sha": module.CONSTITUTION_BASELINE_SHA,
        "baseline": module.baseline_contract(baseline_scan),
        "registered_addition_groups": groups or [],
    }


def _approved(manifest, *, pr_number=145):
    for group in manifest["registered_addition_groups"]:
        group["rationale"] = "Reviewed bounded test addition."
        group["owner"] = "focused test owner"
        group["disposition"] = "TEST_EVIDENCE"
        group["introduced_by_pr"] = pr_number
    return manifest


def _baseline_and_current(module):
    baseline = module.scan_sources(REPO_ROOT, {"pkg/runtime.py": BASE_ENGINE})
    current = module.scan_sources(
        REPO_ROOT,
        {
            "pkg/runtime.py": (
                BASE_ENGINE
                + "    def set_value(self):\n"
                + "        self.state = {}\n"
            )
        },
    )
    return baseline, current


def test_unregistered_mutation_is_rejected():
    module = _load_module()
    baseline, current = _baseline_and_current(module)

    result = module.evaluate(baseline, current, _manifest(module, baseline))

    assert not result["pass"]
    assert result["summary"]["unregistered_addition_occurrences"] > 0
    assert any(
        item["category"] == "mutation" for item in result["unregistered_additions"]
    )


def test_same_pr_registration_allows_exact_delta():
    module = _load_module()
    baseline, current = _baseline_and_current(module)
    manifest = _approved(
        module.suggested_manifest(baseline, current, introduced_by_pr=145)
    )

    result = module.evaluate(
        baseline,
        current,
        manifest,
        base_manifest=_manifest(module, baseline),
        current_pr=145,
    )

    assert result["pass"]
    assert result["summary"]["registered_addition_occurrences"] > 0
    assert result["summary"]["unregistered_addition_occurrences"] == 0
    assert result["summary"]["same_pr_error_count"] == 0


def test_wrong_pr_registration_is_rejected():
    module = _load_module()
    baseline, current = _baseline_and_current(module)
    manifest = _approved(
        module.suggested_manifest(baseline, current, introduced_by_pr=144),
        pr_number=144,
    )

    result = module.evaluate(
        baseline,
        current,
        manifest,
        base_manifest=_manifest(module, baseline),
        current_pr=145,
    )

    assert not result["pass"]
    assert result["summary"]["same_pr_error_count"] > 0


def test_prior_pr_registration_remains_valid_when_unchanged():
    module = _load_module()
    baseline, current = _baseline_and_current(module)
    manifest = _approved(
        module.suggested_manifest(baseline, current, introduced_by_pr=144),
        pr_number=144,
    )
    base_manifest = copy.deepcopy(manifest)

    result = module.evaluate(
        baseline,
        current,
        manifest,
        base_manifest=base_manifest,
        current_pr=145,
    )

    assert result["pass"]
    assert result["summary"]["same_pr_error_count"] == 0


def test_stale_registration_is_rejected():
    module = _load_module()
    baseline, changed = _baseline_and_current(module)
    manifest = _approved(
        module.suggested_manifest(baseline, changed, introduced_by_pr=145)
    )

    result = module.evaluate(baseline, baseline, manifest)

    assert not result["pass"]
    assert result["stale_registrations"]


def test_line_shift_does_not_create_false_delta():
    module = _load_module()
    baseline = module.scan_sources(REPO_ROOT, {"pkg/runtime.py": BASE_ENGINE})
    current = module.scan_sources(
        REPO_ROOT,
        {"pkg/runtime.py": "\n\n" + BASE_ENGINE},
    )

    result = module.evaluate(baseline, current, _manifest(module, baseline))

    assert result["pass"]
    assert result["summary"]["unregistered_addition_occurrences"] == 0


def test_duplicate_occurrences_preserve_counts():
    module = _load_module()
    baseline = module.scan_sources(REPO_ROOT, {"pkg/runtime.py": BASE_ENGINE})
    current = module.scan_sources(
        REPO_ROOT,
        {
            "pkg/runtime.py": (
                BASE_ENGINE
                + "    def add(self, value):\n"
                + "        self.items.append(value)\n"
                + "        self.items.append(value)\n"
            )
        },
    )
    manifest = _approved(
        module.suggested_manifest(baseline, current, introduced_by_pr=145)
    )
    append_counts = []
    for group in manifest["registered_addition_groups"]:
        for fingerprint, count in group["fingerprints"].items():
            evidence = next(
                finding["evidence"]
                for finding in current["findings"]
                if finding["fingerprint"] == fingerprint
            )
            if "mutation_method=self.items.append" in evidence:
                append_counts.append(count)

    assert append_counts == [2]
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


def test_suggested_manifest_fails_until_reviewed():
    module = _load_module()
    baseline, current = _baseline_and_current(module)
    manifest = module.suggested_manifest(baseline, current, introduced_by_pr=145)

    result = module.evaluate(baseline, current, manifest)

    assert not result["pass"]
    assert any("still requires review" in error for error in result["errors"])


def test_manifest_requires_all_review_metadata():
    module = _load_module()
    baseline, current = _baseline_and_current(module)
    manifest = _approved(
        module.suggested_manifest(baseline, current, introduced_by_pr=145)
    )
    del manifest["registered_addition_groups"][0]["rationale"]

    result = module.evaluate(baseline, current, manifest)

    assert not result["pass"]
    assert any("missing fields" in error for error in result["errors"])


def test_baseline_digest_drift_is_rejected():
    module = _load_module()
    baseline = module.scan_sources(REPO_ROOT, {"pkg/runtime.py": BASE_ENGINE})
    manifest = _manifest(module, baseline)
    manifest["baseline"]["counter_sha256"] = "0" * 64

    result = module.evaluate(baseline, baseline, manifest)

    assert not result["pass"]
    assert result["baseline_drift"]


def test_new_parse_error_is_rejected_but_existing_one_is_tolerated():
    module = _load_module()
    baseline_sources = {
        "valid.py": BASE_ENGINE,
        "legacy.py": "BROKEN = [\n",
    }
    baseline = module.scan_sources(REPO_ROOT, baseline_sources)
    same = module.scan_sources(REPO_ROOT, baseline_sources)
    added = module.scan_sources(
        REPO_ROOT,
        {
            **baseline_sources,
            "new.py": "ALSO_BROKEN = {\n",
        },
    )
    manifest = _manifest(module, baseline)

    assert module.evaluate(baseline, same, manifest)["pass"]
    failed = module.evaluate(baseline, added, manifest)
    assert not failed["pass"]
    assert failed["summary"]["new_parse_errors"] == 1
