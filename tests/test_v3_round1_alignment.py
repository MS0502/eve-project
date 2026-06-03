from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_v3_round1_governance_files_exist():
    assert (ROOT / "AGENTS.md").exists()
    assert (ROOT / "CURRENT_STATUS.md").exists()
    assert (ROOT / "ROUND_V3_R1_REPORT.md").exists()


def test_round73_semantic_guards_are_marked_frozen_for_v3():
    text = (ROOT / "adapters" / "orchestrator_adapter.py").read_text(encoding="utf-8")
    assert "EVE v3 ROUND1 FROZEN TEMPORARY GUARDS" in text
    assert "Do not extend them with new" in text
    assert "AppraisalClassifier" in text
