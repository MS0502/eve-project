from copy import deepcopy

from main import build_full_engine
from adapters.agp_adapter import AGPAdapter, AGP_REASON_ANCHORED, AGP_REASON_UNKNOWN_CATEGORY
from adapters.runtime_smoke_runner import run_conversation_smoke_with_agp_trace
from tests.fixtures.korean_conversation_fixtures import fixture_texts


def _engine():
    return build_full_engine()


def test_round47_speech_hub_meaning_carries_explicit_categories():
    engine = _engine()
    hub = engine.speech_hub
    result = {"core": "안녕.", "sub": ["반갑네."]}
    active = hub._activated_categories("greeting_simple", "calm", result)
    meaning = hub._speech_hub_meaning("greeting_simple", {}, 0, "calm", result, active)
    assert meaning["meaning_layer"] == "speech_hub"
    assert meaning["meaning_categories"] == active
    assert meaning["meaning_bridge_source"] == "speech_hub_generation"
    assert meaning["meaning_categories_are_explicit"] is True


def test_round47_compositor_meaning_carries_explicit_categories():
    engine = _engine()
    comp = engine.compositor
    result = comp._compose_rule("힘들어", "공부")
    active = comp._activated_categories(result)
    meaning = comp._composition_meaning(result, active)
    assert meaning["meaning_layer"] == "composition"
    assert meaning["meaning_categories"] == active
    assert meaning["meaning_bridge_source"] == "compositor_generation"
    assert meaning["meaning_categories_are_explicit"] is True


def test_round47_agp_reads_meaning_categories_before_legacy_extraction():
    agp = AGPAdapter()
    meaning = {
        "meaning_layer": "unsupported_layer",
        "meaning_categories": ["speech_hub", "greeting_simple", "calm"],
    }
    result = agp.verify(
        "안녕.",
        meaning=meaning,
        activated_categories=["speech_hub", "greeting_simple", "calm"],
    )
    assert result.passed is True
    assert result.reason == AGP_REASON_ANCHORED
    assert result.candidate_categories == ["speech_hub", "greeting_simple", "calm"]


def test_round47_trace_marks_extraction_source_meaning_bridge():
    agp = AGPAdapter()
    traced = agp.verify_with_trace(
        "안녕.",
        meaning={"meaning_categories": ["speech_hub", "greeting_simple"]},
        activated_categories=["speech_hub", "greeting_simple"],
    )
    trace = traced["trace"]
    assert trace["extraction_source"] == "meaning_bridge"
    assert trace["extraction_method"] == "meaning_bridge:meaning_categories"
    assert trace["does_not_invent_categories"] is True
    assert traced["result"].passed is True


def test_round47_agp_falls_back_to_text_extraction_legacy_path():
    agp = AGPAdapter()
    traced = agp.verify_with_trace("안녕.", meaning={"meaning_layer": "speech_hub"}, activated_categories=[])
    assert traced["trace"]["extraction_source"] == "text_extraction"
    assert traced["trace"]["candidate_count"] == 0
    assert traced["result"].reason == AGP_REASON_UNKNOWN_CATEGORY


def test_round47_meaning_categories_are_sa_subset_at_capture():
    engine = _engine()
    hub = engine.speech_hub
    result = {"core": "안녕.", "sub": ["반갑네."]}
    active = hub._activated_categories("greeting_simple", "calm", result)
    meaning = hub._speech_hub_meaning("greeting_simple", {}, 0, "calm", result, active)
    assert set(meaning["meaning_categories"]).issubset(set(active))


def test_round47_agp_does_not_invent_categories_when_bridge_absent():
    agp = AGPAdapter()
    traced = agp.verify_with_trace("안녕 반갑네", meaning=None, activated_categories=["speech_hub"])
    assert traced["trace"]["candidate_categories_extracted"] == []
    assert traced["trace"]["does_not_invent_categories"] is True
    assert traced["result"].passed is False


def test_round47_bridge_data_is_explicit_not_inferred():
    agp = AGPAdapter()
    meaning = {"meaning_categories": ["speech_hub", "speech_hub", "", None, "has_sub"]}
    traced = agp.verify_with_trace("raw text ignored", meaning=meaning, activated_categories=["speech_hub", "has_sub"])
    assert traced["trace"]["candidate_categories_extracted"] == ["speech_hub", "has_sub"]
    assert "raw" not in traced["trace"]["candidate_categories_extracted"]
    assert traced["result"].passed is True


def test_round47_no_threshold_changes():
    engine = _engine()
    agp = engine.agp_adapter
    before = (agp.category_threshold, agp.hormone_threshold)
    run_conversation_smoke_with_agp_trace(engine, fixture_texts()[:5])
    after = (agp.category_threshold, agp.hormone_threshold)
    assert before == after


def test_round47_no_sa_mechanism_changes_from_bridge():
    engine = _engine()
    before = deepcopy(getattr(engine.activation_adapter, "__dict__", {}))
    run_conversation_smoke_with_agp_trace(engine, fixture_texts()[:4])
    after = getattr(engine.activation_adapter, "__dict__", {})
    assert before.keys() == after.keys()


def test_round47_smoke_fixtures_still_run():
    engine = _engine()
    result = run_conversation_smoke_with_agp_trace(engine, fixture_texts()[:6])
    assert result["fixtures_count"] == 6
    assert len(result["responses"]) == 6
    assert result["traces"]


def test_round47_smoke_with_bridge_populates_candidates():
    engine = _engine()
    result = run_conversation_smoke_with_agp_trace(engine, fixture_texts())
    summary = result["summary"]
    assert summary["trace_count"] > 0
    assert summary["zero_candidate_count"] == 0
    assert min(summary["candidates_per_trace"]) > 0


def test_round47_smoke_with_bridge_changes_agp_pass_rate_above_zero():
    engine = _engine()
    result = run_conversation_smoke_with_agp_trace(engine, fixture_texts())
    total = len(result["traces"])
    passed = sum(1 for row in result["traces"] if getattr(row["agp_trace"]["result"], "passed", False))
    assert total > 0
    assert passed > 0
    assert passed == total


def test_round47_runtime_decision_output_unchanged_shape():
    fixtures = fixture_texts()[:5]
    baseline_engine = _engine()
    baseline = [[str(c) for c in baseline_engine.chat_stream(text)] for text in fixtures]
    trace_engine = _engine()
    smoke = run_conversation_smoke_with_agp_trace(trace_engine, fixtures)
    traced = [row["chunks"] for row in smoke["responses"]]
    assert traced == baseline
