"""EVE v3 round38 — situation_responder fastText parallel observation.

SituationResponder is a user-visible response path. Round38 adds diagnostics
only: existing PMI+SVD self_embedding remains the active path, fastText must not
auto-load, and response selection/text/timing shape must be unchanged.
"""

from pathlib import Path

from main import build_full_engine
from utils.types import Meaning

SELF_EMBEDDING = Path("adapters/self_embedding_adapter.py")
SITUATION_RESPONDER = Path("adapters/situation_responder.py")


def _meaning(text="안녕"):
    m = Meaning(intent="statement", entities=[])
    m.raw_text = text
    return m


def _ctx():
    return {"user_presence": {"user_name": "민석", "intimacy": 0.8}}


def _build_greeting(engine):
    return engine.situation_responder.build_plan("greeting", _ctx(), _meaning("안녕"))


def _chat(engine, text="내가 누구야?"):
    return list(engine.chat_stream(text))


def test_situation_responder_default_uses_self_embedding_and_no_fasttext_trace():
    engine = build_full_engine()

    plan = _build_greeting(engine)

    assert plan is not None
    assert plan.core_message
    assert engine.fasttext_embedding.is_loaded() is True
    assert len(engine.situation_responder.fasttext_trace) >= 1
    assert engine.situation_responder.stats()["in_use_by_generation"] == "wrapper"
    assert engine.situation_responder.stats()["fasttext_parallel_observation"] == "fasttext"


def test_situation_responder_does_not_auto_load_fasttext():
    engine = build_full_engine()

    for _ in range(3):
        _build_greeting(engine)

    assert engine.fasttext_embedding.is_loaded() is True
    assert engine.fasttext_embedding.stats()["vocab_size"] == 30000
    assert len(engine.situation_responder.fasttext_trace) >= 1


def test_situation_responder_parallel_trace_when_fasttext_loaded():
    engine = build_full_engine()
    assert engine.fasttext_embedding.load() is True

    plan = _build_greeting(engine)

    assert plan is not None
    assert len(engine.situation_responder.fasttext_trace) == 1
    trace = engine.situation_responder.fasttext_trace[-1]
    assert trace["meaning_layer"] == "situation_responder"
    assert trace["mode"] == "parallel_observation"
    assert trace["operation"] == "respond"
    assert trace["situation"] == "greeting"
    assert trace["in_use_by_generation"] == "wrapper"
    assert "self_embedding_result" in trace
    assert "fasttext_result" in trace
    assert trace["fasttext_result"]["loaded"] is True
    assert trace["fasttext_result"]["subset_name"] == "cc.ko.300.subset.medium.30k"


def test_situation_responder_response_selection_unaffected_by_fasttext_observation():
    unloaded = build_full_engine()
    loaded = build_full_engine()
    loaded.fasttext_embedding.load()

    u = _build_greeting(unloaded)
    l = _build_greeting(loaded)

    assert loaded.situation_responder.fasttext_trace
    assert u is not None and l is not None
    assert u.intent == l.intent
    assert u.emotional_tone == l.emotional_tone
    assert u.core_message == l.core_message
    assert u.sub_points == l.sub_points
    assert u.skip_finalize == l.skip_finalize


def test_situation_responder_user_visible_output_unaffected_by_fasttext_observation():
    unloaded = build_full_engine()
    loaded = build_full_engine()
    loaded.fasttext_embedding.load()

    unloaded_output = _chat(unloaded, "내가 누구야?")
    loaded_output = _chat(loaded, "내가 누구야?")

    assert loaded.situation_responder.fasttext_trace
    assert unloaded_output == loaded_output


def test_situation_responder_response_timing_shape_unaffected():
    unloaded = build_full_engine()
    loaded = build_full_engine()
    loaded.fasttext_embedding.load()

    unloaded_output = _chat(unloaded, "너는 누구야?")
    loaded_output = _chat(loaded, "너는 누구야?")

    assert unloaded_output == loaded_output
    assert len(unloaded_output) == len(loaded_output)


def test_situation_responder_trace_data_dict_and_operation_counts():
    engine = build_full_engine()
    engine.fasttext_embedding.load()

    _build_greeting(engine)

    assert all(isinstance(item, dict) for item in engine.situation_responder.fasttext_trace)
    assert engine.situation_responder.fasttext_observation_count == 1
    assert engine.situation_responder.fasttext_operation_counts["respond"] == 1
    assert engine.situation_responder.stats()["operations_observed"]["respond"] == 1


def test_situation_responder_fasttext_trace_cap():
    engine = build_full_engine()
    engine.fasttext_embedding.load()
    engine.situation_responder.fasttext_trace_cap = 3

    for _ in range(5):
        _build_greeting(engine)

    assert len(engine.situation_responder.fasttext_trace) == 3
    assert engine.situation_responder.fasttext_observation_count == 5
    assert all(isinstance(item, dict) for item in engine.situation_responder.fasttext_trace)


def test_state_debug_situation_responder_section_unloaded():
    engine = build_full_engine()

    section = engine.state_debug.snapshot_state()["situation_responder"]

    assert section["module"] == "situation_responder"
    assert section["in_use_by_generation"] == "wrapper"
    assert section["parallel_observation"] == "fasttext"
    assert section["fasttext_observation_count"] == 0
    assert section["operations_observed"]["respond"] == 0
    assert section["migration_stage"] == "situation_responder_parallel_observation_only"


def test_state_debug_situation_responder_section_loaded_after_parallel_trace():
    engine = build_full_engine()
    engine.fasttext_embedding.load()
    _build_greeting(engine)

    section = engine.state_debug.snapshot_state()["situation_responder"]

    assert section["parallel_observation"] == "fasttext"
    assert section["fasttext_trace_size"] == 1
    assert section["fasttext_observation_count"] == 1
    assert section["operations_observed"]["respond"] == 1
    assert section["in_use_by_generation"] == "wrapper"


def test_state_debug_is_read_only_for_situation_responder_migration():
    engine = build_full_engine()
    engine.fasttext_embedding.load()
    _build_greeting(engine)
    before_trace = list(engine.situation_responder.fasttext_trace)
    before_counts = dict(engine.situation_responder.fasttext_operation_counts)
    before_observation_count = engine.situation_responder.fasttext_observation_count

    first = engine.state_debug.snapshot_state()["situation_responder"]
    second = engine.state_debug.snapshot_state()["situation_responder"]

    assert first == second
    assert engine.situation_responder.fasttext_trace == before_trace
    assert engine.situation_responder.fasttext_operation_counts == before_counts
    assert engine.situation_responder.fasttext_observation_count == before_observation_count


def test_situation_generation_path_still_uses_pmi_svd_not_fasttext():
    engine = build_full_engine()
    engine.fasttext_embedding.load()

    assert engine.self_embedding.__class__.__name__ == "EmbeddingWrapper"
    assert getattr(engine.self_embedding, "dim") == 300
    assert engine.self_embedding is not engine.fasttext_embedding
    assert engine.situation_responder.stats()["in_use_by_generation"] == "wrapper"


def test_round38_does_not_modify_self_embedding_adapter_or_hardcode_subset_path():
    self_text = SELF_EMBEDDING.read_text(encoding="utf-8")
    situation_text = SITUATION_RESPONDER.read_text(encoding="utf-8")

    assert "FasttextEmbeddingAdapter" not in self_text
    assert "fasttext_embedding" not in self_text
    assert "vectors.npy" not in self_text
    assert "cc.ko.300.subset.medium.30k" not in situation_text
    assert "engine.self_embedding =" not in situation_text


def test_round38_situation_responder_documents_user_visible_observation_only_contract():
    text = SITUATION_RESPONDER.read_text(encoding="utf-8")

    assert "fastText migration is observation-only" in text
    assert "must never affect response selection" in text
    assert "user-visible" in text
