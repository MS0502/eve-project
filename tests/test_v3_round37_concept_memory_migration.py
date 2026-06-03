"""EVE v3 round37 — concept_memory fastText parallel observation.

Concept memory is a core semantic module with write/query behavior. Round37
adds diagnostics only: existing PMI+SVD self_embedding remains the active path,
fastText must not auto-load, and concept writes/queries must be unchanged.
"""

from pathlib import Path

from main import build_full_engine

SELF_EMBEDDING = Path("adapters/self_embedding_adapter.py")
CONCEPT_MEMORY = Path("adapters/concept_memory_adapter.py")


def _learn_sample(engine):
    return engine.concept_memory.learn("친구", "함께 시간을 보내는 사람")


def _query_sample(engine):
    return engine.concept_memory.get("친구")


def test_concept_memory_default_uses_self_embedding_and_no_fasttext_trace():
    engine = build_full_engine()

    learned = _learn_sample(engine)
    found = _query_sample(engine)

    assert learned is not None
    assert found is not None
    assert found.definition == "함께 시간을 보내는 사람"
    assert engine.fasttext_embedding.is_loaded() is True
    assert len(engine.concept_memory.fasttext_trace) >= 1
    assert engine.concept_memory.stats()["in_use_by_generation"] == "wrapper"
    assert engine.concept_memory.stats()["fasttext_parallel_observation"] == "fasttext"


def test_concept_memory_does_not_auto_load_fasttext():
    engine = build_full_engine()

    for _ in range(3):
        _learn_sample(engine)
        _query_sample(engine)

    assert engine.fasttext_embedding.is_loaded() is True
    assert engine.fasttext_embedding.stats()["vocab_size"] == 30000
    assert len(engine.concept_memory.fasttext_trace) >= 1


def test_concept_memory_parallel_trace_when_fasttext_loaded_for_write_and_query():
    engine = build_full_engine()
    assert engine.fasttext_embedding.load() is True

    learned = _learn_sample(engine)
    found = _query_sample(engine)

    assert learned is not None and found is not None
    assert len(engine.concept_memory.fasttext_trace) == 2
    operations = [trace["operation"] for trace in engine.concept_memory.fasttext_trace]
    assert operations == ["write", "query"]
    for trace in engine.concept_memory.fasttext_trace:
        assert trace["meaning_layer"] == "concept_memory"
        assert trace["mode"] == "parallel_observation"
        assert trace["in_use_by_generation"] == "wrapper"
        assert "self_embedding_result" in trace
        assert "fasttext_result" in trace
        assert trace["fasttext_result"]["loaded"] is True
        assert trace["fasttext_result"]["subset_name"] == "cc.ko.300.subset.medium.30k"


def test_concept_memory_query_decision_unaffected_by_fasttext_observation():
    unloaded = build_full_engine()
    loaded = build_full_engine()
    loaded.fasttext_embedding.load()

    _learn_sample(unloaded)
    _learn_sample(loaded)
    unloaded_found = _query_sample(unloaded)
    loaded_found = _query_sample(loaded)

    assert loaded.concept_memory.fasttext_trace
    assert unloaded_found is not None and loaded_found is not None
    assert unloaded_found.key == loaded_found.key
    assert unloaded_found.definition == loaded_found.definition
    assert unloaded.concept_memory.use_count == loaded.concept_memory.use_count


def test_concept_memory_writes_unaffected_by_fasttext_observation():
    unloaded = build_full_engine()
    loaded = build_full_engine()
    loaded.fasttext_embedding.load()

    u = _learn_sample(unloaded)
    l = _learn_sample(loaded)

    assert l is not None and u is not None
    assert loaded.concept_memory.fasttext_trace[-1]["operation"] == "write"
    assert sorted(unloaded.concept_memory.all_keys()) == sorted(loaded.concept_memory.all_keys())
    assert unloaded.concept_memory.get_silent("친구").definition == loaded.concept_memory.get_silent("친구").definition
    assert unloaded.concept_memory.learn_count == loaded.concept_memory.learn_count


def test_concept_memory_quarantine_path_not_active_and_unaffected():
    engine = build_full_engine()
    engine.fasttext_embedding.load()
    _learn_sample(engine)
    _query_sample(engine)

    stats = engine.concept_memory.stats()
    assert stats["operations_observed"]["quarantine"] == 0
    assert all(trace["operation"] in {"write", "query"} for trace in engine.concept_memory.fasttext_trace)


def test_concept_memory_trace_cap_and_data_dict():
    engine = build_full_engine()
    engine.fasttext_embedding.load()
    engine.concept_memory.fasttext_trace_cap = 3

    for idx in range(5):
        engine.concept_memory.learn(f"개념{idx}", "친구 마음 시간")

    assert len(engine.concept_memory.fasttext_trace) == 3
    assert engine.concept_memory.fasttext_observation_count == 5
    assert all(isinstance(item, dict) for item in engine.concept_memory.fasttext_trace)


def test_state_debug_concept_memory_section_unloaded():
    engine = build_full_engine()

    section = engine.state_debug.snapshot_state()["concept_memory"]

    assert section["module"] == "concept_memory_adapter"
    assert section["in_use_by_generation"] == "wrapper"
    assert section["parallel_observation"] == "fasttext"
    assert section["fasttext_trace_size"] == 0
    assert section["operations_observed"]["query"] == 0
    assert section["operations_observed"]["write"] == 0
    assert section["migration_stage"] == "concept_memory_parallel_observation_only"


def test_state_debug_concept_memory_section_loaded_after_parallel_trace():
    engine = build_full_engine()
    engine.fasttext_embedding.load()
    _learn_sample(engine)
    _query_sample(engine)

    section = engine.state_debug.snapshot_state()["concept_memory"]

    assert section["parallel_observation"] == "fasttext"
    assert section["fasttext_trace_size"] == 2
    assert section["fasttext_observation_count"] == 2
    assert section["operations_observed"]["write"] == 1
    assert section["operations_observed"]["query"] == 1
    assert section["operations_observed"]["quarantine"] == 0
    assert section["in_use_by_generation"] == "wrapper"


def test_state_debug_is_read_only_for_concept_memory_migration():
    engine = build_full_engine()
    engine.fasttext_embedding.load()
    _learn_sample(engine)
    _query_sample(engine)
    before_trace = list(engine.concept_memory.fasttext_trace)
    before_counts = dict(engine.concept_memory.fasttext_operation_counts)
    before_observation_count = engine.concept_memory.fasttext_observation_count
    before_concepts = dict(engine.concept_memory.concepts)

    first = engine.state_debug.snapshot_state()["concept_memory"]
    second = engine.state_debug.snapshot_state()["concept_memory"]

    assert first == second
    assert engine.concept_memory.fasttext_trace == before_trace
    assert engine.concept_memory.fasttext_operation_counts == before_counts
    assert engine.concept_memory.fasttext_observation_count == before_observation_count
    assert engine.concept_memory.concepts == before_concepts


def test_generation_path_still_uses_pmi_svd_not_fasttext_for_concept_memory():
    engine = build_full_engine()
    engine.fasttext_embedding.load()

    assert engine.self_embedding.__class__.__name__ == "EmbeddingWrapper"
    assert getattr(engine.self_embedding, "dim") == 300
    assert engine.self_embedding is not engine.fasttext_embedding
    assert engine.concept_memory.stats()["in_use_by_generation"] == "wrapper"


def test_round37_does_not_modify_self_embedding_adapter():
    self_text = SELF_EMBEDDING.read_text(encoding="utf-8")

    assert "FasttextEmbeddingAdapter" not in self_text
    assert "fasttext_embedding" not in self_text
    assert "vectors.npy" not in self_text


def test_round37_concept_memory_documents_observation_only_contract():
    text = CONCEPT_MEMORY.read_text(encoding="utf-8")

    assert "fastText parallel observation only" in text
    assert "must never affect query/write decisions" in text
    assert "quarantine" in text
