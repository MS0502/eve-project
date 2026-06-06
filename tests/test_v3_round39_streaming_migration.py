"""EVE v3 round39 — language/streaming fastText parallel observation.

Streaming is chunk/order/timing-shape sensitive. Round39 adds diagnostics only:
existing PMI+SVD self_embedding remains the active generation path, fastText
must not auto-load, and yielded chunks must remain unchanged.
"""

from pathlib import Path

from main import build_full_engine
from tests.fasttext_observation_stub import install_explicit_loaded_fasttext_stub

SELF_EMBEDDING = Path("adapters/self_embedding_adapter.py")
STREAMING = Path("language/streaming.py")


def _chat(engine, text="민석아 힘들다"):
    return list(engine.chat_stream(text))


def test_streaming_default_uses_self_embedding_and_no_fasttext_trace():
    engine = build_full_engine()

    chunks = _chat(engine)

    assert chunks
    assert engine.fasttext_embedding.is_loaded() is False
    assert len(engine.fasttext_trace) == 0
    assert engine.stats()["in_use_by_generation"] == "wrapper"
    assert engine.stats()["fasttext_parallel_observation"] is None


def test_streaming_does_not_auto_load_fasttext():
    engine = build_full_engine()

    for text in ["민석아 힘들다", "오늘 날씨 좋다", "너는 누구야?"]:
        _chat(engine, text)

    assert engine.fasttext_embedding.is_loaded() is False
    assert engine.fasttext_embedding.stats()["vocab_size"] == 0
    assert len(engine.fasttext_trace) == 0


def test_streaming_parallel_trace_when_fasttext_loaded():
    engine = build_full_engine()
    install_explicit_loaded_fasttext_stub(engine)
    assert engine.fasttext_embedding.is_loaded() is True

    chunks = _chat(engine, "민석아 힘들다")

    assert chunks
    assert len(engine.fasttext_trace) == len(chunks)
    trace = engine.fasttext_trace[-1]
    assert trace["meaning_layer"] == "streaming"
    assert trace["mode"] == "parallel_observation"
    assert trace["operation"] == "stream_chunk"
    assert trace["in_use_by_generation"] == "wrapper"
    assert "stage" in trace
    assert "chunk_index" in trace
    assert "self_embedding_result" in trace
    assert "fasttext_result" in trace
    assert trace["fasttext_result"]["loaded"] is True
    assert trace["fasttext_result"]["subset_name"] == "artifact_free_explicit_loaded_stub"


def test_streaming_output_chunks_unaffected_by_fasttext_observation():
    unloaded = build_full_engine()
    loaded = build_full_engine()
    install_explicit_loaded_fasttext_stub(loaded)

    unloaded_chunks = _chat(unloaded, "민석아 힘들다")
    loaded_chunks = _chat(loaded, "민석아 힘들다")

    assert loaded.fasttext_trace
    assert unloaded_chunks == loaded_chunks


def test_streaming_chunk_order_unaffected_by_fasttext_observation():
    unloaded = build_full_engine()
    loaded = build_full_engine()
    install_explicit_loaded_fasttext_stub(loaded)

    unloaded_chunks = _chat(unloaded, "너는 누구야?")
    loaded_chunks = _chat(loaded, "너는 누구야?")

    assert unloaded_chunks == loaded_chunks
    assert list(enumerate(unloaded_chunks)) == list(enumerate(loaded_chunks))


def test_streaming_chunk_timing_shape_unaffected():
    unloaded = build_full_engine()
    loaded = build_full_engine()
    install_explicit_loaded_fasttext_stub(loaded)

    unloaded_chunks = _chat(unloaded, "오늘 날씨 좋다")
    loaded_chunks = _chat(loaded, "오늘 날씨 좋다")

    assert unloaded_chunks == loaded_chunks
    assert len(unloaded_chunks) == len(loaded_chunks)


def test_streaming_trace_data_dict_and_operation_counts():
    engine = build_full_engine()
    install_explicit_loaded_fasttext_stub(engine)

    chunks = _chat(engine, "민석아 힘들다")

    assert all(isinstance(item, dict) for item in engine.fasttext_trace)
    assert engine.fasttext_observation_count == len(chunks)
    assert engine.fasttext_operation_counts["stream_chunk"] == len(chunks)
    assert engine.stats()["operations_observed"]["stream_chunk"] == len(chunks)


def test_streaming_fasttext_trace_cap():
    engine = build_full_engine()
    install_explicit_loaded_fasttext_stub(engine)
    engine.fasttext_trace_cap = 3

    for text in ["민석아 힘들다", "오늘 날씨 좋다", "너는 누구야?", "내가 누구야?"]:
        _chat(engine, text)

    assert len(engine.fasttext_trace) == 3
    assert engine.fasttext_observation_count >= 4
    assert all(isinstance(item, dict) for item in engine.fasttext_trace)


def test_state_debug_streaming_section_unloaded():
    engine = build_full_engine()

    section = engine.state_debug.snapshot_state()["streaming"]

    assert section["module"] == "language/streaming"
    assert section["in_use_by_generation"] == "wrapper"
    assert section["parallel_observation"] is None
    assert section["fasttext_observation_count"] == 0
    assert section["operations_observed"]["stream_chunk"] == 0
    assert section["migration_stage"] == "streaming_parallel_observation_only"


def test_state_debug_streaming_section_loaded_after_parallel_trace():
    engine = build_full_engine()
    install_explicit_loaded_fasttext_stub(engine)
    chunks = _chat(engine, "민석아 힘들다")

    section = engine.state_debug.snapshot_state()["streaming"]

    assert section["parallel_observation"] == "fasttext"
    assert section["fasttext_trace_size"] == len(chunks)
    assert section["fasttext_observation_count"] == len(chunks)
    assert section["operations_observed"]["stream_chunk"] == len(chunks)
    assert section["in_use_by_generation"] == "wrapper"


def test_state_debug_is_read_only_for_streaming_migration():
    engine = build_full_engine()
    install_explicit_loaded_fasttext_stub(engine)
    _chat(engine, "민석아 힘들다")
    before_trace = list(engine.fasttext_trace)
    before_counts = dict(engine.fasttext_operation_counts)
    before_observation_count = engine.fasttext_observation_count

    first = engine.state_debug.snapshot_state()["streaming"]
    second = engine.state_debug.snapshot_state()["streaming"]

    assert first == second
    assert engine.fasttext_trace == before_trace
    assert engine.fasttext_operation_counts == before_counts
    assert engine.fasttext_observation_count == before_observation_count


def test_streaming_generation_path_still_uses_pmi_svd_not_fasttext():
    engine = build_full_engine()
    install_explicit_loaded_fasttext_stub(engine)

    assert engine.self_embedding.__class__.__name__ == "EmbeddingWrapper"
    assert getattr(engine.self_embedding, "dim") == 300
    assert engine.self_embedding is not engine.fasttext_embedding
    assert engine.stats()["in_use_by_generation"] == "wrapper"


def test_round39_does_not_modify_self_embedding_adapter_or_hardcode_subset_path():
    self_text = SELF_EMBEDDING.read_text(encoding="utf-8")
    streaming_text = STREAMING.read_text(encoding="utf-8")

    assert "FasttextEmbeddingAdapter" not in self_text
    assert "fasttext_embedding" not in self_text
    assert "vectors.npy" not in self_text
    assert "cc.ko.300.subset.medium.30k" not in streaming_text
    assert "engine.self_embedding =" not in streaming_text


def test_round39_streaming_documents_chunk_observation_only_contract():
    text = STREAMING.read_text(encoding="utf-8")

    assert "fastText migration is observation-only" in text
    assert "must never affect chunk text" in text
    assert "chunk order" in text
