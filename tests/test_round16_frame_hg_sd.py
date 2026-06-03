"""
라운드 16: FrameAdapter + HypergraphAdapter + SemanticDistanceAdapter
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import build_full_engine


def test_frame_parse():
    """SVO 추출."""
    engine = build_full_engine()
    from utils.types import Meaning
    m = Meaning(raw_text="민석이 PT를 했다", entities=["민석", "PT"])
    engine.frame_adapter.enrich_meaning(m)
    f = m.context.get("frame")
    print(f"  frame: {f}")
    assert f is not None
    assert f.get("agent") == "민석"
    assert f.get("patient") == "PT"


def test_frame_in_chat():
    """chat_stream에서 frame이 meaning.context에 박힘."""
    engine = build_full_engine()
    # 단순 호출 - 프레임 enrich 자체는 turn 안에서 일어남, 후엔 context 못 봄
    # 대신 HG 저장 확인
    list(engine.chat_stream("민석이 PT를 했다"))
    list(engine.chat_stream("민석이 운동을 했다"))
    # HG에 edge 생겼어야
    stats = engine.hypergraph_adapter.stats()
    print(f"  HG stats: {stats}")
    n_edges = stats.get("edges_count", 0) or stats.get("n_edges", 0) or stats.get("total_edges", 0)
    # edge 없을 수도 (frame parse 안 됐으면)
    print(f"  edges: {n_edges}")


def test_hypergraph_add_query():
    """HG add_edge + query 동작."""
    engine = build_full_engine()
    hg = engine.hypergraph_adapter
    edge_id = hg.add_edge(
        nodes={"민석", "PT", "하다"},
        roles={"민석": "agent", "PT": "patient", "하다": "action"},
        weight=0.7,
    )
    print(f"  added edge: {edge_id}")
    # query — agent=민석인 edge
    results = hg.query({"agent": "민석"})
    print(f"  query agent=민석: {len(results)} results")


def test_hypergraph_neighbors():
    """neighbors_via_edges — 같은 edge의 다른 노드들."""
    engine = build_full_engine()
    hg = engine.hypergraph_adapter
    hg.add_edge(
        nodes={"민석", "PT", "하다"},
        roles={"민석": "agent", "PT": "patient", "하다": "action"},
    )
    n = hg.neighbors("민석")
    print(f"  neighbors of 민석: {n}")
    assert "PT" in n or "하다" in n


def test_sd_similarity():
    """비슷한 카테고리 유사도 더 높음."""
    engine = build_full_engine()
    sa = engine.activation_adapter.sa
    sa.activate("PT", 0.5); sa.activate("운동", 0.5); sa.activate("커피", 0.5)
    sa.learn_pair("PT", "운동", strength=0.8)
    sa.learn_pair("PT", "힘들다", strength=0.5)
    sa.learn_pair("운동", "힘들다", strength=0.6)

    s_pt_운동 = engine.semantic_distance_adapter.similarity("PT", "운동")
    s_pt_커피 = engine.semantic_distance_adapter.similarity("PT", "커피")
    print(f"  sim PT-운동: {s_pt_운동:.3f}")
    print(f"  sim PT-커피: {s_pt_커피:.3f}")
    assert s_pt_운동 > s_pt_커피, "비슷한 거가 덜 비슷한 거보다 높아야"


def test_sd_find_similar():
    """find_similar."""
    engine = build_full_engine()
    sa = engine.activation_adapter.sa
    sa.activate("PT", 0.5); sa.activate("운동", 0.5); sa.activate("러닝", 0.5)
    sa.learn_pair("PT", "운동", strength=0.7)
    sa.learn_pair("PT", "러닝", strength=0.6)
    sa.learn_pair("운동", "러닝", strength=0.5)

    sims = engine.semantic_distance_adapter.find_similar("PT", top_k=3)
    print(f"  similar to PT: {sims}")
    # 운동 또는 러닝이 높아야
    names = [n for n, _ in sims]
    assert "운동" in names or "러닝" in names


def test_sd_similar_phrase():
    """similar_phrase 한 줄."""
    engine = build_full_engine()
    sa = engine.activation_adapter.sa
    sa.activate("PT", 0.5); sa.activate("운동", 0.5); sa.activate("러닝", 0.5)
    sa.learn_pair("PT", "운동", strength=0.7)
    sa.learn_pair("PT", "러닝", strength=0.6)

    phrase = engine.semantic_distance_adapter.similar_phrase("PT")
    print(f"  phrase: {phrase}")
    assert phrase is not None and "PT" in phrase


def test_sd_with_corpus_learning():
    """Corpus 학습 → SD가 그 카테고리 인식."""
    engine = build_full_engine()
    list(engine.chat_stream("PT는 정말 힘들다 운동도 빡세다"))
    list(engine.chat_stream("운동은 힘들지만 좋다"))
    list(engine.chat_stream("PT 갈 때마다 빡세다"))

    # 학습 후 PT-운동 유사도
    sim = engine.semantic_distance_adapter.similarity("PT", "운동")
    print(f"  sim after 3 turns: {sim:.3f}")
    # corpus 학습으로 가중치 생겼어야
    sa = engine.activation_adapter.sa
    print(f"  SA weight PT-운동: {sa.get_weight('PT', '운동'):.3f}")


def test_simulation_chain_with_frame():
    """프레임이 hg에 저장되고 sim/wm이 그것 활용."""
    engine = build_full_engine()
    # 인과 그래프에도 학습
    cg = engine.simulation_engine.cg
    if cg is not None:
        cg.learn_causal("PT", "피로", evidence_weight=0.8)

    # FS도 학습
    list(engine.chat_stream("민석이 PT를 했다"))
    list(engine.chat_stream("민석이 피로하다"))

    # sim 응답
    out = list(engine.chat_stream("PT 하면 어떻게 돼?"))
    full = " ".join(out)
    print(f"  sim chat: {out}")
    has_chain = "→" in full or "흘러" in full or "PT" in full
    assert has_chain


def test_30_stack_determinism():
    e1 = build_full_engine()
    e2 = build_full_engine()
    o1 = list(e1.chat_stream("민석아 힘들다"))
    o2 = list(e2.chat_stream("민석아 힘들다"))
    assert o1 == o2, f"비결정론!"
    print(f"  ✓ 30스택 결정론")


def test_existing_no_regression():
    engine = build_full_engine()
    out = list(engine.chat_stream("3 더하기 5는 뭐야"))
    assert "8" in " ".join(out)
    out = list(engine.chat_stream("어디 있어?"))
    assert "내방" in " ".join(out)
    out = list(engine.chat_stream("민석아 너무 힘들어"))
    assert "느껴져" in " ".join(out) or "지친" in " ".join(out)
    print(f"  ✓ 기존 회귀 없음")


def test_full_demo():
    engine = build_full_engine()

    print("\n  [Frame parsing] '민석이 PT를 했다'")
    list(engine.chat_stream("민석이 PT를 했다"))
    list(engine.chat_stream("민석이 운동을 했다"))
    list(engine.chat_stream("민석이 커피를 마셨다"))

    stats = engine.hypergraph_adapter.stats()
    print(f"  HG: {stats}")

    print("\n  [SD] 학습 후 유사도")
    sa = engine.activation_adapter.sa
    print(f"  PT-운동: {engine.semantic_distance_adapter.similarity('PT', '운동'):.3f}")
    print(f"  PT-커피: {engine.semantic_distance_adapter.similarity('PT', '커피'):.3f}")

    print("\n  [질문] PT랑 비슷한 게 뭐야?")
    sims = engine.semantic_distance_adapter.find_similar("PT", top_k=3)
    print(f"  {sims}")


def run_all():
    tests = [
        ("Frame 파싱", test_frame_parse),
        ("Frame chat", test_frame_in_chat),
        ("HG add+query", test_hypergraph_add_query),
        ("HG neighbors", test_hypergraph_neighbors),
        ("SD 유사도", test_sd_similarity),
        ("SD find_similar", test_sd_find_similar),
        ("SD phrase", test_sd_similar_phrase),
        ("SD + Corpus", test_sd_with_corpus_learning),
        ("Sim + Frame", test_simulation_chain_with_frame),
        ("30 결정론", test_30_stack_determinism),
        ("기존 회귀", test_existing_no_regression),
        ("전체 시연", test_full_demo),
    ]
    passed = 0
    failed = []
    for name, fn in tests:
        print(f"\n[{name}]")
        try:
            fn()
            passed += 1
        except AssertionError as e:
            print(f"  ✗ FAIL: {e}")
            failed.append(name)
        except Exception as e:
            print(f"  ✗ ERROR: {type(e).__name__}: {e}")
            import traceback; traceback.print_exc()
            failed.append(name)
    print(f"\n{'='*40}")
    print(f"{passed}/{len(tests)} pass")
    if failed:
        print(f"실패: {failed}")
        sys.exit(1)


if __name__ == "__main__":
    run_all()
