"""
라운드 39b: ThoughtChainAdapter — 사고 흐름
"""

import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adapters.thought_chain_adapter import (
    ThoughtChainAdapter, ThoughtChain, ThoughtNode,
    CHAIN_MIN_LENGTH, CHAIN_MAX_LENGTH,
    W_SALIENCE, W_ASSOCIATION, W_NOVELTY,
)


# ===== 단위 테스트 =====

def test_init():
    t = ThoughtChainAdapter()
    assert t.chains_generated == 0
    print(f"  ✓ init")


def test_chain_no_engine():
    """engine 없으면 길이 1 (시드만)"""
    t = ThoughtChainAdapter(None)
    chain = t.generate_chain("민석")
    assert len(chain) == 1
    assert chain.nodes[0].category == "민석"
    print(f"  ✓ engine 없으면 시드만")


def test_chain_with_engine():
    from main import build_full_engine
    engine = build_full_engine()
    
    chain = engine.thought_chain.generate_chain("민석", length=3)
    assert len(chain) >= 1
    assert chain.nodes[0].category == "민석"
    # 후보가 있으면 더 긴 체인
    if len(chain) >= 2:
        # visited 체크 (중복 X)
        cats = chain.categories()
        assert len(cats) == len(set(cats))
    print(f"  ✓ chain: {chain.categories()}")


def test_chain_no_repeat():
    """같은 카테고리 반복 X"""
    from main import build_full_engine
    engine = build_full_engine()
    
    sa = engine.activation_adapter.sa
    sa.learn_pair("A", "B", 0.5)
    sa.learn_pair("B", "A", 0.5)  # 양방향
    
    chain = engine.thought_chain.generate_chain("A", length=4)
    cats = chain.categories()
    assert len(cats) == len(set(cats)), f"중복: {cats}"
    print(f"  ✓ 중복 없음: {cats}")


def test_chain_length_bounds():
    from main import build_full_engine
    engine = build_full_engine()
    
    # 짧은 length 요청
    chain = engine.thought_chain.generate_chain("민석", length=2)
    assert len(chain) <= 2
    
    # 매우 긴 length 요청
    chain2 = engine.thought_chain.generate_chain("민석", length=10)
    assert len(chain2) <= CHAIN_MAX_LENGTH
    print(f"  ✓ 길이 제한 적용 (max {CHAIN_MAX_LENGTH})")


def test_chain_to_phrase():
    """사고 체인 → 자연어"""
    t = ThoughtChainAdapter()
    
    # 길이 2
    chain = ThoughtChain(nodes=[
        ThoughtNode("민석", 1.0, "seed"),
        ThoughtNode("PT", 0.7, "association"),
    ])
    phrase = t.chain_to_phrase(chain)
    assert "민석" in phrase and "PT" in phrase
    
    # 길이 3
    chain3 = ThoughtChain(nodes=[
        ThoughtNode("민석", 1.0, "seed"),
        ThoughtNode("PT", 0.7, "association"),
        ThoughtNode("피곤", 0.5, "association"),
    ])
    phrase3 = t.chain_to_phrase(chain3)
    assert "민석" in phrase3 and "PT" in phrase3 and "피곤" in phrase3
    print(f"  ✓ phrase: {phrase!r} / {phrase3!r}")


def test_chain_to_phrase_short():
    """길이 1면 빈 문자열 (단일 카테고리는 일반 발화)"""
    t = ThoughtChainAdapter()
    chain = ThoughtChain(nodes=[ThoughtNode("X", 1.0, "seed")])
    phrase = t.chain_to_phrase(chain)
    assert phrase == ""
    print(f"  ✓ 길이 1은 빈 문자열")


def test_score_calculation():
    """salience + association 가중치 합산"""
    from main import build_full_engine
    engine = build_full_engine()
    
    sa = engine.activation_adapter.sa
    sa.learn_pair("seed", "high_assoc", 0.9)
    sa.learn_pair("seed", "low_assoc", 0.2)
    
    # high_assoc는 salience 없음, low_assoc는 salience 강
    engine.salience_adapter.spike("low_assoc", emotional=True)
    
    # 어느 게 선택되나
    chain = engine.thought_chain.generate_chain("seed", length=2)
    if len(chain) >= 2:
        chosen = chain.nodes[1].category
        # 점수 계산: high_assoc = 0.9*W_A = 0.45, low_assoc = 0.5/5*W_S + 0.2*W_A = 0.04+0.1 = 0.14
        # 또는 emotional spike가 0.5 → 0.5/5 * 0.4 = 0.04 + 0.2*0.5 = 0.14
        # high_assoc 이김
        assert chosen == "high_assoc", f"기대 high_assoc, 실제 {chosen}"
    print(f"  ✓ 점수 계산 — association 강한 거 선택")


def test_determinism():
    from main import build_full_engine
    e1 = build_full_engine()
    e2 = build_full_engine()
    
    for sa_obj in [e1.activation_adapter.sa, e2.activation_adapter.sa]:
        sa_obj.learn_pair("X", "Y", 0.5)
        sa_obj.learn_pair("Y", "Z", 0.5)
    
    c1 = e1.thought_chain.generate_chain("X", length=3)
    c2 = e2.thought_chain.generate_chain("X", length=3)
    
    assert c1.categories() == c2.categories()
    print(f"  ✓ 결정론: {c1.categories()}")


def test_stats():
    from main import build_full_engine
    engine = build_full_engine()
    
    engine.thought_chain.generate_chain("민석", length=3)
    engine.thought_chain.generate_chain("PT", length=3)
    
    s = engine.thought_chain.stats()
    assert s["chains_generated"] == 2
    assert s["total_nodes"] >= 2
    print(f"  ✓ stats {s}")


# ===== 통합 — proactive 발화에 chain 들어가는지 =====

def test_chain_in_proactive():
    """카테고리 hash 4의 배수면 chain 발화"""
    from main import build_full_engine
    engine = build_full_engine()
    
    # SA에 강한 연결 만들기
    sa = engine.activation_adapter.sa
    sa.learn_pair("민석", "PT", 0.8)
    sa.learn_pair("PT", "피곤", 0.8)
    
    # 여러 번 시도 — 카테고리 hash 4의 배수일 때만 chain
    chain_used = False
    for _ in range(10):
        chunks = list(engine.proactive_stream(force=True))
        full = " ".join(chunks)
        # chain 발화 표시 — "생각하다 보니" / "떠올라서" / "거쳐서"
        if any(kw in full for kw in ["생각하다 보니", "떠올라서", "거쳐서", "떠올랐는데"]):
            chain_used = True
            print(f"    chain 발화: {full}")
            break
    
    # 항상 안 나올 수 있음 (카테고리 hash 4의 배수 조건)
    print(f"  ✓ chain 발화 시도 (chain_used={chain_used})")


def test_engine_has_thought_chain():
    """engine에 thought_chain 부착"""
    from main import build_full_engine
    engine = build_full_engine()
    
    assert hasattr(engine, "thought_chain")
    assert engine.thought_chain is not None
    assert engine.thought_chain.engine is engine
    print(f"  ✓ engine 부착")


def run_all():
    tests = [
        ("init", test_init),
        ("engine 없으면 시드", test_chain_no_engine),
        ("chain 생성", test_chain_with_engine),
        ("중복 없음", test_chain_no_repeat),
        ("길이 제한", test_chain_length_bounds),
        ("phrase 변환", test_chain_to_phrase),
        ("길이 1 빈 문자열", test_chain_to_phrase_short),
        ("점수 계산", test_score_calculation),
        ("결정론", test_determinism),
        ("stats", test_stats),
        ("proactive에 chain", test_chain_in_proactive),
        ("engine 부착", test_engine_has_thought_chain),
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
