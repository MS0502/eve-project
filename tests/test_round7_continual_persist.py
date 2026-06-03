"""
라운드 7: ContinualAdapter + AllostaticAdapter + PersistenceAdapter
"""

import sys, os, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from language.streaming import StreamingEngine
from adapters.hormone_adapter import HormoneAdapter
from adapters.activation_adapter import ActivationAdapter
from adapters.memory_adapter import MemoryAdapter
from adapters.nl_adapter import NLAdapter
from adapters.sd_adapter import SDAdapter
from adapters.dmn_adapter import DMNAdapter
from adapters.vsa_adapter import VSAAdapter
from adapters.ai_adapter import AIAdapter
from adapters.goal_adapter import GoalAdapter
from adapters.norm_adapter import NormAdapter
from adapters.continual_adapter import ContinualAdapter
from adapters.allostatic_adapter import AllostaticAdapter
from adapters.persistence_adapter import PersistenceAdapter


def make_engine_full():
    h = HormoneAdapter()
    a = ActivationAdapter(hormone_adapter=h)
    m = MemoryAdapter(activation_adapter=a, hormone_adapter=h)
    n = NLAdapter(hormone_adapter=h, activation_adapter=a)
    s = SDAdapter(hormone_adapter=h, activation_adapter=a)
    d = DMNAdapter(hormone_adapter=h, activation_adapter=a)
    v = VSAAdapter(hormone_adapter=h, activation_adapter=a)
    ai = AIAdapter(hormone_adapter=h, activation_adapter=a, sd_adapter=s)
    g = GoalAdapter(hormone_adapter=h, activation_adapter=a)
    nr = NormAdapter(hormone_adapter=h, activation_adapter=a, nl_adapter=n, dmn_adapter=d)
    cr = ContinualAdapter(hormone_adapter=h, activation_adapter=a)
    al = AllostaticAdapter(hormone_adapter=h, activation_adapter=a)
    return StreamingEngine(
        delays={"stage1": 0, "core": 0, "expand": 0, "final": 0},
        hormone_adapter=h, activation_adapter=a, memory_adapter=m,
        nl_adapter=n, sd_adapter=s, dmn_adapter=d,
        vsa_adapter=v, ai_adapter=ai,
        goal_adapter=g, norm_adapter=nr,
        continual_adapter=cr, allostatic_adapter=al,
    )


def test_continual_protects_default_categories():
    """기본 정체성 카테고리(민석, EVE, 친구, 대화)가 보호됨."""
    engine = make_engine_full()
    protected = engine.continual_adapter.protected_categories()
    print(f"  protected: {protected}")
    assert "민석" in protected, f"민석 보호 안 됨: {protected}"


def test_continual_tick_works():
    """tick 호출 정상."""
    engine = make_engine_full()
    engine.continual_adapter.tick(dt=1.0)
    print(f"  ✓ tick 호출 정상")


def test_continual_runs_in_chat():
    """chat_stream 내부에서 continual tick 호출됨 (안 죽음)."""
    engine = make_engine_full()
    out = list(engine.chat_stream("민석아 안녕"))
    print(f"  chat: {out}")
    # 보호 카테고리 활성도 확인
    sa = engine.activation_adapter.sa
    minseok_lvl = sa.get_activation_level("민석")
    print(f"  민석 활성도: {minseok_lvl}")
    assert len(out) >= 2


def test_allostatic_observe_works():
    """대화 후 allostatic.observe 호출 (안 죽음)."""
    engine = make_engine_full()
    list(engine.chat_stream("민석아 힘들다"))
    list(engine.chat_stream("PT 빡세"))
    # patterns 누적 가능성 (시간 부족하면 비어있어도 OK)
    patterns = engine.allostatic_adapter.get_patterns()
    print(f"  patterns: {len(patterns)}")
    print(f"  ✓ AL 흐름 안 깨짐")


def test_persistence_save_returns_path():
    """save 호출 → 결과 dict + 파일 생성."""
    engine = make_engine_full()
    list(engine.chat_stream("민석아 PT 힘들다"))
    list(engine.chat_stream("쉬어"))

    persistence = PersistenceAdapter(engine)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "eve_state.pkl")
        result = persistence.save(path)
        print(f"  save result keys: {list(result.keys())}")
        assert "v40" in result
        # 파일이 어딘가 만들어졌을 것
        # (compress=True 면 .gz 붙음)
        files = os.listdir(tmp)
        print(f"  saved files: {files}")
        assert len(files) >= 1


def test_persistence_save_load_roundtrip():
    """save → load → 호르몬 상태 복원 확인."""
    engine = make_engine_full()
    # 호르몬 변동
    engine.hormone_adapter.hs.hormones["oxytocin"].level = 0.85
    engine.hormone_adapter.hs.hormones["cortisol"].level = 0.15

    persistence = PersistenceAdapter(engine)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "eve_state.pkl")
        result = persistence.save(path)
        save_path = result.get("path") or path
        print(f"  saved at: {save_path}")

        # 새 엔진 만들어서 load
        new_engine = make_engine_full()
        # 새 엔진은 baseline 호르몬
        assert abs(new_engine.hormone_adapter.hs.hormones["oxytocin"].level - 0.3) < 0.01

        new_persist = PersistenceAdapter(new_engine)
        load_result = new_persist.load(save_path)
        print(f"  load result: {load_result}")

        # 복원됐는지 — oxytocin level 비교
        new_ot = new_engine.hormone_adapter.hs.hormones["oxytocin"].level
        new_cort = new_engine.hormone_adapter.hs.hormones["cortisol"].level
        print(f"  복원 후: ot={new_ot}, cort={new_cort}")
        assert abs(new_ot - 0.85) < 0.05, f"oxytocin 복원 안 됨: {new_ot}"
        assert abs(new_cort - 0.15) < 0.05, f"cortisol 복원 안 됨: {new_cort}"


def test_full_12_adapter_chat():
    """12 어댑터 풀스택 응답 정상."""
    engine = make_engine_full()
    out = list(engine.chat_stream("민석아 PT 힘들다"))
    full = " ".join(out)
    print(f"  12-stack: {out}")
    assert len(out) >= 4
    assert "지친" in full or "에너지" in full


def test_12_stack_determinism():
    e1 = make_engine_full()
    e2 = make_engine_full()
    out1 = list(e1.chat_stream("민석아 힘들다"))
    out2 = list(e2.chat_stream("민석아 힘들다"))
    assert out1 == out2, f"비결정론!\n{out1}\nvs\n{out2}"
    print(f"  ✓ 12스택 결정론")


def run_all():
    tests = [
        ("CR 보호", test_continual_protects_default_categories),
        ("CR tick", test_continual_tick_works),
        ("CR in chat", test_continual_runs_in_chat),
        ("AL observe", test_allostatic_observe_works),
        ("Persistence save", test_persistence_save_returns_path),
        ("Persistence roundtrip", test_persistence_save_load_roundtrip),
        ("12스택 응답", test_full_12_adapter_chat),
        ("12스택 결정론", test_12_stack_determinism),
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
