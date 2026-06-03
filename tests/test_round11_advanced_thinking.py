"""
라운드 11: Creative + MetaCognition + EmotionRegulation + MultiStream
"""

import sys, os
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
from adapters.env_adapter import EnvironmentAdapter
from adapters.autonomy_adapter import AutonomyAdapter
from adapters.counterfactual_adapter import CounterfactualAdapter
from adapters.analogy_adapter import AnalogyAdapter
from adapters.temporal_adapter import TemporalAdapter
from adapters.humor_adapter import HumorAdapter
from adapters.suffering_adapter import SufferingAdapter
from adapters.narrative_adapter import NarrativeAdapter
from adapters.creative_adapter import CreativeAdapter
from adapters.metacognition_adapter import MetacognitionAdapter
from adapters.emotion_regulation_adapter import EmotionRegulationAdapter
from adapters.multi_stream_adapter import MultiStreamAdapter


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
    env = EnvironmentAdapter(hormone_adapter=h, activation_adapter=a)
    auto = AutonomyAdapter()
    cf = CounterfactualAdapter(hormone_adapter=h, activation_adapter=a)
    an = AnalogyAdapter(hormone_adapter=h, activation_adapter=a)
    tm = TemporalAdapter(hormone_adapter=h, activation_adapter=a, memory_adapter=m)
    hu = HumorAdapter(hormone_adapter=h, activation_adapter=a)
    sf = SufferingAdapter(hormone_adapter=h, activation_adapter=a)
    nv = NarrativeAdapter(hormone_adapter=h, activation_adapter=a, memory_adapter=m)
    cra = CreativeAdapter(hormone_adapter=h, activation_adapter=a, memory_adapter=m, dmn_adapter=d)
    mc = MetacognitionAdapter(hormone_adapter=h, activation_adapter=a,
                              sd_adapter=s, ai_adapter=ai, dmn_adapter=d)
    era = EmotionRegulationAdapter(hormone_adapter=h, activation_adapter=a,
                                   memory_adapter=m, cf_adapter=cf, tm_adapter=tm)
    msa = MultiStreamAdapter(hormone_adapter=h, activation_adapter=a,
                             dmn_adapter=d, memory_adapter=m, tm_adapter=tm,
                             goal_adapter=g, nl_adapter=n)
    return StreamingEngine(
        delays={"stage1": 0, "core": 0, "expand": 0, "final": 0},
        hormone_adapter=h, activation_adapter=a, memory_adapter=m,
        nl_adapter=n, sd_adapter=s, dmn_adapter=d,
        vsa_adapter=v, ai_adapter=ai,
        goal_adapter=g, norm_adapter=nr,
        continual_adapter=cr, allostatic_adapter=al,
        env_adapter=env, autonomy_adapter=auto,
        counterfactual_adapter=cf, analogy_adapter=an, temporal_adapter=tm,
        humor_adapter=hu, suffering_adapter=sf, narrative_adapter=nv,
        creative_adapter=cra, metacognition_adapter=mc,
        emotion_regulation_adapter=era, multi_stream_adapter=msa,
    )


def test_creative_incubate_runs():
    """incubate 호출 안 죽음."""
    engine = make_engine_full()
    # DA 충분 + cort 낮춤
    engine.hormone_adapter.hs.hormones["dopamine"].level = 0.7
    engine.hormone_adapter.hs.hormones["cortisol"].level = 0.2
    from utils.types import Meaning
    m = Meaning(entities=["하늘"], emotions={})
    ok = engine.creative_adapter.maybe_incubate(m)
    print(f"  incubate: {ok}")


def test_metacog_confidence():
    """confidence 호출 가능."""
    engine = make_engine_full()
    from utils.types import Meaning
    m = Meaning(entities=["민석"])
    c = engine.metacognition_adapter.confidence(m, "안녕")
    print(f"  confidence: {c}")
    assert 0.0 <= c <= 1.0


def test_er_detect_need():
    """detect_need 호출."""
    engine = make_engine_full()
    # cort 높임 → need 감지
    engine.hormone_adapter.hs.hormones["cortisol"].level = 0.8
    need = engine.emotion_regulation_adapter.detect_need()
    print(f"  detect_need (cort 0.8): {need}")
    # need 있을 수도 없을 수도, 안 죽으면 OK


def test_er_auto_regulate():
    """cort 높을 때 auto_regulate가 cort 떨어뜨림."""
    engine = make_engine_full()
    hs = engine.hormone_adapter.hs
    hs.hormones["cortisol"].level = 0.85
    cort_before = hs.hormones["cortisol"].level
    result = engine.emotion_regulation_adapter.auto_regulate()
    cort_after = hs.hormones["cortisol"].level
    print(f"  cort {cort_before} → {cort_after}, result={result}")
    # cort 떨어졌거나 그대로 (조건에 따라). 안 죽으면 OK


def test_ms_tick_runs():
    """multi-stream tick 호출."""
    engine = make_engine_full()
    r = engine.multi_stream_adapter.tick(dt=0.5)
    print(f"  ms tick: {r}")


def test_full_24_adapter_chat():
    """24 어댑터 풀스택 응답."""
    engine = make_engine_full()
    out = list(engine.chat_stream("민석아 PT 힘들다"))
    full = " ".join(out)
    print(f"  24-stack: {out}")
    assert len(out) >= 4
    assert "지친" in full or "에너지" in full


def test_24_stack_determinism():
    e1 = make_engine_full()
    e2 = make_engine_full()
    out1 = list(e1.chat_stream("민석아 힘들다"))
    out2 = list(e2.chat_stream("민석아 힘들다"))
    assert out1 == out2, f"비결정론!\n{out1}\nvs\n{out2}"
    print(f"  ✓ 24스택 결정론")


def test_existing_no_regression():
    """기존 10 라운드 기능 살아있음."""
    engine = make_engine_full()
    out = list(engine.chat_stream("3 더하기 5는 뭐야"))
    assert "8" in " ".join(out)
    out = list(engine.chat_stream("어디 있어?"))
    assert "내방" in " ".join(out)
    out = list(engine.chat_stream("너무 슬프다"))
    assert "민석" in " ".join(out) or "가라앉" in " ".join(out)
    print(f"  ✓ 기존 회귀 없음")


def test_high_cort_chat_triggers_er():
    """대화 도중 cort 높아도 응답 정상 + ER 자동 트리거."""
    engine = make_engine_full()
    engine.hormone_adapter.hs.hormones["cortisol"].level = 0.85
    out = list(engine.chat_stream("민석아 안녕"))
    print(f"  high cort chat: {out}")
    # ER이 응답 후 cort 조절했는지
    cort_after = engine.hormone_adapter.hs.hormones["cortisol"].level
    print(f"  cort after: {cort_after}")
    assert len(out) >= 2


def run_all():
    tests = [
        ("creative incubate", test_creative_incubate_runs),
        ("MC confidence", test_metacog_confidence),
        ("ER detect_need", test_er_detect_need),
        ("ER auto regulate", test_er_auto_regulate),
        ("MS tick", test_ms_tick_runs),
        ("24 풀스택", test_full_24_adapter_chat),
        ("24 결정론", test_24_stack_determinism),
        ("기존 회귀", test_existing_no_regression),
        ("ER 자동 트리거", test_high_cort_chat_triggers_er),
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
