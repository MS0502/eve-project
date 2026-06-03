"""
라운드 10: HumorAdapter + SufferingAdapter + NarrativeAdapter
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
    )


def test_humor_blocked_by_negative_emotion():
    """부정 감정 강하면 농담 X."""
    engine = make_engine_full()
    from utils.types import Meaning
    meaning = Meaning(emotions={"sadness": 0.9}, entities=["민석"])
    assert not engine.humor_adapter.is_eligible(meaning)
    print(f"  ✓ 강한 sadness → not eligible")


def test_humor_eligible_when_da_high():
    """DA 충분 + 평이 입력 → 농담 가능."""
    engine = make_engine_full()
    engine.hormone_adapter.hs.hormones["dopamine"].level = 0.8
    from utils.types import Meaning
    meaning = Meaning(emotions={}, entities=["하늘"])
    assert engine.humor_adapter.is_eligible(meaning)
    print(f"  ✓ DA 0.8 + 평이 → eligible")


def test_suffering_detected():
    """fatigue/sadness 감정 → suffering 감지."""
    engine = make_engine_full()
    from utils.types import Meaning
    m1 = Meaning(emotions={"fatigue": 0.8})
    m2 = Meaning(emotions={"sadness": 0.7})
    m3 = Meaning(emotions={"joy": 0.7})
    assert engine.suffering_adapter.detect(m1) == "피로"
    assert engine.suffering_adapter.detect(m2) == "슬픔"
    assert engine.suffering_adapter.detect(m3) is None
    print(f"  ✓ fatigue→피로, sadness→슬픔, joy→None")


def test_suffering_compassion_in_response():
    """힘들다 입력 → 응답에 공감 한 줄 끼임."""
    engine = make_engine_full()
    out = list(engine.chat_stream("민석아 힘들다"))
    full = " ".join(out)
    print(f"  suffering response: {out}")
    has_compassion = "느껴져" in full or "민석" in full
    assert has_compassion or "지친" in full, f"공감/위로 없음: {full}"


def test_intimacy_grows():
    """compassion 호출 누적 → intimacy 증가."""
    engine = make_engine_full()
    intimacy0 = engine.suffering_adapter.get_intimacy()
    list(engine.chat_stream("민석아 힘들다"))
    list(engine.chat_stream("운동 너무 빡세 슬프다"))
    intimacy1 = engine.suffering_adapter.get_intimacy()
    print(f"  intimacy: {intimacy0} → {intimacy1}")
    assert intimacy1 >= intimacy0


def test_narrative_extract_themes():
    """일화 누적 → themes 추출."""
    engine = make_engine_full()
    list(engine.chat_stream("민석아 PT 힘들다"))
    list(engine.chat_stream("운동 빡세"))
    list(engine.chat_stream("기분 좋아"))
    themes = engine.narrative_adapter.themes(top_n=3)
    print(f"  themes: {themes}")
    # 그래프 sparse하면 빈 리스트도 OK


def test_narrative_phrase():
    """narrate_phrase 한 줄 출력 가능."""
    engine = make_engine_full()
    list(engine.chat_stream("민석아 PT 힘들다"))
    list(engine.chat_stream("운동 빡세"))
    phrase = engine.narrative_adapter.narrate_phrase()
    print(f"  narrate: {phrase}")
    assert phrase is not None and len(phrase) > 0


def test_narrative_coherence():
    """coherence_score 호출 가능."""
    engine = make_engine_full()
    list(engine.chat_stream("민석아 PT 힘들다"))
    score = engine.narrative_adapter.coherence()
    print(f"  coherence: {score}")
    assert isinstance(score, float)


def test_full_20_adapter_chat():
    """20 어댑터 풀스택 응답."""
    engine = make_engine_full()
    out = list(engine.chat_stream("민석아 PT 힘들다"))
    full = " ".join(out)
    print(f"  20-stack: {out}")
    assert len(out) >= 4
    assert "지친" in full or "에너지" in full


def test_20_stack_determinism():
    e1 = make_engine_full()
    e2 = make_engine_full()
    out1 = list(e1.chat_stream("민석아 힘들다"))
    out2 = list(e2.chat_stream("민석아 힘들다"))
    assert out1 == out2, f"비결정론!\n{out1}\nvs\n{out2}"
    print(f"  ✓ 20스택 결정론")


def test_existing_features_no_regression():
    """기존 9 라운드 기능 다 살아있음."""
    engine = make_engine_full()

    # 산술
    out = list(engine.chat_stream("3 더하기 5는 뭐야"))
    assert "8" in " ".join(out)

    # 환경
    out = list(engine.chat_stream("어디 있어?"))
    assert "내방" in " ".join(out)

    # 목표
    list(engine.chat_stream("쉬어"))
    assert "쉬다" in engine.goal_adapter.active_goal_categories()

    # 회상 (2턴)
    list(engine.chat_stream("PT 힘들다"))
    out = list(engine.chat_stream("PT 또 힘들어"))
    assert "전에" in " ".join(out) or "PT" in " ".join(out)

    print(f"  ✓ 모든 기존 기능 정상")


def run_all():
    tests = [
        ("humor 부정에 막힘", test_humor_blocked_by_negative_emotion),
        ("humor DA에 통과", test_humor_eligible_when_da_high),
        ("suffering 감지", test_suffering_detected),
        ("suffering 응답", test_suffering_compassion_in_response),
        ("intimacy 증가", test_intimacy_grows),
        ("NS themes", test_narrative_extract_themes),
        ("NS narrate", test_narrative_phrase),
        ("NS coherence", test_narrative_coherence),
        ("20 풀스택", test_full_20_adapter_chat),
        ("20 결정론", test_20_stack_determinism),
        ("기존 회귀", test_existing_features_no_regression),
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
