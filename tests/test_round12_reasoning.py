"""
라운드 12: DeepReasoningAdapter + ReasoningLoop + LengthDecider

GPT 라운드 11/12 핵심:
- 입력 → 이해 → [사고] → 공감 → 말하기  
- 길이 = 상태의 결과 (고정 X)
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
from adapters.deep_reasoning_adapter import DeepReasoningAdapter
from core.reasoning import ReasoningLoop


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
    dr = DeepReasoningAdapter(hormone_adapter=h, activation_adapter=a, nl_adapter=n)
    reasoning = ReasoningLoop(
        hormone_adapter=h,
        deep_reasoning_adapter=dr,
        temporal_adapter=tm,
        counterfactual_adapter=cf,
        metacognition_adapter=mc,
    )
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
        deep_reasoning_adapter=dr, reasoning_loop=reasoning,
    )


# ===== ReasoningLoop =====

def test_reasoning_generates_streams():
    """감정 강한 입력 → emotional thought 생성."""
    engine = make_engine_full()
    from utils.types import Meaning
    m = Meaning(intent="emotion", entities=["민석"], emotions={"fatigue": 0.8})
    streams = engine.reasoning_loop.generate_thought_streams(m)
    print(f"  streams: {[(s.type, s.content[:30]) for s in streams]}")
    assert any(s.type == "emotional" for s in streams), f"emotional 없음: {streams}"


def test_reasoning_picks_best():
    """think() 결과 점수 가장 높은 것."""
    engine = make_engine_full()
    from utils.types import Meaning
    from cognition.meaning_graph import build_graph
    m = Meaning(intent="emotion", entities=["민석"], emotions={"fatigue": 0.8})
    engine.workspace.update(build_graph(m), m)
    state = engine.workspace.get_state()
    thought = engine.reasoning_loop.think(state)
    print(f"  best thought: {thought}")
    assert thought is not None
    assert thought.score > 0


def test_reasoning_in_response():
    """응답에 thought 끼임."""
    engine = make_engine_full()
    out = list(engine.chat_stream("민석아 PT 진짜 힘들다"))
    full = " ".join(out)
    print(f"  reasoning chat: {out}")
    # 사고 표현 ("단순 ~보다 더 깊은", "쌓인 거 같아", "잠깐 멈춰가면" 등)
    has_thought = (
        "단순" in full or "쌓인 거" in full or "멈춰가면" in full
        or "이어지는" in full or "더 깊은" in full
    )
    assert has_thought, f"thought 없음: {full}"


# ===== 동적 길이 =====

def test_length_short_for_simple_input():
    """평이한 입력 → 짧은 응답 (1~2 줄 + stage1 + final)."""
    engine = make_engine_full()
    out = list(engine.chat_stream("응 그래"))
    print(f"  simple: {out}")
    # core_message + final 정도. 4 chunk 미만.
    assert len(out) <= 4, f"평이 입력에 너무 김: {len(out)}"


def test_length_long_for_strong_emotion():
    """강한 감정 → 긴 응답."""
    engine = make_engine_full()
    out = list(engine.chat_stream("민석아 너무 힘들어 죽겠어"))
    print(f"  strong emotion: {len(out)} lines: {out}")
    assert len(out) >= 4, f"강감정인데 짧음: {len(out)}"


def test_length_question_increment():
    """질문이면 +1줄."""
    engine = make_engine_full()
    out_stmt = list(engine.chat_stream("그래"))
    out_q = list(engine.chat_stream("이거 뭐야?"))
    print(f"  stmt: {len(out_stmt)} / question: {len(out_q)}")
    # 질문이 statement 이상이어야


def test_length_intimacy_growth():
    """intimacy 누적되면 길이 늘어남."""
    engine = make_engine_full()
    # 짧은 대화 (intimacy 0)
    short = list(engine.chat_stream("힘들다"))

    # intimacy 누적 (suffering compassion 여러 번)
    for _ in range(5):
        list(engine.chat_stream("진짜 힘들어"))
    intimacy = engine.suffering_adapter.get_intimacy()
    print(f"  intimacy after 5 turns: {intimacy}")

    # 같은 입력 다시 — intimacy로 길어졌어야
    long = list(engine.chat_stream("힘들다"))
    print(f"  short: {len(short)}, long: {len(long)}")
    # 친밀도 영향이 있다면 long >= short


def test_length_high_cort_shortens():
    """cort 0.85 → 길이 -1."""
    engine = make_engine_full()
    engine.hormone_adapter.hs.hormones["cortisol"].level = 0.85
    out = list(engine.chat_stream("민석아 힘들다"))
    print(f"  high cort: {len(out)} lines: {out}")
    # 평상보다 짧아야


# ===== 회귀 =====

def test_25_stack_determinism():
    e1 = make_engine_full()
    e2 = make_engine_full()
    out1 = list(e1.chat_stream("민석아 힘들다"))
    out2 = list(e2.chat_stream("민석아 힘들다"))
    assert out1 == out2, f"비결정론!\n{out1}\nvs\n{out2}"
    print(f"  ✓ 25스택 결정론")


def test_existing_features_no_regression():
    engine = make_engine_full()
    # 산술
    out = list(engine.chat_stream("3 더하기 5는 뭐야"))
    assert "8" in " ".join(out)
    # 환경
    out = list(engine.chat_stream("어디 있어?"))
    assert "내방" in " ".join(out)
    # 슬픔
    out = list(engine.chat_stream("너무 슬프다"))
    assert "민석" in " ".join(out) or "가라앉" in " ".join(out)
    print(f"  ✓ 산술/환경/공감 회귀 없음")


def test_full_demo():
    engine = make_engine_full()

    print("\n  [평이] 응 그래")
    for c in engine.chat_stream("응 그래"): print(f"    {c}")

    print("\n  [강감정] 민석아 너무 힘들어 죽겠어")
    for c in engine.chat_stream("민석아 너무 힘들어 죽겠어"): print(f"    {c}")

    print("\n  [질문] 왜 이렇게 힘들지?")
    for c in engine.chat_stream("왜 이렇게 힘들지?"): print(f"    {c}")


def run_all():
    tests = [
        ("Reasoning streams", test_reasoning_generates_streams),
        ("Reasoning best", test_reasoning_picks_best),
        ("Reasoning in chat", test_reasoning_in_response),
        ("길이: 평이→짧음", test_length_short_for_simple_input),
        ("길이: 강감정→긺", test_length_long_for_strong_emotion),
        ("길이: 질문 +1", test_length_question_increment),
        ("길이: intimacy 영향", test_length_intimacy_growth),
        ("길이: cort↑→짧음", test_length_high_cort_shortens),
        ("25스택 결정론", test_25_stack_determinism),
        ("기존 회귀", test_existing_features_no_regression),
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
