"""
라운드 8: EnvironmentAdapter + AutonomyAdapter

확인:
1. 환경 어댑터 없으면 기존 동작
2. env 켜면 meaning.context에 location/visible_objects 들어감
3. "어디?" 입력 → 환경 묘사 응답
4. "뭐 보여?" 입력 → 객체 묘사
5. autonomy idle threshold 작동 (시간 주입 결정론)
6. 14 어댑터 풀 + 결정론
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


def make_engine_full(time_fn=None):
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
    auto_kwargs = {"idle_threshold_sec": 30.0, "cooldown_sec": 60.0}
    if time_fn is not None:
        auto_kwargs["time_fn"] = time_fn
    auto = AutonomyAdapter(**auto_kwargs)
    return StreamingEngine(
        delays={"stage1": 0, "core": 0, "expand": 0, "final": 0},
        hormone_adapter=h, activation_adapter=a, memory_adapter=m,
        nl_adapter=n, sd_adapter=s, dmn_adapter=d,
        vsa_adapter=v, ai_adapter=ai,
        goal_adapter=g, norm_adapter=nr,
        continual_adapter=cr, allostatic_adapter=al,
        env_adapter=env, autonomy_adapter=auto,
    )


def test_env_meaning_context():
    """env 켜면 meaning.context에 location 들어감."""
    engine = make_engine_full()
    meaning = engine.lu.parse("안녕")
    engine.env_adapter.enrich_meaning(meaning)
    assert meaning.context.get("location") == "내방", f"location: {meaning.context}"
    assert "침대" in meaning.context.get("visible_objects", []), \
        f"visible: {meaning.context}"
    print(f"  ✓ context: location={meaning.context['location']}, "
          f"objs={meaning.context['visible_objects'][:3]}...")


def test_location_query_response():
    """'어디?' → 환경 묘사."""
    engine = make_engine_full()
    out = list(engine.chat_stream("너 어디 있어?"))
    full = " ".join(out)
    print(f"  location query: {out}")
    assert "내방" in full, f"내방 없음: {full}"


def test_what_visible_query():
    """'뭐 보여?' → 객체 묘사."""
    engine = make_engine_full()
    out = list(engine.chat_stream("뭐 보여?"))
    full = " ".join(out)
    print(f"  visible query: {out}")
    # 객체 중 하나라도 들어가면 OK
    has_obj = any(o in full for o in ["침대", "책상", "의자", "창문", "일기장", "책"])
    assert has_obj, f"객체 없음: {full}"


def test_autonomy_idle_due():
    """idle 임계 넘으면 is_due() True."""
    fake_time = [1000.0]
    def fake_now(): return fake_time[0]
    engine = make_engine_full(time_fn=fake_now)
    auto = engine.autonomy_adapter

    # 입력 직후 — idle 0초
    auto.mark_input()
    assert not auto.is_due()
    print(f"  ✓ 직후 is_due={auto.is_due()}")

    # 시간 흐름 — 35초 (threshold 30초)
    fake_time[0] += 35
    assert auto.is_due(), f"35s 지났는데 due 아님: idle={auto.get_idle_seconds()}"
    print(f"  ✓ 35s 후 is_due={auto.is_due()}, idle={auto.get_idle_seconds()}s")


def test_autonomy_cooldown():
    """proactive 후 cooldown 동안 is_due 막힘."""
    fake_time = [1000.0]
    def fake_now(): return fake_time[0]
    engine = make_engine_full(time_fn=fake_now)
    auto = engine.autonomy_adapter

    auto.mark_input()
    fake_time[0] += 35       # idle 충족
    auto.mark_proactive()    # 발화 했음
    fake_time[0] += 35       # 입력 더 이상 없고 35초 지남

    # idle은 충족이지만 cooldown은 35초 < 60초 → 막힘
    assert not auto.is_due(), f"cooldown 안 됨"
    print(f"  ✓ cooldown 작동")

    # cooldown 마저 지나면 다시 OK
    fake_time[0] += 30       # 65초 누적
    assert auto.is_due(), f"cooldown 지났는데 막힘"
    print(f"  ✓ cooldown 지나서 다시 due")


def test_chat_marks_input_time():
    """chat_stream 호출 시 last_input_time 갱신."""
    fake_time = [1000.0]
    def fake_now(): return fake_time[0]
    engine = make_engine_full(time_fn=fake_now)

    auto = engine.autonomy_adapter
    fake_time[0] = 2000.0
    list(engine.chat_stream("안녕"))
    # last_input_time이 2000 근처여야 함
    print(f"  last_input_time: {auto.last_input_time}")
    assert abs(auto.last_input_time - 2000.0) < 1


def test_proactive_marks_proactive_time():
    """proactive_stream 호출 시 last_proactive_time 갱신."""
    fake_time = [1000.0]
    def fake_now(): return fake_time[0]
    engine = make_engine_full(time_fn=fake_now)
    auto = engine.autonomy_adapter

    fake_time[0] = 3000.0
    list(engine.proactive_stream(force=True))
    print(f"  last_proactive: {auto.last_proactive_time}")
    assert auto.last_proactive_time > 0


def test_full_14_adapter_chat():
    engine = make_engine_full()
    out = list(engine.chat_stream("민석아 PT 힘들다"))
    print(f"  14-stack: {out}")
    full = " ".join(out)
    assert len(out) >= 4
    assert "지친" in full or "에너지" in full


def test_14_stack_determinism():
    e1 = make_engine_full()
    e2 = make_engine_full()
    out1 = list(e1.chat_stream("민석아 힘들다"))
    out2 = list(e2.chat_stream("민석아 힘들다"))
    assert out1 == out2, f"비결정론!\n{out1}\nvs\n{out2}"
    print(f"  ✓ 14스택 결정론")


def test_env_persists_across_turns():
    """환경은 대화 거쳐도 유지."""
    engine = make_engine_full()
    list(engine.chat_stream("민석아 안녕"))
    list(engine.chat_stream("힘들다"))
    out = list(engine.chat_stream("어디 있어?"))
    full = " ".join(out)
    print(f"  3턴 후 location query: {out}")
    assert "내방" in full


def run_all():
    tests = [
        ("env meaning context", test_env_meaning_context),
        ("location 질문", test_location_query_response),
        ("뭐 보여 질문", test_what_visible_query),
        ("idle due", test_autonomy_idle_due),
        ("cooldown", test_autonomy_cooldown),
        ("chat mark input", test_chat_marks_input_time),
        ("proactive mark", test_proactive_marks_proactive_time),
        ("14 풀스택 응답", test_full_14_adapter_chat),
        ("14 결정론", test_14_stack_determinism),
        ("환경 유지", test_env_persists_across_turns),
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
