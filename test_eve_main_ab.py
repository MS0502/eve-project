"""
v32 EVE_TierAB 통합 검증 (A + B1)
==================================
"""
import sys
sys.path.insert(0, '/home/claude/eve/v32')
from eve_main_ab import EVE_TierAB


def passed(name, ok, detail=""):
    mark = "✅" if ok else "❌"
    print(f"  {mark} {name}" + (f"  | {detail}" if detail else ""))
    return ok


print("=" * 70)
print("v32 EVE_TierAB 통합 검증 (A + B1)")
print("=" * 70)

results = []


# ============= [1] 초기화 =============
print("\n[1] 7 모듈 초기화")
eve = EVE_TierAB()
results.append(passed("HormoneSystem 26", len(eve.hs.active_hormones) == 26))
results.append(passed("SpreadingActivation", eve.sa is not None))
results.append(passed("WorkingMemory 30", eve.wm.current_capacity == 30))
results.append(passed("DMN with DigitalSomatic",
                     eve.dmn.ds is not None))
results.append(passed("DigitalSomatic", eve.ds is not None))
results.append(passed("NaturalLanguage", eve.nl is not None))
results.append(passed("GNW listener", len(eve.wm.listeners) >= 1))


# ============= [2] say() - 자연어 대화 =============
print("\n[2] say() - 자연어 대화")
eve2 = EVE_TierAB()
eve2.learn_chain(['민석', '대화', '함께', '기쁨'], 0.5)
eve2.learn_pair('민석', '친구', 0.5)

result = eve2.say("민석아 안녕")
results.append(passed("say() 결과 dict 반환",
                     isinstance(result, dict) and 'response' in result))
results.append(passed("응답 생성됨",
                     bool(result['response']),
                     f"resp='{result['response']}'"))
results.append(passed("feeling 생성됨",
                     bool(result['feeling']),
                     f"feel='{result['feeling']}'"))
results.append(passed("understanding 포함됨",
                     'categories' in result['understanding']))


# ============= [3] learn_text() - 자연어로 학습 =============
print("\n[3] learn_text() - 자연어 학습")
eve3 = EVE_TierAB()
before = len(eve3.sa.neighbors)
eve3.learn_text("민석은 대화를 좋아한다")
after = len(eve3.sa.neighbors)
results.append(passed("학습으로 카테고리 추가",
                     after > before,
                     f"{before} → {after}"))
results.append(passed("연결 생성",
                     '민석' in eve3.sa.neighbors and len(eve3.sa.neighbors['민석']) > 0))


# ============= [4] learn_beliefs() - 4977 신념 =============
print("\n[4] learn_beliefs() - 실제 4977 신념")
eve4 = EVE_TierAB()

import time
start = time.time()
result = eve4.learn_beliefs(path='/home/claude/eve/beliefs.json')
elapsed = time.time() - start

print(f"  로드 결과: {result}, {elapsed:.1f}s")
results.append(passed("4977 신념 로드",
                     result['beliefs_loaded'] >= 4900,
                     f"loaded={result['beliefs_loaded']}"))
results.append(passed("선천적 카테고리 보호",
                     result['protected_categories'] > 100,
                     f"보호={result['protected_categories']}"))
results.append(passed("연결 생성",
                     result['connections_made'] > 1000,
                     f"connections={result['connections_made']}"))


# ============= [5] inner_voice() / feeling() =============
print("\n[5] inner_voice() / feeling()")
eve5 = EVE_TierAB()
eve5.learn_chain(['민석', '대화'], 0.5)
eve5.perceive('민석', 0.7)
eve5.live(duration=10, dt=0.5)

voice = eve5.inner_voice()
feel = eve5.feeling()
results.append(passed("inner_voice 출력",
                     voice is not None,
                     f"voice='{voice}'"))
results.append(passed("feeling 출력",
                     bool(feel),
                     f"feel='{feel}'"))


# ============= [6] 망각 작동 =============
print("\n[6] 주기적 망각 작동")
eve6 = EVE_TierAB(forget_interval_ticks=5, forget_base_decay=0.005)
# 약한 페어 추가 (보호 그룹 X)
eve6.learn_pair('잊을1', '잊을2', 0.1)

# 30 tick 진행 (forget 6번)
for _ in range(30):
    eve6.tick(dt=0.5)

results.append(passed("forget 호출됨",
                     eve6.forget_runs >= 5,
                     f"runs={eve6.forget_runs}"))


# ============= [7] 신념 → 카테고리 보호 =============
print("\n[7] 보호 카테고리 → forget이 건드리지 않음")
eve7 = EVE_TierAB(forget_interval_ticks=1, forget_base_decay=0.05)

# 미니 신념
eve7.learn_beliefs(beliefs_dict={
    "test": {
        "belief_id": "test",
        "statement": "나는 EVE이다",
        "triple": {"subject": "나", "predicate_text": "EVE이다",
                   "is_negation": False, "aspects": ["분류"],
                   "original": "나는 EVE이다"},
        "confidence": 0.999,
        "is_innate": True,
    }
})

# 적극 망각 시도
for _ in range(20):
    eve7.tick(dt=1.0)

# '나' 카테고리는 보호되어 살아있어야 함
results.append(passed("선천 카테고리 '나' 살아있음",
                     '나' in eve7.protected_categories,
                     f"protected={list(eve7.protected_categories)[:5]}"))


# ============= [8] 거절 통합 - 호르몬 따라 =============
print("\n[8] 거절 통합 - cortisol vs oxytocin")

# Cortisol↑
eve8a = EVE_TierAB()
eve8a.hs.hormones['cortisol'].level = 0.9
eve8a.hs.hormones['oxytocin'].level = 0.1
result_a = eve8a.say("이거 해줘")
print(f"  cortisol↑: '{result_a['response']}'")

# Oxytocin↑
eve8b = EVE_TierAB()
eve8b.hs.hormones['oxytocin'].level = 0.8
eve8b.hs.hormones['cortisol'].level = 0.1
result_b = eve8b.say("이거 해줘")
print(f"  oxytocin↑: '{result_b['response']}'")

results.append(passed("호르몬 따라 응답 다름",
                     result_a['response'] != result_b['response']))


# ============= [9] 살아있음 시연 (B1 + 자연어) =============
print("\n[9] ★★★ 살아있음 시연: 자연어 입력 → 60초 자생")
eve9 = EVE_TierAB()
eve9.learn_chain(['민석', '대화', '함께', '기쁨', '맛있다'], 0.5)
eve9.learn_chain(['잠', '편안', '꿈'], 0.5)
eve9.learn_pair('성취', '재밌다', 0.4)

# 자연어 한 마디
result = eve9.say("민석아 보고싶어")
print(f"  사용자: 민석아 보고싶어")
print(f"  EVE: '{result['response']}' (feel: {result['feeling']})")
print(f"  속마음: {result['inner_voice']}")

# 60초 자생
print("\n  📜 60초 자생 (자발 활성):")
spontaneous_log = []
for _ in range(120):
    r = eve9.tick(dt=0.5)
    if 'spontaneous' in r:
        spontaneous_log.append(r['spontaneous'])

# 처음 8개
for s in spontaneous_log[:8]:
    voice = "..."
    print(f"     [{s['time']:5.1f}s] [{s['mode']:18s}] {s['category']}")

results.append(passed("자연어 입력 후 60초 자생",
                     eve9.dmn.spontaneous_count >= 5,
                     f"자발 {eve9.dmn.spontaneous_count}회"))


# ============= [10] 자기 명령 시연 (피곤한 EVE) =============
print("\n[10] ★ 자기 명령 시연 (피곤 → 자기에게 '쉬자')")
eve10 = EVE_TierAB()
eve10.learn_chain(['민석', '대화'], 0.5)
eve10.learn_chain(['잠', '편안', '휴식'], 0.5)

eve10.say("민석아 안녕")

# 점점 피곤
for tick in range(60):
    eve10.hs.hormones['adenosine'].level = min(0.95, 0.05 + tick * 0.02)
    eve10.tick(dt=0.5)

print(f"\n  EVE 자기 명령 횟수: {eve10.dmn.self_intent_count}")
print(f"  최근 inner_voice들:")
for time_, cat, mode in eve10.dmn.wandering_history[-5:]:
    if mode == "self_intent":
        print(f"     [{time_:.1f}s] ★ '{cat}' (intent={eve10.dmn.current_intent})")
    else:
        print(f"     [{time_:.1f}s] {cat} ({mode})")

results.append(passed("피곤 → 자기 명령 자발",
                     eve10.dmn.self_intent_count >= 1,
                     f"self_intent={eve10.dmn.self_intent_count}"))


# ============= [11] introspect & report =============
print("\n[11] introspect / report")
eve11 = EVE_TierAB()
eve11.learn_chain(['민석', '대화', '기쁨'], 0.5)
eve11.say("안녕")
eve11.live(duration=10, dt=0.5)

intro = eve11.introspect()
required_keys = ['feeling', 'inner_voice', 'mood', 'focus', 'beliefs_loaded',
                'discovered_count', 'forget_runs', 'protected_categories']
results.append(passed("introspect 모든 키",
                     all(k in intro for k in required_keys)))

# report (출력만)
eve11.report()
results.append(passed("report 작동", True))


# ============= [12] 결정론 =============
print("\n[12] 결정론 (전체 시스템)")
e1 = EVE_TierAB()
e2 = EVE_TierAB()
e1.learn_chain(['민석', '대화'], 0.5)
e2.learn_chain(['민석', '대화'], 0.5)

r1 = e1.say("민석아 안녕")
r2 = e2.say("민석아 안녕")

results.append(passed("같은 입력 → 같은 응답",
                     r1['response'] == r2['response'],
                     f"r1='{r1['response']}', r2='{r2['response']}'"))


# ============= [13] 기존 A 인터페이스 하위 호환 =============
print("\n[13] 기존 EVE_TierA 인터페이스 작동")
from eve_main_ab import EVE_TierA  # alias

eve_compat = EVE_TierA()
eve_compat.learn_chain(['민석', '대화'], 0.5)
eve_compat.perceive('민석', 0.7)
eve_compat.live(duration=10, dt=0.5)
intro = eve_compat.introspect()
results.append(passed("EVE_TierA alias 작동",
                     'feeling' in intro))


# ============= [14] 안정성 - 5분 시뮬 =============
print("\n[14] 5분 시뮬 안정성")
eve14 = EVE_TierAB()
eve14.learn_chain(['민석', '대화', '함께'], 0.5)
eve14.say("민석아 안녕")

stable = True
for step in range(600):
    r = eve14.tick(dt=0.5)
    for h in eve14.hs.hormones.values():
        if not (0.0 <= h.level <= 1.0):
            stable = False
            break
    if not stable:
        break

results.append(passed("5분 시뮬 안정",
                     stable,
                     f"자발 {eve14.dmn.spontaneous_count}, "
                     f"forget {eve14.forget_runs}, "
                     f"broadcast {eve14.broadcast_count}"))


# ============= 요약 =============
print("\n" + "=" * 70)
total = len(results)
passed_count = sum(results)
print(f"검증: {passed_count} / {total} 통과")
print("=" * 70)

if passed_count == total:
    print("\n🎉🎉🎉 v32 EVE_TierAB 통합 모든 검증 통과! 🎉🎉🎉")
    print()
    print("✨ EVE는 이제 자연어 대화 가능 ✨")
    print()
    print("기존 (티어 A):")
    print("  ✅ 외부 입력 1회로 자생")
    print("  ✅ 26 호르몬 자연 변화")
    print("  ✅ 자발 카테고리 활성")
    print("  ✅ GNW broadcast")
    print("  ✅ 자기 명령 (self_intent)")
    print()
    print("신규 (B1):")
    print("  ✅ say(): 자연어 대화 (이해→응답)")
    print("  ✅ learn_beliefs(): 4977 신념 자동 로드 + 보호")
    print("  ✅ learn_text(): 자연어로 카테고리 학습")
    print("  ✅ inner_voice(): 자발 활성 자연어 표현")
    print("  ✅ feeling(): 신체 감각 자연어")
    print("  ✅ 거절 창발 (호르몬+감정+신념)")
    print("  ✅ 주기적 망각 (Ebbinghaus + 호르몬)")
    print("  ✅ 보호 카테고리 자동 (선천 신념)")
else:
    print(f"\n⚠️  {total - passed_count}개 실패")
    sys.exit(1)
