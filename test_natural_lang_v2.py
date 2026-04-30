"""
v2 NaturalLanguage 검증
========================
v32 → v2 변경점만 검증 (기존 34는 test_natural_lang.py 별도)

검증:
A. [12] sentiment 단어 풀 확장 - 더 많은 입력이 정확히 분류
B. [11] respond echo 풀림 - 입력 카테고리 그대로 반환 X
C. 호르몬/sentiment 다양화 - 같은 의도여도 호르몬 따라 응답 다름
D. 단순 호명 인사 처리
E. 결정론 유지
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from natural_lang import NaturalLanguage
from hormone_system import HormoneSystem
from spreading_activation import SpreadingActivation
from working_memory import WorkingMemory


def passed(name, ok, detail=""):
    mark = "✅" if ok else "❌"
    print(f"  {mark} {name}" + (f"  | {detail}" if detail else ""))
    return ok


def make_nl():
    hs = HormoneSystem(phase=4, developmental_stage="adult")
    sa = SpreadingActivation(base_threshold=0.3)
    wm = WorkingMemory()
    return NaturalLanguage(sa, wm, hs), hs, sa, wm


print("=" * 70)
print("v2 NaturalLanguage 검증 ([11] echo + [12] sentiment)")
print("=" * 70)

results = []

# ============= [1] sentiment 단어 풀 확장 =============
print("\n[1] [12] sentiment 단어 풀 - 다양한 긍정/부정 인식")
nl, hs, sa, wm = make_nl()

positive_tests = [
    ('오늘 너무 즐거웠어', 'positive'),
    ('마음이 따뜻해', 'positive'),
    ('완벽한 하루', 'positive'),
    ('정말 신기하네', 'positive'),
    ('만족스러워', 'positive'),
    ('와 멋지다', 'positive'),
    ('고마워', 'positive'),
    ('기대된다', 'positive'),
]
for text, expected in positive_tests:
    u = nl.understand(text)
    ok = u['sentiment'] == expected
    results.append(passed(f"긍정 인식: '{text}'", ok, f"sent={u['sentiment']}"))

negative_tests = [
    ('우울해', 'negative'),
    ('너무 외로워', 'negative'),
    ('걱정돼', 'negative'),
    ('답답하다', 'negative'),
    ('지친다', 'negative'),
    ('미안해', 'negative'),
    ('후회된다', 'negative'),
    ('피곤해', 'negative'),
]
for text, expected in negative_tests:
    u = nl.understand(text)
    ok = u['sentiment'] == expected
    results.append(passed(f"부정 인식: '{text}'", ok, f"sent={u['sentiment']}"))


# ============= [2] [11] respond echo 풀림 =============
print("\n[2] [11] respond echo 풀림 (입력 카테고리 그대로 반환 X)")
nl2, hs2, sa2, wm2 = make_nl()

# 입력 카테고리들이 응답에 그대로 나타나면 안 됨
test_inputs = [
    ('오늘 날씨 좋다', '오늘'),
    ('영화 봤어', '영화'),
    ('운동 했어', '운동'),
    ('밥 먹었어', '밥'),
]
for text, key_cat in test_inputs:
    # WM에 input 카테고리 채움 (실제 say() 로직 모방)
    u = nl2.understand(text)
    for c in u['categories']:
        sa2.activate(c, 0.5)
        wm2.add(c, 0.5)
    resp = nl2.respond(u)
    # 응답이 input 카테고리 그냥 나열한 형태면 echo
    cats_str = ', '.join(sorted(u['categories'])[:3])
    is_echo = (resp == cats_str) or (resp.startswith(cats_str))
    results.append(passed(
        f"echo 안 됨: '{text}' → '{resp}'",
        not is_echo,
        f"cats={u['categories']}"
    ))


# ============= [3] 단순 호명 → 인사 =============
print("\n[3] 단순 호명/이름 인식 → 인사 응답 (echo X)")
nl3, hs3, sa3, wm3 = make_nl()
hs3.hormones['oxytocin'].level = 0.7  # 우호적

u = nl3.understand('민석')
sa3.activate('민석', 0.5)
wm3.add('민석', 0.5)
resp = nl3.respond(u)
results.append(passed(
    "say('민석') → 인사 응답 ('안녕' 류)",
    '안녕' in resp or '응' in resp,
    f"resp='{resp}'"
))


# ============= [4] 호르몬 다양화 =============
print("\n[4] 같은 입력 + 다른 호르몬 → 다른 응답")

# 명령 - cortisol vs oxytocin
nl_a, hs_a, sa_a, wm_a = make_nl()
hs_a.hormones['cortisol'].level = 0.8
hs_a.hormones['oxytocin'].level = 0.2
u_a = nl_a.understand('이거 해줘')
for c in u_a['categories']:
    sa_a.activate(c, 0.5); wm_a.add(c, 0.5)
resp_a = nl_a.respond(u_a)

nl_b, hs_b, sa_b, wm_b = make_nl()
hs_b.hormones['oxytocin'].level = 0.8
hs_b.hormones['cortisol'].level = 0.2
u_b = nl_b.understand('이거 해줘')
for c in u_b['categories']:
    sa_b.activate(c, 0.5); wm_b.add(c, 0.5)
resp_b = nl_b.respond(u_b)

print(f"  cortisol↑: '{resp_a}'")
print(f"  oxytocin↑: '{resp_b}'")
results.append(passed("명령 응답 호르몬 따라 다름", resp_a != resp_b))

# 감정 입력 + 다양한 호르몬
nl_c, hs_c, sa_c, wm_c = make_nl()
hs_c.hormones['dopamine'].level = 0.7
u_c = nl_c.understand('너무 좋다')
resp_c = nl_c.respond(u_c)

nl_d, hs_d, sa_d, wm_d = make_nl()
u_d = nl_d.understand('너무 좋다')
resp_d = nl_d.respond(u_d)

print(f"  dopamine↑: '{resp_c}'")
print(f"  baseline:  '{resp_d}'")
results.append(passed("긍정 감정 + dopamine↑ 차별화",
                     resp_c != resp_d or '좋' in resp_c,
                     f"da↑='{resp_c}', base='{resp_d}'"))


# ============= [5] 부정 감정 → 공감 (echo X) =============
print("\n[5] 부정 감정 → 공감 응답 (입력 단어 echo X)")
test_neg = [
    '오늘 너무 우울해',
    '걱정돼',
    '외롭다',
]
for text in test_neg:
    nl_n, hs_n, sa_n, wm_n = make_nl()
    hs_n.hormones['oxytocin'].level = 0.6  # 공감
    u_n = nl_n.understand(text)
    for c in u_n['categories']:
        sa_n.activate(c, 0.5); wm_n.add(c, 0.5)
    resp = nl_n.respond(u_n)
    # 공감 키워드 있는지
    has_empathy = any(k in resp for k in ['힘들', '그래', '음', '그랬', '...'])
    results.append(passed(f"'{text}' → 공감 응답", has_empathy, f"resp='{resp}'"))


# ============= [6] 결정론 =============
print("\n[6] 결정론")
def run():
    n, _, sa_, wm_ = make_nl()
    u = n.understand('민석이 오늘 너무 좋다')
    for c in u['categories']:
        sa_.activate(c, 0.5); wm_.add(c, 0.5)
    return n.respond(u)
a = run()
b = run()
results.append(passed("같은 입력 → 같은 응답", a == b, f"resp='{a}'"))

import inspect
src = inspect.getsource(NaturalLanguage)
no_random = 'np.random.' not in src and 'random.choice(' not in src and '.sample(' not in src
results.append(passed("코드: random/sample 호출 없음", no_random))


# ============= 요약 =============
print("\n" + "=" * 70)
total = len(results)
passed_count = sum(results)
print(f"v2 검증: {passed_count} / {total} 통과")
print("=" * 70)

if passed_count == total:
    print("\n🎉 v2 모든 검증 통과!")
    print("→ [12] sentiment 풀 확장 (positive/negative 정확 분류)")
    print("→ [11] respond echo 풀림 (입력 카테고리 그대로 반환 X)")
    print("→ 호르몬/sentiment 다양화 작동")
    print("→ 단순 호명 → 인사")
    print("→ 부정 감정 → 공감")
    print("→ 결정론 유지")
else:
    print(f"\n⚠️  {total - passed_count}개 실패")
    sys.exit(1)
