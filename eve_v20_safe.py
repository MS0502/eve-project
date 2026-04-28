"""
EVE v20 SAFE - 모든 패치 안전하게
============================================
- add_belief 원복 (무한 재귀 X)
- smart_chat 새로 (ID 충돌 X)
- chat 작동
- v18 + v19 다 통합 (안전 버전)

사용:
exec(open('/content/drive/MyDrive/eve_v20_safe.py').read())

# 그 다음:
eve.chat("강아지는 동물")
eve.learn_batch(["문장1", "문장2"])
eve.show_status()
"""

import time, math
import numpy as np
from collections import defaultdict, deque, Counter

print("=" * 60)
print("EVE v20 SAFE")
print("=" * 60)

# 1. add_belief 원복 (무한 재귀 방지)
if hasattr(eve, '_direct_add_belief'):
    eve.add_belief = eve._direct_add_belief
    print("✅ add_belief = 클래스 원본 (재귀 X)")

# 2. 카운터 (ID 충돌 방지)
eve._chat_counter = 0
eve._learn_counter = 0

# 3. smart_chat 새로
def safe_smart_chat(t):
    eve._chat_counter += 1
    bid = 'user_' + str(eve._chat_counter) + '_' + str(int(time.time() * 1000))
    eve.add_belief(bid, '사용자: ' + str(t), confidence=0.7, source='user_chat')
    speech = '?'
    if hasattr(eve, 'long_chat'):
        try:
            speech = eve.long_chat(t)
        except:
            speech = 'EVE: 들었어'
    elif hasattr(eve, 'pure_chat'):
        try:
            speech = eve.pure_chat(t)
        except:
            speech = 'EVE: 들었어'
    return {'speech': speech, 'input': t, 'bid': bid}

eve.smart_chat = safe_smart_chat
print("✅ smart_chat = 안전 (ID 카운터)")

# 4. STDP (안전 버전)
def stdp_update_safe(pre, post, dt_ms):
    if abs(dt_ms) > 20:
        return 0
    if dt_ms > 0:
        delta_w = 0.1 * math.exp(-dt_ms / 20)
    else:
        delta_w = -0.05 * math.exp(dt_ms / 20)
    if pre in eve.knowledge and post in eve.knowledge:
        if eve.knowledge.has_edge(pre, post):
            current = eve.knowledge.edges[pre, post].get('weight', 1.0)
            new_w = max(0.01, min(10.0, current + delta_w))
            eve.knowledge.edges[pre, post]['weight'] = new_w
            eve.knowledge.edges[pre, post]['type'] = 'stdp'
        elif delta_w > 0:
            eve.knowledge.add_edge(pre, post, weight=1.0 + delta_w, type='stdp')
    return delta_w

eve.stdp_update = stdp_update_safe
print("✅ stdp_update")

# 5. 학습 = chat + STDP 자동
def learn(text):
    eve._learn_counter += 1
    bid = 'learn_' + str(eve._learn_counter) + '_' + str(int(time.time() * 1000))
    eve.add_belief(bid, str(text), confidence=0.7, source='taught')
    fired = list(eve.hybrid_brain.recent_concepts)[-10:] if hasattr(eve.hybrid_brain, 'recent_concepts') else []
    words = [w.rstrip('을를이가는도에서로') for w in str(text).split() if w.rstrip('을를이가는도에서로') in eve.knowledge]
    stdp_count = 0
    for i, f in enumerate(fired):
        for w in words[:3]:
            if f != w:
                d = stdp_update_safe(f, w, dt_ms=(i + 1) * 5)
                if d != 0:
                    stdp_count += 1
    return {'bid': bid, 'words_known': len(words), 'stdp_updates': stdp_count}

eve.learn = learn
print("✅ learn (단일 학습)")

def learn_batch(texts):
    return [learn(t) for t in texts]

eve.learn_batch = learn_batch
print("✅ learn_batch (배치 학습)")

# 6. Hopfield (전체 4970)
eve._patterns = {}
for bid, b in eve.beliefs.items():
    words = list(set(b.statement.split() + bid.split('_')))
    eve._patterns[bid] = set(w.rstrip('을를이가는도에서로') for w in words if len(w) >= 1)

def hopfield_recall(partial, threshold=0.2, top=10):
    if isinstance(partial, str):
        parts = set(w.rstrip('을를이가는도에서로') for w in (partial.split() if ' ' in partial else [partial]))
    else:
        parts = set(partial)
    matches = []
    for name, pattern in eve._patterns.items():
        if not pattern:
            continue
        overlap = len(parts & pattern) / max(1, min(len(parts), len(pattern)))
        if overlap >= threshold:
            matches.append((name, round(overlap, 2)))
    matches.sort(key=lambda x: -x[1])
    return matches[:top]

eve.hopfield_recall = hopfield_recall
print('✅ Hopfield ' + str(len(eve._patterns)) + ' 패턴')

# 7. 호르몬 칵테일 15
eve._cocktails = {
    '기쁨': {'dopamine': 0.8, 'serotonin': 0.7},
    '슬픔': {'cortisol': 0.6, 'dopamine': 0.2},
    '분노': {'cortisol': 0.8, 'norepinephrine': 0.9},
    '두려움': {'cortisol': 0.9, 'norepinephrine': 0.8},
    '놀람': {'norepinephrine': 0.7, 'dopamine': 0.5},
    '혐오': {'cortisol': 0.6},
    '신뢰': {'oxytocin': 0.8},
    '예측': {'dopamine': 0.6},
    '사랑': {'oxytocin': 0.9, 'dopamine': 0.7},
    '평온': {'gaba': 0.7, 'serotonin': 0.6},
    '호기심': {'dopamine': 0.7, 'norepinephrine': 0.5},
    '몰입': {'dopamine': 0.8},
    '수면': {'melatonin': 0.9, 'gaba': 0.8},
    '각성': {'cortisol': 0.5, 'norepinephrine': 0.7},
    '유대감': {'oxytocin': 0.9, 'serotonin': 0.7},
}

def get_cocktail():
    if not hasattr(eve, 'full_nt'):
        return 'X'
    best = None
    score = -1
    for name, profile in eve._cocktails.items():
        s = sum(1 - abs(getattr(eve.full_nt, k, 0.5) - v) for k, v in profile.items()) / len(profile)
        if s > score:
            score = s
            best = name
    return {'state': best, 'score': round(score, 2)}

def trigger_cocktail(name):
    if name not in eve._cocktails:
        return 'X'
    for k, v in eve._cocktails[name].items():
        if hasattr(eve.full_nt, k):
            setattr(eve.full_nt, k, v)
    return name + ' 발동'

eve.get_cocktail = get_cocktail
eve.trigger_cocktail = trigger_cocktail
print('✅ 호르몬 칵테일 ' + str(len(eve._cocktails)))

# 8. 동의어
eve._synonyms = {
    '기쁨': ['행복', '즐거움', '환희'],
    '슬픔': ['우울', '비통', '서글픔'],
    '분노': ['화남', '격노', '울분'],
    '두려움': ['공포', '무서움', '겁'],
    '사랑': ['애정', '연모', '좋아함'],
    '강아지': ['개', '멍멍이', '댕댕이'],
    '고양이': ['냥이', '고양', '냐옹이'],
    '엄마': ['어머니', '모친', '맘'],
    '아빠': ['아버지', '부친', '파파'],
    '집': ['주거', '가옥', '댁'],
    '학교': ['학원', '교실'],
    '먹다': ['섭취', '식사', '드시다'],
    '말하다': ['이야기', '대화', '발언'],
    '보다': ['시각', '관찰'],
    '좋다': ['훌륭', '괜찮', '멋지'],
    '나쁘다': ['불쾌', '싫다'],
    'EVE': ['이브', '이브이'],
    '김민석': ['민석', '창조자'],
}

def find_synonyms(word):
    if word in eve._synonyms:
        return eve._synonyms[word]
    for key, syns in eve._synonyms.items():
        if word in syns:
            return [key] + [s for s in syns if s != word]
    return []

eve.find_synonyms = find_synonyms
print('✅ 동의어 ' + str(len(eve._synonyms)))

# 9. co-occurrence
eve._cooccur = defaultdict(Counter)

def record_cooccur(text):
    words = str(text).split()
    known = [w.rstrip('을를이가는도에서로') for w in words if w.rstrip('을를이가는도에서로') in eve.knowledge]
    for i, w1 in enumerate(known):
        for w2 in known[i+1:]:
            if w1 != w2:
                eve._cooccur[w1][w2] += 1
                eve._cooccur[w2][w1] += 1

eve.record_cooccur = record_cooccur

def cooccur_score(w1, w2):
    return eve._cooccur.get(w1, {}).get(w2, 0)

eve.cooccur_score = cooccur_score
print('✅ co-occurrence')

# 10. 청각 (주파수)
eve._sound_freq = {
    '동물소리': {'멍멍': (200, 800), '야옹': (300, 1500), '음매': (100, 400), '꼬꼬': (1000, 3000), '꽥꽥': (400, 1200), '짹짹': (2000, 8000)},
    '자연소리': {'바람': (100, 1000), '비': (500, 5000), '천둥': (20, 200), '파도': (50, 500)},
    '인간소리': {'웃음': (300, 3000), '울음': (200, 2000), '말': (85, 1100), '노래': (80, 1100)},
    '기계소리': {'엔진': (50, 500), '시계': (1000, 5000), '벨': (500, 3000)},
    '음악': {'피아노': (27, 4186), '기타': (82, 1318), '드럼': (50, 5000)},
    '충격음': {'쾅': (20, 200), '쿵': (30, 150), '탕': (200, 1500)},
    '생활음': {'문': (100, 2000), '발소리': (50, 800), '박수': (500, 4000)},
}
eve._heard_sounds = deque(maxlen=100)

def hear(sound_text):
    cat = '미지의 소리'
    matched = None
    freq = (500, 2000)
    for category, sounds in eve._sound_freq.items():
        for name, f in sounds.items():
            if name in str(sound_text):
                cat = category
                matched = name
                freq = f
                break
        if matched:
            break
    eve._heard_sounds.append({'time': time.time(), 'sound': sound_text, 'category': cat, 'matched': matched, 'freq': freq})
    bid = 'heard_' + cat + '_' + str(int(time.time() * 1000))
    eve.add_belief(bid, cat + ' (' + str(matched or sound_text) + ', ' + str(int((freq[0]+freq[1])/2)) + 'Hz)', confidence=0.7, source='auditory')
    if hasattr(eve, 'full_nt'):
        if cat == '동물소리':
            eve.full_nt.dopamine = min(1.0, eve.full_nt.dopamine + 0.05)
        elif cat == '충격음':
            eve.full_nt.cortisol = min(1.0, eve.full_nt.cortisol + 0.1)
        elif cat == '음악':
            eve.full_nt.serotonin = min(1.0, eve.full_nt.serotonin + 0.05)
    return {'sound': sound_text, 'matched': matched, 'category': cat, 'freq_Hz': freq, 'belief': bid}

eve.hear = hear
print('✅ 청각 (주파수)')

# 11. 행동 결정 (SNN)
def neural_decide():
    fired = list(eve.hybrid_brain.recent_concepts)[-5:] if hasattr(eve.hybrid_brain, 'recent_concepts') else []
    if not fired:
        return ('idle', None, '활성 X')
    main = fired[0]
    cor = getattr(eve.full_nt, 'cortisol', 0.3) if hasattr(eve, 'full_nt') else 0.3
    dop = getattr(eve.full_nt, 'dopamine', 0.5) if hasattr(eve, 'full_nt') else 0.5
    if any(d in main for d in ['죽음', '위험', '아픔']):
        return ('avoid', main, '위험 회피')
    if any(s in main for s in ['엄마', '아빠', '친구']):
        return ('approach', main, '사회적')
    if any(f in main for f in ['밥', '음식', '먹']):
        return ('PickupObject', main, '음식')
    if cor > 0.7:
        return ('avoid', main, '스트레스')
    if dop > 0.6:
        return ('explore', main, '탐험')
    return ('MoveAhead', None, '기본')

eve.neural_decide = neural_decide
print('✅ neural_decide')

# 12. 환각 차단
def check_hallucination(text):
    words = str(text).split()
    known = sum(1 for w in words if w.rstrip('을를이가는도에서로') in eve.knowledge or any(w in bid for bid in eve.beliefs))
    ratio = known / max(1, len(words))
    return {'hallucination': ratio < 0.3, 'ratio': round(ratio, 2), 'known': known, 'total': len(words)}

eve.check_hallucination = check_hallucination
print('✅ check_hallucination')

# 13. 상태 표시
def show_status():
    print('=' * 50)
    print('EVE 상태')
    print('=' * 50)
    print('  노드: ' + str(eve.knowledge.number_of_nodes()))
    print('  신념: ' + str(len(eve.beliefs)))
    print('  SNN running: ' + str(eve.hybrid_brain.running))
    print('  tick: ' + str(eve.hybrid_brain.tick_count))
    print('  thoughts: ' + str(eve.hybrid_brain.thoughts_triggered))
    print('  recent: ' + str(list(eve.hybrid_brain.recent_concepts)[-5:]))
    print('  패턴: ' + str(len(eve._patterns)))
    print('  co-occur: ' + str(len(eve._cooccur)) + ' 단어')
    print('  들은 소리: ' + str(len(eve._heard_sounds)))
    print('  학습 카운터: ' + str(eve._learn_counter))
    print('  대화 카운터: ' + str(eve._chat_counter))
    print('  현재 호르몬: ' + str(get_cocktail()))

eve.show_status = show_status

# 결과
print()
print('=' * 60)
print('✅ EVE v20 SAFE 부착 완료')
print('=' * 60)
print()
print('사용:')
print('  eve.chat("강아지는 동물")        # 대화 (자동 학습)')
print('  eve.learn("새 지식")              # 단일 학습 + STDP')
print('  eve.learn_batch([t1, t2, ...])   # 배치')
print('  eve.hopfield_recall("강아지")    # 패턴')
print('  eve.hear("멍멍")                  # 청각')
print('  eve.find_synonyms("기쁨")        # 동의어')
print('  eve.cooccur_score("a", "b")     # 동시 출현')
print('  eve.get_cocktail()               # 현재 감정')
print('  eve.trigger_cocktail("호기심")   # 감정 발동')
print('  eve.neural_decide()              # 행동 결정')
print('  eve.check_hallucination(text)   # 환각 차단')
print('  eve.show_status()                # 전체 상태')
