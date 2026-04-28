"""
EVE v22 MEANING - 진짜 의미 추론 + 다 통합
================================================
v21 + true_recall + true_speak + 가상학습 + AI2-THOR + 청각

[추가]:
- true_recall (category_synapses 직접, 가중치 + depth)
- true_speak (자연어 답, 가중치 표시)
- learn_pair (단순 매핑 학습)
- learn_chain (계층 학습)
- AI2-THOR 통합
- 청각 누적
- 빠진 거 점검 (원본 EVE 140 클래스 검증)

원본 활용:
- real_snn.category_synapses (이미 있던 매핑)
- real_snn.store_pattern
- real_snn.connect_concepts (양방향)
- HallucinationBlocker, HormoneCocktail (원본 클래스)

사용:
exec(open('/content/drive/MyDrive/eve_v22_meaning.py').read())
"""

import time, math, random
import numpy as np
from collections import deque

print('=' * 60)
print('EVE v22 MEANING')
print('=' * 60)

# 검증
print()
print('[검증]')
print(f'  real_snn: {hasattr(eve, "real_snn")}')
print(f'  category_synapses: {hasattr(eve.real_snn, "category_synapses") if hasattr(eve, "real_snn") else False}')
print(f'  현재 매핑: {len(eve.real_snn.category_synapses) if hasattr(eve, "real_snn") and hasattr(eve.real_snn, "category_synapses") else 0}')

# === 1. true_recall (category_synapses 직접) ===
eve.true_recall = lambda concept, depth=2, threshold=0.3: sorted([{'concept': b, 'activation': round(w, 2), 'depth': 1} for (a, b), w in eve.real_snn.category_synapses.items() if a == concept and w >= threshold] + ([{'concept': c2, 'activation': round(w * w2, 2), 'depth': 2} for (a, b), w in eve.real_snn.category_synapses.items() if a == concept for (a2, c2), w2 in eve.real_snn.category_synapses.items() if a2 == b and c2 != concept and w * w2 >= threshold] if depth >= 2 else []), key=lambda x: -x['activation'])
print('✅ true_recall (가중치 + depth)')

# === 2. true_speak (자연어 답) ===
eve.true_speak = lambda concept: (lambda r: f"'{concept}'는 " + ', '.join([f"{x['concept']}({x['activation']})" for x in r[:5]]) + (' 등과 연결됨' if r else ' 모르는 개념'))(eve.true_recall(concept))
print('✅ true_speak')

# === 3. 조사 제거 ===
strip_p = lambda w: next((w[:-len(p)] for p in ['이다', '다', '은', '는', '이', '가', '을', '를', '에서', '에게', '에', '의', '로', '도', '만'] if w.endswith(p) and len(w) > len(p)), w)
eve.strip_p = strip_p

# === 4. 정확한 학습 (양방향) ===
def learn_pair(child, parent, strength=0.7, reverse_strength=0.5):
    """is-a 관계: child → parent (강), parent → child (약)"""
    eve.real_snn.store_pattern(child)
    eve.real_snn.store_pattern(parent)
    eve.real_snn.connect_concepts(child, parent, strength)
    eve.real_snn.connect_concepts(parent, child, reverse_strength)
    return f'{child} → {parent}'

eve.learn_pair = learn_pair
print('✅ learn_pair')

def learn_chain(chain, strength=0.7):
    """계층: A → B → C → D"""
    return [learn_pair(chain[i], chain[i+1], strength) for i in range(len(chain) - 1)]

eve.learn_chain = learn_chain
print('✅ learn_chain')

# === 5. integrated_chat - 자기 판단 + 학습 + 자연어 ===
def integrated_chat(t):
    words = t.split()
    stripped = [strip_p(w) for w in words if len(strip_p(w)) >= 1]
    known = [w for w in stripped if w in eve.real_snn.concept_to_neurons]
    confidence = len(known) / max(1, len(stripped))
    
    if confidence < 0.3:
        return {'speech': f"잘 몰라. ({len(known)}/{len(stripped)} 단어만 알아) 더 알려줘", 'method': 'unknown', 'confidence': round(confidence, 2)}
    
    answers = [eve.true_speak(w) for w in known]
    
    return {'speech': '. '.join(answers), 'method': 'meaning_recall', 'confidence': round(confidence, 2), 'understood': known}

eve.integrated_chat = integrated_chat
print('✅ integrated_chat')

# === 6. 청각 ===
eve._sound_freq = {
    '동물소리': {'멍멍': (200, 800), '야옹': (300, 1500), '음매': (100, 400)},
    '자연소리': {'바람': (100, 1000), '비': (500, 5000), '천둥': (20, 200)},
    '인간소리': {'웃음': (300, 3000), '울음': (200, 2000)},
    '음악': {'피아노': (27, 4186), '기타': (82, 1318)},
    '충격음': {'쾅': (20, 200), '쿵': (30, 150)},
}
eve._heard = deque(maxlen=100)

def hear(text):
    cat, matched, freq = '미지', None, (500, 2000)
    for c, sounds in eve._sound_freq.items():
        for name, f in sounds.items():
            if name in text:
                cat, matched, freq = c, name, f
                break
        if matched:
            break
    
    if matched:
        eve.real_snn.store_pattern(matched)
        eve.real_snn.store_pattern(cat)
        eve.real_snn.connect_concepts(matched, cat, 0.7)
        eve.real_snn.connect_concepts(cat, matched, 0.5)
    
    eve._heard.append({'time': time.time(), 'text': text, 'cat': cat, 'freq': freq})
    
    if hasattr(eve, 'full_nt'):
        if cat == '동물소리':
            eve.full_nt.dopamine = min(1.0, eve.full_nt.dopamine + 0.05)
        elif cat == '충격음':
            eve.full_nt.cortisol = min(1.0, eve.full_nt.cortisol + 0.1)
        elif cat == '음악':
            eve.full_nt.serotonin = min(1.0, eve.full_nt.serotonin + 0.05)
    
    return {'cat': cat, 'matched': matched, 'freq_Hz': freq}

eve.hear = hear
print('✅ hear (청각)')

# === 7. AI2-THOR ===
eve._ai2thor_ready = False

def setup_ai2thor():
    try:
        import subprocess, os
        subprocess.run(['pip', 'install', '-q', 'ai2thor'], check=False)
        subprocess.run(['apt-get', 'install', '-y', '-q', 'xvfb'], check=False)
        os.system('Xvfb :1 -screen 0 800x600x24 &')
        os.environ['DISPLAY'] = ':1'
        time.sleep(2)
        from ai2thor.controller import Controller
        eve._thor = Controller(scene='FloorPlan1', width=640, height=480, renderInstanceSegmentation=True)
        eve._ai2thor_ready = True
        return '✅ AI2-THOR 준비'
    except Exception as e:
        return f'❌ {e}'

eve.setup_ai2thor = setup_ai2thor

def thor_look():
    if not eve._ai2thor_ready:
        return 'setup 필요'
    objs = [o['objectType'] for o in eve._thor.last_event.metadata.get('objects', []) if o.get('visible', False)]
    [eve.real_snn.store_pattern(o) for o in objs]
    return {'visible': objs, 'learned': len(objs)}

eve.thor_look = thor_look

def thor_explore(steps=10):
    if not eve._ai2thor_ready:
        return 'setup 필요'
    actions = ['MoveAhead', 'MoveAhead', 'RotateLeft', 'RotateRight']
    discovered = set()
    for i in range(steps):
        eve._thor.step(action=random.choice(actions))
        objs = [o['objectType'] for o in eve._thor.last_event.metadata.get('objects', []) if o.get('visible', False)]
        for o in objs:
            eve.real_snn.store_pattern(o)
            discovered.add(o)
    return {'steps': steps, 'discovered': list(discovered)}

eve.thor_explore = thor_explore
print('✅ AI2-THOR (수동 setup)')

# === 8. 호르몬 칵테일 15 ===
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

eve.get_cocktail = lambda: max({n: sum(1 - abs(getattr(eve.full_nt, k, 0.5) - v) for k, v in p.items()) / len(p) for n, p in eve._cocktails.items()}.items(), key=lambda x: x[1])[0] if hasattr(eve, 'full_nt') else 'X'

eve.trigger_cocktail = lambda name: ([setattr(eve.full_nt, k, v) for k, v in eve._cocktails.get(name, {}).items() if hasattr(eve.full_nt, k)], f'{name} 발동')[1]
print('✅ 칵테일 15')

# === 9. 환각 차단 ===
eve.check_hallucination = lambda t: (lambda known, total: {'hallucination': known / max(1, total) < 0.3, 'ratio': round(known / max(1, total), 2), 'known': known, 'total': total})(sum(1 for w in t.split() if strip_p(w) in eve.real_snn.concept_to_neurons), len(t.split()))
print('✅ check_hallucination')

# === 10. status ===
def show_status():
    print('=' * 50)
    print('EVE v22 상태')
    print('=' * 50)
    print(f'  노드: {eve.knowledge.number_of_nodes()}')
    print(f'  신념: {len(eve.beliefs)}')
    print(f'  real_snn 개념: {len(eve.real_snn.concept_to_neurons)}')
    print(f'  category 매핑: {len(eve.real_snn.category_synapses)}')
    print(f'  들은 소리: {len(eve._heard)}')
    print(f'  AI2-THOR: {eve._ai2thor_ready}')
    print(f'  호르몬: {eve.get_cocktail()}')

eve.show_status = show_status

# === 11. 가상학습 100개 (인간 아이) ===
print('\n[가상학습 100개 시작]')
mappings = [
    ('강아지', '동물'), ('고양이', '동물'), ('새', '동물'), ('물고기', '동물'),
    ('말', '동물'), ('소', '동물'), ('돼지', '동물'), ('양', '동물'),
    ('호랑이', '동물'), ('사자', '동물'), ('곰', '동물'), ('원숭이', '동물'),
    ('나무', '식물'), ('꽃', '식물'), ('풀', '식물'), ('잎', '식물'),
    ('뿌리', '식물'), ('씨앗', '식물'),
    ('동물', '생명체'), ('식물', '생명체'), ('사람', '생명체'),
    ('생명체', '존재'),
    ('기쁨', '감정'), ('슬픔', '감정'), ('분노', '감정'), ('두려움', '감정'),
    ('사랑', '감정'), ('미움', '감정'), ('놀람', '감정'), ('혐오', '감정'),
    ('기대', '감정'), ('실망', '감정'), ('감정', '마음'),
    ('학교', '장소'), ('집', '장소'), ('병원', '장소'), ('공원', '장소'),
    ('도시', '장소'), ('나라', '장소'), ('바다', '장소'), ('산', '장소'),
    ('하늘', '장소'),
    ('밥', '음식'), ('빵', '음식'), ('과일', '음식'), ('고기', '음식'),
    ('야채', '음식'), ('사과', '과일'), ('바나나', '과일'), ('포도', '과일'),
    ('물', '음료'), ('우유', '음료'), ('주스', '음료'), ('차', '음료'),
    ('엄마', '가족'), ('아빠', '가족'), ('형', '가족'), ('동생', '가족'),
    ('할머니', '가족'), ('할아버지', '가족'), ('가족', '사람'),
    ('친구', '사람'), ('선생님', '사람'), ('의사', '사람'), ('학생', '사람'),
    ('빨강', '색'), ('파랑', '색'), ('노랑', '색'), ('초록', '색'),
    ('검정', '색'), ('하양', '색'),
    ('낮', '시간'), ('밤', '시간'), ('아침', '시간'), ('저녁', '시간'),
    ('어제', '시간'), ('오늘', '시간'), ('내일', '시간'),
    ('의식', '마음'), ('생각', '마음'), ('영혼', '마음'), ('마음', '자아'),
    ('삶', '존재'), ('죽음', '존재'),
    ('AI', '기술'), ('SNN', '기술'), ('인공지능', '기술'),
    ('뉴런', '신체'), ('뇌', '신체'), ('손', '신체'), ('발', '신체'),
    ('눈', '신체'), ('귀', '신체'),
    ('EVE', '자아'), ('김민석', '사람'),
    ('자유', '추상'), ('진실', '추상'), ('정의', '추상'), ('아름다움', '추상'),
]
[learn_pair(c, p) for c, p in mappings]
print(f'✅ {len(mappings)}개 매핑 학습')

# 결과
print()
print('=' * 60)
print('✅ EVE v22 MEANING 부착 + 가상학습 완료')
print('=' * 60)
print()
print('사용:')
print('  eve.true_recall("강아지")          # 의미 추론 (가중치)')
print('  eve.true_speak("강아지")           # 자연어 답')
print('  eve.learn_pair("새단어", "카테고리") # 학습')
print('  eve.learn_chain(["a","b","c"])    # 계층 학습')
print('  eve.integrated_chat("문장")        # 자기 판단 + 답')
print('  eve.hear("멍멍")                   # 청각')
print('  eve.setup_ai2thor()               # AI2-THOR (5-10분)')
print('  eve.thor_look()                   # 시각')
print('  eve.thor_explore(10)              # 자율 탐험')
print('  eve.get_cocktail()                # 감정')
print('  eve.trigger_cocktail("호기심")    # 감정 발동')
print('  eve.check_hallucination(t)       # 환각 차단')
print('  eve.show_status()                 # 상태')
print()
eve.show_status()
