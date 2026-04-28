"""
EVE v21 REAL - real_snn 직접 활용 + 오늘 발견 통합
================================================
v18~v20 내가 멍청하게 새로 만든 거 다 버림
원본 real_snn 메서드 직접 사용
오늘 발견한 거 통합

원본 활용 (이미 있던 거):
- real_snn.store_pattern (Hopfield 자동)
- real_snn.activate_concept (spreading)
- real_snn.stdp_learn (LTP/LTD 자동)
- real_snn.pattern_complete
- real_snn.trace_activation (multi-hop)
- real_snn.hebbian_strengthen
- real_snn.connect_concepts

오늘 발견:
- 양방향 connect_concepts
- 조사 제거 (strip_p)
- real_learn (자동 store + 양방향 연결)
- true_chat (spreading 자연어)
- 자기 판단 chat
- 청각 통합 (real_snn 자극)
- AI2-THOR 객체 → real_snn 학습
"""

import time, math
import numpy as np
from collections import deque

print('=' * 60)
print('EVE v21 REAL - 원본 활용 + 오늘 발견')
print('=' * 60)

# 검증 - real_snn 있나
print()
print('[검증]')
print('  real_snn:', hasattr(eve, 'real_snn'))
print('  store_pattern:', hasattr(eve.real_snn, 'store_pattern') if hasattr(eve, 'real_snn') else False)
print('  activate_concept:', hasattr(eve.real_snn, 'activate_concept') if hasattr(eve, 'real_snn') else False)

# === 1. 조사 제거 ===
strip_p = lambda w: next((w[:-len(p)] for p in ['이다', '다', '은', '는', '이', '가', '을', '를', '에서', '에게', '에', '의', '로', '도', '만'] if w.endswith(p) and len(w) > len(p)), w)
eve.strip_p = strip_p
print('✅ strip_p (조사 제거)')

# === 2. real_learn (양방향 자동 학습) ===
eve.real_learn = lambda t: ([eve.real_snn.store_pattern(strip_p(w)) for w in t.split() if len(strip_p(w)) >= 1], [(eve.real_snn.connect_concepts(strip_p(w1), strip_p(w2), 0.5), eve.real_snn.connect_concepts(strip_p(w2), strip_p(w1), 0.5)) for i, w1 in enumerate(t.split()) for w2 in t.split()[i+1:] if strip_p(w1) != strip_p(w2) and len(strip_p(w1)) >= 1 and len(strip_p(w2)) >= 1], f'학습: {t}')[2]
print('✅ real_learn (양방향 자동)')

# === 3. true_chat (spreading 추론) ===
eve.true_chat = lambda t: [{'word': strip_p(w), 'spreads': eve.real_snn.trace_activation(strip_p(w))} for w in t.split() if strip_p(w) in eve.real_snn.concept_to_neurons]
print('✅ true_chat (spreading)')

# === 4. 자연어 답 (자기 판단) ===
def natural_speak(t):
    spreads = eve.true_chat(t)
    if not spreads:
        return f"'{t}' 모르는 단어들. 가르쳐줄래?"
    parts = []
    for s in spreads:
        word = s['word']
        related = [r['concept'] for r in s['spreads'][:3]]
        if related:
            parts.append(f"{word}는 {', '.join(related)}와 연결됨")
        else:
            parts.append(f"{word} 알아")
    return '. '.join(parts)

eve.natural_speak = natural_speak
print('✅ natural_speak (자연어 답)')

# === 5. 자기 판단 chat ===
def self_judging_chat(t):
    has_concepts = sum(1 for w in t.split() if strip_p(w) in eve.real_snn.concept_to_neurons)
    total = len(t.split())
    confidence = has_concepts / max(1, total)
    
    if confidence < 0.3:
        return {'speech': f"잘 모르겠어. ({has_concepts}/{total}만 알아) 더 알려줘", 'method': 'unknown', 'confidence': round(confidence, 2)}
    
    if confidence > 0.7:
        eve.real_learn(t)
        spreads = eve.true_chat(t)
        speech = natural_speak(t)
        return {'speech': speech, 'method': 'understood_learned', 'confidence': round(confidence, 2), 'spreads': spreads}
    
    eve.real_learn(t)
    return {'speech': natural_speak(t), 'method': 'partial', 'confidence': round(confidence, 2)}

eve.self_judging_chat = self_judging_chat
print('✅ self_judging_chat (자기 판단)')

# === 6. 청각 → real_snn 통합 ===
sound_freq = {
    '동물소리': {'멍멍': (200, 800), '야옹': (300, 1500), '음매': (100, 400)},
    '자연소리': {'바람': (100, 1000), '비': (500, 5000), '천둥': (20, 200)},
    '인간소리': {'웃음': (300, 3000), '울음': (200, 2000), '말': (85, 1100)},
    '기계소리': {'엔진': (50, 500), '벨': (500, 3000)},
    '음악': {'피아노': (27, 4186), '기타': (82, 1318)},
    '충격음': {'쾅': (20, 200), '쿵': (30, 150)},
}
eve._sound_freq = sound_freq
eve._heard = deque(maxlen=100)

def hear(text):
    cat, matched, freq = '미지', None, (500, 2000)
    for c, sounds in sound_freq.items():
        for name, f in sounds.items():
            if name in text:
                cat, matched, freq = c, name, f
                break
        if matched:
            break
    
    eve.real_snn.store_pattern(cat)
    if matched:
        eve.real_snn.store_pattern(matched)
        eve.real_snn.connect_concepts(matched, cat, 0.7)
        eve.real_snn.connect_concepts(cat, matched, 0.7)
    
    eve._heard.append({'time': time.time(), 'text': text, 'cat': cat, 'freq': freq})
    
    if hasattr(eve, 'full_nt'):
        if cat == '동물소리':
            eve.full_nt.dopamine = min(1.0, eve.full_nt.dopamine + 0.05)
        elif cat == '충격음':
            eve.full_nt.cortisol = min(1.0, eve.full_nt.cortisol + 0.1)
        elif cat == '음악':
            eve.full_nt.serotonin = min(1.0, eve.full_nt.serotonin + 0.05)
    
    return {'cat': cat, 'matched': matched, 'freq_Hz': freq, 'real_snn_neurons': len(eve.real_snn.get_active_neurons(cat))}

eve.hear = hear
print('✅ hear (real_snn 통합)')

# === 7. AI2-THOR 가상 신체 (real_snn 학습) ===
eve._ai2thor_ready = False

def setup_ai2thor():
    try:
        import subprocess, os
        subprocess.run(['pip', 'install', '-q', '--break-system-packages', 'ai2thor'], check=False)
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
        return 'AI2-THOR setup 필요'
    objs = [o['objectType'] for o in eve._thor.last_event.metadata.get('objects', []) if o.get('visible', False)]
    for o in objs:
        eve.real_snn.store_pattern(o)
    return {'visible': objs, 'learned': len(objs)}

eve.thor_look = thor_look

def thor_move(direction='forward'):
    if not eve._ai2thor_ready:
        return 'setup 필요'
    action_map = {'forward': 'MoveAhead', 'back': 'MoveBack', 'left': 'RotateLeft', 'right': 'RotateRight'}
    eve._thor.step(action=action_map.get(direction, 'MoveAhead'))
    return thor_look()

eve.thor_move = thor_move

def thor_explore(steps=10):
    if not eve._ai2thor_ready:
        return 'setup 필요'
    import random
    discovered = set()
    for i in range(steps):
        d = random.choice(['forward', 'forward', 'left', 'right'])
        r = thor_move(d)
        if isinstance(r, dict):
            discovered.update(r.get('visible', []))
    return {'steps': steps, 'discovered': list(discovered), 'real_snn_patterns': len(eve.real_snn.concept_to_neurons)}

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

eve.get_cocktail = lambda: (lambda nt, scores: max(scores, key=scores.get) if scores else 'X')(eve.full_nt if hasattr(eve, 'full_nt') else None, {name: sum(1 - abs(getattr(eve.full_nt, k, 0.5) - v) for k, v in p.items()) / len(p) for name, p in eve._cocktails.items()} if hasattr(eve, 'full_nt') else {})

eve.trigger_cocktail = lambda name: ([setattr(eve.full_nt, k, v) for k, v in eve._cocktails.get(name, {}).items() if hasattr(eve.full_nt, k)] if name in eve._cocktails else None, f'{name} 발동' if name in eve._cocktails else 'X')[1]

print('✅ 호르몬 칵테일 15')

# === 9. 동의어 ===
eve._synonyms = {
    '기쁨': ['행복', '즐거움'], '슬픔': ['우울', '비통'], '분노': ['화남'],
    '두려움': ['공포'], '사랑': ['애정'], '강아지': ['개', '멍멍이'],
    '고양이': ['냥이'], '엄마': ['어머니'], '아빠': ['아버지'],
    '집': ['주거'], '학교': ['교실'], 'EVE': ['이브'],
}

eve.find_synonyms = lambda w: eve._synonyms.get(w, []) or next(([k] + [s for s in v if s != w] for k, v in eve._synonyms.items() if w in v), [])
print('✅ 동의어 12')

# === 10. 환각 차단 (real_snn 활용) ===
eve.check_hallucination = lambda t: (lambda known: {'hallucination': known / max(1, len(t.split())) < 0.3, 'ratio': round(known / max(1, len(t.split())), 2), 'known': known, 'total': len(t.split())})(sum(1 for w in t.split() if strip_p(w) in eve.real_snn.concept_to_neurons))
print('✅ 환각 차단 (real_snn 기반)')

# === 11. 통합 chat - 자기 판단 + 자연어 + 학습 ===
def integrated_chat(t):
    halluc = eve.check_hallucination(t)
    if halluc['hallucination']:
        return {'speech': f"잘 몰라. ({halluc['known']}/{halluc['total']} 단어만 알아) 더 알려줘", 'method': 'unknown', 'detail': halluc}
    
    eve.hear(t)
    eve.real_learn(t)
    spreads = eve.true_chat(t)
    speech = natural_speak(t)
    cocktail = eve.get_cocktail()
    
    return {
        'speech': speech,
        'method': 'integrated',
        'understanding': halluc['ratio'],
        'spreads': spreads,
        'emotion': cocktail,
    }

eve.integrated_chat = integrated_chat
print('✅ integrated_chat (자기 판단 + 자연어 + 학습)')

# === 12. 상태 ===
def show_status():
    print('=' * 50)
    print('EVE v21 상태')
    print('=' * 50)
    print(f'  노드: {eve.knowledge.number_of_nodes()}')
    print(f'  신념: {len(eve.beliefs)}')
    print(f'  real_snn 개념: {len(eve.real_snn.concept_to_neurons)}')
    print(f'  real_snn 시냅스: {len(eve.real_snn.weights) if hasattr(eve.real_snn, "weights") else "X"}')
    print(f'  들은 소리: {len(eve._heard)}')
    print(f'  AI2-THOR: {eve._ai2thor_ready}')
    print(f'  호르몬: {eve.get_cocktail()}')

eve.show_status = show_status

# 결과
print()
print('=' * 60)
print('✅ EVE v21 REAL 부착 완료')
print('=' * 60)
print()
print('사용:')
print('  eve.real_learn("문장")            # 학습 (real_snn 자동)')
print('  eve.true_chat("문장")             # spreading 추론')
print('  eve.natural_speak("문장")         # 자연어 답')
print('  eve.integrated_chat("문장")       # 자기 판단 + 학습 + 답')
print('  eve.hear("멍멍")                  # 청각 (real_snn 자극)')
print('  eve.setup_ai2thor()              # AI2-THOR 시작 (5-10분)')
print('  eve.thor_look()                  # 시각')
print('  eve.thor_explore(10)             # 자율 탐험')
print('  eve.get_cocktail()               # 감정')
print('  eve.find_synonyms("기쁨")        # 동의어')
print('  eve.check_hallucination(t)      # 환각 차단')
print('  eve.show_status()                # 전체 상태')
