"""
EVE v18 COMPLETE - 빠진 거 다 추가 + 학습 SNN 기반
==========================================================
v15 + v16 + v17 + 어제 거 다 + 학습 SNN

기존 (절대 X 빠뜨림):
- 19,442 노드 + 4,970+ 신념
- 시냅스 그래프 (780,084 연결)
- 카테고리 (lexical/semantic + auto_*)
- 자아 답 + self_will + why chain
- Mental + Repetition + instant_learn + ctx
- BDI + listen + 자기 종료
- Compositional Semantics + Bayesian
- Metacognition
- AI2-THOR body (눈 + 운동 + 자율)
- 일화 기억, World Model, Counterfactual, Curiosity
- Grid world (어제)
- HybridBrain SNN (Izhikevich 1000 뉴런)

v18 추가 - 다 복원:
1. ✅ Pattern Completion (Hopfield)
2. ✅ 환각 차단기
3. ✅ 호르몬 칵테일 15개
4. ✅ 동의어 인식
5. ✅ co-occurrence
6. ✅ 청각 시스템 (Grid 사운드 카테고리)
7. ✅ 학습 = SNN 발화 패턴 기반 (Hebbian)
8. ✅ 행동 결정 = SNN 발화 패턴 기반
9. ✅ 카테고리 = SNN 활성 + substring 결합

사용:
exec(open('/content/drive/MyDrive/eve_v18_complete.py').read())

eve.hopfield_recall('강아지')
eve.check_hallucination(text)
eve.get_cocktail()
eve.find_synonyms('기쁨')
eve.cooccur_score(w1, w2)
eve.hear(sound_text)
eve.neural_decide()
eve.classify_neural(bid)
"""

import time, random, math
from collections import defaultdict, deque, Counter

print("=" * 60)
print("EVE v18 COMPLETE - 빠진 거 다 + 학습 SNN")
print("=" * 60)

# === 진단 ===
print(f"\n[진단]")
print(f"  노드: {eve.knowledge.number_of_nodes()}")
print(f"  신념: {len(eve.beliefs)}")
print(f"  SNN running: {eve.hybrid_brain.running}")
print(f"  Grid world: {hasattr(eve, 'world')}")
print(f"  카테고리: {len(eve._categories) if hasattr(eve, '_categories') else 'X'}")

# ============================================
# [1] Pattern Completion (Hopfield)
# ============================================
print("\n[1] Hopfield Pattern Completion 추가")

eve._patterns = {}

def hopfield_store(name, pattern_words):
    eve._patterns[name] = set(pattern_words)
    return f"패턴 '{name}' 저장: {len(pattern_words)}개 단어"

def hopfield_recall(partial_input, threshold=0.3):
    if isinstance(partial_input, str):
        partial = set(partial_input.split() if ' ' in partial_input else [partial_input])
    else:
        partial = set(partial_input)
    matches = []
    for name, pattern in eve._patterns.items():
        overlap = len(partial & pattern) / max(1, len(partial))
        if overlap >= threshold:
            matches.append((name, overlap, list(pattern)))
    matches.sort(key=lambda x: -x[1])
    return matches[:5]

eve.hopfield_store = hopfield_store
eve.hopfield_recall = hopfield_recall

# 신념에서 패턴 자동 추출
[hopfield_store(bid.split('_')[0], list(set(b.statement.split() + bid.split('_')))) for bid, b in list(eve.beliefs.items())[:200] if not any(k in bid for k in ['punct_', 'space_', '_명사', '_동사', '_부사', '_형용사', 'auto_', '어미_'])]

print(f"  ✅ 패턴 {len(eve._patterns)}개 저장")

# ============================================
# [2] 환각 차단기
# ============================================
print("\n[2] 환각 차단기 추가")

def check_hallucination(text):
    if not text:
        return {'hallucination': False, 'reason': '빈 텍스트'}
    words = text.split() if isinstance(text, str) else list(text)
    known = sum(1 for w in words if w in eve.knowledge or any(w in bid for bid in eve.beliefs))
    ratio = known / max(1, len(words))
    if ratio < 0.3:
        return {'hallucination': True, 'reason': f'아는 단어 비율 {round(ratio*100)}% (낮음)', 'known': known, 'total': len(words)}
    return {'hallucination': False, 'reason': f'아는 단어 비율 {round(ratio*100)}%', 'known': known, 'total': len(words)}

eve.check_hallucination = check_hallucination
print(f"  ✅ check_hallucination 부착")

# ============================================
# [3] 호르몬 칵테일 15개
# ============================================
print("\n[3] 호르몬 칵테일 15개 추가")

eve._cocktails = {
    '기쁨': {'dopamine': 0.8, 'serotonin': 0.7, 'oxytocin': 0.5},
    '슬픔': {'cortisol': 0.6, 'dopamine': 0.2, 'serotonin': 0.3},
    '분노': {'cortisol': 0.8, 'norepinephrine': 0.9, 'dopamine': 0.4},
    '두려움': {'cortisol': 0.9, 'norepinephrine': 0.8},
    '놀람': {'norepinephrine': 0.7, 'dopamine': 0.5},
    '혐오': {'cortisol': 0.6, 'serotonin': 0.3},
    '신뢰': {'oxytocin': 0.8, 'serotonin': 0.6},
    '예측': {'dopamine': 0.6, 'acetylcholine': 0.7},
    '사랑': {'oxytocin': 0.9, 'dopamine': 0.7, 'serotonin': 0.7},
    '평온': {'gaba': 0.7, 'serotonin': 0.6, 'cortisol': 0.2},
    '호기심': {'dopamine': 0.7, 'norepinephrine': 0.5, 'acetylcholine': 0.6},
    '몰입': {'dopamine': 0.8, 'norepinephrine': 0.6, 'acetylcholine': 0.7},
    '수면': {'melatonin': 0.9, 'gaba': 0.8, 'cortisol': 0.1},
    '각성': {'cortisol': 0.5, 'norepinephrine': 0.7, 'dopamine': 0.5},
    '유대감': {'oxytocin': 0.9, 'serotonin': 0.7, 'dopamine': 0.5},
}

def get_cocktail():
    nt = eve.full_nt if hasattr(eve, 'full_nt') else None
    if not nt:
        return '호르몬 시스템 X'
    best_match = None
    best_score = -1
    for name, profile in eve._cocktails.items():
        score = sum(1 - abs(getattr(nt, k, 0.5) - v) for k, v in profile.items()) / len(profile)
        if score > best_score:
            best_score = score
            best_match = name
    return {'state': best_match, 'score': round(best_score, 2)}

def trigger_cocktail(name):
    if name not in eve._cocktails:
        return f"'{name}' 칵테일 X"
    profile = eve._cocktails[name]
    if hasattr(eve, 'full_nt'):
        for k, v in profile.items():
            if hasattr(eve.full_nt, k):
                setattr(eve.full_nt, k, v)
    return f"'{name}' 칵테일 발동: {profile}"

eve.get_cocktail = get_cocktail
eve.trigger_cocktail = trigger_cocktail
print(f"  ✅ {len(eve._cocktails)}개 칵테일 부착")

# ============================================
# [4] 동의어 인식
# ============================================
print("\n[4] 동의어 시스템 추가")

eve._synonyms = {
    '기쁨': ['행복', '즐거움', '환희', '쾌락'],
    '슬픔': ['우울', '비통', '서글픔', '애통'],
    '분노': ['화남', '격노', '울분', '짜증'],
    '두려움': ['공포', '무서움', '겁', '불안'],
    '사랑': ['애정', '연모', '좋아함'],
    '강아지': ['개', '멍멍이', '댕댕이', '강쥐'],
    '고양이': ['냥이', '고양', '냐옹이'],
    '엄마': ['어머니', '모친', '맘'],
    '아빠': ['아버지', '부친', '파파'],
    '집': ['주거', '가옥', '댁'],
    '학교': ['학원', '교실', '학교'],
    '먹다': ['섭취', '식사', '드시다', '잡수다'],
    '말하다': ['이야기', '대화', '발언', '말씀'],
    '보다': ['시각', '관찰', '쳐다보다'],
    '좋다': ['훌륭', '괜찮', '멋지'],
    '나쁘다': ['안 좋다', '불쾌', '싫다'],
    'EVE': ['이브', '이브이', 'eve'],
    '김민석': ['민석', '창조자', '주인'],
}

def find_synonyms(word):
    if word in eve._synonyms:
        return eve._synonyms[word]
    for key, syns in eve._synonyms.items():
        if word in syns:
            return [key] + [s for s in syns if s != word]
    return []

def add_synonym(word, syn):
    eve._synonyms.setdefault(word, []).append(syn)
    return f"'{word}' ↔ '{syn}' 동의어 등록"

eve.find_synonyms = find_synonyms
eve.add_synonym = add_synonym
print(f"  ✅ {len(eve._synonyms)}개 동의어 그룹")

# ============================================
# [5] co-occurrence
# ============================================
print("\n[5] co-occurrence 시스템 추가")

eve._cooccur = defaultdict(Counter)

def record_cooccur(text):
    words = text.split() if isinstance(text, str) else list(text)
    known_words = [w.rstrip('을를이가는도에서로') for w in words if w.rstrip('을를이가는도에서로') in eve.knowledge]
    for i, w1 in enumerate(known_words):
        for w2 in known_words[i+1:]:
            if w1 != w2:
                eve._cooccur[w1][w2] += 1
                eve._cooccur[w2][w1] += 1

def cooccur_score(w1, w2):
    return eve._cooccur.get(w1, {}).get(w2, 0)

def similar_by_context(word, top=5):
    return [(w, c) for w, c in eve._cooccur.get(word, Counter()).most_common(top)]

eve.record_cooccur = record_cooccur
eve.cooccur_score = cooccur_score
eve.similar_by_context = similar_by_context

# 기존 신념에서 누적
[record_cooccur(b.statement) for b in list(eve.beliefs.values())[:500] if not any(k in b.statement for k in ['명사로', '동사로', '부사로'])]

print(f"  ✅ co-occurrence {len(eve._cooccur)}개 단어")

# ============================================
# [6] 청각 시스템 (Grid 사운드)
# ============================================
print("\n[6] 청각 시스템 추가")

eve._sound_categories = {
    '동물소리': ['멍멍', '야옹', '음매', '꼬꼬', '꽥꽥', '짹짹', '히힝', '으르릉', '캥캥'],
    '자연소리': ['바람', '비', '천둥', '파도', '나뭇잎', '시냇물'],
    '인간소리': ['웃음', '울음', '말소리', '노래', '한숨', '비명'],
    '기계소리': ['엔진', '시계', '벨', '알람', '버튼', '키보드'],
    '음악': ['피아노', '기타', '드럼', '바이올린', '노래'],
    '충격음': ['쾅', '쿵', '탕', '뻥', '와장창'],
    '생활음': ['문 여닫기', '발소리', '숨소리', '손뼉', '박수'],
}

eve._heard_sounds = deque(maxlen=100)

def hear(sound_text):
    cat = None
    for category, sounds in eve._sound_categories.items():
        if any(s in sound_text for s in sounds):
            cat = category
            break
    cat = cat or '미지의 소리'
    
    eve._heard_sounds.append({'time': time.time(), 'sound': sound_text, 'category': cat})
    
    bid = f'heard_{cat}_{int(time.time())}'
    eve.add_belief(bid, f'{cat} 소리: {sound_text}', confidence=0.7, source='auditory')
    
    if hasattr(eve, '_categories'):
        eve._categories.setdefault('auditory', set()).add(bid)
    
    if hasattr(eve, 'full_nt'):
        if cat == '동물소리':
            eve.full_nt.dopamine = min(1.0, eve.full_nt.dopamine + 0.05)
        elif cat == '충격음':
            eve.full_nt.cortisol = min(1.0, eve.full_nt.cortisol + 0.1)
        elif cat == '음악':
            eve.full_nt.serotonin = min(1.0, eve.full_nt.serotonin + 0.05)
    
    return {'sound': sound_text, 'category': cat, 'belief': bid}

eve.hear = hear
print(f"  ✅ {len(eve._sound_categories)}개 사운드 카테고리")

# ============================================
# [7] 학습 = SNN 발화 패턴 기반
# ============================================
print("\n[7] 학습 SNN 기반으로 패치")

original_add_belief = eve.add_belief

def neural_add(bid, statement, confidence=0.5, source='unknown'):
    result = original_add_belief(bid, statement, confidence, source)
    
    fired_concepts = list(eve.hybrid_brain.recent_concepts)[-10:] if hasattr(eve.hybrid_brain, 'recent_concepts') else []
    words = bid.split('_') + statement.split()
    
    for w in words[:5]:
        w_clean = w.rstrip('을를이가는도에서로')
        if w_clean in eve.knowledge:
            for f in fired_concepts:
                if f != w_clean and f in eve.knowledge:
                    if eve.knowledge.has_edge(w_clean, f):
                        eve.knowledge.edges[w_clean, f]['weight'] = eve.knowledge.edges[w_clean, f].get('weight', 1) + 1
                    else:
                        eve.knowledge.add_edge(w_clean, f, weight=1, type='neural_hebbian')
    
    record_cooccur(statement)
    
    if hasattr(eve, '_categories'):
        eve._categories.setdefault('neural_learned', set()).add(bid)
    
    return result

eve.add_belief = neural_add
print(f"  ✅ 학습 = SNN 발화 패턴 + Hebbian 강화 + co-occur")

# ============================================
# [8] 행동 결정 = SNN 기반 (정교화)
# ============================================
print("\n[8] 행동 결정 SNN 기반 정교화")

def neural_decide_v2():
    fired = list(eve.hybrid_brain.recent_concepts)[-5:] if hasattr(eve.hybrid_brain, 'recent_concepts') else []
    if not fired:
        return ('idle', None, '활성 개념 X')
    
    nt = eve.full_nt if hasattr(eve, 'full_nt') else None
    cortisol = getattr(nt, 'cortisol', 0.3) if nt else 0.3
    dopamine = getattr(nt, 'dopamine', 0.5) if nt else 0.5
    
    main = fired[0]
    
    food_words = ['밥', '음식', '먹', '빵', '사과', '물', 'Apple', 'Bread', 'Egg', 'Tomato']
    object_words = ['냉장고', '문', '서랍', '뚜껑', 'Fridge', 'Drawer', 'Cabinet', 'Microwave']
    danger_words = ['죽음', '위험', '아픔', '무서움']
    social_words = ['엄마', '아빠', '친구', '김민석', '사람']
    
    if any(d in main for d in danger_words) or cortisol > 0.7:
        return ('avoid', main, f'{main} 위험 감지 → 회피')
    
    if any(s in main for s in social_words):
        return ('approach_social', main, f'{main} 사회적 자극 → 다가감')
    
    if any(f in main for f in food_words):
        return ('PickupObject', main, f'{main} 음식 인식 → 집기')
    
    if any(o in main for o in object_words):
        return ('OpenObject', main, f'{main} 가능한 객체 → 열기')
    
    if dopamine > 0.6:
        return ('explore', main, f'도파민 ↑ → 탐험')
    
    return ('MoveAhead', None, f'기본 → 이동')

eve.neural_decide = neural_decide_v2
print(f"  ✅ neural_decide_v2 (호르몬 + 발화 패턴 + 카테고리)")

# ============================================
# [9] 카테고리 분류 = SNN + substring 결합
# ============================================
print("\n[9] 카테고리 분류 정교화")

def classify_neural(bid, statement=''):
    if any(k in bid for k in ['_명사', '_동사', '_형용사', '_부사', 'punct_', 'space_', '어미_', '겉표지']):
        return 'lexical'
    if 'intent_' in bid:
        return 'intent'
    if any(k in bid for k in ['heard_', 'sound_']):
        return 'auditory'
    if any(k in bid for k in ['experienced_', 'episodic_', 'embodied_']):
        return 'embodied'
    if any(k in bid for k in ['action_', 'rule_']):
        return 'action_rules'
    if any(k in bid for k in ['q_', 'p_', 'ctx_', 'meta_', 'talk_']):
        return 'meta'
    if '나는_' in bid or 'EVE이다' in bid:
        return 'identity'
    if any(k in bid for k in ['learned_', 'taught_']):
        return 'learned'
    if any(k in bid for k in ['_정의', '_특징', '_분류']):
        return 'semantic_concept'
    
    fired = list(eve.hybrid_brain.recent_concepts)[-5:] if hasattr(eve.hybrid_brain, 'recent_concepts') else []
    bid_words = bid.split('_')
    if any(w in fired for w in bid_words):
        return 'neural_active'
    
    return 'semantic'

eve.classify_neural = classify_neural

# 재분류
eve._categories_v2 = defaultdict(set)
for bid, b in eve.beliefs.items():
    cat = classify_neural(bid, b.statement)
    eve._categories_v2[cat].add(bid)

print(f"  ✅ 재분류 {len(eve._categories_v2)} 카테고리")
for cat, bids in sorted(eve._categories_v2.items(), key=lambda x: -len(x[1])):
    print(f"    {cat}: {len(bids)}")

# ============================================
# [10] smart_chat 통합
# ============================================
print("\n[10] smart_chat v18 통합")

original_smart_v18 = eve.smart_chat

def complete_chat(t):
    record_cooccur(t)
    
    halluc = check_hallucination(t)
    if halluc['hallucination']:
        return {'speech': f"미안, {halluc['reason']}. 더 알려줄래?", 'method': 'hallucination_block', 'detail': halluc}
    
    if any(k in t for k in ['소리', '들려', '귀']):
        recent = list(eve._heard_sounds)[-3:]
        return {'speech': f"최근 들은 소리: {[s['category'] for s in recent] if recent else '없음'}", 'method': 'auditory'}
    
    if '내 기분' in t or '내 감정' in t:
        c = get_cocktail()
        return {'speech': f"지금 기분: {c.get('state', '?')}", 'method': 'cocktail', 'detail': c}
    
    if '비슷한' in t or '동의어' in t:
        for w in t.split():
            syns = find_synonyms(w)
            if syns:
                return {'speech': f"'{w}'의 동의어: {', '.join(syns[:5])}", 'method': 'synonym'}
    
    return original_smart_v18(t)

eve.smart_chat = complete_chat

# 결과
print("\n" + "=" * 60)
print("✅ EVE v18 COMPLETE 부착")
print("=" * 60)
print(f"\n[다 추가됨]:")
print(f"  ✅ Hopfield 패턴: {len(eve._patterns)}")
print(f"  ✅ 환각 차단기")
print(f"  ✅ 호르몬 칵테일: {len(eve._cocktails)}")
print(f"  ✅ 동의어: {len(eve._synonyms)} 그룹")
print(f"  ✅ co-occurrence: {len(eve._cooccur)} 단어")
print(f"  ✅ 청각 카테고리: {len(eve._sound_categories)}")
print(f"  ✅ 학습 = SNN Hebbian")
print(f"  ✅ 행동 = SNN 발화 패턴")
print(f"  ✅ 카테고리 = SNN + 정교화")
print(f"\n[기존 다 유지]:")
print(f"  노드: {eve.knowledge.number_of_nodes()}")
print(f"  신념: {len(eve.beliefs)}")
print(f"  SNN: {eve.hybrid_brain.running}")
print(f"  Grid world: {hasattr(eve, 'world')}")
print(f"\n사용:")
print(f"  eve.hopfield_recall('강아지')")
print(f"  eve.check_hallucination(text)")
print(f"  eve.get_cocktail()")
print(f"  eve.trigger_cocktail('호기심')")
print(f"  eve.find_synonyms('기쁨')")
print(f"  eve.cooccur_score('강아지', '동물')")
print(f"  eve.hear('멍멍')")
print(f"  eve.neural_decide()")
print(f"  eve.classify_neural(bid)")
print(f"  eve.smart_chat('지금 무슨 소리 들렸어?')")
