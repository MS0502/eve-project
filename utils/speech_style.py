"""
speech_style.py — 라운드 45

GPT 짚은 핵심 3개:
1. 설명체 갇힘 → 다양한 말투 풀 (이지/야/같은 거야/그냥 ~)
2. 호르몬 톤 → 발화 변형 (활기/피곤/평온 따라 다른 표현)
3. 길이 조절 → Salience 높으면 짧게

학계:
- Brown & Levinson 1987 — politeness theory (상황별 말투)
- Damasio Somatic Marker — 신체 상태 → 발화 길이/단순성
"""

from typing import Optional


# === 정의 응답 풀 (호르몬별) ===
# {ji} = 이지/지 (받침 따라), {ya} = 이야/야, {rago} = 이라고/라고
DEFINITION_TEMPLATES_BY_STATE = {
    # 활기 (dopamine 높) — 짧고 단호
    "energetic": [
        "{target}? 그거 {def_speech}{ya}!",
        "{target}? {def_speech}, 그거지!",
        "{target}? {def_speech}.",
    ],
    # 피곤 (cortisol 높 + da 낮) — 늘어진 말투
    "tired": [
        "{target}... {def_speech}{ji} 뭐...",
        "{target}? 그냥 {def_speech}{ya}...",
        "{target}? {def_speech}, 그런 거...",
    ],
    # 외로움 (oxytocin 낮) — 부드러움
    "lonely": [
        "{target}? {def_speech}{ya}.",
        "{target}이/가 {def_speech}{ji}.",
        "{target}? 음, {def_speech}.",
    ],
    # 호기심 (dopamine 높 단독)
    "curious": [
        "{target}? 그거 {def_speech} 같은 거야.",
        "{target}? 음, {def_speech}{rago} 보면 돼.",
        "{target}? {def_speech}{ji}.",
    ],
    # 불안 (cortisol 높)
    "anxious": [
        "{target}? 음... {def_speech}{ji}.",
        "{target}? 그게 {def_speech}{ya}.",
    ],
    # 평온 (cortisol 낮)
    "calm": [
        "{target}? {def_speech}{ya}.",
        "{target}? {def_speech}{ji}.",
        "{target}? {def_speech} 같은 거.",
    ],
    # 디폴트 (호르몬 중립)
    "neutral": [
        "{target}? {def_speech}{ji}.",
        "{target}? {def_speech}{ya}.",
        "{target}? {def_speech} 같은 거야.",
        "{target}? {def_speech}{rago} 보면 돼.",
        "그냥 {def_speech}{ya}.",
        "{target}? 음, {def_speech}.",
    ],
}


# === 학습 응답 풀 (호르몬별) ===
LEARN_TEMPLATES_BY_STATE = {
    "energetic": [
        ("아!", "{key_t} {def_speech}!", "기억할게."),
        ("오케이!", "{key_t} {def_speech} 그거.", "알겠어."),
    ],
    "tired": [
        ("음...", "{key_t} {def_speech}...", "기억해둘게..."),
        ("아...", "{key_t} {def_speech}, 그런 거.", "알아둘게."),
    ],
    "curious": [
        ("오?", "{key_t} {def_speech}{quote_end} 거구나.", "재밌네, 기억할게."),
        ("그렇구나!", "{key_t} {def_speech} 그거.", "알아둘게, 더 알려줄래?"),
    ],
    "neutral": [
        ("음...", "{key_t} {def_speech}{quote_end} 거구나.", "기억해둘게."),
        ("아,", "{key_t} {def_speech}.", "알겠어, 그걸로 기억할게."),
        ("오케이.", "{key_t} {def_speech}{quote_end} 말이지.", "음, 기억해둘게."),
        ("그렇구나.", "{key_t} {def_speech}.", "알아둘게."),
    ],
}


def get_emotional_state(engine) -> str:
    """엔진의 호르몬 상태 → 감정 카테고리. _select_state_pool과 동일 룰."""
    if engine is None or not hasattr(engine, "hormone_adapter") or engine.hormone_adapter is None:
        return "neutral"
    h = engine.hormone_adapter.hs.hormones
    cort = h["cortisol"].level if "cortisol" in h else 0.5
    da = h["dopamine"].level if "dopamine" in h else 0.5
    ot = h["oxytocin"].level if "oxytocin" in h else 0.5
    ne = h["norepinephrine"].level if "norepinephrine" in h else 0.5
    mel = h["melatonin"].level if "melatonin" in h else 0.3
    
    if (cort > 0.6 and da < 0.4) or mel > 0.6:
        return "tired"
    if ot < 0.30:
        return "lonely"
    if da > 0.65 and ne > 0.55:
        return "energetic"
    if da > 0.6:
        return "curious"
    if cort > 0.65:
        return "anxious"
    if cort < 0.35 and ot >= 0.30:
        return "calm"
    return "neutral"


def pick_definition_template(state: str, hash_seed: int) -> str:
    """호르몬 상태 + 결정론 hash → 템플릿 선택."""
    pool = DEFINITION_TEMPLATES_BY_STATE.get(state, DEFINITION_TEMPLATES_BY_STATE["neutral"])
    return pool[hash_seed % len(pool)]


def pick_learn_template(state: str, hash_seed: int) -> tuple:
    """학습 응답 — (s1, core, fin) 3-tuple."""
    pool = LEARN_TEMPLATES_BY_STATE.get(state, LEARN_TEMPLATES_BY_STATE["neutral"])
    return pool[hash_seed % len(pool)]


# === 길이 조절 ===

def should_be_short(engine, target: str) -> bool:
    """
    짧게 답해야 하나? (Salience 높음 = 익숙한 거 = 짧게)
    GPT 짚은 핵심: 항상 풀 설명 X.
    """
    if engine is None or not hasattr(engine, "salience_adapter"):
        return False
    sal = engine.salience_adapter.get(target)
    # salience > 1.0 = 사용자가 자주 언급 → 짧게 OK
    return sal > 1.0


# === 키워드 추출 (ConceptMemory related용) ===

# 한국어 조사 — 키워드 추출 시 제거
PARTICLES = ('은', '는', '이', '가', '을', '를', '의', '에', '도', '만',
             '와', '과', '도', '같은', '거', '것', '말')

def extract_keywords(text: str, max_n: int = 5) -> list:
    """
    정의 텍스트 → 핵심 키워드 추출 (간단).
    "운동 도와주는 트레이닝" → ["운동", "트레이닝"]
    """
    if not text:
        return []
    
    # 어절 분리
    eojeols = text.split()
    keywords = []
    
    for e in eojeols:
        if len(e) < 2:
            continue
        # 동사/형용사 활용형 제외 — "는/한/된/하는" 등으로 끝나면 skip
        if e.endswith(("는", "한", "된", "하는", "되는", "하기", "되기")):
            continue
        # 조사 제거
        clean = e
        for p in PARTICLES:
            if clean.endswith(p) and len(clean) > len(p):
                clean = clean[:-len(p)]
                break
        if len(clean) < 2:
            continue
        keywords.append(clean)
    
    # 중복 제거 + 최대 N
    seen = set()
    result = []
    for k in keywords:
        if k not in seen:
            seen.add(k)
            result.append(k)
            if len(result) >= max_n:
                break
    return result
