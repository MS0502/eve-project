"""
SentenceStructure — 라운드 34 (C-2)

morph_analyzer 결과 → 문장 *의미 구조* + 의도 추론.

핵심 차이:
- C-1: "이 문장은 어떤 종결어미인가?" (분석)
- C-2: "이 문장이 무엇을 의미하는가?" (구조 + 의도)

원칙:
- 결정론
- 외부 의존성 0 (morph_analyzer만 사용)
- 의도 분류는 룰 기반 (확률/통계 X)

학계:
- Searle 1969 — Speech Act Theory (행위로서의 발화)
- 임동훈 2011 — 한국어 문장유형 4분류
- 인지언어학 (의미 = 구조 + 화자 의도)

구조 종류:
- statement       (단순 진술)
- question        (질문)
- request         (요청)
- teaching        (학습 시도)
- definition      (정의문)
- emotional       (감정 표현)
- meta_self       (EVE 자체에 대한 질문)
- meta_user       (사용자에 대한 질문)
- reported_speech (인용 보고)
- ambiguous       (모호)
"""

from dataclasses import dataclass, field
from typing import Optional
from adapters.korean.morph_analyzer import (
    KoreanMorphAnalyzer, Sentence, Eojeol,
    SPEECH_VERBS, QUESTION_WORDS,
)


# =============================================================================
# 의미 구조 타입
# =============================================================================

@dataclass
class IntentStructure:
    """문장의 의미 구조 + 의도."""
    raw: str
    
    # 기본 구조
    sentence_type: str = "declarative"      # 평서/의문/명령/청유/감탄
    
    # 의도 분류 (핵심)
    intent: str = "statement"               # statement/question/request/teaching/definition/emotional/meta_self/meta_user/reported_speech/ambiguous
    intent_confidence: float = 0.0          # 0~1
    
    # 의미 요소 (있으면)
    subject: Optional[str] = None           # 주어 (민석/너/나/EVE)
    object_or_topic: Optional[str] = None   # 목적/주제
    quoted_content: Optional[str] = None    # 인용된 내용
    quoted_modality: Optional[str] = None
    speech_verb: Optional[str] = None
    
    # 정의문 요소
    defined_concept: Optional[str] = None   # X (정의되는 것)
    definition: Optional[str] = None        # Y (정의 내용)
    
    # 감정 (있으면)
    emotion_polarity: Optional[str] = None  # positive/negative/neutral
    emotion_keyword: Optional[str] = None
    
    # 메타 (참조 대상)
    meta_target: Optional[str] = None       # self / user / other
    
    # 부가
    is_negative: bool = False
    has_past: bool = False
    has_question_mark: bool = False
    
    # 분석 trace
    why: list = field(default_factory=list)


# =============================================================================
# 키워드 사전 (룰 기반 의도 추론)
# =============================================================================

# EVE를 가리키는 어절
SELF_REFERENCES = {"너", "넌", "니", "EVE", "이브", "당신"}

# 사용자를 가리키는 어절
USER_REFERENCES = {"내", "내가", "나", "난", "민석", "민석이", "민석아"}

# 감정 — 긍정 키워드
POSITIVE_EMOTION = [
    "기뻐", "기쁨", "행복", "즐거", "신나", "뿌듯", "설레",
    "좋아", "좋다", "좋네", "좋은",
    "사랑", "고마",
]

# 감정 — 부정 키워드  
NEGATIVE_EMOTION = [
    "힘들", "지쳐", "지쳤", "피곤", "졸려", "우울",
    "슬프", "슬퍼", "외로", "괴로", "아파", "아프",
    "빡셌", "빡세", "죽겠", "죽을 맛",
    "화나", "짜증", "싫어", "미워", "답답", "찝찝", "열받", "빡쳐",
    "무서워", "불안", "두려", "걱정", "초조",
]

# 메타 self 의문 키워드 (EVE 자체)
META_SELF_QUESTIONS = [
    "누구", "뭐", "뭔지", "뭔가",
    "살아", "존재", "진짜",
    "기분", "상태", "어때",
    "좋아하", "싫어하",       # 너는 뭘 좋아해
    "할 수 있", "할 수 없",
]

# 정의문 표현 (X는 Y야)
DEFINITION_BE_ENDING = ["야", "이야", "이다", "이지", "지", "거야", "것이다"]

# 청구/요청 표현
REQUEST_KEYWORDS = ["해줘", "도와줘", "해봐", "줄래", "줄수", "줘"]


# =============================================================================
# SentenceStructure 분석기
# =============================================================================

class SentenceStructureAnalyzer:
    """
    morph_analyzer 위에서 의미 구조 + 의도 추론.
    """

    def __init__(self, morph_analyzer: Optional[KoreanMorphAnalyzer] = None):
        self.morph = morph_analyzer or KoreanMorphAnalyzer()

    # ===== 메인 진입점 =====

    def analyze(self, text: str) -> IntentStructure:
        """텍스트 → 의미 구조."""
        morph = self.morph.analyze(text)
        struct = IntentStructure(raw=text)
        
        # 기본 형태소 분석 결과 복사
        struct.sentence_type = morph.sentence_type
        struct.is_negative = morph.is_negative
        struct.has_past = morph.has_past
        struct.has_question_mark = text.rstrip().endswith("?")
        
        if morph.is_quoted:
            struct.quoted_content = morph.quoted_content
            struct.quoted_modality = morph.quoted_modality
            struct.speech_verb = morph.quoted_speech_verb
        
        # 주어/대상 추출
        struct.subject, struct.meta_target = self._extract_subject(morph)
        
        # 감정 키워드 검출
        emotion = self._detect_emotion(text)
        if emotion:
            struct.emotion_polarity, struct.emotion_keyword = emotion
        
        # 의도 분류 (핵심)
        intent, confidence, why = self._classify_intent(text, morph, struct)
        struct.intent = intent
        struct.intent_confidence = confidence
        struct.why = why
        
        # 정의문이면 X/Y 추출
        if intent == "definition":
            self._extract_definition(text, struct)
        
        return struct

    # ===== 의도 분류 (핵심 룰) =====

    def _classify_intent(self, text: str, morph: Sentence,
                         struct: IntentStructure) -> tuple[str, float, list]:
        """
        규칙 기반 의도 분류 — 우선순위 명확:
        1. 인용 + 명령형 발화 = teaching (확실)
        2. 정의문 (X는 Y야) = definition (의문 X)
        3. 의문문 = meta_self/meta_user/question
        4. 인용 + 비명령 = reported_speech
        5. 감정 = emotional
        6. 명령/청유 = request
        7. 기본 = statement
        """
        why = []

        # === 1. 가르침 (인용 + 명령형 발화) — 가장 확실 ===
        if morph.is_quoted and morph.quoted_speech_verb:
            v = morph.quoted_speech_verb
            # 명령형 = "해", "말해", "부르", "불러", "답해"
            # 비명령형 = "했어", "했다", "했었다", "한다", "했었"
            # 비명령형 키워드를 먼저 검사
            non_imperative_speech = ["했었", "한다", "했어", "했다", "한", "할"]
            is_non_imperative = False
            for nis in non_imperative_speech:
                if v == nis or v.startswith(nis):
                    # "한" / "할" 단독은 모호 — "했어" 우선
                    if v in {"한", "할"}:
                        continue
                    is_non_imperative = True
                    break
            if is_non_imperative:
                why.append(f"인용+비명령발화({v})=reported_speech")
                return ("reported_speech", 0.7, why)
            
            teaching_speech = {"해", "말해", "부르", "불러", "답해", "얘기해"}
            for ts in teaching_speech:
                if v == ts or v.startswith(ts) and len(v) - len(ts) <= 2:
                    why.append(f"인용+명령형발화동사({v})=teaching")
                    return ("teaching", 0.9, why)
            # 기타 = 보고
            why.append(f"인용+기타발화({v})=reported_speech")
            return ("reported_speech", 0.6, why)
        
        # 2. 인용만 있고 발화동사 없음 — 짧은 가르침 ("응이라고")
        if morph.is_quoted and not morph.quoted_speech_verb:
            why.append("인용+동사없음=teaching(짧은 가르침)")
            return ("teaching", 0.6, why)

        # === 2. 정의문 (감정보다 우선!) ===
        if struct.sentence_type == "declarative" and not struct.has_question_mark:
            if self._looks_like_definition(text, morph):
                why.append("정의문 패턴(X는 Y야)")
                return ("definition", 0.75, why)

        # === 3. 의문문 ===
        if struct.sentence_type == "interrogative":
            self_ref = self._has_self_reference(morph)
            user_ref = self._has_user_reference(morph)
            
            if self_ref:
                for kw in META_SELF_QUESTIONS:
                    if kw in text:
                        why.append(f"의문+self참조+메타키워드({kw})=meta_self")
                        return ("meta_self", 0.9, why)
                why.append("의문+self참조=meta_self")
                return ("meta_self", 0.7, why)
            
            if user_ref:
                why.append("의문+user참조=meta_user")
                return ("meta_user", 0.85, why)
            
            why.append("의문문(일반)=question")
            return ("question", 0.7, why)
        
        # === 4. 감정 키워드 + 평서/감탄 = emotional ===
        if struct.emotion_polarity:
            if struct.sentence_type in ("declarative", "exclamative"):
                why.append(f"감정키워드({struct.emotion_keyword})+{struct.sentence_type}=emotional")
                return ("emotional", 0.8, why)
        
        # === 5. 명령/청유 = request ===
        if struct.sentence_type in ("imperative", "propositive"):
            why.append(f"종결어미={struct.sentence_type}=request")
            return ("request", 0.8, why)
        
        # === 6. 요청 키워드 ===
        if any(rk in text for rk in REQUEST_KEYWORDS):
            why.append("요청키워드=request")
            return ("request", 0.6, why)
        
        # === 7. 디폴트 ===
        why.append("기본 평서=statement")
        return ("statement", 0.5, why)

    # ===== 헬퍼 =====

    def _extract_subject(self, morph: Sentence) -> tuple[Optional[str], Optional[str]]:
        """첫 번째 주격/주제 어절 → 주어, 메타 대상."""
        for e in morph.eojeols:
            # 주격 또는 주제 조사
            if e.particle in ("이", "가", "은", "는"):
                if e.main:
                    subject = e.main.surface
                    # 메타 대상 분류
                    if subject in SELF_REFERENCES:
                        return (subject, "self")
                    if subject in USER_REFERENCES:
                        return (subject, "user")
                    return (subject, "other")
            # "내가/너는" 같은 결합형
            if e.surface in {"내가", "너는", "넌"}:
                if e.surface in SELF_REFERENCES or e.surface == "넌":
                    return (e.surface, "self")
                if e.surface in {"내가", "내", "나"}:
                    return (e.surface, "user")
        return (None, None)

    def _has_self_reference(self, morph: Sentence) -> bool:
        for e in morph.eojeols:
            surface = e.surface
            if surface in SELF_REFERENCES:
                return True
            # "넌" 같은 줄임
            for s in SELF_REFERENCES:
                if surface.startswith(s) and len(surface) - len(s) <= 2:
                    return True
        return False

    def _has_user_reference(self, morph: Sentence) -> bool:
        for e in morph.eojeols:
            surface = e.surface
            for u in USER_REFERENCES:
                if surface.startswith(u) and len(surface) - len(u) <= 3:
                    return True
        return False

    def _detect_emotion(self, text: str) -> Optional[tuple[str, str]]:
        """감정 키워드 검출 → (polarity, keyword)."""
        for kw in NEGATIVE_EMOTION:
            if kw in text:
                return ("negative", kw)
        for kw in POSITIVE_EMOTION:
            if kw in text:
                return ("positive", kw)
        return None

    def _looks_like_definition(self, text: str, morph: Sentence) -> bool:
        """
        "X는 Y야" 형태 인지.
        X는 1~2 어절 (예: "진짜 친구는") 가능.
        """
        if not morph.eojeols or len(morph.eojeols) < 2:
            return False
        # 의문 어휘 있으면 정의 X
        for qw in QUESTION_WORDS:
            if qw in text:
                return False
        
        topic_particles = ("은", "는", "란", "이란")
        modifier_quotes = ("라는", "이라는", "다는", "는다는")
        topic_endings = ("건", "것은", "것")
        
        # 처음 1~2 어절에서 주제 마커 검사
        has_topic = False
        for i in (0, 1):
            if i >= len(morph.eojeols):
                break
            ej = morph.eojeols[i]
            if ej.particle in topic_particles:
                has_topic = True
                break
            for mq in modifier_quotes:
                if ej.surface.endswith(mq):
                    has_topic = True
                    break
            if has_topic:
                break
            for te in topic_endings:
                if ej.surface.endswith(te):
                    has_topic = True
                    break
            if has_topic:
                break
        
        if not has_topic:
            return False
        
        # 마지막 어절이 정의 종결로 끝남
        last = morph.eojeols[-1]
        for end in DEFINITION_BE_ENDING:
            if last.surface.endswith(end):
                return True
        return False

    def _extract_definition(self, text: str, struct: IntentStructure):
        """정의문 X와 Y 분리."""
        morph = self.morph.analyze(text)
        if not morph.eojeols:
            return
        
        topic_particles = ("은", "는", "란", "이란")
        modifier_quotes = ("라는", "이라는", "다는", "는다는")
        topic_endings = ("건", "것은", "것")
        
        # 주제 마커 어디까지 인가? (1~2 어절)
        topic_idx = -1
        for i in (0, 1):
            if i >= len(morph.eojeols):
                break
            ej = morph.eojeols[i]
            if ej.particle in topic_particles:
                topic_idx = i
                break
            for mq in modifier_quotes:
                if ej.surface.endswith(mq):
                    topic_idx = i
                    break
            if topic_idx >= 0:
                break
            for te in topic_endings:
                if ej.surface.endswith(te):
                    topic_idx = i
                    break
            if topic_idx >= 0:
                break
        
        if topic_idx < 0:
            return
        
        # 주제까지의 어절들 합쳐서 concept
        concept_eojeols = morph.eojeols[:topic_idx + 1]
        # 마지막 어절의 main만 추출 (조사 떼고)
        concept_parts = [e.surface for e in concept_eojeols[:-1]]
        last_topic_eojeol = concept_eojeols[-1]
        if last_topic_eojeol.main:
            concept_parts.append(last_topic_eojeol.main.surface)
        else:
            concept_parts.append(last_topic_eojeol.surface)
        struct.defined_concept = " ".join(concept_parts).strip()
        
        # 나머지 어절 = Y (정의)
        rest_eojeols = morph.eojeols[topic_idx + 1:]
        rest = " ".join(e.surface for e in rest_eojeols).strip()
        # 마지막 정의 종결 떼고
        for end in DEFINITION_BE_ENDING:
            if rest.endswith(end):
                rest = rest[:-len(end)].strip()
                break
        struct.definition = rest
