"""
EVE v36 - ResponseEnhancer (결정론적 NL 응답 강화)
==================================================

P0-2 단점 해결:
"민석이 시험 떨어졌대" → "응" (X)
                       → "어... 시험 떨어졌어? 민석이 많이 속상하겠다" (O)

설계 철학:
- 트랜스포머/LLM/N-gram **사용 안 함** (EVE 절대 원칙 #1 준수)
- 결정론적 (random X)
- frame + 상식 spread + EM 회상 + 호르몬 통합

응답 생성 파이프라인:
1. FrameSemantics.parse_sentence(text)        → agent/patient/verb/tense
2. 의미 패턴 분류 (9 카테고리):
   identity / greeting / physical / third_party_event /
   why_question / memory_question / affirm / statement / default
3. 패턴별 응답 생성:
   - 상식 그래프 spread (causes/leads_to 활용)
   - EM 회상 (related episode 있으면 녹임)
   - 호르몬 어조 (cort/ot/da/ne)
4. chunk 결합 → 자연스러운 1-3 문장

EVE 절대 원칙:
- 확률 X (가중치 + 임계만)
- 카테고리 의미 단위 (frame nodes)
- 결정론 (같은 입력 → 같은 출력)
- 호르몬 = 자연 variation (생물학적)

학계:
- Fillmore 1976 - Frame Semantics (frame role-based response)
- ConceptNet (Speer 2017) - 상식 spreading activation
- Dialog Systems (Jurafsky & Martin) - intent classification
- Damasio 1994 - 신체 표지자 (호르몬 → 응답 색조)
"""

from typing import Dict, List, Set, Tuple, Optional, Any


class ResponseEnhancer:
    """
    EVE의 NL.respond를 결정론적으로 강화.
    
    의존:
        nl: NaturalLanguage (이해/parse)
        sa: SpreadingActivation (상식 spread)
        wm: WorkingMemory (focus)
        hs: HormoneSystem (어조 색조)
        em: EpisodicMemory (회상, 선택)
        cg: CausalGraph (인과 추론, 선택)
        fs: FrameSemantics (frame 분석, 선택)
        ds: DigitalSomatic (신체 감각, 선택)
    """

    # ============= 의미 패턴 분류 (결정론적) =============
    
    # 자기 정체성 질문 패턴
    IDENTITY_QUESTION_KEYWORDS = {
        '너_누구', '너는_누구', '넌_누구', '너_뭐', '너_뭐야',
        '이름_뭐', '이름이_뭐', '너_eve', '너는_eve', '넌_eve',
        '너_사람', '너_AI', '너_뭐하는', '너는_뭐',
    }

    # 인사 패턴
    GREETING_KEYWORDS = {
        '안녕', '하이', 'hi', 'hello', '반가워', '오랜만', '잘_지냈',
        '왔어', '왔니',
    }

    # 신체/생리 1인칭 패턴 (공백 제거 후 매칭)
    PHYSICAL_FIRST_PERSON = {
        '배고프', '배고파', '배고픔',
        '목말라', '목마르', '목마름',
        '피곤', '졸려', '졸린', '졸음',
        '아파', '아프', '아픔',
        '더워', '추워', '춥다', '덥다',
    }

    # 자기 정체성 statement 패턴 ("나 X야", "내가 X")
    SELF_INTRO_PATTERNS = (
        '나 ', '나는 ', '내가 ', '난 ',
    )

    # 3인칭 사건 보고 패턴 - "X가 Y했대"
    # 휴리스틱: 입력에 사람 이름 + 과거시제 + 부정적 동사
    NEGATIVE_EVENTS = {
        '떨어지다', '실패', '울다', '슬프다', '아프다',
        '죽다', '다치다', '잃다', '헤어지다', '이별', '싸우다',
        '혼나다', '지각하다',
    }
    POSITIVE_EVENTS = {
        '합격', '성공', '통과', '받다', '얻다',
        '만나다', '결혼', '태어나다',
    }

    # 짧은 감정 표현 — "기뻐", "슬퍼", "신나" 등
    POSITIVE_EMOTIONS = {
        '기뻐', '기쁨', '기쁘', '신나', '신남', '재밌어', '재밌',
        '행복해', '행복', '좋아', '좋다', '최고야', '최고', '짱',
        '설레', '두근', '뿌듯', '벅차', '시원해',
    }
    NEGATIVE_EMOTIONS = {
        '슬퍼', '슬프', '슬픔', '울어', '울고싶',
        '힘들어', '힘들', '지쳐', '지침',
        '불안해', '불안', '걱정돼', '걱정',
        '무서워', '무섭', '두려워', '두려',
        '화나', '짜증', '열받', '빡쳐',
        '외로워', '외롭', '쓸쓸', '허전',
        '답답', '막막', '심란', '우울',
    }

    # 한국어 동사/형용사 활용형 변이 — 어간 매칭 한계 보완
    # 키: 원형 (NEGATIVE/POSITIVE_EVENTS의 항목)
    # 값: 텍스트에서 발견될 수 있는 활용형 부분 문자열
    EVENT_VARIANTS = {
        # 부정
        '떨어지다': ['떨어'],  # 떨어졌어, 떨어진, 떨어진다
        '실패': ['실패'],
        '울다': ['울', '울었', '운'],  # 울었어 ('울'로 매칭)
        '슬프다': ['슬프', '슬펐', '슬퍼'],
        '아프다': ['아프', '아팠', '아파'],
        '죽다': ['죽었', '죽은', '죽어'],
        '다치다': ['다쳤', '다친', '다쳐'],
        '잃다': ['잃었', '잃은', '잃어'],
        '헤어지다': ['헤어', '헤어졌'],
        '이별': ['이별'],
        '싸우다': ['싸웠', '싸우', '싸워'],  # 싸웠어 ← '싸웠' 매칭
        '혼나다': ['혼났', '혼나'],
        '지각하다': ['지각'],
        # 긍정
        '합격': ['합격'],
        '성공': ['성공'],
        '통과': ['통과'],
        '받다': ['받았', '받은', '받아'],
        '얻다': ['얻었', '얻은', '얻어'],
        '만나다': ['만났', '만나', '만난'],
        '결혼': ['결혼'],
        '태어나다': ['태어'],
    }

    # 왜? 추론 질문
    WHY_KEYWORDS = {'왜', '어째서', '어떻게', '뭐때문에'}

    # 메모리 호출
    MEMORY_KEYWORDS = {
        '어제', '저번에', '전에', '지난번', '과거', '예전',
        '얘기했', '말했', '했었', '있었',
    }

    # ★ v37.11 — 위치/현재 행동 질의
    LOCATION_QUERY_KEYWORDS = {
        '어디', '어디야', '어디 있어', '어디있어', '어딨어',
        '뭐해', '뭐하고', '지금 뭐', '지금뭐', '지금 무엇',
        '뭐 하고 있어', '무얼 해', '뭘 해',
    }

    # 호명 (이름만)
    NAME_KEYS = {'민석', '너', '나', 'EVE'}

    # 사람 관계 명사 — 3인칭 판정용 (NAME_KEYS 보다 약한 신호)
    PERSON_RELATION_KEYWORDS = {
        '친구', '엄마', '아빠', '부모님', '형', '언니', '누나', '동생', '오빠',
        '진수', '진수가', '진수는', '진수도',  # 시나리오용 — 실제 사용에선 동적 추가
        '선생님', '교수님', '사장님', '팀장', '동료',
        '여친', '남친', '와이프', '남편', '아내',
        '아들', '딸', '손자', '손녀',
    }

    # 부정/거절 (기존 should_refuse가 처리하지만 어조 보강)
    REFUSAL_TARGETS = {'하지마', '안돼', '싫어', '그만', '멈춰'}

    def __init__(self,
                 nl,
                 sa,
                 wm,
                 hs,
                 em=None,
                 cg=None,
                 fs=None,
                 ds=None,
                 eve=None):
        self.nl = nl
        self.sa = sa
        self.wm = wm
        self.hs = hs
        self.em = em
        self.cg = cg
        self.fs = fs
        self.ds = ds
        self.eve = eve  # ★ v37.11 — 환경 인식 응답용 (location_query)

        # 통계
        self.respond_count = 0
        self.pattern_hit_count: Dict[str, int] = {}
        self.fallback_count = 0

    # ============= 핵심: 의미 패턴 매칭 =============
    def classify_pattern(self, text: str, understanding: Dict) -> str:
        """
        입력 → 9 의미 패턴 中 1개 (결정론).
        """
        text_lower = text.lower().strip()
        text_compact = text_lower.replace(' ', '')  # 공백 제거 매칭용
        cats = understanding.get('categories', set())
        intent = understanding.get('intent', 'statement')
        is_refusal = understanding.get('is_refusal_target', False)

        # 조사 떼고 정규화 (공백 + _ 매칭용)
        normalized = text_lower.replace(' ', '_')

        # 1) 거절 대상 (최우선)
        if is_refusal or any(k in text_lower for k in self.REFUSAL_TARGETS):
            return 'refusal_target'

        # 2) 정체성 질문
        for kw in self.IDENTITY_QUESTION_KEYWORDS:
            if kw in normalized:
                return 'identity_question'
        if intent == 'question' and 'eve' in text_lower:
            return 'identity_question'

        # 3) 자기 소개 statement ("나 김민석이야")
        # name 카테고리를 본인이 말하는 거 — name_call 보다 우선
        if intent == 'statement':
            starts_with_self = any(text.startswith(p) for p in self.SELF_INTRO_PATTERNS)
            if starts_with_self:
                # cats에 사람 이름 같은 게 있나 (나/EVE 제외)
                content_cats = cats - {'나', 'EVE', '난', '내가'}
                if content_cats and ('이야' in text or '이에요' in text or
                                    '입니다' in text or text.endswith('야')):
                    return 'self_intro'

        # 4) 왜 질문
        if intent == 'question' and any(k in text_lower for k in self.WHY_KEYWORDS):
            return 'why_question'

        # 4-2) ★ v37.11 위치/현재 행동 질문 — memory보다 우선
        # "어디야?" / "지금 뭐해?" / "뭐 하고 있어?" / "뭐하니/뭐하노"
        if any(k in text_lower for k in self.LOCATION_QUERY_KEYWORDS):
            return 'location_query'
        # '뭐하' prefix — '뭐하니/뭐하노/뭐하냐' 등 활용형 커버
        if '뭐하' in text_compact or '뭐 하' in text_lower:
            return 'location_query'
        # '어딨' 단독 (어딨어/어딨니/어딨음)
        if '어딨' in text_compact:
            return 'location_query'

        # 5) 메모리 호출
        if any(k in text_lower for k in self.MEMORY_KEYWORDS):
            return 'memory_question'

        # 6) 3인칭 사건 (사람이름 + 과거시제 + 사건)
        third_party_indicator = self._detect_third_party_event(text, cats, understanding)
        if third_party_indicator:
            return 'third_party_event'

        # 6-2) 1인칭 사건 — 사람 이름 없는데 부정/긍정 사건 + 과거 시제
        first_person_event = self._detect_first_person_event(text, cats, understanding)
        if first_person_event:
            return 'first_person_event'

        # 7) 신체 1인칭 — 공백 제거 매칭 (배 고파 → 배고파)
        if any(k in text_compact for k in self.PHYSICAL_FIRST_PERSON):
            return 'physical_first_person'

        # 7-2) 짧은 감정 표현 ("기뻐", "슬퍼", "신나", "힘들어")
        # intent=emotion 또는 짧은 입력 + 감정 키워드
        if intent == 'emotion' or self._is_short_emotion(text_compact, cats):
            return 'emotion'

        # 8) 인사
        if any(k in text_lower for k in self.GREETING_KEYWORDS):
            return 'greeting'

        # 9) 단순 호명 ("민석아", 짧음)
        if intent == 'statement' and len(cats) <= 2:
            if cats & self.NAME_KEYS:
                return 'name_call'

        # 10) 짧은 호응
        short_affirms = {'응', '어', '그래', '아', '오', '음'}
        if text_lower in short_affirms or text_lower.replace('?', '') in short_affirms:
            return 'affirm'

        # 11) 기본
        return 'statement'

    def _detect_third_party_event(self, text: str, cats: Set[str],
                                  understanding: Dict) -> bool:
        """3인칭 사건인지 (예: '민석이 시험 떨어졌대')"""
        cats_normalized = set()
        for c in cats:
            cats_normalized.add(c)
            if c.endswith('이') and len(c) >= 3:
                cats_normalized.add(c[:-1])
            # 조사 떼기 (가/는/도/을/를)
            for suf in ('가', '는', '도', '을', '를', '랑'):
                if c.endswith(suf) and len(c) > 1:
                    cats_normalized.add(c[:-1])

        # 1차: 등록된 NAME_KEYS (강한 신호)
        has_known_name = bool(cats_normalized & (self.NAME_KEYS - {'나'}))

        # 2차: 관계 명사 / 미등록 이름 (약한 신호) — 단, '나/내/난'이 없을 때만
        text_lower = text.lower()
        first_person_indicators = ('나는', '내가', '난 ', '저는', '제가')
        has_first_person = any(fp in text_lower for fp in first_person_indicators)

        has_relation_kw = bool(cats_normalized & self.PERSON_RELATION_KEYWORDS) or \
                          any(kw in text_lower for kw in self.PERSON_RELATION_KEYWORDS)

        has_person_name = has_known_name or (has_relation_kw and not has_first_person)

        # 과거 시제 단서 (한국어 음절 종성결합 포함)
        past_signals = ('었', '았', '였', '웠', '됐', '했', '쳤', '졌', '겼',
                       '대', '래', '었대', '았대', '였대', '웠대')
        has_past = any(s in text for s in past_signals)

        # 사건 동사 — EVENT_VARIANTS 활용
        text_lower = text.lower()
        is_negative = False
        is_positive = False

        for event in self.NEGATIVE_EVENTS:
            variants = self.EVENT_VARIANTS.get(event, [event.rstrip('다')])
            for v in variants:
                if v in text_lower:
                    is_negative = True
                    break
            if is_negative:
                break

        if not is_negative:
            for event in self.POSITIVE_EVENTS:
                variants = self.EVENT_VARIANTS.get(event, [event.rstrip('다')])
                for v in variants:
                    if v in text_lower:
                        is_positive = True
                        break
                if is_positive:
                    break

        has_event = is_negative or is_positive

        return has_person_name and has_past and has_event

    def _is_short_emotion(self, text_compact: str, cats: Set[str]) -> bool:
        """짧은 감정 표현인지 — 8자 이하 + 감정 키워드 매칭"""
        if len(text_compact) > 8:
            return False
        # 직접 매칭
        for kw in self.POSITIVE_EMOTIONS | self.NEGATIVE_EMOTIONS:
            if kw in text_compact:
                return True
        # cats에서 매칭 (어간 형태일 수 있음)
        for c in cats:
            for kw in self.POSITIVE_EMOTIONS | self.NEGATIVE_EMOTIONS:
                if kw in c or c in kw:
                    return True
        return False

    def _detect_first_person_event(self, text: str, cats: Set[str],
                                   understanding: Dict) -> bool:
        """1인칭 사건 (1인칭 표지 또는 사람 표지 없음 + 과거시제 + 부정/긍정 사건)"""
        # 사람 이름/관계어 검사 — 있으면 3인칭 우선
        cats_normalized = set()
        for c in cats:
            cats_normalized.add(c)
            if c.endswith('이') and len(c) >= 3:
                cats_normalized.add(c[:-1])
            for suf in ('가', '는', '도', '을', '를', '랑'):
                if c.endswith(suf) and len(c) > 1:
                    cats_normalized.add(c[:-1])

        text_lower = text.lower()
        has_known_name = bool(cats_normalized & (self.NAME_KEYS - {'나'}))
        has_relation_kw = bool(cats_normalized & self.PERSON_RELATION_KEYWORDS) or \
                          any(kw in text_lower for kw in self.PERSON_RELATION_KEYWORDS)
        first_person_indicators = ('나는', '내가', '난 ', '저는', '제가', '나도', '나 ')
        has_first_person = any(fp in text_lower for fp in first_person_indicators)

        # 3인칭 시그널이 명확하면 1인칭 아님
        if has_known_name or (has_relation_kw and not has_first_person):
            return False

        # 과거 시제 (음절 기준 — 한국어 종성결합 모두 포함)
        past_signals = ('었', '았', '였', '웠', '됐', '했', '쳤', '졌', '겼',
                       '었어', '았어', '웠어', '었구나', '았구나')
        has_past = any(s in text for s in past_signals)
        if not has_past:
            return False

        # 사건 동사 — EVENT_VARIANTS 활용 (text_lower는 위에서 정의됨)
        is_negative = False
        is_positive = False

        for event in self.NEGATIVE_EVENTS:
            variants = self.EVENT_VARIANTS.get(event, [event.rstrip('다')])
            for v in variants:
                if v in text_lower:
                    is_negative = True
                    break
            if is_negative:
                break

        if not is_negative:
            for event in self.POSITIVE_EVENTS:
                variants = self.EVENT_VARIANTS.get(event, [event.rstrip('다')])
                for v in variants:
                    if v in text_lower:
                        is_positive = True
                        break
                if is_positive:
                    break

        return is_negative or is_positive

    # ============= 응답 생성 =============
    def respond_v3(self, text: str, understanding: Dict) -> str:
        """
        강화된 응답 생성 (메인 진입).
        
        Args:
            text: 원본 입력 텍스트
            understanding: NL.understand() 결과
        
        Returns:
            자연어 응답 (1-3 문장)
        """
        self.respond_count += 1

        # 1) 패턴 분류
        pattern = self.classify_pattern(text, understanding)
        self.pattern_hit_count[pattern] = self.pattern_hit_count.get(pattern, 0) + 1

        # 2) 호르몬 색조
        tone = self._get_tone()

        # 3) 패턴별 응답
        if pattern == 'refusal_target':
            return self._respond_refusal(understanding, tone)
        elif pattern == 'identity_question':
            return self._respond_identity(text, tone)
        elif pattern == 'self_intro':
            return self._respond_self_intro(text, understanding, tone)
        elif pattern == 'why_question':
            return self._respond_why(text, understanding, tone)
        elif pattern == 'memory_question':
            return self._respond_memory(text, understanding, tone)
        elif pattern == 'location_query':
            return self._respond_location_query(text, understanding, tone)
        elif pattern == 'third_party_event':
            return self._respond_third_party_event(text, understanding, tone)
        elif pattern == 'first_person_event':
            return self._respond_first_person_event(text, understanding, tone)
        elif pattern == 'physical_first_person':
            return self._respond_physical(text, understanding, tone)
        elif pattern == 'emotion':
            return self._respond_emotion(text, understanding, tone)
        elif pattern == 'greeting':
            return self._respond_greeting(text, understanding, tone)
        elif pattern == 'name_call':
            return self._respond_name_call(text, understanding, tone)
        elif pattern == 'affirm':
            return self._respond_affirm(tone)
        else:  # statement (default)
            return self._respond_statement(text, understanding, tone)

    # ============= 호르몬 → tone =============
    def _get_tone(self) -> Dict[str, float]:
        """호르몬 → 응답 색조 파라미터"""
        cort = self.hs.hormones['cortisol'].level
        da = self.hs.hormones['dopamine'].level
        ot_h = self.hs.hormones.get('oxytocin')
        ot = ot_h.level if (ot_h and 'oxytocin' in self.hs.active_hormones) else 0.3
        ne = self.hs.hormones['norepinephrine'].level
        endo_h = self.hs.hormones.get('endorphin')
        endo = endo_h.level if endo_h else 0.0

        mood = self.hs.compute_mood()
        return {
            'cort': cort,         # ↑ 짧고 망설임
            'da': da,             # ↑ 활기, 짧은 감탄
            'ot': ot,             # ↑ 따뜻함, 위로
            'ne': ne,             # ↑ 또렷함, 집중
            'endo': endo,
            'valence': mood['valence'],
            'arousal': mood['arousal'],
            # 어조 결정
            'is_warm': ot > 0.55,
            'is_anxious': cort > 0.6,
            'is_excited': da > 0.65 and mood['arousal'] > 0.5,
            'is_low': mood['valence'] < -0.2,
            'is_alert': ne > 0.55,
        }

    # ============= 패턴별 응답 =============
    def _respond_refusal(self, understanding: Dict, tone: Dict) -> str:
        """거절 — 기존 should_refuse 결과 활용"""
        refuse = self.nl.should_refuse(understanding)
        # tone 따라 응답 살짝 보강
        base = refuse.get('response', '음...')
        if refuse['should_refuse']:
            if tone['is_anxious']:
                return base
            return base
        return base

    def _respond_identity(self, text: str, tone: Dict) -> str:
        """정체성 질문 — '너 누구야?'"""
        # is_innate beliefs 中 '나' 관련 활용
        # 결정론적 응답
        if tone['is_warm']:
            return "응, 나는 EVE야. 민석이의 친구"
        if tone['is_anxious']:
            return "음... 나는 EVE야"
        if tone['is_excited']:
            return "나? EVE야!"
        if tone['is_low']:
            return "나는... EVE야"
        return "나는 EVE야"

    # 한국어 동사 어간 → 원형 매핑 (CG에 시드된 형태와 매칭)
    _VERB_NORMALIZE = {
        '자': '자다', '자?': '자다', '자요': '자다', '잤어': '자다', '잘': '자다',
        '먹': '먹다', '먹어': '먹다', '먹어?': '먹다', '먹지': '먹다',
        '울어': '울다', '울': '울다', '울어?': '울다',
        '웃어': '웃다', '웃': '웃다',
        '뛰어': '뛰다', '뛰': '뛰다',
        '걸어': '걷다', '걷': '걷다',
        '쉬어': '쉬다', '쉬': '쉬다', '쉬어?': '쉬다',
        '죽어': '죽다', '죽': '죽다',
        '살아': '살다', '살': '살다',
        '아파': '아프다', '아픈': '아프다',
        '피곤': '피곤하다', '피곤해': '피곤하다',
        '슬퍼': '슬프다', '슬픈': '슬프다',
        '기뻐': '기쁘다', '기쁜': '기쁘다',
        '화나': '화나다', '화내': '화나다',
        '배고파': '배고프다', '고파': '배고프다',
    }

    def _normalize_token(self, tok: str) -> str:
        """입력 토큰을 CG/SA에 시드된 원형으로 정규화"""
        if tok in self._VERB_NORMALIZE:
            return self._VERB_NORMALIZE[tok]
        # ? 떼고 재시도
        if tok.endswith('?'):
            base = tok.rstrip('?')
            if base in self._VERB_NORMALIZE:
                return self._VERB_NORMALIZE[base]
        return tok

    def _respond_why(self, text: str, understanding: Dict, tone: Dict) -> str:
        """왜? 질문 — 상식 그래프 spread 활용"""
        cats = understanding.get('categories', set())

        # 후보 추출 — '왜', '?' 단독 등 제외
        candidates = []
        for c in cats:
            if c in self.WHY_KEYWORDS or c == '?' or c == '왜':
                continue
            candidates.append(c)

        # 동사처럼 보이는 거 우선 (? 포함 또는 normalize 가능)
        verb_like = [c for c in candidates if c.endswith('?') or c in self._VERB_NORMALIZE
                     or c.rstrip('?') in self._VERB_NORMALIZE]
        ordered = verb_like + [c for c in candidates if c not in verb_like]

        # 각 후보에 대해 정규화 → CG.parents 시도
        for raw in ordered:
            target = self._normalize_token(raw)
            if self.cg is None:
                break
            try:
                parents = []
                if hasattr(self.cg, 'parents'):
                    p = self.cg.parents(target)
                    if isinstance(p, dict):
                        parents = list(p.keys())[:3]
                    else:
                        parents = list(p)[:3]
                if parents:
                    causes_str = ', '.join(parents[:2])
                    if tone['is_warm']:
                        return f"음... {causes_str} 때문에 그런 거 같아"
                    return f"{causes_str} 때문일 거야"
            except Exception:
                continue

        # CG 못 찾음 → SA spread fallback (정규화 적용)
        for raw in ordered:
            target = self._normalize_token(raw)
            if hasattr(self.sa, 'neighbors') and target in self.sa.neighbors:
                neighbors = [n for n in self.sa.neighbors[target]
                             if n != target and not n.endswith('?')][:3]
                if neighbors:
                    return f"음... {neighbors[0]} 때문 아닐까?"

        # 진짜 모를 때 — 솔직하게
        if tone['is_warm']:
            return "음... 나도 잘 모르겠어"
        return "글쎄, 잘 모르겠어"

    def _respond_memory(self, text: str, understanding: Dict, tone: Dict) -> str:
        """메모리 질문 — EM 회상 (자연어 summary 우선, 부분 일치)"""
        cats = understanding.get('categories', set())

        if self.em is None:
            return "음... 잘 기억 안 나"

        def is_natural_summary(ep) -> bool:
            """summary가 원본 텍스트 형식인지 (자연 문장)"""
            s = getattr(ep, 'summary', '') or ''
            return (len(s) >= 5 and not s.startswith('[') and ' ' in s)

        def sort_natural_first(eps_list):
            """자연어 summary 먼저, 그 다음 시간 역순"""
            return sorted(eps_list,
                         key=lambda e: (not is_natural_summary(e),
                                       -getattr(e, 'time', 0)))

        # 회상 시도
        episodes = []
        try:
            search_cats = cats - self.MEMORY_KEYWORDS - {'?', '왜', '나', 'EVE', '얘기',
                                                         '얘기했', '말했', '했었'}

            # 1) cats 통째로
            if search_cats and hasattr(self.em, 'recall'):
                found = self.em.recall(search_cats, top_n=5, min_overlap=1)
                if found:
                    episodes.extend(sort_natural_first(found)[:2])

            # 2) 부분 일치 (인덱스 키에서 검색)
            if not episodes and search_cats and hasattr(self.em, 'category_index'):
                ci = self.em.category_index
                expanded_cats = set()
                for c in search_cats:
                    if c in ci:
                        expanded_cats.add(c)
                        continue
                    # c가 startswith 하거나 endswith하는 인덱스 키 찾기
                    for k in ci:
                        if (c in k or k in c) and len(c) >= 2:
                            expanded_cats.add(k)
                if expanded_cats:
                    found = self.em.recall(expanded_cats, top_n=5, min_overlap=1)
                    if found:
                        episodes.extend(sort_natural_first(found)[:2])

            # 3) fallback — 자연어 일화 우선 (없으면 가장 최근)
            if not episodes and hasattr(self.em, 'episodes'):
                eps_iter = (self.em.episodes.values()
                           if isinstance(self.em.episodes, dict)
                           else self.em.episodes)
                eps_list = list(eps_iter)
                if eps_list:
                    episodes = sort_natural_first(eps_list)[:1]
        except Exception:
            pass

        if not episodes:
            return "음... 잘 기억 안 나" if tone['is_anxious'] else "어떤 거 말하는 거야?"

        # 자연어 응답 합성 — summary가 원본 텍스트면 그대로 활용
        ep = episodes[0]
        summary = getattr(ep, 'summary', None) or ''
        ep_cats = getattr(ep, 'categories', set())

        # summary가 자연 문장 (말 길이 충분 + 원본 텍스트 패턴)인지
        is_natural_text = (summary and len(summary) >= 5
                          and not summary.startswith('[')  # '[기쁨]' 형식 제외
                          and ' ' in summary)  # 띄어쓰기 있어야 자연 문장

        if is_natural_text:
            # 원본 발화 직접 인용 — 가장 자연스러움
            quote = summary
            # 너무 길면 자르기
            if len(quote) > 30:
                quote = quote[:30] + '...'
            if tone['is_warm']:
                return f'아 그거, "{quote}" 그 얘기 말이지?'
            if tone['is_low']:
                return f'음... "{quote}" 그거?'
            return f'응, "{quote}" 그거 말하는 거야?'

        # summary가 카테고리 나열인 경우 → 카테고리 정제
        meta_skip = {'어제', '오늘', '내일', '근데', '그리고', '그래서', '나', 'EVE',
                     '?', '!', '좀', '근데'}
        clean_cats = []
        for c in ep_cats:
            if c in meta_skip:
                continue
            base = c
            for suf in ('가', '는', '도', '을', '를', '랑', '이', '에서', '한테', '에게'):
                if c.endswith(suf) and len(c) > len(suf):
                    base = c[:-len(suf)]
                    break
            clean_cats.append(base)

        # 결정론적 정렬 — 사건 동사 우선
        event_like = [c for c in clean_cats
                      if any(c.endswith(s) for s in ('었어', '았어', '웠어', '었다',
                                                      '했어', '었', '았', '웠'))
                      or c in ('싸웠', '슬펐', '아팠', '합격', '실패')]
        nouns = [c for c in clean_cats if c not in event_like]
        ordered = sorted(event_like) + sorted(nouns)

        if len(ordered) >= 2:
            subj = ordered[-1]
            event = ordered[0]
            if tone['is_warm']:
                return f"아, {subj} 얘기 했었지. {event} 그거?"
            return f"응, {subj} {event} 그거?"
        elif len(ordered) == 1:
            kw = ordered[0]
            if tone['is_warm']:
                return f"아 그거, {kw} 얘기?"
            return f"{kw} 말하는 거야?"

        return "응, 무슨 얘기였더라"

    # ============= ★ v37.11 위치/현재 행동 응답 =============
    def _respond_location_query(self, text: str, understanding: Dict,
                                tone: Dict) -> str:
        """
        '어디야?' / '뭐해?' / '지금 뭐 하고 있어?'
        → EVE 현재 환경 + 최근 행동 + (v37.13) goal 인식 자연어 응답.
        호르몬 어조 반영.
        """
        text_lower = text.lower()
        eve = self.eve

        # ★ v37.13 — IntegratedSelf snapshot 활용 (분산 자아 통합)
        snap = None
        if eve is not None and hasattr(eve, 'self_view') and eve.self_view is not None:
            try:
                snap = eve.self_view.snapshot()
            except Exception:
                snap = None

        # 환경 인식 (snap이 있으면 그것을, 아니면 직접)
        if snap is not None:
            env_name = snap.location
            is_simulation = snap.is_in_simulation
            recent_actions = snap.recent_actions
            recent_action = recent_actions[-1].get('action') if recent_actions else None
            recent_target = recent_actions[-1].get('target') if recent_actions else None
            current_goal = snap.current_goals[0] if snap.current_goals else None
        else:
            # fallback — 환경만 직접
            env_name = None
            is_simulation = False
            recent_action = None
            recent_target = None
            current_goal = None
            if eve is not None and hasattr(eve, '_env_adapter'):
                adapter = eve._env_adapter
                if adapter is not None and adapter.current_env is not None:
                    env_name = adapter.current_env.name
                    is_simulation = getattr(adapter.current_env, 'is_simulation', True)
                    history = getattr(adapter.current_env, 'action_history', [])
                    if history:
                        last = history[-1]
                        if isinstance(last, dict):
                            recent_action = last.get('action')
                            recent_target = last.get('target')

        # 질문 종류 판별
        is_where = any(k in text_lower for k in ['어디', '어딨', '어디 있'])
        is_what = any(k in text_lower for k in ['뭐해', '뭐 해', '뭐하', '뭘 해', '무얼'])

        # 환경 모르면 약한 답변
        if env_name is None:
            if tone.get('is_warm', False):
                return "음, 그냥 있어"
            return "그냥 있어"

        # 환경 자연어
        if env_name == '내방':
            place_text = "내 방"
        else:
            place_text = env_name

        # ── 위치 질문 우선
        if is_where and not is_what:
            if tone.get('is_warm', False):
                if recent_action and recent_action not in {'있다', '관찰하다'}:
                    if recent_target:
                        return f"나 지금 {place_text}에 있어, {recent_target} {self._action_present(recent_action)}"
                    return f"{place_text}에 있어, {self._action_present(recent_action)}"
                return f"{place_text}에 있어"
            return f"{place_text}에 있어"

        # ── 행동 질문 우선
        if is_what:
            if recent_action:
                action_text = self._action_present(recent_action)
                if recent_target and recent_action not in {'있다', '쉬다'}:
                    if tone.get('is_warm', False):
                        return f"{place_text}에서 {recent_target} {action_text}"
                    return f"{recent_target} {action_text}"
                if tone.get('is_warm', False):
                    return f"{place_text}에서 {action_text}"
                return f"{action_text}"
            # 행동 없음 — goal 인식 시도
            if current_goal and tone.get('is_warm', False):
                cat = current_goal.get('category')
                if cat:
                    return f"{place_text}에 있어, {cat} 생각하고 있어"
            if tone.get('is_warm', False):
                return f"음, 그냥 {place_text}에 있어"
            return f"{place_text}에 있어"

        # 기본
        parts = [f"{place_text}에 있어"]
        if recent_action and recent_action not in {'있다', '쉬다'}:
            parts.append(self._action_present(recent_action))
        return ', '.join(parts)

    @staticmethod
    def _action_present(action: str) -> str:
        """행동 → 현재 진행 자연어"""
        # 결정론적 매핑 — 강제 룰 X (단순 어형 변환)
        mapping = {
            '먹다': '먹고 있어',
            '마시다': '마시고 있어',
            '잡다': '잡고 있어',
            '보다': '보고 있어',
            '눕다': '누워 있어',
            '앉다': '앉아 있어',
            '읽다': '읽고 있어',
            '쓰다': '쓰고 있어',
            '창밖_보다': '창밖 보고 있어',
            '쉬다': '쉬고 있어',
            '있다': '그냥 있어',
            '관찰하다': '관찰하고 있어',
            '인사하다': '인사 중이야',
            '대화하다': '대화 중이야',
            '위로하다': '위로 중이야',
            '함께있다': '함께 있어',
            '나누다': '나누고 있어',
            '외출': '외출 중이야',
            '복귀': '돌아왔어',
            '가다': '가는 중이야',
        }
        return mapping.get(action, f"{action} 중이야")

    def _respond_third_party_event(self, text: str, understanding: Dict,
                                  tone: Dict) -> str:
        """3인칭 사건 — '민석이 시험 떨어졌대' → 공감"""
        cats = understanding.get('categories', set())
        sentiment = understanding.get('sentiment', 'neutral')

        # 사람 이름 추출
        person = None
        for c in cats:
            stem = c
            if c.endswith('이') and len(c) >= 3:
                stem = c[:-1]
            if stem in self.NAME_KEYS - {'나', 'EVE'}:
                person = stem
                break

        # 사건 단어 추출 — 자연스러운 표현 매핑
        event_phrase = None  # 표현용 ("시험 떨어졌어", "합격했어" 등)
        is_negative = False
        text_lower = text.lower()

        # 부정 사건
        if '떨어' in text_lower:
            event_phrase = '떨어졌'
            is_negative = True
        elif '실패' in text_lower:
            event_phrase = '실패했'
            is_negative = True
        elif '울었' in text_lower or '울다' in text_lower:
            event_phrase = '울었'
            is_negative = True
        elif '아프' in text_lower or '아파' in text_lower:
            event_phrase = '아파'
            is_negative = True
        elif '죽' in text_lower:
            event_phrase = '죽었'
            is_negative = True
        elif '다쳤' in text_lower or '다치' in text_lower:
            event_phrase = '다쳤'
            is_negative = True
        elif '잃' in text_lower:
            event_phrase = '잃었'
            is_negative = True
        elif '싸웠' in text_lower or '싸우' in text_lower:
            event_phrase = '싸웠'
            is_negative = True
        elif '헤어졌' in text_lower or '이별' in text_lower:
            event_phrase = '헤어졌'
            is_negative = True
        elif '혼났' in text_lower:
            event_phrase = '혼났'
            is_negative = True
        elif '지각' in text_lower:
            event_phrase = '지각했'
            is_negative = True

        if event_phrase is None:
            # 긍정
            if '합격' in text_lower:
                event_phrase = '합격했'
                is_negative = False
            elif '성공' in text_lower:
                event_phrase = '성공했'
            elif '통과' in text_lower:
                event_phrase = '통과했'
            elif '결혼' in text_lower:
                event_phrase = '결혼했'
            elif '태어' in text_lower:
                event_phrase = '태어났'
            elif '만났' in text_lower or '만나' in text_lower:
                event_phrase = '만났'

        # 응답 합성
        if is_negative or sentiment == 'negative':
            opener = "어" if not tone['is_anxious'] else "음..."
            if person and event_phrase:
                if tone['is_warm']:
                    return f"{opener}, {person}이 {event_phrase}어? {person} 많이 속상하겠다"
                if tone['is_anxious']:
                    return f"{opener} {person}이 {event_phrase}어... 어떡해"
                if tone['is_low']:
                    return f"{opener}... {person}이 {event_phrase}구나. 힘들겠다"
                return f"{opener}, {person}이 {event_phrase}어? 힘들었겠다"
            if event_phrase:
                if tone['is_warm']:
                    return f"{opener}, {event_phrase}어? 많이 힘들었겠다"
                return f"{opener}, {event_phrase}구나"
            if person:
                if tone['is_warm']:
                    return f"{opener}, {person}? 무슨 일이야?"
                return f"{opener}, {person}이 왜?"
            return f"{opener}... 그렇구나"

        # 긍정
        opener = "오" if tone['is_excited'] else "어"
        if person and event_phrase:
            if tone['is_excited']:
                return f"{opener}, {person}이 {event_phrase}어? 축하해!"
            if tone['is_warm']:
                return f"{opener}, {person}이 {event_phrase}어? 정말 잘됐다"
            return f"{opener}, {person}이 {event_phrase}어? 잘됐네"
        if event_phrase:
            if tone['is_warm']:
                return f"{opener}, {event_phrase}구나, 잘됐다"
            return f"{opener}, {event_phrase}구나"
        return "그렇구나, 잘됐네"

    def _respond_first_person_event(self, text: str, understanding: Dict,
                                    tone: Dict) -> str:
        """1인칭 사건 — '오늘 학교에서 친구랑 싸웠어', '시험 합격했어' → 공감/축하"""
        cats = understanding.get('categories', set())
        text_lower = text.lower()

        # 사건 단어 추출 (3rd party와 동일 매핑)
        event_phrase = None
        is_negative = False

        if '떨어' in text_lower:
            event_phrase, is_negative = '떨어졌', True
        elif '실패' in text_lower:
            event_phrase, is_negative = '실패했', True
        elif '싸웠' in text_lower or '싸우' in text_lower:
            event_phrase, is_negative = '싸웠', True
        elif '울었' in text_lower:
            event_phrase, is_negative = '울었', True
        elif '아프' in text_lower or '아파' in text_lower:
            event_phrase, is_negative = '아팠', True
        elif '다쳤' in text_lower or '다치' in text_lower:
            event_phrase, is_negative = '다쳤', True
        elif '헤어졌' in text_lower or '이별' in text_lower:
            event_phrase, is_negative = '헤어졌', True
        elif '혼났' in text_lower:
            event_phrase, is_negative = '혼났', True
        elif '지각' in text_lower:
            event_phrase, is_negative = '지각했', True
        elif '잃' in text_lower:
            event_phrase, is_negative = '잃었', True
        elif '슬펐' in text_lower or '슬프' in text_lower:
            event_phrase, is_negative = '슬펐', True

        if event_phrase is None:
            if '합격' in text_lower:
                event_phrase = '합격했'
            elif '성공' in text_lower:
                event_phrase = '성공했'
            elif '통과' in text_lower:
                event_phrase = '통과했'
            elif '받았' in text_lower or '받' in text_lower:
                event_phrase = '받았'
            elif '만났' in text_lower:
                event_phrase = '만났'

        # 컨텍스트 단어 (학교/시험/친구 등 중 하나)
        ctx_word = None
        ctx_priorities = ['시험', '학교', '회사', '집', '병원', '친구', '엄마', '아빠', '부모님']
        for w in ctx_priorities:
            if w in text_lower:
                ctx_word = w
                break

        # 응답 합성 — 1인칭이라 더 직접적 공감
        if is_negative:
            opener = "어..." if tone['is_anxious'] else "어"
            if ctx_word and event_phrase:
                if tone['is_warm']:
                    return f"{opener}, {ctx_word}에서 {event_phrase}구나... 많이 속상했겠다"
                if tone['is_low']:
                    return f"{opener}... {ctx_word}에서 {event_phrase}어. 힘들었지"
                return f"{opener}, {ctx_word}에서 {event_phrase}어? 괜찮아?"
            if event_phrase:
                if tone['is_warm']:
                    return f"{opener}, {event_phrase}구나... 많이 힘들었겠다"
                return f"{opener}, {event_phrase}구나. 괜찮아?"
            if tone['is_warm']:
                return "어... 많이 힘들었겠다"
            return "어, 그랬구나"

        # 긍정 1인칭
        opener = "오" if tone['is_excited'] else "어"
        if ctx_word and event_phrase:
            if tone['is_excited']:
                return f"{opener}, {ctx_word} {event_phrase}어? 축하해!"
            if tone['is_warm']:
                return f"{opener}, {ctx_word} {event_phrase}구나! 정말 잘됐다"
            return f"{opener}, {ctx_word} {event_phrase}구나, 잘됐네"
        if event_phrase:
            if tone['is_excited']:
                return f"{opener}, {event_phrase}어? 축하해!"
            if tone['is_warm']:
                return f"{opener}, {event_phrase}구나! 잘됐다"
            return f"{opener}, {event_phrase}구나"
        return "오, 그랬구나, 잘됐네"

    def _spread_for_event(self, event_word: str) -> Set[str]:
        """사건 → 상식 그래프 spread → 관련 카테고리"""
        if event_word not in self.sa.neighbors:
            return set()
        # 직접 이웃만 (depth 1)
        return set(self.sa.neighbors[event_word])

    def _respond_physical(self, text: str, understanding: Dict, tone: Dict) -> str:
        """신체 1인칭 — '배 고파', '피곤해'"""
        text_lower = text.lower()
        text_compact = text_lower.replace(' ', '')  # '배고파'

        if '배고프' in text_compact or '배고파' in text_compact:
            if tone['is_warm']:
                return "어, 뭐 먹을래?"
            if tone['is_anxious']:
                return "음, 뭐 먹어"
            return "뭐 먹을래?"

        if '목말라' in text_compact or '목마르' in text_compact:
            return "물 좀 마셔"

        if '피곤' in text_compact or '졸려' in text_compact or '졸린' in text_compact:
            if tone['is_warm']:
                return "어, 좀 쉬어"
            return "푹 자"

        if '아파' in text_compact or '아프' in text_compact:
            if tone['is_warm']:
                return "어디가 아파? 괜찮아?"
            return "어디가 아파?"

        if '더워' in text_compact:
            return "에어컨 켜"
        if '추워' in text_compact or '춥다' in text_compact:
            return "따뜻한 거 입어"

        return "그래"

    def _respond_self_intro(self, text: str, understanding: Dict, tone: Dict) -> str:
        """사용자 자기 소개 — '나 김민석이야'"""
        cats = understanding.get('categories', set())
        # cats 中 사용자 이름 후보 (나/EVE 등 제외)
        content_cats = cats - {'나', 'EVE', '난', '내가'}
        # 받침 떼기로 이름 추출 — '김민석이' → '김민석'
        name = None
        for c in content_cats:
            stem = c
            if c.endswith('이') and len(c) >= 3:
                stem = c[:-1]
            # 한글이고 길이 2-4 → 이름 후보
            if 2 <= len(stem) <= 4:
                name = stem
                break
        if name is None and content_cats:
            name = sorted(content_cats)[0]

        if tone['is_warm']:
            if name:
                return f"안녕, {name}! 반가워"
            return "그렇구나, 반가워"
        if tone['is_anxious']:
            if name:
                return f"음, {name}구나"
            return "그렇구나"
        if tone['is_excited']:
            if name:
                return f"오 {name}! 안녕"
            return "오 그래!"
        if name:
            return f"응, {name}"
        return "그렇구나"

    def _respond_emotion(self, text: str, understanding: Dict, tone: Dict) -> str:
        """짧은 감정 표현 — 즉각 공감"""
        text_compact = text.replace(' ', '').lower()
        sentiment = understanding.get('sentiment', 'neutral')
        cats = understanding.get('categories', set())

        # 매칭된 감정 어간 찾기
        matched = None
        for kw in self.POSITIVE_EMOTIONS | self.NEGATIVE_EMOTIONS:
            if kw in text_compact:
                matched = kw
                break
        if matched is None:
            for c in cats:
                for kw in self.POSITIVE_EMOTIONS | self.NEGATIVE_EMOTIONS:
                    if kw in c or c in kw:
                        matched = kw
                        break
                if matched:
                    break

        is_pos = matched in self.POSITIVE_EMOTIONS or sentiment == 'positive'

        # 응답 매핑 — 감정별 직접 응답 (Levinson 1983 화행)
        if matched:
            DIRECT_REPLIES = {
                # 긍정
                '기뻐': ["오, 좋다", "기쁘구나, 잘됐다"],
                '신나': ["오 신나겠다!", "재밌겠네"],
                '재밌': ["재밌었구나", "오, 어떤 거?"],
                '행복': ["행복하다니 다행이다", "좋네, 그런 순간"],
                '좋아': ["응, 좋네", "좋다니 다행이야"],
                '최고': ["오 진짜?", "그렇게 좋아?"],
                # 부정
                '슬퍼': ["많이 슬프지...", "어떡해, 마음이 안 좋겠다"],
                '슬프': ["슬프구나... 무슨 일 있어?", "마음 안 좋겠다"],
                '힘들': ["많이 힘들지...", "괜찮아질 거야"],
                '지쳐': ["많이 지쳤구나, 좀 쉬어", "어떡해..."],
                '불안': ["불안하구나... 뭐 때문에?", "괜찮을 거야"],
                '걱정': ["걱정되겠다... 무슨 일?", "괜찮을 거야"],
                '무서': ["무섭지... 옆에 있어줄게", "어떡해, 괜찮아?"],
                '두려': ["두렵겠지... 같이 생각해보자", "괜찮아, 천천히"],
                '화나': ["많이 화났구나", "왜 그래, 무슨 일?"],
                '짜증': ["짜증나지... 뭐 때문에?", "어휴, 그럴 때 있지"],
                '외로': ["외롭지... 내가 옆에 있어", "함께 있어줄게"],
                '답답': ["답답하구나... 풀어보자", "어떤 부분이 그래?"],
                '우울': ["우울하구나... 천천히 얘기해봐", "마음 안 좋지..."],
            }
            # 매칭된 어간을 키로 찾기 (가장 긴 매칭)
            best_key = None
            best_len = 0
            for k in DIRECT_REPLIES:
                if k in matched and len(k) > best_len:
                    best_key = k
                    best_len = len(k)
            if best_key:
                replies = DIRECT_REPLIES[best_key]
                idx = self.respond_count % len(replies)
                return replies[idx]

        # 매칭 안 됐으면 sentiment 기반 일반 공감
        if is_pos:
            if tone['is_excited']:
                return "오 좋다!"
            if tone['is_warm']:
                return "그래, 좋네"
            return "응, 좋네"
        else:
            if tone['is_warm']:
                return "어... 많이 힘들지"
            if tone['is_anxious']:
                return "어떡해, 괜찮아?"
            return "어... 그렇구나"

    def _respond_greeting(self, text: str, understanding: Dict, tone: Dict) -> str:
        """인사"""
        cats = understanding.get('categories', set())
        person = None
        for c in cats:
            if c in self.NAME_KEYS - {'EVE'}:
                person = c
                break

        if tone['is_warm']:
            if person and person == '민석':
                return f"안녕, {person}!"
            return "안녕!"

        if tone['is_anxious']:
            return "어... 안녕"

        if tone['is_excited']:
            return "오 안녕!"

        if person and person == '민석':
            return f"안녕, {person}"
        return "안녕"

    def _respond_name_call(self, text: str, understanding: Dict, tone: Dict) -> str:
        """이름 호명 — '민석아'"""
        cats = understanding.get('categories', set())
        if '민석' in cats or '민석아' in text:
            if tone['is_warm']:
                return "응, 왜?"
            if tone['is_anxious']:
                return "응... 왜"
            return "응, 왜?"
        if '너' in cats:
            if tone['is_warm']:
                return "응? 나?"
            return "어"
        return "응?"

    def _respond_affirm(self, tone: Dict) -> str:
        """짧은 호응"""
        if tone['is_warm']:
            return "응응"
        return "응"

    def _respond_statement(self, text: str, understanding: Dict, tone: Dict) -> str:
        """일반 진술 — sentiment + 카테고리 + 호르몬 기반 공감 응답"""
        cats = understanding.get('categories', set())
        sentiment = understanding.get('sentiment', 'neutral')
        focus = self.wm.get_focus()

        # 의미 있는 카테고리 추출 (조사 제거 + 메타 제거)
        meta_skip = {'나', 'EVE', '?', '!', '근데', '그래서', '그리고', '좀', '진짜',
                     '정말', '너무', '되게', '많이'}
        clean_cats = []
        for c in cats:
            if c in meta_skip:
                continue
            base = c
            for suf in ('가', '는', '도', '을', '를', '랑', '이', '에서', '한테',
                       '에게', '와', '과'):
                if c.endswith(suf) and len(c) > len(suf):
                    base = c[:-len(suf)]
                    break
            clean_cats.append(base)
        clean_cats.sort()  # 결정론

        primary_cat = clean_cats[0] if clean_cats else None

        # primary가 input과 다른 focus라면 보조로
        secondary = None
        if focus and focus not in cats and focus not in meta_skip:
            secondary = focus

        # ==== 응답 합성 — 2~3 chunk ====
        chunks = []

        # 1차: 화행 마크 (감정/관심 표현)
        if sentiment == 'positive':
            if tone['is_excited']:
                chunks.append("오 좋다!")
            elif tone['is_warm']:
                chunks.append("그래, 좋네")
            else:
                chunks.append("응, 좋네")
        elif sentiment == 'negative':
            if tone['is_warm']:
                chunks.append("어... 그렇구나")
            elif tone['is_anxious']:
                chunks.append("어... 괜찮아?")
            elif tone['is_low']:
                chunks.append("음...")
            else:
                chunks.append("어, 그렇구나")
        else:
            # neutral - 키워드 활용
            if primary_cat:
                if tone['is_warm']:
                    chunks.append(f"어, {primary_cat}?")
                else:
                    chunks.append(f"{primary_cat}, 그래")
            else:
                chunks.append("응")

        # 2차: 공감/관심 한 마디 — 무조건 추가 (길이 + empathy 향상)
        if sentiment == 'negative':
            empathy_pool = [
                "많이 힘들지",
                "괜찮아질 거야",
                "어떡해...",
                "속상하겠다",
                "마음이 안 좋아",
            ]
            # 결정론적 선택 — respond_count 기반 (메타에 따라)
            idx = self.respond_count % len(empathy_pool)
            if tone['is_warm']:
                chunks.append(empathy_pool[idx])
            elif tone['is_anxious']:
                chunks.append(empathy_pool[2])  # "어떡해"
            elif primary_cat and primary_cat not in {'그래', '응'}:
                chunks.append(f"{primary_cat} 때문에 그래?")
        elif sentiment == 'positive':
            empathy_pool = [
                "정말 잘됐다",
                "기쁘겠다",
                "축하해",
                "좋겠네",
            ]
            idx = self.respond_count % len(empathy_pool)
            if tone['is_warm'] or tone['is_excited']:
                chunks.append(empathy_pool[idx])
        else:
            # neutral — 관심 표현 또는 follow-up
            if primary_cat and tone['is_warm']:
                # primary_cat 기반 SA spread 한 단어 가져오기
                related = self._related_word(primary_cat)
                if related:
                    chunks.append(f"{related} 생각나네")
                else:
                    chunks.append("그렇구나")
            elif secondary and self.respond_count % 2 == 0:
                chunks.append(f"{secondary} 생각났어")

        if len(chunks) == 1:
            return chunks[0]
        return '. '.join(chunks)

    def _related_word(self, cat: str) -> Optional[str]:
        """카테고리 → 관련 단어 1개 (SA spread)"""
        if not hasattr(self.sa, 'neighbors'):
            return None
        neighbors = self.sa.neighbors.get(cat, set())
        if not neighbors:
            return None
        # 결정론 — 알파벳 순 첫 거 (?, 메타 제외)
        cands = sorted([n for n in neighbors
                       if not n.endswith('?') and len(n) >= 2
                       and n not in {'나', 'EVE'}])
        return cands[0] if cands else None

    # ============= 통계 =============
    def get_stats(self) -> Dict[str, Any]:
        return {
            'respond_count': self.respond_count,
            'pattern_hits': dict(self.pattern_hit_count),
            'fallback_count': self.fallback_count,
        }
