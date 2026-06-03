"""
KoreanMorphAnalyzer — 라운드 33 (C-1)

학계 기반 한국어 룰 베이스 형태소 분석기.

원칙:
- 결정론 (random 0)
- 신경망 X (트랜스포머 0, 통계 학습 X)
- 외부 의존성 0 (Python 표준만)
- 학계 자료 기반 (1985 학교문법 + 표준국어대사전)

학계 출처:
- 한국민족문화대백과사전 (문체법, 인용문)
- TOPIK 공식 문법 (라고/다고/자고/냐고)
- 학교문법 5분법: 평서/의문/명령/청유/감탄
- 임동훈 2011 "한국어의 문장 유형과 용법" (간접인용절 기준 4유형)

처리 단위:
- 어절 (띄어쓰기 단위)
- 어절 = 어간 + 어미 (용언) 또는 체언 + 조사 (체언)

분석 단계:
1. 어절 분리 (split)
2. 각 어절 → 어간/어미 또는 체언/조사
3. 종결어미 → 문장 유형 (평서/의문/명령/청유)
4. 인용 조사 → 직접/간접 인용 구조
"""

from dataclasses import dataclass, field
from typing import Optional


# =============================================================================
# 한국어 형태소 자료 (학계 기반)
# =============================================================================

# === 종결어미 (학교문법 5분법) ===
# 평서형
DECLARATIVE_ENDINGS = [
    # 해라체
    "ㄴ다", "는다", "다", "라",                              # 가-ㄴ다, 먹는다, 좋다, 하느니라
    # 해체 (반말)
    "아", "어", "여",                                        # 가, 먹어, 해
    # 해요체
    "아요", "어요", "여요", "예요",                          # 가요, 먹어요, 해요, 학생이에요
    # 합쇼체
    "ㅂ니다", "습니다",                                       # 갑니다, 먹습니다
    # 하오체 / 하게체 / 약속
    "오", "소", "네", "지", "거든", "ㄹ게", "을게",          # 가오, 좋소, 좋네, 좋지, 좋거든
    # 시제 + 다
    "었다", "았다", "였다", "겠다",                          # 갔다, 좋았다, 했다, 가겠다
    # 단축 인용 (간접인용 + 어요)
    "대", "래", "재", "냬",                                   # 좋대, 친구래, 가재, 어디냬
]

# 의문형
INTERROGATIVE_ENDINGS = [
    "까", "습니까", "ㅂ니까",                                # ?
    "냐", "느냐", "는가", "ㄴ가", "는지", "ㄴ지",
    "니", "나", "ㄹ까", "을까",
    "어", "아", "지",                                         # (억양으로 의문 — 모호)
]

# 명령형
IMPERATIVE_ENDINGS = [
    "아라", "어라", "여라",
    "거라", "너라",
    "라",                                                     # 가라
    "세요", "셔요", "십시오", "으십시오",
    "으세요", "어요", "아요",                                 # 부드러운 명령
    "지마", "지마라", "지말아", "지말아라",                  # 부정 명령
]

# 청유형
PROPOSITIVE_ENDINGS = [
    "자",                                                     # 가자
    "세",                                                     # 가세
    "ㅂ시다", "읍시다",                                      # 갑시다
    "시지요", "시죠",
]

# 감탄형
EXCLAMATIVE_ENDINGS = [
    "구나", "는구나", "ㄴ구나",
    "군", "는군", "ㄴ군",
    "네",                                                     # 좋네! (억양)
    "로다",
    "아라", "어라",                                          # 형용사 + 아라/어라 = 감탄 (귀여워라)
]

# === 인용 조사 (간접/직접) ===
QUOTE_PARTICLES = {
    # 직접 인용 (체언 뒤)
    "라고":   {"type": "direct_or_noun", "modality": "any"},
    "이라고": {"type": "noun_indirect", "modality": "declarative"},
    "하고":   {"type": "direct_sound", "modality": "any"},
    # 간접 인용 — 평서
    "다고":   {"type": "indirect", "modality": "declarative"},
    "ㄴ다고": {"type": "indirect", "modality": "declarative"},
    "는다고": {"type": "indirect", "modality": "declarative"},
    "았다고": {"type": "indirect", "modality": "declarative_past"},
    "었다고": {"type": "indirect", "modality": "declarative_past"},
    "겠다고": {"type": "indirect", "modality": "declarative_future"},
    "아니라고": {"type": "indirect", "modality": "declarative_negation"},
    # 간접 인용 — 의문
    "냐고":   {"type": "indirect", "modality": "interrogative"},
    "느냐고": {"type": "indirect", "modality": "interrogative"},
    "으냐고": {"type": "indirect", "modality": "interrogative"},
    # 간접 인용 — 명령
    "라고":   {"type": "indirect", "modality": "imperative"},     # 동사 어간 + 라고
    "으라고": {"type": "indirect", "modality": "imperative"},
    # 간접 인용 — 청유
    "자고":   {"type": "indirect", "modality": "propositive"},
    # 변형 — 며/는/단/란
    "다며":   {"type": "indirect", "modality": "declarative", "variant": "and"},
    "라며":   {"type": "indirect", "modality": "any", "variant": "and"},
    "이라며": {"type": "indirect", "modality": "noun", "variant": "and"},
    "다는":   {"type": "indirect", "modality": "declarative", "variant": "modifier"},
    "라는":   {"type": "indirect", "modality": "any", "variant": "modifier"},
    "이라는": {"type": "indirect", "modality": "noun", "variant": "modifier"},
    "는다는": {"type": "indirect", "modality": "declarative", "variant": "modifier"},
}

# === 격조사 ===
CASE_PARTICLES = {
    # 주격
    "이":    "subject", "가": "subject",
    # 목적격
    "을":    "object",  "를": "object",
    # 관형격 (소유)
    "의":    "possessive",
    # 부사격 (장소/시간/도구/방향)
    "에":    "locative",
    "에서":  "ablative",
    "에게":  "dative",  "한테": "dative",
    "께":    "honorific_dative",
    "로":    "instrumental", "으로": "instrumental",
    "에로":  "directional",
    "부터":  "from",
    "까지":  "until",
    # 호격
    "야":    "vocative", "아": "vocative",
}

# === 보조사 (의미 더하기) ===
TOPIC_PARTICLES = {
    "은":    "topic", "는": "topic",
    "도":    "also",
    "만":    "only",
    "조차":  "even",
    "마저":  "even",
    "라도":  "even_if",
    "이라도": "even_if",
    "보다":  "than",
}

# === 부정 표현 ===
NEGATION_MARKERS = [
    "안",         # 짧은 부정
    "못",         # 능력 부정
    "지 않",      # 긴 부정
    "지 못",
    "지 말",      # 명령 부정
    "지마",       # 줄임
    "지 마",
    "아니",       # 명사 부정
]

# === 시제 선어말어미 ===
TENSE_PAST = ["었", "았", "였"]
TENSE_FUTURE = ["겠", "ㄹ 거", "을 거"]

# === 높임 선어말어미 ===
HONORIFIC_MARKERS = ["시", "으시", "십니", "셨", "셔", "세요", "셨어"]

# === 의문 어휘 (5W1H) ===
QUESTION_WORDS = [
    "누구", "누가",
    "뭐", "뭔", "무엇", "무슨",
    "언제", "어제",
    "어디", "어느",
    "왜", "어째서", "어찌",
    "어떻", "어떤", "어떡",
    "얼마", "몇",
]


# =============================================================================
# 한글 자모 분리 (시제/높임 같은 받침 기반 마커 인식용)
# =============================================================================

JONGSEONG_LIST = [
    '', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ',
    'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ',
    'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ'
]


def get_jongseong(ch: str) -> str:
    """한글 음절 → 종성 (받침). 한글 아니면 ''."""
    if not ch:
        return ''
    code = ord(ch) - 0xAC00
    if code < 0 or code > 11171:
        return ''
    return JONGSEONG_LIST[code % 28]


def has_past_tense_marker(text: str) -> bool:
    """
    과거 시제 — 받침 ㅆ + (다음 글자에 -어/-아) 패턴.
    "갔어, 했어, 먹었어, 빡셌어" → True
    "있다, 있는" → False (있 단독은 시제 아님)
    """
    for i, ch in enumerate(text):
        if get_jongseong(ch) != 'ㅆ':
            continue
        # 다음 글자가 어/았/었/...
        next_ch = text[i+1] if i+1 < len(text) else ''
        if next_ch in {'어', '아', '여', '었', '았', '여'}:
            # 또는 ㅆ 받침 음절의 *모음*이 ㅏ/ㅓ/ㅕ — 시제 융합 (갔/했/했)
            return True
        # 다음 없으면 "갔" 단독은 안 됨 — 보통 "갔어" 가 표준
    # 명시적 "었/았/였" 단어 검사
    for marker in ["었어", "았어", "였어", "었다", "았다", "였다",
                    "었던", "았던", "였던", "었지", "았지", "였지"]:
        if marker in text:
            return True
    return False


# =============================================================================
# Token 자료 구조
# =============================================================================

@dataclass
class Morpheme:
    """형태소 — 어간/어미/조사/단어"""
    surface: str             # 표면형 ("좋아한다고")
    lemma: str = ""           # 기본형 ("좋아하")
    pos: str = "unknown"      # 품사 (verb/adj/noun/particle/ending)
    features: dict = field(default_factory=dict)


@dataclass
class Eojeol:
    """어절 — 띄어쓰기 단위 (체언+조사 또는 어간+어미)"""
    surface: str
    morphs: list = field(default_factory=list)
    main: Optional[Morpheme] = None       # 의미 핵심 (체언 또는 어간)
    ending: Optional[str] = None          # 종결어미 (있으면)
    particle: Optional[str] = None        # 조사 (있으면)
    sentence_type: Optional[str] = None   # 평서/의문/명령/청유 (어절이 종결일 때)
    quote_particle: Optional[str] = None  # 인용 조사 (있으면)


@dataclass
class Sentence:
    """문장 분석 결과"""
    raw: str
    eojeols: list = field(default_factory=list)
    sentence_type: str = "declarative"     # 기본 평서
    has_question_word: bool = False
    is_negative: bool = False
    has_past: bool = False
    has_future: bool = False
    has_honorific: bool = False
    is_quoted: bool = False                # 간접/직접 인용 포함?
    quoted_content: Optional[str] = None
    quoted_modality: Optional[str] = None  # 인용 내용의 modality
    quoted_speech_verb: Optional[str] = None  # 인용 뒤 동사 (말해/물어/답해 등)


# =============================================================================
# 분석기
# =============================================================================

# 발화 동사 — 인용 뒤에 오는 (말하다, 묻다, 답하다, 부르다 등)
SPEECH_VERBS = {
    "말하", "말해", "말한", "말할", "말함", "말했",
    "얘기하", "얘기해", "얘기한", "얘기했",
    "이야기하", "이야기해",
    "답하", "답해", "답한", "답했",
    "대답하", "대답해", "대답했",
    "묻", "물어", "물은", "물을", "물으", "물었", "물었어",
    "질문하", "질문해", "질문했",
    "부르", "불러", "부른", "부를", "불렀",
    "외치", "외쳐", "외친", "외쳤",
    "외", "쳐",
    "표현하", "표현해", "표현했",
    "해", "한", "할", "함", "했",            # 보편 (해/했/할/했어/했다/했었)
    "해봐", "해줘", "했어", "했다", "했었",
}


class KoreanMorphAnalyzer:
    """한국어 룰 기반 형태소 분석기."""

    def __init__(self):
        # 종결어미를 길이 역순으로 정렬 (긴 거 먼저 매칭)
        self._all_endings = self._build_ending_table()
        # 인용 조사도 길이 역순
        self._quote_particles_sorted = sorted(
            QUOTE_PARTICLES.keys(), key=len, reverse=True
        )
        # 격조사 + 보조사
        self._all_particles = {**CASE_PARTICLES, **TOPIC_PARTICLES}
        self._particles_sorted = sorted(
            self._all_particles.keys(), key=len, reverse=True
        )

    def _build_ending_table(self):
        """
        모든 종결어미 → (어미, 문장유형) 매핑.
        - 길이 역순 (긴 거 먼저)
        - 같은 길이 어미 우선순위:
          * 명확한 어미 (긴 것, ㄴ다/구나/십시오 등) → 그 type
          * 모호한 어미 (-아/-어/-지) → 평서 디폴트 (의문/명령은 ?와 문맥으로)
        """
        # 학계 분석:
        # -아/-어/-여 = 해체 평서 (가장 흔함). 의문은 '?' 동반.
        # -지 = 평서 (확인). 의문은 '?'.
        # -까 = 의문 (명확)
        # -자 = 청유 (명확)
        # -아라/-어라 = 명령 (명확)
        # -구나 = 감탄 (명확)
        ambiguous = {"아", "어", "여", "지"}    # 평서 디폴트
        priority = {
            "imperative": 0,
            "propositive": 1,
            "interrogative": 2,
            "exclamative": 3,
            "declarative": 4,
        }
        table = []
        seen = set()
        all_endings = (
            [(e, "imperative") for e in IMPERATIVE_ENDINGS] +
            [(e, "propositive") for e in PROPOSITIVE_ENDINGS] +
            [(e, "interrogative") for e in INTERROGATIVE_ENDINGS] +
            [(e, "exclamative") for e in EXCLAMATIVE_ENDINGS] +
            [(e, "declarative") for e in DECLARATIVE_ENDINGS]
        )
        # 모호 어미는 평서 강제, 그 외는 우선순위
        for ending, stype in sorted(all_endings, key=lambda x: priority[x[1]]):
            if ending in seen:
                continue
            if ending in ambiguous:
                table.append((ending, "declarative"))    # 모호는 평서 디폴트
            else:
                table.append((ending, stype))
            seen.add(ending)
        # 길이 역순 정렬
        table.sort(key=lambda x: -len(x[0]))
        return table

    # ===== 메인 진입점 =====

    def analyze(self, text: str) -> Sentence:
        """문장 → 구조 분석."""
        text = text.strip()
        # 구두점 제거 (분석용 — 의문 표시는 따로 보존)
        ends_with_question = text.endswith("?")
        ends_with_exclaim = text.endswith("!")
        cleaned = text.rstrip("?!.").strip()

        sentence = Sentence(raw=text)
        eojeols_str = cleaned.split()

        for w in eojeols_str:
            eojeol = self._analyze_eojeol(w)
            sentence.eojeols.append(eojeol)
            # 인용 조사 발견
            if eojeol.quote_particle:
                sentence.is_quoted = True
                sentence.quoted_modality = QUOTE_PARTICLES[eojeol.quote_particle].get("modality")

        # 의문 어휘 체크
        for qw in QUESTION_WORDS:
            if qw in cleaned:
                sentence.has_question_word = True
                break

        # 부정 — 단어 경계 확인 (어절 단위)
        # "안"은 단독 어절이거나 "안+공백+동사"일 때만 부정
        # "안녕" 같은 단어 안에 들어있으면 X
        eojeols_str = cleaned.split()
        for n in NEGATION_MARKERS:
            if n == "안":
                # "안" 단독 어절 또는 "안" + 공백 + 다음 어절
                if "안" in eojeols_str:
                    sentence.is_negative = True
                    break
            elif n == "못":
                if "못" in eojeols_str:
                    sentence.is_negative = True
                    break
            elif n == "아니":
                # "아니" 단독 또는 "아니다/아니야"
                if "아니" in eojeols_str or any(e.startswith("아니") for e in eojeols_str):
                    sentence.is_negative = True
                    break
            else:
                # 긴 부정 표현 — substring으로
                if n in cleaned:
                    sentence.is_negative = True
                    break

        # 과거 시제 — 받침 ㅆ 또는 명시적 었/았/였
        if has_past_tense_marker(cleaned):
            sentence.has_past = True
        else:
            for t in TENSE_PAST:
                if t in cleaned:
                    sentence.has_past = True
                    break
        # 미래/추측
        for t in TENSE_FUTURE:
            if t in cleaned:
                sentence.has_future = True
                break

        # 높임 — 키워드 매칭
        for hm in HONORIFIC_MARKERS:
            if hm in cleaned:
                sentence.has_honorific = True
                break

        # 문장 유형 결정 (마지막 어절의 종결어미 기준)
        sentence.sentence_type = self._determine_sentence_type(
            sentence, ends_with_question, ends_with_exclaim
        )

        # 인용된 발화 추출
        if sentence.is_quoted:
            self._extract_quoted_content(sentence, cleaned)

        return sentence

    # ===== 어절 분석 =====

    def _analyze_eojeol(self, w: str) -> Eojeol:
        """어절 → 어간/어미/조사 분리."""
        e = Eojeol(surface=w)

        # 1. 어절 내부에 *인용 조사 + 발화 동사* 패턴 (띄어쓰기 누락)
        # 예: "좋아한다고해" = "좋아한" + "다고" + "해"
        for qp in self._quote_particles_sorted:
            idx = w.find(qp)
            if idx <= 0:
                continue
            # qp 뒤에 발화 동사가 있는가?
            after_qp = w[idx + len(qp):]
            if not after_qp:
                continue
            # 짧은 발화 동사 시작 ("해/한/했/말/물/...")
            short_speech = ["해", "한", "했", "할", "함", "말", "물", "답", "부",
                           "얘기", "이야기", "외", "표"]
            for ss in short_speech:
                if after_qp.startswith(ss):
                    # 인용 + 동사 패턴 인식
                    e.quote_particle = qp
                    e.main = Morpheme(
                        surface=w[:idx],
                        lemma=w[:idx],
                        pos="quoted_clause",
                    )
                    # speech_verb_after 표시
                    e.features = getattr(e, "features", {}) or {}
                    if not hasattr(e, 'features') or e.features is None:
                        pass
                    return e

        # 2. 인용 조사 우선 매칭 (어절 끝)
        for qp in self._quote_particles_sorted:
            if w.endswith(qp):
                e.quote_particle = qp
                e.main = Morpheme(
                    surface=w[:-len(qp)],
                    lemma=w[:-len(qp)],
                    pos="quoted_clause",
                )
                return e

        # 3. 종결어미 매칭
        for ending, stype in self._all_endings:
            if w.endswith(ending) and len(w) > len(ending):
                e.ending = ending
                e.sentence_type = stype
                e.main = Morpheme(
                    surface=w[:-len(ending)],
                    lemma=w[:-len(ending)],
                    pos="verb_or_adj_stem",
                )
                return e

        # 4. 조사 매칭
        for p in self._particles_sorted:
            if w.endswith(p) and len(w) > len(p):
                e.particle = p
                e.main = Morpheme(
                    surface=w[:-len(p)],
                    lemma=w[:-len(p)],
                    pos="noun",
                    features={"particle_type": self._all_particles[p]},
                )
                return e

        # 5. 매칭 없음
        e.main = Morpheme(surface=w, lemma=w, pos="word")
        return e

    # ===== 문장 유형 결정 =====

    def _determine_sentence_type(self, sent: Sentence,
                                 ends_q: bool, ends_e: bool) -> str:
        """
        문장 유형 결정.
        우선순위:
          0. ! 표시 → 감탄 강제 (가장 명확)
          1. ? + 의문 어휘 → 의문 강제
          2. 마지막 어절 종결어미
          3. ? 단독 → 의문
          4. 디폴트 평서
        """
        # 0. ! 표시 = 명확한 감탄 (사용자 의도 우선)
        if ends_e and not ends_q:
            return "exclamative"
        
        # 1. ? + 의문 어휘 동시
        if ends_q and sent.has_question_word:
            return "interrogative"

        # 2. 마지막 어절 종결어미
        if sent.eojeols:
            last = sent.eojeols[-1]
            if last.sentence_type:
                # 모호 어미 — ? 있으면 의문
                ambiguous = {"아", "어", "여", "지"}
                if last.ending in ambiguous and ends_q:
                    return "interrogative"
                return last.sentence_type

        # 3. ? 표시
        if ends_q or sent.has_question_word:
            return "interrogative"
        return "declarative"

    # ===== 인용된 발화 추출 =====

    def _extract_quoted_content(self, sent: Sentence, cleaned: str):
        """
        인용 조사 위치 찾아서 인용된 부분 + 발화 동사 추출.
        어절 내부 또는 다음 어절에서 동사 검색.
        """
        for i, e in enumerate(sent.eojeols):
            if not e.quote_particle:
                continue
            # 인용 조사 *앞*까지가 인용 내용
            quoted = e.main.surface if e.main else ""
            sent.quoted_content = quoted
            
            # A. 같은 어절 내부에 발화 동사가 있나? (띄어쓰기 누락 케이스)
            qp = e.quote_particle
            qp_idx = e.surface.find(qp)
            if qp_idx >= 0:
                after_qp = e.surface[qp_idx + len(qp):]
                if after_qp:
                    for v in SPEECH_VERBS:
                        if after_qp.startswith(v):
                            sent.quoted_speech_verb = after_qp
                            return
            
            # B. 다음 어절에서 발화 동사
            if i + 1 < len(sent.eojeols):
                next_eojeol = sent.eojeols[i + 1]
                next_main = next_eojeol.main.surface if next_eojeol.main else next_eojeol.surface
                for v in SPEECH_VERBS:
                    if next_main.startswith(v) or next_eojeol.surface.startswith(v):
                        sent.quoted_speech_verb = next_eojeol.surface
                        return
            return

    # ===== 단순 헬퍼 =====

    def is_question(self, text: str) -> bool:
        return self.analyze(text).sentence_type == "interrogative"

    def is_imperative(self, text: str) -> bool:
        return self.analyze(text).sentence_type == "imperative"

    def is_propositive(self, text: str) -> bool:
        return self.analyze(text).sentence_type == "propositive"

    def is_declarative(self, text: str) -> bool:
        return self.analyze(text).sentence_type == "declarative"

    def has_quote(self, text: str) -> bool:
        return self.analyze(text).is_quoted

    def is_teaching_quote(self, text: str) -> bool:
        """
        가르침 인용? — 간접 인용 + 발화 동사가 명령형 (해/말해/부르)
        """
        s = self.analyze(text)
        if not s.is_quoted:
            return False
        if not s.quoted_speech_verb:
            return False
        # 발화 동사가 명령/평서 짧은형 ("해", "말해", "부르")이면 가르침일 가능성
        v = s.quoted_speech_verb
        teaching_speech = {"해", "말해", "부르", "불러", "답해", "얘기해"}
        for t in teaching_speech:
            if v.startswith(t):
                return True
        return False
