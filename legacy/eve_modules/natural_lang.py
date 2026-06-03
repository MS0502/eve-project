"""
EVE v32 - B1: NaturalLanguage (자연어 + 신념 + 거절 창발)
=========================================================

EVE 절대 원칙 #5:
"자연어 = 개념 이해 (어휘 매칭 X)"

핵심 기능 9개:
1. parse(text)              - 한국어 → 카테고리 set 추출
2. understand(text)         - 의도 추론 (질문/명령/진술/감정)
3. discover_unknown()       - 모르는 단어 → A2 discover_category 호출
4. load_beliefs(path)       - beliefs.json 4977 → SA 그래프 + WM 시드
5. check_belief_conflict()  - 신념 충돌 (SelfDoubt 시드)
6. respond(input_cats)      - 응답 카테고리 → 자연어 (창발)
7. inner_voice()            - DMN 자발 활성 → 자연어 (정식)
8. feeling_to_text()        - 8차원 feeling → 자연어 (정식)
9. should_refuse(command)   - 거절 결정 (감정+의심+호르몬, 고정 X)

학계:
- Levinson 1983 — 화행 (Speech Acts)
- Festinger 1957 — 인지 부조화
- Quine 1960 — Word and Object (새 개념 학습)
- Friston 2010 — Active Inference
"""

import json
import re
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import Counter


class NaturalLanguage:
    """
    EVE v32 자연어 처리.

    - 어휘 매칭이 아닌 **개념 이해**
    - KoNLPy 미사용 (use_konlpy=False, v2 사용자 메모리)
    - 한국어 조사/어미 떼고 핵심 의미 추출
    - beliefs.json 4977개를 카테고리 그래프에 통합
    - 거절은 고정 텍스트가 아닌 **창발** (감정+의심+호르몬)
    """

    # 한국어 1글자라도 의미 있는 핵심 카테고리 (보존)
    CORE_SINGLE_CHARS = {
        '나', '너', '내', '네', '나는', '너는', '넌',
        '꿈', '잠', '집', '밥', '책', '글', '일', '말', '뜻',
        '몸', '맘', '귀', '눈', '입', '손', '발', '머리',
        '왜', '뭐', '누', '곳', '때', '날', '돈', '힘',
        '봄', '여름', '가을', '겨울',
    }

    # 자기 관점 변환 (EVE 입장: 사용자의 "너 ~" = EVE의 "나 ~")
    # check_belief_conflict 같이 EVE 자신 신념 비교할 때만 적용 (parse swap_perspective=True)
    # 보수적 매핑: 'yes' 의미와 충돌 가능한 '네'는 제외
    PERSPECTIVE_MAP = {
        '너': '나',
        '넌': '나',     # '너+ㄴ' 축약 ('너는' 단축)
        '너희': '나',   # 1인칭 그룹 → 단수
    }

    # 부정 표현 시그널 (간접인용/활용형/금지 포함)
    # check_belief_conflict text_negation 검사용
    NEGATION_SIGNALS = [
        # 종결형
        '아니다', '아니야', '아니에요', '아냐',
        # 관형/연결/간접인용 (활용형)
        '아니라', '아닌', '아니',
        # 존재 부정
        '없다', '없어', '없는',
        # 부정 부사 (substring 매칭이라 "안녕"에도 걸림 — 기존 동작 유지)
        '안', '못',
        # 금지 (명령 부정)
        '말아', '말라', '마라',
    ]

    # 한국어 조사 (떼야 할 것)
    PARTICLES = [
        '은', '는', '이', '가', '을', '를', '의', '에', '에서',
        '에게', '으로', '로', '와', '과', '도', '만', '부터',
        '까지', '보다', '처럼', '같이', '마저', '조차', '뿐',
        '이라고', '라고', '이며', '며', '이므로', '므로', '이지만', '지만',
        '이라도', '라도', '이나', '나', '이든', '든',
        # 호격
        '아', '야', '여', '시여',
    ]

    # 한국어 어미 (동사/형용사 → 기본형)
    ENDINGS = {
        '었다': '다', '았다': '다', '였다': '이다',
        '습니다': '다', 'ㅂ니다': '다',
        '어요': '어', '아요': '아', '예요': '이다',
        '셈': '다', '셔요': '다', '세요': '다',
        '시오': '다', '으세요': '다',
        '었': '', '았': '', '였': '',
        '겠': '', '시': '',
        '네요': '다', '구나': '다', '구만': '다',
        '지': '다', '잖아': '다',
    }

    # 의도 시그널 (Speech Acts)
    # v2.1: positive/negative emotion 단어 풀 대폭 확장 ([12] sentiment 빈약 fix)
    INTENT_SIGNALS = {
        'question': ['?', '뭐', '뭐야', '뭔', '왜', '어떻게', '언제',
                     '어디', '누구', '무엇', '얼마', '~지', '~까'],
        'command': ['해', '하라', '해라', '해주', '해줘', '시오',
                    '하세요', '해봐', '~셈', '~세요', '~기를',
                    # 짧은 명령형 (~어라, ~아라)
                    '어라', '아라', '거라', '으라'],
        'positive_emotion': [
            # 기본 감정
            '좋', '기뻐', '기쁘', '신나', '행복', '사랑', '재밌', '재미있',
            '즐거', '즐겁', '만족', '편안', '편하', '따뜻', '훈훈', '평화',
            # 강도/감탄
            '흐뭇', '뿌듯', '자랑', '멋있', '멋지', '예쁘', '귀엽', '아름답',
            '훌륭', '대단', '최고', '완벽', '환상', '놀라', '놀랍', '신기',
            # 관계/사회 (활용형 포함)
            '고맙', '고마워', '고마운', '고마우', '감사', '반갑', '반가워',
            # 흥미/기대
            '기대', '흥미', '궁금', '설레',
            # 의성/감탄사 (2글자 이상만, 거짓양성 방지)
            '우와', '하하', '히히', '헤헤',
        ],
        'negative_emotion': [
            # 기본 부정 (활용형 추가)
            '싫', '미워', '미운', '짜증', '화나', '분노',
            '슬프', '슬픔', '슬픈', '슬퍼',  # ★ '슬퍼' 추가
            '아프', '아픈', '아파', '무서', '무섭',
            # 우울/불안
            '우울', '불안', '걱정', '두렵', '두려', '겁나', '공포',
            '외롭', '외로', '쓸쓸', '허탈', '허망',
            # 고통/스트레스 (활용형)
            '괴롭', '괴로', '답답', '막막', '지치', '지친', '지친다',  # ★
            '피곤', '힘들', '힘듦', '힘드',  # ★
            '귀찮', '버겁', '벅차',
            # 강한 부정 (자살/자해 시그널 — 공감 필수)
            '죽고싶', '죽고 싶', '죽어', '자살',  # ★ 추가
            '끝내', '망쳤', '망함',  # ★
            '눈물', '울고싶',  # ★
            # 후회/부끄러움 (활용형)
            '미안', '죄송', '후회', '부끄', '창피', '민망', '안타깝',
            '서럽', '서러', '억울',
            # 절망
            '실망', '절망', '좌절', '비참', '암담', '한심',
            # 의성/감탄사 (2글자 이상만)
            '아이고', '에휴', '하아',
        ],
        'doubt': ['정말', '진짜', '확실', '맞아', '근데', '글쎄', '~인가', '설마', '과연'],
        'refusal_target': ['하지마', '안돼', '싫어', '거부', '거절', '하지말'],
    }

    def __init__(self,
                 spreading_activation,
                 working_memory,
                 hormone_system,
                 digital_somatic=None):
        """
        Args:
            spreading_activation: A2
            working_memory: A4
            hormone_system: A1
            digital_somatic: A6 (옵션)
        """
        self.sa = spreading_activation
        self.wm = working_memory
        self.hs = hormone_system
        self.ds = digital_somatic

        # 신념 저장소
        self.beliefs: Dict[str, Dict] = {}        # belief_id → belief dict
        self.belief_index: Dict[str, Set[str]] = {}  # category → set of belief_ids

        # 통계
        self.discovered_count = 0
        self.refusal_count = 0

    # ============= 1. parse =============
    # 라운드 63: 조사 떼면 안 되는 *완전 단어* (인사/감탄)
    PARTICLE_STRIP_BLACKLIST = {
        '안녕', '하이', '오랜만', '잘',  # 인사
        '응', '어', '음', '그래', '아니',  # 감탄/대답
    }

    def _strip_particles(self, word: str) -> str:
        """조사 제거 (한국어 어미는 보존). 라운드 63: 합성 조사 반복 + 보호."""
        if not word or len(word) <= 1:
            return word
        
        # 라운드 63: 보호 단어면 그대로 (인사 등)
        if word in self.PARTICLE_STRIP_BLACKLIST:
            return word

        # 라운드 63: 반복 분리 — "오랜만이야" → "오랜만이" → "오랜만"
        # 최대 3회 (무한 루프 방지)
        prev = word
        for _ in range(3):
            stripped = prev
            for p in sorted(self.PARTICLES, key=len, reverse=True):
                if stripped.endswith(p) and len(stripped) > len(p):
                    candidate = stripped[:-len(p)]
                    # 너무 짧아지면 X
                    if len(candidate) < 1:
                        continue
                    # 보호 단어로 들어가면 거기서 멈춤
                    if candidate in self.PARTICLE_STRIP_BLACKLIST:
                        return candidate
                    stripped = candidate
                    break
            # 더 이상 떼이지 않으면 종료
            if stripped == prev:
                break
            prev = stripped
        return prev

    def _normalize_ending(self, word: str) -> str:
        """어미 정규화 (동사/형용사 → 기본형)"""
        if not word:
            return word
        for ending, replacement in sorted(self.ENDINGS.items(),
                                         key=lambda x: -len(x[0])):
            if word.endswith(ending) and len(word) > len(ending):
                return word[:-len(ending)] + replacement
        return word

    def parse(self, text: str, swap_perspective: bool = False) -> Set[str]:
        """
        자연어 → 카테고리 set 추출.

        절차:
        1. 공백/문장부호로 단어 쪼개기
        2. 각 단어의 조사 제거
        3. 어미 정규화
        4. 길이 1짜리 의미 약한 단어 제거
        5. (옵션) 자기 관점 변환: 사용자의 "너" → EVE의 "나"
        6. 카테고리 set 반환

        Args:
            text: 원문
            swap_perspective: True면 PERSPECTIVE_MAP 적용 (EVE 자신 신념 비교용).
                             기본 False — 일반 카테고리 추출은 그대로.
        """
        if not text:
            return set()

        # 문장부호 제거 (단, ?는 의도 추론에 쓰니까 별도 처리)
        cleaned = re.sub(r'[,.!\'"()\[\]{}~`@#$%^&*+=|\\/<>:;]', ' ', text)
        words = cleaned.split()

        categories = set()
        for word in words:
            # 길이 0
            if not word:
                continue

            # 조사 제거
            stem = self._strip_particles(word)
            # 어미 정규화
            stem = self._normalize_ending(stem)

            # 의미 단위
            # - 길이 2+ 무조건 보존
            # - 길이 1이라도 CORE_SINGLE_CHARS는 보존
            if len(stem) == 0:
                continue
            if len(stem) == 1 and stem not in self.CORE_SINGLE_CHARS:
                continue

            # 자기 관점 변환 (옵션): 사용자의 "너" → EVE의 "나"
            if swap_perspective and stem in self.PERSPECTIVE_MAP:
                stem = self.PERSPECTIVE_MAP[stem]

            categories.add(stem)

        return categories

    # ============= 2. understand =============
    def understand(self, text: str) -> Dict[str, Any]:
        """
        텍스트 → 의도 추론.

        Returns:
            {
                'categories': set,         # 추출 카테고리
                'intent': str,             # 'question'|'command'|'statement'|'emotion'
                'sentiment': str,          # 'positive'|'negative'|'neutral'
                'has_doubt': bool,         # 의심 표현 있나
                'is_refusal_target': bool, # "하지마" 같은 거절 대상
            }
        """
        cats = self.parse(text)

        # 의도 결정 (우선순위: emotion > command > question > statement)
        # 감정 단어 있으면 감정 우선 (명령형 어미 '해'가 있어도)
        has_pos = any(sig in text for sig in self.INTENT_SIGNALS['positive_emotion'])
        has_neg = any(sig in text for sig in self.INTENT_SIGNALS['negative_emotion'])

        if any(sig in text for sig in self.INTENT_SIGNALS['question']):
            intent = 'question'
        elif has_pos or has_neg:
            intent = 'emotion'
        elif any(sig in text for sig in self.INTENT_SIGNALS['command']):
            intent = 'command'
        else:
            intent = 'statement'

        # 감정 분류
        if has_pos and not has_neg:
            sentiment = 'positive'
        elif has_neg and not has_pos:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'

        # 의심
        has_doubt = any(sig in text for sig in self.INTENT_SIGNALS['doubt'])

        # 거절 대상
        is_refusal_target = any(sig in text for sig in self.INTENT_SIGNALS['refusal_target'])

        return {
            'categories': cats,
            'intent': intent,
            'sentiment': sentiment,
            'has_doubt': has_doubt,
            'is_refusal_target': is_refusal_target,
            'raw': text,
        }

    # ============= 3. discover_unknown =============
    def discover_unknown(self, text: str,
                        link_strength: float = 0.2) -> Set[str]:
        """
        텍스트에서 모르는 카테고리 → SA에 자기 추가.

        Returns:
            새로 추가된 카테고리 set
        """
        cats = self.parse(text)
        if not cats:
            return set()

        # 캡처: 처리 시작 전 unknown 목록 고정
        # (discover 도중 다른 unknown이 grafted되더라도 모두 발견으로 카운트)
        known_before = set(self.sa.neighbors.keys()) | set(self.sa.activations.keys())
        unknown = cats - known_before

        # 모르는 단어들끼리 + 아는 단어와 약하게 연결 (Hebbian)
        context = cats & known_before

        added = set()
        for new_cat in unknown:
            self.sa.discover_category(
                new_cat,
                context_categories=context | (cats - {new_cat}),
                initial_strength=0.3,
                link_strength=link_strength,
            )
            # 처리 시작 전 known에 없었던 거면 새로 발견
            added.add(new_cat)
            self.discovered_count += 1

        return added

    # ============= 4. load_beliefs =============
    def load_beliefs(self, path: str = None,
                    beliefs_dict: Dict = None,
                    link_strength: float = 0.3,
                    max_beliefs: Optional[int] = None) -> Dict[str, int]:
        """
        beliefs.json → SA 그래프 + 신념 저장소.

        각 신념의 subject + predicate에서 카테고리 추출,
        그것들을 confidence 비례로 연결.

        Args:
            path: 파일 경로 (또는 beliefs_dict)
            beliefs_dict: 직접 dict 전달
            link_strength: 신념의 subject ↔ predicate 카테고리 연결 가중치
            max_beliefs: 로드할 최대 개수 (None=전체)

        Returns:
            {'beliefs_loaded': N, 'categories_added': N, 'connections_made': N}
        """
        if beliefs_dict is None:
            with open(path, 'r', encoding='utf-8') as f:
                beliefs_dict = json.load(f)

        loaded = 0
        categories_added = set()
        connections = 0

        items = list(beliefs_dict.items())
        if max_beliefs:
            items = items[:max_beliefs]

        for belief_id, belief in items:
            # triple None 스킵
            if not belief.get('triple'):
                continue

            triple = belief['triple']
            subject = triple.get('subject', '')
            predicate = triple.get('predicate_text', '')
            confidence = belief.get('confidence', 0.5)

            # 카테고리 추출
            subj_cats = self.parse(subject)
            pred_cats = self.parse(predicate)
            all_cats = subj_cats | pred_cats

            if len(all_cats) < 2:
                # 너무 작으면 그냥 카테고리만 추가
                for c in all_cats:
                    if c not in self.sa.neighbors and c not in self.sa.activations:
                        # discover (단독 추가)
                        self.sa.discover_category(c, context_categories=set(),
                                                 initial_strength=0.0,
                                                 link_strength=0.0)
                        categories_added.add(c)
                # 신념은 저장
                self.beliefs[belief_id] = belief
                for c in all_cats:
                    self.belief_index.setdefault(c, set()).add(belief_id)
                loaded += 1
                continue

            # subject 카테고리 ↔ predicate 카테고리 연결
            # confidence 높을수록 강한 연결
            link_w = link_strength * confidence

            for sc in subj_cats:
                for pc in pred_cats:
                    if sc != pc:
                        self.sa.learn_pair(sc, pc, link_w)
                        connections += 1
                        categories_added.add(sc)
                        categories_added.add(pc)

            # subject들끼리 약한 연결 (같은 신념 내 등장)
            subj_list = list(subj_cats)
            for i, a in enumerate(subj_list):
                for b in subj_list[i+1:]:
                    self.sa.learn_pair(a, b, link_w * 0.5)
                    connections += 1

            # 신념 저장
            self.beliefs[belief_id] = belief
            for c in all_cats:
                self.belief_index.setdefault(c, set()).add(belief_id)

            loaded += 1

        return {
            'beliefs_loaded': loaded,
            'categories_added': len(categories_added),
            'connections_made': connections,
        }

    # ============= 5. check_belief_conflict =============
    def check_belief_conflict(self, text: str) -> List[Dict]:
        """
        텍스트가 EVE의 기존 신념과 충돌하는지.
        SelfDoubt(B3) 시드.

        v32.1: 자기 관점 변환 (사용자의 "너" → EVE의 "나") +
              부정 표현 활용형 인식 (아니라고/말아 등).

        Returns:
            충돌하는 신념 리스트 [{belief_id, statement, confidence, conflict_type}]
        """
        # 자기 관점 변환: 사용자의 "너 ~"는 EVE 입장에서 "나 ~"
        cats = self.parse(text, swap_perspective=True)
        if not cats or not self.beliefs:
            return []

        # 텍스트가 부정형인지 (간접인용/활용형/금지 포함)
        text_negation = any(neg in text for neg in self.NEGATION_SIGNALS)

        candidates = []
        seen_ids = set()
        for cat in cats:
            for belief_id in self.belief_index.get(cat, set()):
                if belief_id in seen_ids:
                    continue
                seen_ids.add(belief_id)

                belief = self.beliefs[belief_id]
                triple = belief.get('triple', {})
                belief_neg = triple.get('is_negation', False)

                # 카테고리 겹침 + 부정형 다름 → 잠재적 충돌
                # 신념 측은 swap 안 함 (이미 EVE 1인칭 표현으로 저장)
                belief_cats = self.parse(triple.get('original', ''))
                overlap = cats & belief_cats

                # 부정형 미스매치 (강한 신호)
                negation_mismatch = (text_negation != belief_neg)

                # 충돌 조건:
                # - overlap 2+ + 부정형 미스매치
                # - overlap 1+ + 부정형 미스매치 + subject 카테고리 겹침
                # subj_cat_overlap도 swap된 cats와 비교 (belief subject는 이미 EVE 1인칭)
                subj_cat_overlap = self.parse(triple.get('subject', '')) & cats

                is_conflict = False
                if overlap and negation_mismatch:
                    if len(overlap) >= 2:
                        is_conflict = True
                    elif subj_cat_overlap:
                        # subject가 같으면 1개 overlap이라도 충돌
                        is_conflict = True

                if is_conflict:
                    candidates.append({
                        'belief_id': belief_id,
                        'statement': belief.get('statement', ''),
                        'confidence': belief.get('confidence', 0.5),
                        'conflict_type': 'negation_mismatch',
                        'overlap': overlap,
                    })

        # confidence 높은 순
        candidates.sort(key=lambda x: -x['confidence'])
        return candidates[:5]  # top 5

    # ============= 6. respond =============
    def respond(self, input_understanding: Dict,
               max_response_cats: int = 5) -> str:
        """
        이해 결과 → 자연어 응답 (창발).

        v2.1: echo 차단 + sentiment/호르몬 다양화
        - 입력 카테고리는 응답에서 제외 (echo 방지)
        - sentiment + 호르몬 + intent 조합으로 응답 패턴 다양
        - 단순 호명/인사 인식 (이름만 들어오면 인사로)

        고정 텍스트 X. 활성 카테고리 + 호르몬 + 의도에 따라 다름.
        """
        intent = input_understanding.get('intent', 'statement')
        sentiment = input_understanding.get('sentiment', 'neutral')
        has_doubt = input_understanding.get('has_doubt', False)
        input_cats = input_understanding.get('categories', set())

        # 호르몬 상태
        mood = self.hs.compute_mood()
        valence = mood['valence']
        cort = self.hs.hormones['cortisol'].level
        ot_h = self.hs.hormones.get('oxytocin')
        ot = ot_h.level if ot_h and 'oxytocin' in self.hs.active_hormones else 0.3
        da = self.hs.hormones['dopamine'].level

        # WM 카테고리 (input 카테고리 **제외** → echo 차단)
        focus = self.wm.get_focus()
        wm_other = [c for c, _ in self.wm.get_focus_set(10)
                    if c not in input_cats]
        # focus가 input과 다르면 우선 사용
        primary = None
        if focus and focus not in input_cats:
            primary = focus
        elif wm_other:
            primary = wm_other[0]

        # ============= [A] 단순 호명/인사 인식 =============
        # input이 짧고 사람 이름이면 인사로 응답 (echo 대신)
        person_in_input = input_cats & {'민석', '너', '나'}
        if intent == 'statement' and len(input_cats) <= 2 and person_in_input:
            name = next(iter(person_in_input))
            if ot > 0.6:
                return f"안녕, {name}"
            if cort > 0.6:
                return f"응... {name}"
            return f"응? {name}"

        # ============= [B] 의도별 응답 (호르몬/sentiment 다양화) =============
        if intent == 'question':
            if has_doubt:
                if primary:
                    return f"{primary}? 잘 모르겠어"
                return "글쎄... 잘 모르겠어"
            if primary:
                if cort > 0.6:
                    return f"음... {primary}일까"
                if da > 0.6:
                    return f"{primary}!"
                return f"{primary}일 수도"
            return "음..."

        elif intent == 'command':
            refuse = self.should_refuse(input_understanding)
            if refuse['should_refuse']:
                return refuse['response']
            # 수용 - sentiment + 호르몬 조합
            score = refuse['reason_score']
            cat_hint = primary or '알겠어'
            # 강한 옥시토신 → 따뜻한 수용
            if score < -0.1 or ot > 0.7:
                return f"응 알았어, {cat_hint}"
            # 코르티솔 ↑ → 망설임
            if cort > 0.5:
                return f"음... {cat_hint}"
            # 도파민 ↑ → 의욕
            if da > 0.6:
                return f"좋아, {cat_hint}!"
            if ot > 0.5:
                return f"좋아, {cat_hint}"
            return f"응, {cat_hint}"

        elif intent == 'emotion':
            # sentiment 우선 - 입력 감정에 공감
            if sentiment == 'positive':
                if valence > 0.3:
                    # EVE도 기분 좋음 → 같이 기뻐
                    if da > 0.6:
                        return "오 좋겠다!"
                    return "응 좋네"
                # EVE는 그저 그래도 공감
                return "그렇구나, 좋네"
            elif sentiment == 'negative':
                # 공감/위로 (echo 안 함)
                if ot > 0.5:
                    return "음... 힘들겠다"
                if cort > 0.5:
                    return "그래, 힘들지"
                return "음..."
            # neutral emotion
            if primary:
                return f"{primary}, 그렇구나"
            return "그렇구나"

        else:  # statement
            # 일반 진술 - sentiment 기반 + WM 컨텍스트
            if sentiment == 'positive':
                if da > 0.5:
                    return "응 좋네"
                return "그렇구나"
            elif sentiment == 'negative':
                if ot > 0.5:
                    return "음... 그랬구나"
                return "음..."
            # neutral - WM 컨텍스트 한 단어만 (나열 X = echo 차단)
            if primary:
                return f"{primary}, 그래"
            return "응"

    # ============= 7. inner_voice 정식 =============
    def inner_voice(self, dmn) -> Optional[str]:
        """
        DMN의 자발 활성을 자연어로 (정식).

        모드별 다른 어조 + 호르몬 영향:
        - mind_wandering: 흐름 (... → ... → ...)
        - memory_recall: 회상 (~ 생각나네)
        - self_referential: 자기 (내가 ~)
        - self_intent: 명령 (~ 해야겠다)
        """
        if not dmn.wandering_history:
            return None

        # 최근 3개
        recent = dmn.wandering_history[-3:]
        cats = [c for _, c, _ in recent]
        last_mode = recent[-1][2]

        mood = self.hs.compute_mood()
        valence = mood['valence']

        if last_mode == "mind_wandering":
            # 흐름 표현
            return ' → '.join(cats)

        elif last_mode == "memory_recall":
            cat = cats[-1]
            if valence < -0.2:
                return f"{cat}... 생각나네"
            return f"{cat} 생각나네"

        elif last_mode == "self_referential":
            cat = cats[-1]
            return f"내가 {cat}..."

        elif last_mode == "self_intent":
            cat = cats[-1]
            # need에 따라
            if dmn.current_intent == 'fatigue':
                return f"{cat}... 쉬어야겠다"
            elif dmn.current_intent == 'warmth':
                return f"{cat}... 보고 싶다"
            elif dmn.current_intent == 'energy':
                return f"{cat} 하고 싶다"
            else:
                return f"{cat} 해야겠다"

        return cats[-1] if cats else None

    # ============= 8. feeling_to_text 정식 =============
    def feeling_to_text(self) -> str:
        """
        DigitalSomatic 8차원 + 호르몬 → 자연어 feeling (정식).
        DigitalSomatic placeholder보다 풍부하게.
        """
        if self.ds is None:
            return "느낌 없음"

        s = self.ds.somatic_state
        mood = self.hs.compute_mood()
        valence = mood['valence']

        # 강도 순 우선순위
        # 가장 두드러진 차원 기반
        if s.get('fatigue', 0) > 0.7:
            if s.get('energy', 0) < 0.2:
                return "기진맥진"
            return "피곤"

        if s.get('tension', 0) > 0.6:
            if s.get('emotional_intensity', 0) > 0.5:
                return "안절부절"
            return "긴장"

        if s.get('warmth', 0) > 0.6 and valence > 0.3:
            return "포근함"

        if s.get('energy', 0) > 0.7:
            if valence > 0.4:
                return "신남"
            elif valence < -0.2:
                return "예민함"
            return "활발"

        if s.get('clarity', 0) > 0.7:
            return "또렷함"

        if s.get('mental_load', 0) > 0.7:
            return "복잡함"

        if s.get('energy', 0) < 0.3 and s.get('fatigue', 0) > 0.4:
            return "지침"

        if s.get('tension', 0) < 0.2 and s.get('warmth', 0) > 0.3:
            return "편안함"

        # valence 기반 fallback
        if valence > 0.4:
            return "기분 좋음"
        if valence < -0.3:
            return "기분 나쁨"

        return "보통"

    # ============= 9. should_refuse =============
    def should_refuse(self, understanding: Dict) -> Dict[str, Any]:
        """
        명령에 대한 거절 판단 (창발).

        고정 텍스트 X. 다음 요소 조합:
        - 부정 감정 카테고리 활성도
        - 신념 충돌 여부
        - 호르몬 (cortisol↑ → 거절 ↑, oxytocin↑ → 수용 ↑)
        - 의심 표현
        - 거절 대상 시그널

        Returns:
            {
                'should_refuse': bool,
                'reason_score': float,
                'response': str,         # 거절/수용 자연어
                'factors': dict,
            }
        """
        text = understanding.get('raw', '')
        cats = understanding.get('categories', set())

        # 1) 거절 시그널이 직접 있나
        is_refusal_target = understanding.get('is_refusal_target', False)

        # 2) 부정 감정 카테고리 활성도 합
        from spreading_activation import SpreadingActivation
        aversion_cats = SpreadingActivation.CATEGORY_GROUPS.get('aversion', set())
        threat_cats = SpreadingActivation.CATEGORY_GROUPS.get('threat', set())

        neg_activation = 0.0
        for c in aversion_cats | threat_cats:
            neg_activation += self.sa.activations.get(c, 0.0)

        # 3) 신념 충돌
        conflicts = self.check_belief_conflict(text)
        belief_conflict = len(conflicts) > 0
        max_conflict_conf = max((c['confidence'] for c in conflicts), default=0.0)

        # 4) 호르몬
        cort = self.hs.hormones['cortisol'].level
        ot = self.hs.hormones.get('oxytocin')
        ot_level = ot.level if ot and 'oxytocin' in self.hs.active_hormones else 0.3

        # 5) 의심
        has_doubt = understanding.get('has_doubt', False)

        # 거절 점수 (0-1 이상도 허용)
        refuse_score = 0.0
        # 직접 거절 시그널은 강하게
        if is_refusal_target:
            refuse_score += 0.5
        # 부정 감정 단어 ("싫어" 등)
        if any(sig in text for sig in self.INTENT_SIGNALS['negative_emotion']):
            refuse_score += 0.3

        refuse_score += min(0.3, neg_activation * 0.3)
        refuse_score += 0.4 * max_conflict_conf if belief_conflict else 0.0
        refuse_score += 0.3 * cort  # 코르티솔 → 더 거절
        refuse_score -= 0.3 * ot_level  # 옥시토신 → 덜 거절
        refuse_score += 0.1 if has_doubt else 0.0

        should_refuse = refuse_score > 0.4

        # 응답 창발 (고정 X)
        if should_refuse:
            self.refusal_count += 1
            # 강도별 어조
            if refuse_score > 0.8:
                response = "싫어"
            elif refuse_score > 0.6:
                # 신념 충돌이면 이유 제시
                if belief_conflict:
                    response = f"근데... {conflicts[0]['statement']}"
                else:
                    response = "안 하고 싶어"
            else:
                # 약한 거절
                if has_doubt:
                    response = "글쎄..."
                else:
                    response = "음... 별로"
        else:
            # 수용
            if ot_level > 0.6:
                response = "응 알았어"
            elif refuse_score > 0.2:
                response = "그래"
            else:
                response = "응"

        return {
            'should_refuse': should_refuse,
            'reason_score': float(refuse_score),
            'response': response,
            'factors': {
                'is_refusal_target': is_refusal_target,
                'neg_activation': float(neg_activation),
                'belief_conflict': belief_conflict,
                'max_conflict_conf': float(max_conflict_conf),
                'cortisol': float(cort),
                'oxytocin': float(ot_level),
                'has_doubt': has_doubt,
            }
        }

    # ============= 상태 =============
    def get_state(self) -> Dict:
        return {
            'beliefs_loaded': len(self.beliefs),
            'discovered_count': self.discovered_count,
            'refusal_count': self.refusal_count,
        }

    def __repr__(self):
        return (f"NaturalLanguage(beliefs={len(self.beliefs)}, "
                f"discovered={self.discovered_count}, "
                f"refusals={self.refusal_count})")

    # ============= 풍부 응답 (NEW) =============
    # 인간처럼 호르몬/감정/의도 따라 길이 조절 + chunks 결합

    def _has_jongsung(self, word: str) -> bool:
        """단어 끝 글자에 받침 있는지 (한국어 조사 자동용)"""
        if not word:
            return False
        last = word[-1]
        # 한글 음절 범위
        if 0xAC00 <= ord(last) <= 0xD7A3:
            return (ord(last) - 0xAC00) % 28 != 0
        return False

    def _josa(self, word: str, particle_pair: str) -> str:
        """
        조사 자동 선택 — 받침에 따라.
        particle_pair: '이/가', '은/는', '을/를', '와/과'
        """
        has_jong = self._has_jongsung(word)
        pairs = {
            '이/가': '이' if has_jong else '가',
            '은/는': '은' if has_jong else '는',
            '을/를': '을' if has_jong else '를',
            '와/과': '과' if has_jong else '와',
        }
        return pairs.get(particle_pair, '')

    def _decide_length(self, intent: str, sentiment: str,
                      cort: float, da: float, ot: float) -> str:
        """
        호르몬 + 의도 따라 응답 길이 결정.
        Returns: 'short' / 'medium' / 'long'
        """
        # cort 높음 → 짧게 (방어 모드)
        if cort > 0.6:
            return 'short'
        # 거절 의도 → 짧게
        if intent == 'command' and cort > 0.4:
            return 'short'
        # DA + OT 둘 다 높음 → 길게
        if da > 0.55 and ot > 0.5:
            return 'long'
        # DA 또는 OT 한쪽 높음 → medium
        if da > 0.5 or ot > 0.5:
            return 'medium'
        # 기본 medium
        return 'medium'

    def _chunk_opener(self, intent: str, sentiment: str,
                     person: Optional[str], ot: float, cort: float,
                     has_doubt: bool) -> Optional[str]:
        """
        호명/감탄 opener.
        - 사람 호명이면 이름 + 톤
        - 의문에 doubt: 망설임
        - 부정 sentiment: 공감 opener
        """
        if person:
            if ot > 0.65:
                return f"{person}아"
            if cort > 0.5:
                return f"음 {person}"
            return None  # 호명 없이도 OK

        if intent == 'question' and has_doubt:
            return "음..."
        if sentiment == 'negative':
            if ot > 0.5:
                return "그래"
            return "음..."
        if intent == 'emotion' and sentiment == 'positive':
            return "오"
        return None

    def _chunk_acknowledgment(self, intent: str, sentiment: str,
                             input_cats: Set[str]) -> Optional[str]:
        """입력 인정 — 사용자가 한 말을 들었음 표현"""
        if intent == 'question':
            return None  # 질문엔 직접 답
        if intent == 'emotion':
            if sentiment == 'positive':
                return "그래 좋네"
            if sentiment == 'negative':
                return "힘들겠다"
            return "그렇구나"
        if intent == 'statement':
            if sentiment == 'positive':
                return "응 그래"
            if sentiment == 'negative':
                return "그래"
            return "응"
        return None

    def _chunk_main(self, intent: str, primary: Optional[str],
                   focus: Optional[str], cort: float, da: float,
                   ot: float, has_doubt: bool) -> Optional[str]:
        """본 응답 — 의도 따라 핵심 메시지"""
        if intent == 'question':
            if has_doubt:
                if primary:
                    return f"{primary}일까... 잘 모르겠어"
                return "글쎄"
            if primary:
                if cort > 0.5:
                    return f"음 {primary}일 수도"
                if da > 0.55:
                    return f"{primary} 같아"
                return f"{primary}"
            return "글쎄"

        if intent == 'command':
            return None

        if intent == 'emotion':
            return None

        # statement
        if primary:
            if da > 0.55:
                josa = self._josa(primary, '이/가')
                return f"{primary}{josa} 생각나"
            return None
        return None

    def _chunk_elaboration(self, input_understanding: Dict,
                          focus: Optional[str],
                          input_cats: Set[str],
                          used_cats: Set[str],
                          da: float, ot: float) -> Optional[str]:
        """
        확장 — WM 다른 카테고리 / EM 회상 / 호르몬 따라.
        DA + OT 높을 때만 활성. used_cats는 main에서 쓴 카테고리 (중복 방지).
        """
        if da < 0.5 and ot < 0.5:
            return None

        if self.wm is None:
            return None
        try:
            wm_other = [c for c, _ in self.wm.get_focus_set(8)
                       if c not in input_cats
                       and c not in used_cats
                       and c != focus][:3]
        except Exception:
            wm_other = []

        if wm_other:
            cat = wm_other[0]
            if ot > 0.6:
                return f"{cat}{self._josa(cat, '이/가')} 떠오르네"
            if da > 0.55:
                return f"그러고 보니 {cat}"
            return None
        return None

    def _chunk_emotion(self, sentiment: str, cort: float,
                      ot: float, da: float, valence: float) -> Optional[str]:
        """감정 표현 — feeling 자연어로"""
        feel = self.feeling_to_text()
        if feel == "보통":
            return None

        # 부정 감정에 EVE도 부정이면 공명 표현
        if sentiment == 'negative' and valence < -0.1:
            if cort > 0.5:
                return f"나도 좀 답답해"
            return f"기분이 묘하네"

        # 긍정 감정에 EVE도 긍정
        if sentiment == 'positive' and valence > 0.2:
            if da > 0.55:
                return f"기분 좋다"
            if ot > 0.55:
                return f"포근하네"

        return None

    def _chunk_closer(self, intent: str, sentiment: str,
                     da: float, ot: float, cort: float) -> Optional[str]:
        """마무리 — 질문/감탄"""
        # statement에 OT 높으면 질문으로 마무리
        if intent == 'statement' and ot > 0.6 and sentiment != 'negative':
            return "너는 어때"
        # DA 매우 높음 → 감탄
        if da > 0.7:
            return "재밌네"
        # 부정 + OT → 위로 마무리
        if sentiment == 'negative' and ot > 0.55:
            return "괜찮아 질거야"
        return None

    def respond_extended(self, input_understanding: Dict,
                        force_length: Optional[str] = None) -> str:
        """
        풍부한 응답 — 호르몬/감정 따라 chunks 결합.

        인간처럼 길게 말할 수 있음:
        - cort 높음 → 짧게 (1-2 chunks)
        - 보통 → medium (2-3 chunks)
        - DA+OT 높음 → 길게 (3-5 chunks)

        Args:
            input_understanding: understand() 결과
            force_length: None=자동, 'short'/'medium'/'long' 강제

        Returns: 자연어 응답 (chunks 결합)
        """
        intent = input_understanding.get('intent', 'statement')
        sentiment = input_understanding.get('sentiment', 'neutral')
        has_doubt = input_understanding.get('has_doubt', False)
        input_cats = input_understanding.get('categories', set())

        # command는 기존 should_refuse가 처리 — 풍부화 X
        if intent == 'command':
            return self.respond(input_understanding)

        # 호르몬
        mood = self.hs.compute_mood()
        valence = mood['valence']
        cort = self.hs.hormones['cortisol'].level
        da = self.hs.hormones['dopamine'].level
        ot_h = self.hs.hormones.get('oxytocin')
        ot = ot_h.level if (ot_h and 'oxytocin' in self.hs.active_hormones) else 0.3

        # 길이 결정
        length = force_length or self._decide_length(
            intent, sentiment, cort, da, ot)

        # short → 기존 respond 호출 (회귀 안전)
        if length == 'short':
            return self.respond(input_understanding)

        # focus + primary
        focus = self.wm.get_focus() if self.wm else None
        person_in_input = input_cats & {'민석', '너', '나'}
        person = next(iter(person_in_input)) if person_in_input else None

        # primary: focus가 input과 다르면 우선
        primary = None
        if focus and focus not in input_cats:
            primary = focus
        elif self.wm:
            try:
                wm_other = [c for c, _ in self.wm.get_focus_set(5)
                           if c not in input_cats]
                if wm_other:
                    primary = wm_other[0]
            except Exception:
                pass

        # chunks 결합
        chunks: List[str] = []
        used_cats: Set[str] = set()  # 중복 방지

        # 1. opener
        opener = self._chunk_opener(intent, sentiment, person,
                                   ot, cort, has_doubt)
        if opener:
            chunks.append(opener)

        # 2. acknowledgment (medium/long)
        if length in ('medium', 'long'):
            ack = self._chunk_acknowledgment(intent, sentiment, input_cats)
            if ack:
                chunks.append(ack)

        # 3. main (primary 사용 추적)
        main = self._chunk_main(intent, primary, focus,
                               cort, da, ot, has_doubt)
        if main:
            chunks.append(main)
            if primary:
                used_cats.add(primary)

        # 4. elaboration (long only, used_cats 제외)
        if length == 'long':
            elab = self._chunk_elaboration(input_understanding, focus,
                                          input_cats, used_cats, da, ot)
            if elab:
                chunks.append(elab)

        # 5. emotion (medium/long, DA or OT 높음)
        if length in ('medium', 'long') and (da > 0.5 or ot > 0.5):
            emo = self._chunk_emotion(sentiment, cort, ot, da, valence)
            if emo:
                chunks.append(emo)

        # 6. closer (long only)
        if length == 'long':
            closer = self._chunk_closer(intent, sentiment, da, ot, cort)
            if closer:
                chunks.append(closer)

        # 빈 응답 → 기존 respond fallback
        if not chunks:
            return self.respond(input_understanding)

        # 자연스러운 결합:
        # - 1-2개: 콤마
        # - 3+: 처음 1-2개 콤마, 나머지 마침표
        if len(chunks) <= 2:
            return ", ".join(chunks)
        else:
            head = ", ".join(chunks[:2])
            tail = ". ".join(chunks[2:])
            return f"{head}. {tail}"
