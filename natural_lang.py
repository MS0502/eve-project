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
        '나', '너', '내', '네', '나는', '너는',
        '꿈', '잠', '집', '밥', '책', '글', '일', '말', '뜻',
        '몸', '맘', '귀', '눈', '입', '손', '발', '머리',
        '왜', '뭐', '누', '곳', '때', '날', '돈', '힘',
        '봄', '여름', '가을', '겨울',
    }

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
    INTENT_SIGNALS = {
        'question': ['?', '뭐', '뭐야', '뭔', '왜', '어떻게', '언제',
                     '어디', '누구', '무엇', '얼마', '~지', '~까'],
        'command': ['해', '하라', '해라', '해주', '해줘', '시오',
                    '하세요', '해봐', '~셈', '~세요', '~기를'],
        'negative_emotion': [
            '싫', '미워', '짜증', '화', '슬프', '아프', '무서',
            '우울', '외로', '외롭', '걱정', '답답', '지친', '지치', '미안',
            '후회', '피곤', '힘들',
        ],
        'positive_emotion': [
            '좋', '기뻐', '신나', '행복', '사랑', '재밌',
            '즐거', '따뜻', '완벽', '신기', '만족', '멋지',
            '고마', '기대',
        ],
        'doubt': ['정말', '진짜', '확실', '맞아', '근데', '글쎄', '~인가'],
        'refusal_target': ['하지마', '안돼', '싫어', '거부', '거절'],
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
    def _strip_particles(self, word: str) -> str:
        """조사 제거 (한국어 어미는 보존)"""
        if not word or len(word) <= 1:
            return word

        # 가장 긴 조사부터 매칭 (그리디)
        for p in sorted(self.PARTICLES, key=len, reverse=True):
            if word.endswith(p) and len(word) > len(p):
                stem = word[:-len(p)]
                # 너무 짧아지면 원본 유지
                if len(stem) >= 1:
                    return stem
        return word

    def _normalize_ending(self, word: str) -> str:
        """어미 정규화 (동사/형용사 → 기본형)"""
        if not word:
            return word
        for ending, replacement in sorted(self.ENDINGS.items(),
                                         key=lambda x: -len(x[0])):
            if word.endswith(ending) and len(word) > len(ending):
                return word[:-len(ending)] + replacement
        return word

    def parse(self, text: str) -> Set[str]:
        """
        자연어 → 카테고리 set 추출.

        절차:
        1. 공백/문장부호로 단어 쪼개기
        2. 각 단어의 조사 제거
        3. 어미 정규화
        4. 길이 1짜리 의미 약한 단어 제거
        5. 카테고리 set 반환
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

        Returns:
            충돌하는 신념 리스트 [{belief_id, statement, confidence, conflict_type}]
        """
        cats = self.parse(text)
        if not cats or not self.beliefs:
            return []

        # 텍스트가 부정형인지
        text_negation = any(neg in text for neg in ['아니다', '아냐', '아닌', '없다', '없어', '안'])

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
                belief_cats = self.parse(triple.get('original', ''))
                overlap = cats & belief_cats

                # 부정형 미스매치 (강한 신호)
                negation_mismatch = (text_negation != belief_neg)

                # 충돌 조건:
                # - overlap 2+ + 부정형 미스매치
                # - overlap 1+ + 부정형 미스매치 + subject 카테고리 겹침
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

        고정 텍스트 X. 활성 카테고리 + 호르몬 + 의도에 따라 다름.
        """
        intent = input_understanding.get('intent', 'statement')
        sentiment = input_understanding.get('sentiment', 'neutral')
        has_doubt = input_understanding.get('has_doubt', False)

        # 현재 호르몬 상태
        mood = self.hs.compute_mood()
        valence = mood['valence']

        # WM에서 강한 카테고리 (focus + top)
        focus = self.wm.get_focus()
        wm_cats = sorted(self.wm.slots.keys(),
                        key=lambda c: -self.wm.slots[c].salience)[:max_response_cats]

        # 응답 카테고리 = WM 강한 거 + input 카테고리
        response_cats = []
        if focus:
            response_cats.append(focus)
        for c in wm_cats:
            if c not in response_cats:
                response_cats.append(c)
        response_cats = response_cats[:max_response_cats]

        # 의도별 응답 형식 (창발 - 호르몬 따라 어조 변화)
        if intent == 'question':
            if not response_cats:
                return "음..."
            if has_doubt:
                return f"{response_cats[0]}? 잘 모르겠어"
            return f"{', '.join(response_cats[:3])}"

        elif intent == 'command':
            # should_refuse() 호출 결과에 따라
            refuse = self.should_refuse(input_understanding)
            if refuse['should_refuse']:
                return refuse['response']
            # 수용 - refuse_score, 호르몬에 따라 어조 다양 (창발)
            cort = self.hs.hormones['cortisol'].level
            ot = self.hs.hormones.get('oxytocin')
            ot_level = ot.level if ot and 'oxytocin' in self.hs.active_hormones else 0.3
            cat = response_cats[0] if response_cats else '알겠어'
            score = refuse['reason_score']
            # 점수 음수 (옥시토신 강함) = 따뜻한 수용
            if score < -0.1:
                return f"응 알았어, {cat}"
            # 점수 0~0.4 (중간) = 호르몬 따라
            if cort > 0.5:
                return f"음... {cat}"
            if ot_level > 0.5:
                return f"좋아, {cat}"
            return f"응, {cat}"

        elif intent == 'emotion':
            if sentiment == 'positive':
                if valence > 0.3:
                    return f"{response_cats[0] if response_cats else '응'}!"
                return f"{response_cats[0] if response_cats else '응'}..."
            elif sentiment == 'negative':
                return f"{response_cats[0] if response_cats else '음'}..."
            return response_cats[0] if response_cats else "..."

        else:  # statement
            raw = input_understanding.get('raw', '')
            input_cats = input_understanding.get('categories') or set()
            direct_address = (
                len(input_cats) == 1
                and len(raw.split()) == 1
                and bool(response_cats)
                and response_cats[0] in input_cats
            )
            if direct_address:
                ot = self.hs.hormones.get('oxytocin')
                ot_level = ot.level if ot and 'oxytocin' in self.hs.active_hormones else 0.3
                return "안녕..." if ot_level >= 0.5 else "응..."
            return ', '.join(response_cats[:3]) if response_cats else "..."

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
