"""
EVE v35 - E8: FrameSemantics (한국어 SVO frame 자동 추출)
==========================================================

목표: 자연어 → frame (semantic roles) → HyperGraph 자동 등록.

기존 한계:
- NL.parse: 카테고리 set 추출 (역할 정보 없음)
- E2 HyperGraph: 역할 매핑 *수동* 필요
- 일상 대화 → 사건 자동 등록 불가

E8 해결:
- 한국어 격조사 분석 → 격(case) → 역할(role) 자동 매핑
- 동사 추출 (마지막 어절, 어미 정규화)
- 시간/장소 단어 사전으로 'time' vs 'location' 구분
- 결과: HyperEdge format → E2.add_edge 직접 호출 가능

EVE 절대 원칙:
- 결정론적 (정렬 + 사전 기반, no random)
- KoNLPy 등 외부 라이브러리 X (사용자 원칙)
- 카테고리 = 의미 단위 유지
- read-only on NL/HG (E8은 *추가* 모듈)

학계 부합:
- Fillmore 1968 — Case Grammar (격 문법)
- Fillmore 1976 — Frame Semantics (frame = role + filler)
- 한국어 격조사 연구 (이익섭, 임홍빈)
- FrameNet (Berkeley 1997~) — 동사 → frame
- PropBank — semantic roles labeling

핵심 함수:
1. parse_sentence(text)              → frame dict
2. extract_frames(paragraph)          → List[frame]
3. classify_word(word)                → 'verb'/'time'/'location'/'noun'
4. to_hyperedge(frame, hg)            → edge_id (자동 등록)
5. text_to_hg(text, hg)               → edges 자동 등록
6. compose_question(...)              → frame 부분 매칭 query

Author: 김민석 + Claude v35
"""

from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict


class FrameSemantics:
    """
    한국어 frame 추출 + HyperGraph 자동 통합.

    의존:
        natural_lang: NL (선택, parse 활용)
        hypergraph: HG (선택, to_hyperedge 자동 등록)
        spreading_activation: SA (선택, 카테고리 검증)
        hormone_system: HS (선택)
    """

    # ============= 격조사 → role 매핑 =============
    # 한국어 격조사 (case markers) → semantic role
    # 길이 긴 거 먼저 (greedy match)
    PARTICLE_TO_ROLE = [
        # 처소격 (location/time) — 시간 단어면 time, 아니면 location
        ('에서는', 'location'),
        ('에서도', 'location'),
        ('에서', 'location'),
        ('에게서', 'source'),
        ('한테서', 'source'),
        ('에게로', 'recipient'),
        ('에게', 'recipient'),
        ('한테', 'recipient'),
        # 도구격
        ('으로', 'instrument'),
        ('으로써', 'instrument'),
        # 공동격
        ('와는', 'companion'),
        ('과는', 'companion'),
        ('와', 'companion'),
        ('과', 'companion'),
        # 목적격
        ('을', 'patient'),
        ('를', 'patient'),
        # 주격 (높임 포함)
        ('께서', 'agent'),
        ('이', 'agent'),
        ('가', 'agent'),
        # 주제격 (보조사)
        ('은', 'topic'),
        ('는', 'topic'),
        # 처소/시간 단순 '에' (마지막)
        ('에는', 'location_or_time'),
        ('에도', 'location_or_time'),
        ('에', 'location_or_time'),
        # 보조사 (맥락 dependent)
        ('만', 'restriction'),
        ('도', 'addition'),
        # 도구 단순 '로'
        ('로', 'instrument'),
    ]

    # 시간 단어 사전 (location vs time 구분)
    TIME_WORDS = {
        '어제', '오늘', '내일', '모레', '그제',
        '지금', '방금', '아까', '나중에', '이따',
        '아침', '점심', '저녁', '밤', '새벽',
        '오전', '오후', '낮', '밤중',
        '월요일', '화요일', '수요일', '목요일',
        '금요일', '토요일', '일요일',
        '봄', '여름', '가을', '겨울',
        '작년', '올해', '내년',
        '지난주', '이번주', '다음주',
        '지난달', '이번달', '다음달',
    }

    # 동사 어미 — 정규화 (NL.ENDINGS 보충)
    VERB_ENDINGS = {
        '었다': '다', '았다': '다', '였다': '다',
        '했다': '하다', '됐다': '되다',
        '한다': '하다', '된다': '되다',
        '해': '하다', '돼': '되다',
        '었어': '다', '았어': '다',
        '었어요': '다', '았어요': '다', '였어요': '다',
        '어요': '다', '아요': '다',
        '었던': '다', '았던': '다',
        '습니다': '다',
        '겠다': '다', '겠어': '다',
        '려고': '다', '려': '다',
    }

    # 동사로 인식할 만한 끝 (휴리스틱)
    VERB_HINTS = ('다', '해', '돼', '나', '려', '면', '게', '데',
                  '요', '어', '아', '네', '죠', '지', '구나', '습니다')

    def __init__(self,
                 natural_lang=None,
                 hypergraph=None,
                 spreading_activation=None,
                 hormone_system=None,
                 use_hormone_modulation: bool = True):
        self.nl = natural_lang
        self.hg = hypergraph
        self.sa = spreading_activation
        self.hs = hormone_system
        self.use_hormone_modulation = bool(use_hormone_modulation)

        # 통계
        self.parse_count = 0
        self.frame_count = 0
        self.hg_added_count = 0
        self.failed_parse = 0

    # ============= 1. 어절 분석 =============
    def _split_particle(self, word: str) -> Tuple[str, Optional[str]]:
        """
        어절 → (본체, 조사).
        가장 긴 매치 우선.

        Returns:
            (stem, particle) — particle None이면 조사 없음
        """
        if not word:
            return word, None
        # PARTICLE_TO_ROLE는 이미 길이 ↓ 정렬됨 (수동)
        for particle, _ in self.PARTICLE_TO_ROLE:
            if word.endswith(particle) and len(word) > len(particle):
                stem = word[:-len(particle)]
                return stem, particle
        return word, None

    def _normalize_verb(self, word: str) -> str:
        """동사 어미 정규화 → 기본형 추정"""
        for ending, replacement in sorted(self.VERB_ENDINGS.items(),
                                         key=lambda x: -len(x[0])):
            if word.endswith(ending):
                stem = word[:-len(ending)] + replacement
                return stem
        return word

    def classify_word(self, word: str) -> str:
        """
        어절 분류: 'verb' / 'time' / 'noun'

        간단 휴리스틱:
        - TIME_WORDS에 있으면 time
        - VERB_HINTS로 끝나면 verb 후보
        - 그 외 noun
        """
        if not word:
            return 'noun'
        if word in self.TIME_WORDS:
            return 'time'
        # 동사 휴리스틱
        for hint in self.VERB_HINTS:
            if word.endswith(hint) and len(word) >= 2:
                return 'verb'
        return 'noun'

    def _has_ss_jongseong(self, char: str) -> bool:
        """
        한글 음절의 종성이 ㅆ인가? (갔/했/왔/봤/됐 등)
        한국어 과거 시제 핵심 마커.
        """
        if not char or len(char) != 1:
            return False
        code = ord(char) - 0xAC00
        if not (0 <= code < 11172):  # 한글 음절 범위 외
            return False
        jongseong = code % 28
        return jongseong == 20  # ㅆ 받침

    def _has_n_jongseong(self, char: str) -> bool:
        """
        종성이 ㄴ인가? (간/한/된 등 → 현재시제 ~ㄴ다)
        """
        if not char or len(char) != 1:
            return False
        code = ord(char) - 0xAC00
        if not (0 <= code < 11172):
            return False
        jongseong = code % 28
        return jongseong == 4  # ㄴ 받침

    def _detect_tense(self, last_word: str) -> Optional[str]:
        """동사 어절 → 시제 추정 (한국어 음절 분석 포함)"""
        if not last_word or len(last_word) < 2:
            return None

        # 1) 명시적 패턴 매칭 (가장 신뢰)
        past_suffixes = ['었다', '았다', '였다', '했다', '었어', '았어', '었던',
                        '었습니다', '았습니다', '었네', '았네',
                        '었어요', '았어요', '였어요', '었었다', '았었다']
        if any(last_word.endswith(s) for s in past_suffixes):
            return 'past'

        future_suffixes = ['겠다', '겠어', '려고', '려', '겠습니다',
                          '을 것이다', '리라', '겠네']
        if any(last_word.endswith(s) for s in future_suffixes):
            return 'future'

        present_suffixes = ['한다', '된다', '하다', '되다', '습니다',
                           'ㅂ니다', '어요', '아요', '예요']
        if any(last_word.endswith(s) for s in present_suffixes):
            return 'present'

        # 2) 음절 분석 — '~ㅆ다' = 과거 (갔다/왔다/됐다/봤다/...)
        if last_word.endswith('다') and len(last_word) >= 2:
            second_to_last = last_word[-2]
            if self._has_ss_jongseong(second_to_last):
                return 'past'
            # '~ㄴ다' = 현재 (간다/한다/온다/...)
            if self._has_n_jongseong(second_to_last):
                return 'present'

        return None
    def parse_sentence(self, text: str) -> Optional[Dict[str, Any]]:
        """
        한 문장 → frame.

        예: "민석이 어제 학교에서 사과를 먹었다"
            → {
                'nodes': {민석, 어제, 학교, 사과, 먹다},
                'roles': {민석: agent, 어제: time,
                         학교: location, 사과: patient, 먹다: action},
                'verb': '먹다',
                'tense': 'past',  (어미 기반)
                'raw': "민석이 어제..."
              }

        Returns:
            frame dict 또는 None (파싱 실패)
        """
        if not text or not isinstance(text, str):
            return None

        text = text.strip()
        if not text:
            return None

        # 어절 분리 (공백)
        words = text.split()
        if len(words) < 1:
            return None

        nodes: Set[str] = set()
        roles: Dict[str, str] = {}
        verb: Optional[str] = None
        tense: Optional[str] = None

        # 마지막 어절 = 동사 (보통)
        last = words[-1]
        cls = self.classify_word(last)
        if cls == 'verb':
            verb = self._normalize_verb(last)
            # 시제 추정 (음절 분석 포함)
            tense = self._detect_tense(last)

            if verb and len(verb) >= 1:
                nodes.add(verb)
                roles[verb] = 'action'

        # 다른 어절들 분석
        for w in words[:-1] if verb else words:
            stem, particle = self._split_particle(w)
            if not stem:
                continue

            # role 결정
            role = None
            if particle:
                # 격조사 → role
                for p, r in self.PARTICLE_TO_ROLE:
                    if particle == p:
                        role = r
                        break

            # location_or_time 모호성 해결
            if role == 'location_or_time':
                # stem이 시간 단어면 time
                if stem in self.TIME_WORDS:
                    role = 'time'
                else:
                    role = 'location'

            # 시간 단어 자동 인식 (조사 없을 때)
            if not role and stem in self.TIME_WORDS:
                role = 'time'

            # role 없으면 추정 (위치, 순서)
            if not role:
                # 첫 번째 명사형 → topic 가정
                if not roles and stem:
                    role = 'topic'
                else:
                    role = 'modifier'

            # 노드 추가
            if stem and role:
                nodes.add(stem)
                # role 충돌 시 (같은 cat이 여러 role) — 첫 거 유지
                if stem not in roles:
                    roles[stem] = role

        if len(nodes) < 1:
            self.failed_parse += 1
            return None

        # 호르몬 (학습 신호 — frame 추출 = 깊은 이해)
        if self.use_hormone_modulation and self.hs is not None and len(nodes) >= 2:
            self.hs.hormones['acetylcholine'].stimulate(0.05)

        self.parse_count += 1
        if len(nodes) >= 2:
            self.frame_count += 1

        return {
            'nodes': nodes,
            'roles': roles,
            'verb': verb,
            'tense': tense,
            'raw': text,
        }

    # ============= 3. extract_frames — 여러 문장 =============
    def extract_frames(self, paragraph: str) -> List[Dict[str, Any]]:
        """문단 → 여러 frame"""
        if not paragraph:
            return []

        # 문장 분리 (단순)
        sentences = []
        cur = []
        for char in paragraph:
            cur.append(char)
            if char in '.!?。':
                sentences.append(''.join(cur).strip())
                cur = []
        if cur:
            sentences.append(''.join(cur).strip())

        frames = []
        for s in sentences:
            if not s:
                continue
            f = self.parse_sentence(s)
            if f:
                frames.append(f)
        return frames

    # ============= 4. to_hyperedge — HG 자동 등록 =============
    def to_hyperedge(self,
                     frame: Dict[str, Any],
                     hg=None,
                     weight: float = 0.6,
                     epistemic: str = 'belief',
                     source: str = 'frame_extracted') -> Optional[str]:
        """
        Frame → HyperGraph edge.

        Args:
            frame: parse_sentence 결과
            hg: HG 인스턴스 (None이면 self.hg 사용)
            weight: edge 강도
            epistemic: 'fact'/'belief'/'hypothesis'
            source: 라벨

        Returns:
            edge_id 또는 None
        """
        target_hg = hg if hg is not None else self.hg
        if target_hg is None or frame is None:
            return None

        nodes = frame.get('nodes', set())
        roles = frame.get('roles', {})

        if len(nodes) < 2:
            return None

        # temporal 정보 (tense + time role)
        temporal = None
        if frame.get('tense') or any(r == 'time' for r in roles.values()):
            temporal = {
                'tense': frame.get('tense'),
            }
            for cat, role in roles.items():
                if role == 'time':
                    temporal['when'] = cat
                    break

        edge_id = target_hg.add_edge(
            nodes=nodes,
            roles=roles,
            weight=weight,
            temporal=temporal,
            epistemic=epistemic,
            source=source,
        )
        self.hg_added_count += 1
        return edge_id

    # ============= 5. text_to_hg — text → HG 자동 =============
    def text_to_hg(self, text: str,
                   hg=None,
                   weight: float = 0.6,
                   epistemic: str = 'belief',
                   source: str = 'frame_extracted') -> List[str]:
        """
        문단 → 자동 frame 추출 → HG 등록.

        Returns:
            추가된 edge_id list
        """
        target_hg = hg if hg is not None else self.hg
        if target_hg is None:
            return []

        frames = self.extract_frames(text)
        edge_ids = []
        for f in frames:
            eid = self.to_hyperedge(f, target_hg, weight=weight,
                                   epistemic=epistemic, source=source)
            if eid:
                edge_ids.append(eid)
        return edge_ids

    # ============= 6. compose_question =============
    def compose_question(self,
                        question: str) -> Optional[Dict[str, Any]]:
        """
        한국어 질문 → query frame.

        예: "민석이 뭐 했어?" → {agent: '민석', action: ?}
            "민석이 어디서 공부했어?" → {agent: '민석', action: '공부', location: ?}

        간단 휴리스틱:
        - 의문사 (뭐, 어디, 언제, 누구) → 해당 role의 unknown
        """
        WH_TO_ROLE = {
            '뭐': 'patient',
            '무엇': 'patient',
            '뭣': 'patient',
            '어디': 'location',
            '언제': 'time',
            '누가': 'agent',
            '누구': 'recipient',
            '왜': 'cause',
            '어떻게': 'manner',
        }

        frame = self.parse_sentence(question.replace('?', '').strip())
        if not frame:
            return None

        # WH 단어 찾기 → unknown role
        unknowns = []
        nodes_to_remove = set()
        for cat in list(frame['nodes']):
            for wh, role in WH_TO_ROLE.items():
                if wh in cat or cat == wh:
                    unknowns.append(role)
                    nodes_to_remove.add(cat)
                    if cat in frame['roles']:
                        del frame['roles'][cat]
                    break

        for n in nodes_to_remove:
            frame['nodes'].discard(n)

        frame['unknowns'] = unknowns
        frame['is_question'] = True
        return frame

    # ============= 7. tick + stats =============
    def tick(self, dt: float = 0.5):
        """no-op"""
        pass

    def stats(self) -> Dict[str, Any]:
        return {
            'parse_count': self.parse_count,
            'frame_count': self.frame_count,
            'hg_added_count': self.hg_added_count,
            'failed_parse': self.failed_parse,
            'parse_success_rate': (
                self.frame_count / max(1, self.parse_count + self.failed_parse)
            ),
        }

    def __repr__(self):
        s = self.stats()
        return (f"FrameSemantics(parses={s['parse_count']}, "
                f"frames={s['frame_count']}, "
                f"hg_added={s['hg_added_count']})")
