"""
EVE v35 - E1: CorpusLearner (대규모 텍스트 학습)
=================================================

목표: 카테고리 14 → 50,000+ 폭증.
양 부족이 모든 한계의 근본. 양이 많아지면 모든 모듈 즉시 향상.

EVE 절대 원칙 부합:
- 결정론적 (random/sampling/probability 사용 X)
- 자체 형태소 X — NL.parse 호출 (책임 단일화)
- 호르몬 자연 변조 (학습 = '새로운_학습' 칵테일 자극)
- 카테고리 = 의미 단위 (문자열 그대로)

학계 기반:
- Distributional Hypothesis (Firth 1957):
    "You shall know a word by the company it keeps."
    같은 맥락에 자주 나오는 단어는 의미가 가깝다.
- HAL/LSA (Lund 1996, Landauer 1997): co-occurrence matrix
- Hasselmo 1995: ACh = 학습 신호 (LTP 강화)
- Bjork & Bjork 1992: 반복 등장 → stability ↑ (망각 둔화)
- Continual learning (Branzan 2025): 점진적 추가, 재학습 X

알고리즘:
1. 텍스트 → 단락/문장 분리
2. 각 문장 → NL.parse (조사/어미 제거된 카테고리)
3. 모르는 단어 → NL.discover_unknown (자동 그래프 추가)
4. Hebbian 학습 (3 단계):
   a. 같은 문장 내 페어 → 강한 연결 (sentence_strength, freq 보정)
   b. 인접 문장 (같은 단락) → 중간 연결 (paragraph_strength)
   c. 같은 문서 내 멀리 → 학습 X (노이즈 제거)
5. Cooccurrence 카운트 → 자동 그룹 발견 (E1.4)
6. 호르몬: 학습 칵테일 (DA + NE + ACh + Glu) + acetylcholine 비례 강도

핵심 함수:
1. learn_sentence(text)              → dict (cats, new, links)
2. learn_paragraph(text)             → dict (sentences 통합)
3. learn_document(text)              → dict (paragraphs 통합)
4. learn_corpus(texts: List[str])    → dict (문서 누적)
5. discover_groups(min_cooc, min_strength) → Dict[group_name, Set[cat]]
6. stats()                           → dict (학습 통계 전체)
7. tick(dt)                          → 망각 가속 보호 (no-op, 호환성)

Author: 김민석 + Claude v35
"""

import re
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict, Counter


class CorpusLearner:
    """
    EVE v35 대규모 텍스트 학습기.

    의존성:
        - SpreadingActivation (그래프 + Hebbian)
        - NaturalLanguage (parse + discover_unknown)
        - HormoneSystem (선택, 학습 호르몬 칵테일)

    설계:
        - 모든 학습은 즉시 SA에 반영 (incremental)
        - cooccurrence 별도 카운트 → 그룹 발견 + 통계
        - 한 문서 내 다른 페어는 약하게만 (전체 페어 연결 X — O(n²) 폭발 방지)
    """

    # 한국어 문장 종결 (마침표/물음표/느낌표/줄바꿈)
    SENTENCE_SPLIT_RE = re.compile(r'[.!?。]+\s*|\n+')
    # 단락 분리 (빈 줄 또는 두 번 이상 줄바꿈)
    PARAGRAPH_SPLIT_RE = re.compile(r'\n\s*\n+')

    def __init__(self,
                 spreading_activation,
                 natural_lang,
                 hormone_system=None,
                 sentence_strength: float = 0.3,
                 paragraph_strength: float = 0.1,
                 sentence_window: int = 0,
                 max_sentence_pairs: int = 50,
                 saturation_factor: float = 1.0,
                 use_hormone_modulation: bool = True):
        """
        Args:
            spreading_activation: SA 인스턴스 (graph + Hebbian)
            natural_lang: NL 인스턴스 (parse + discover_unknown)
            hormone_system: HS 인스턴스 (선택)
            sentence_strength: 같은 문장 내 페어 기본 강도
            paragraph_strength: 인접 문장 간 페어 강도
            sentence_window: 문장 내 거리 제한 (0=모든 페어, 5=±5단어 이내)
            max_sentence_pairs: 한 문장에서 만들 최대 페어 수 (long sentence 폭발 방지)
            saturation_factor: 반복 등장 시 강도 감쇠 (1.0=일정, 0.5=빠르게 약해짐)
            use_hormone_modulation: 학습 시 호르몬 칵테일 자극 여부
        """
        self.sa = spreading_activation
        self.nl = natural_lang
        self.hs = hormone_system

        self.sentence_strength = float(sentence_strength)
        self.paragraph_strength = float(paragraph_strength)
        self.sentence_window = int(sentence_window)
        self.max_sentence_pairs = int(max_sentence_pairs)
        self.saturation_factor = float(saturation_factor)
        self.use_hormone_modulation = bool(use_hormone_modulation)

        # 통계 (cooccurrence 영구 저장 — 그룹 발견용)
        self.cooc: Dict[Tuple[str, str], int] = defaultdict(int)
        self.category_freq: Dict[str, int] = defaultdict(int)

        # 누적 카운터
        self.documents_processed = 0
        self.paragraphs_processed = 0
        self.sentences_processed = 0
        self.sentences_skipped = 0  # 카테고리 < 2개 (학습 불가)
        self.categories_added = 0   # 신규 발견 누적
        self.pairs_strengthened = 0 # learn_pair 호출 횟수

    # ============= 1. 문장/단락 분리 =============
    def split_sentences(self, text: str) -> List[str]:
        """텍스트 → 문장 리스트 (빈 문장 제거)"""
        if not text:
            return []
        parts = self.SENTENCE_SPLIT_RE.split(text)
        return [p.strip() for p in parts if p and p.strip()]

    def split_paragraphs(self, text: str) -> List[str]:
        """텍스트 → 단락 리스트"""
        if not text:
            return []
        parts = self.PARAGRAPH_SPLIT_RE.split(text)
        return [p.strip() for p in parts if p and p.strip()]

    # ============= 2. 문장 학습 (가장 작은 단위) =============
    def learn_sentence(self, sentence: str) -> Dict[str, Any]:
        """
        한 문장 학습.

        절차:
        1. NL.parse → 카테고리 추출
        2. NL.discover_unknown → 모르는 카테고리 자동 그래프 추가
        3. 카테고리 페어 → Hebbian 강화 (freq 보정)
        4. cooc 카운트 갱신
        5. 호르몬 자극 (선택)

        Returns:
            {
                'sentence': str,
                'categories': set,    # 추출된 모든 카테고리
                'new_categories': set, # 새로 발견된 것
                'pairs_made': int,    # 강화된 페어 수
                'skipped': bool,      # 카테고리 < 2 (학습 X)
            }
        """
        if not sentence:
            return {'sentence': '', 'categories': set(),
                    'new_categories': set(), 'pairs_made': 0, 'skipped': True}

        cats = self.nl.parse(sentence)
        result = {
            'sentence': sentence,
            'categories': cats,
            'new_categories': set(),
            'pairs_made': 0,
            'skipped': False,
        }

        # 카테고리 < 2 → 페어 못 만듦 (학습 X)
        if len(cats) < 2:
            self.sentences_skipped += 1
            result['skipped'] = True
            # 그래도 빈도는 갱신
            for c in cats:
                self.category_freq[c] += 1
            return result

        # 모르는 카테고리 자동 발견
        new_cats = self.nl.discover_unknown(sentence,
                                           link_strength=self.paragraph_strength)
        result['new_categories'] = new_cats
        self.categories_added += len(new_cats)

        # 빈도 갱신
        for c in cats:
            self.category_freq[c] += 1

        # Hebbian 강화 — 결정론적 정렬 후 페어 생성 (재현성)
        cat_list = sorted(cats)

        # window=0 (전체 페어) 또는 long sentence 자르기
        pairs = self._enumerate_pairs(cat_list)

        for a, b in pairs:
            # 정렬 키 (cooc 대칭)
            key = (a, b) if a < b else (b, a)
            cur_count = self.cooc[key]

            # Saturation 보정 — 반복일수록 강도 ↓
            # cur_count=0 → factor=1.0
            # cur_count=10 → factor=1/(1+10*sat)
            factor = 1.0 / (1.0 + cur_count * self.saturation_factor)
            strength = self.sentence_strength * factor

            self.sa.learn_pair(a, b, strength)
            self.cooc[key] += 1
            result['pairs_made'] += 1
            self.pairs_strengthened += 1

        # 호르몬 자극 — 학습 칵테일
        if self.use_hormone_modulation and self.hs is not None:
            # '새로운_학습' 칵테일 약하게 (강하면 매 문장마다 saturate)
            self.hs.trigger_cocktail('새로운_학습', intensity=0.05)
            # 활성된 카테고리 → 호르몬 변조 (group-aware)
            self.hs.category_to_hormone(cats, strength=0.03)

        self.sentences_processed += 1
        return result

    def _enumerate_pairs(self, cat_list: List[str]) -> List[Tuple[str, str]]:
        """
        문장 내 페어 생성.
        - window > 0이면 거리 제한
        - max_sentence_pairs 초과 시 처음 N개만 (결정론)
        """
        n = len(cat_list)
        pairs = []
        if self.sentence_window > 0:
            # 거리 제한
            for i in range(n):
                for j in range(i + 1, min(i + 1 + self.sentence_window, n)):
                    pairs.append((cat_list[i], cat_list[j]))
        else:
            # 전체 페어 (모든 i<j)
            for i in range(n):
                for j in range(i + 1, n):
                    pairs.append((cat_list[i], cat_list[j]))

        # 폭발 방지 — 결정론적 자르기
        if len(pairs) > self.max_sentence_pairs:
            pairs = pairs[:self.max_sentence_pairs]
        return pairs

    # ============= 3. 단락 학습 =============
    def learn_paragraph(self, paragraph: str) -> Dict[str, Any]:
        """
        한 단락 학습. 문장별로 강한 연결 + 인접 문장 간 약한 연결.

        Returns:
            {
                'paragraph': str,
                'sentences': int,
                'all_categories': set,
                'new_categories': set,
                'pairs_made': int,
            }
        """
        if not paragraph:
            return {'paragraph': '', 'sentences': 0, 'all_categories': set(),
                    'new_categories': set(), 'pairs_made': 0}

        sentences = self.split_sentences(paragraph)
        all_cats: Set[str] = set()
        all_new: Set[str] = set()
        total_pairs = 0
        per_sentence_cats: List[Set[str]] = []

        # 1) 각 문장 학습 (강한 sentence-level 연결)
        for s in sentences:
            r = self.learn_sentence(s)
            all_cats |= r['categories']
            all_new |= r['new_categories']
            total_pairs += r['pairs_made']
            per_sentence_cats.append(r['categories'])

        # 2) 인접 문장 간 약한 연결 (1-step만 — 너무 멀면 노이즈)
        for i in range(len(per_sentence_cats) - 1):
            cats_a = per_sentence_cats[i]
            cats_b = per_sentence_cats[i + 1]
            if not cats_a or not cats_b:
                continue
            # 공통 카테고리 제외 (이미 같은 문장에서 강하게 연결)
            unique_a = sorted(cats_a - cats_b)
            unique_b = sorted(cats_b - cats_a)
            # 결정론적 페어 생성 (작은 set 위주)
            for a in unique_a[:5]:  # 결정론적 자르기
                for b in unique_b[:5]:
                    self.sa.learn_pair(a, b, self.paragraph_strength)
                    self.pairs_strengthened += 1
                    total_pairs += 1

        self.paragraphs_processed += 1
        return {
            'paragraph': paragraph[:80] + ('...' if len(paragraph) > 80 else ''),
            'sentences': len(sentences),
            'all_categories': all_cats,
            'new_categories': all_new,
            'pairs_made': total_pairs,
        }

    # ============= 4. 문서 학습 =============
    def learn_document(self, document: str) -> Dict[str, Any]:
        """
        한 문서 학습. 단락별로 처리.

        Returns:
            {'paragraphs': int, 'sentences': int, ...}
        """
        if not document:
            return {'paragraphs': 0, 'sentences': 0,
                    'all_categories': set(), 'new_categories': set(),
                    'pairs_made': 0}

        paragraphs = self.split_paragraphs(document)
        all_cats: Set[str] = set()
        all_new: Set[str] = set()
        total_pairs = 0
        total_sentences = 0

        for p in paragraphs:
            r = self.learn_paragraph(p)
            all_cats |= r['all_categories']
            all_new |= r['new_categories']
            total_pairs += r['pairs_made']
            total_sentences += r['sentences']

        self.documents_processed += 1
        return {
            'paragraphs': len(paragraphs),
            'sentences': total_sentences,
            'all_categories': all_cats,
            'new_categories': all_new,
            'pairs_made': total_pairs,
        }

    # ============= 5. 코퍼스 학습 (대량) =============
    def learn_corpus(self, texts: List[str],
                    progress_every: int = 100) -> Dict[str, Any]:
        """
        다수 문서 학습. progress_every 문서마다 통계 출력 옵션.

        Returns:
            누적 통계
        """
        for i, text in enumerate(texts):
            self.learn_document(text)
            if progress_every > 0 and (i + 1) % progress_every == 0:
                s = self.stats()
                print(f"  [{i+1}/{len(texts)}] cats={s['total_categories']}, "
                      f"links={s['total_connections']}, "
                      f"sentences={s['sentences_processed']}")

        return self.stats()

    # ============= 6. 자동 그룹 발견 (E1.4) =============
    def discover_groups(self,
                       min_cooc: int = 3,
                       min_strength: float = 0.2,
                       max_groups: int = 50) -> Dict[str, Set[str]]:
        """
        Cooccurrence 기반 자동 그룹 발견 (clustering).

        알고리즘 (결정론):
        1. cooc[(a,b)] >= min_cooc + sa.weights[(a,b)] >= min_strength → 강한 페어
        2. 강한 페어들 → 무방향 그래프
        3. Connected components 찾기 (BFS, 알파벳 순 시드)
        4. 각 component → 그룹 (이름 = component 내 가장 빈도 높은 카테고리)

        Returns:
            {group_name: {cat1, cat2, ...}}
        """
        # 강한 페어만
        strong_pairs: List[Tuple[str, str]] = []
        for (a, b), count in self.cooc.items():
            if count < min_cooc:
                continue
            if self.sa.get_weight(a, b) < min_strength:
                continue
            strong_pairs.append((a, b))

        # 인접 리스트
        adj: Dict[str, Set[str]] = defaultdict(set)
        all_nodes: Set[str] = set()
        for a, b in strong_pairs:
            adj[a].add(b)
            adj[b].add(a)
            all_nodes.add(a)
            all_nodes.add(b)

        # Connected components (BFS, 결정론적 시작점)
        visited: Set[str] = set()
        components: List[Set[str]] = []
        for seed in sorted(all_nodes):
            if seed in visited:
                continue
            comp: Set[str] = set()
            stack = [seed]
            while stack:
                node = stack.pop()
                if node in visited:
                    continue
                visited.add(node)
                comp.add(node)
                # 결정론적 순서로 이웃 push
                for n in sorted(adj[node]):
                    if n not in visited:
                        stack.append(n)
            if len(comp) >= 2:  # 최소 2개 이상
                components.append(comp)

        # 그룹 이름 — 컴포넌트 내 가장 빈도 높은 카테고리
        groups: Dict[str, Set[str]] = {}
        # 컴포넌트는 크기 ↓, 시드 알파벳 ↑ 순 정렬
        components.sort(key=lambda c: (-len(c), sorted(c)[0]))

        for i, comp in enumerate(components[:max_groups]):
            # 컴포넌트 내 가장 빈도 높은 카테고리 → 이름
            comp_freq = [(c, self.category_freq[c]) for c in comp]
            comp_freq.sort(key=lambda x: (-x[1], x[0]))
            best = comp_freq[0][0]
            group_name = f"group_{i+1}_{best}"
            groups[group_name] = comp

        return groups

    # ============= 7. 통계 =============
    def stats(self) -> Dict[str, Any]:
        """학습 통계 전체"""
        return {
            'documents_processed': self.documents_processed,
            'paragraphs_processed': self.paragraphs_processed,
            'sentences_processed': self.sentences_processed,
            'sentences_skipped': self.sentences_skipped,
            'categories_added': self.categories_added,
            'pairs_strengthened': self.pairs_strengthened,
            'total_categories': len(self.sa.neighbors),
            'total_connections': len(self.sa.weights),
            'cooc_pairs': len(self.cooc),
            'unique_categories_seen': len(self.category_freq),
            'top_categories': self._top_categories(10),
        }

    def _top_categories(self, n: int = 10) -> List[Tuple[str, int]]:
        """가장 빈도 높은 카테고리 N개"""
        items = sorted(self.category_freq.items(),
                      key=lambda x: (-x[1], x[0]))
        return items[:n]

    # ============= 8. tick (호환성) =============
    def tick(self, dt: float = 0.5):
        """매 tick no-op (CorpusLearner는 외부 자극으로만 학습)"""
        pass

    def __repr__(self):
        s = self.stats()
        return (f"CorpusLearner(docs={s['documents_processed']}, "
                f"sents={s['sentences_processed']}, "
                f"cats={s['total_categories']}, "
                f"links={s['total_connections']})")
