"""
SelfEmbeddingAdapter — 라운드 47

EVE 자체 학습 임베딩. 외부 모델 0. numpy만.

원칙:
- 사용자 입력 + ConceptMemory 정의들로 co-occurrence matrix 구축
- PMI (Pointwise Mutual Information) 변환
- SVD 차원 축소 (50차원)
- Cosine similarity로 의미 유사도
- 결정론 (같은 데이터 → 같은 임베딩)

학계:
- Deerwester et al 1990 — Latent Semantic Analysis
- Church & Hanks 1990 — Pointwise Mutual Information
- Levy & Goldberg 2014 — Neural Word Embeddings as Implicit Matrix Factorization
  (word2vec = PMI 행렬 SVD와 수학적 등가)
- Pennington et al 2014 — GloVe

사용:
- ThoughtChain: 다음 카테고리 후보 점수에 임베딩 유사도 추가
- ConceptMemory: 유사 개념 fallback ("운동" 안 가르쳤어도 "PT" 정의 활용)
- Salience: 의미 유사 단어 자동 spike
"""

from collections import defaultdict
from typing import Optional

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


# === 토크나이저 (한국어 단순 분리) ===

# 한국어 조사 — 토큰 끝에 있으면 제거
KOREAN_PARTICLES = ('은', '는', '이', '가', '을', '를', '의', '에', '에서',
                     '로', '으로', '와', '과', '도', '만', '까지', '부터',
                     '한테', '에게', '아', '야')

# 의미 없는 토큰 (Stopwords)
STOPWORDS = {
    '그', '그것', '저', '저것', '이', '이것', '그래', '그러', '근데',
    '그냥', '음', '어', '아', '오', '응', '네',
    '뭐', '뭐야', '뭔지', '왜', '어떻게', '어디', '언제',
    '나', '내', '너', '니',
}


def simple_tokenize(text: str) -> list:
    """한국어 단순 토크나이저 — 어절 + 조사 제거.
    한국어 1글자 단어 (책, 일, 잠 등) 허용.
    라운드 52: 동사 활용형 이상한 거 정리."""
    if not text:
        return []
    tokens = []
    for eojeol in text.split():
        cleaned = eojeol
        # 라운드 52: 이상한 동사 활용형 정리
        # "하고있었는다" / "물어본거" / "했는다" 같은 거 처리
        replacements = [
            ("었는다", "었다"),
            ("했는다", "했다"),
            ("는다구", "는다고"),
            ("이라구", "이라고"),
            ("했는지", "했는지"),  # 통과
        ]
        for old, new in replacements:
            if old in cleaned:
                cleaned = cleaned.replace(old, new)
        
        # 조사 제거
        for p in KOREAN_PARTICLES:
            if cleaned.endswith(p) and len(cleaned) > len(p):
                cleaned = cleaned[:-len(p)]
                break
        # 의문/마침표 제거
        cleaned = cleaned.rstrip('?！.,!？')
        if not cleaned:
            continue
        # 한국어는 1글자 단어 허용 (책, 일, 잠), 영어는 2글자 이상 (PT, AI 등)
        if not (0xAC00 <= ord(cleaned[0]) <= 0xD7A3):
            if len(cleaned) < 2:
                continue
        if cleaned in STOPWORDS:
            continue
        # 합성 키 (+, @, _) 제외 — 노이즈
        if any(c in cleaned for c in '+@_'):
            continue
        tokens.append(cleaned)
    return tokens


class SelfEmbeddingAdapter:
    """
    EVE 자체 학습 임베딩.
    
    핵심 알고리즘:
    1. observe(text) — co-occurrence 누적
    2. train() — PMI 행렬 + SVD → 임베딩
    3. similarity(a, b) — cosine
    4. most_similar(word, n) — top-N 유사 단어
    """

    def __init__(self, engine=None, dim: int = 50,
                 window: int = 5, train_interval: int = 30,
                 min_word_freq: int = 2,
                 max_train_vocab: int = 128):
        self.engine = engine
        self.dim = dim
        self.window = window               # co-occurrence window 크기
        self.train_interval = train_interval  # N 입력마다 재학습
        self.min_word_freq = min_word_freq # 최소 등장 횟수
        # 라운드 73 안정화: build_full_engine()가 테스트마다 호출되므로
        # 부트스트랩 SVD가 전체 테스트를 멈추지 않게 학습 vocab 상한을 둔다.
        # 상위 빈도 + 가나다/알파벳 tie-break라 결정론은 유지된다.
        self.max_train_vocab = max_train_vocab
        self._suspend_auto_train = False
        # build_full_engine()에서는 부트스트랩을 돌리지 않는다. 대신 실제로
        # 임베딩이 필요한 경로에서 ensure_ready()가 한 번만 lazy bootstrap한다.
        self._bootstrap_attempted = False
        
        # 누적 카운트
        self.cooc_counts = defaultdict(int)  # (a, b) → count (a < b)
        self.word_counts = defaultdict(int)  # word → count
        self.total_obs = 0
        
        # 학습된 임베딩
        self.embeddings: dict = {}     # word → np.array(dim)
        self.last_train_obs = 0
        self.train_count = 0
        self.last_word_count = 0       # 마지막 학습 시 단어 수

    def observe(self, text: str):
        """입력 텍스트에서 co-occurrence 추출. 자동 재학습 트리거."""
        if not NUMPY_AVAILABLE:
            return
        tokens = simple_tokenize(text)
        if len(tokens) < 2:
            return
        
        # word 카운트 + co-occurrence
        for i, w in enumerate(tokens):
            self.word_counts[w] += 1
            for j in range(max(0, i - self.window),
                            min(len(tokens), i + self.window + 1)):
                if i == j:
                    continue
                pair = tuple(sorted([w, tokens[j]]))
                if pair[0] != pair[1]:
                    self.cooc_counts[pair] += 1
        
        self.total_obs += 1
        
        # 자동 재학습 — 일정 간격 + 단어 수 늘었을 때만
        if (not self._suspend_auto_train
            and self.total_obs - self.last_train_obs >= self.train_interval
            and len(self.word_counts) > self.last_word_count):
            self.train()

    def bootstrap_from_engine(self):
        """라운드 47: 엔진의 다른 지식 소스로 초기 학습.
        - ConceptMemory 정의들 → 텍스트로 observe
        - SA 가중치 → 가상 문장 (이웃들끼리 묶어서)
        - 첫 사용 시 빈 임베딩 방지.

        라운드 73 안정화:
        observe()의 interval 자동 학습을 부트스트랩 중 잠깐 끈 뒤,
        마지막에 한 번만 train()한다. build_full_engine()가 테스트마다
        호출되는 구조에서 반복 SVD로 멈추는 문제를 막는다.
        """
        self._bootstrap_attempted = True
        if self.engine is None:
            return False

        old_suspend = self._suspend_auto_train
        self._suspend_auto_train = True
        try:
            # 1. ConceptMemory 정의들 — "key 정의" 형태
            cm = getattr(self.engine, "concept_memory", None)
            if cm is not None:
                for k, c in cm.concepts.items():
                    synthetic = f"{k} {c.definition}"
                    self.observe(synthetic)

            # 2. SA 가중치 → 각 노드의 이웃들로 *가상 문장* 만들기
            # 강한 페어들 묶어서 같이 등장한 것처럼
            if self.engine.activation_adapter is not None:
                sa = self.engine.activation_adapter.sa
                if hasattr(sa, "weights"):
                    node_neighbors = defaultdict(list)
                    for (a, b), w in sa.weights.items():
                        if any(ch in a + b for ch in '+@_?'):
                            continue
                        if w < 0.3:
                            continue
                        node_neighbors[a].append((b, w))
                        node_neighbors[b].append((a, w))

                    # 빌드 시간 보호: 강한 노드부터 결정론적으로 제한
                    ranked_nodes = sorted(
                        node_neighbors.items(),
                        key=lambda kv: (-max(w for _, w in kv[1]), kv[0])
                    )[: self.max_train_vocab]

                    for node, neighbors in ranked_nodes:
                        neighbors.sort(key=lambda x: (-x[1], x[0]))
                        top_neighbors = [n for n, _ in neighbors[:5]]
                        if top_neighbors:
                            synthetic_sentence = " ".join([node] + top_neighbors)
                            max_w = neighbors[0][1] if neighbors else 0.5
                            repeat = max(1, min(2, int(max_w * 3)))
                            for _ in range(repeat):
                                self.observe(synthetic_sentence)
        finally:
            self._suspend_auto_train = old_suspend

        return self.train()

    def ensure_ready(self) -> bool:
        """임베딩 소비자가 호출하는 lazy 준비 단계.

        build_full_engine()는 모바일/테스트 속도를 위해 SVD를 돌리지 않는다.
        대신 실제로 similarity/composition/attention 경로가 임베딩을 요구할 때
        엔진 지식으로 딱 한 번 부트스트랩한다. 이미 학습됐거나 한 번 시도한
        뒤 실패한 경우에는 반복 SVD를 막기 위해 즉시 현재 상태만 반환한다.
        """
        if not NUMPY_AVAILABLE:
            return False
        if self.embeddings:
            return True
        if self._bootstrap_attempted:
            return bool(self.embeddings)

        # 이미 observe()로 충분한 로컬 데이터가 쌓였으면 전체 엔진 SA를
        # 훑지 않고 그 데이터만 학습한다. ConceptMemory가 넣은 seeded data도
        # 이 경로를 탄다.
        ready_words = [w for w, c in self.word_counts.items()
                       if c >= self.min_word_freq]
        if len(ready_words) >= 5:
            self._bootstrap_attempted = True
            return self.train()

        return self.bootstrap_from_engine()

    def train(self):
        """PMI 행렬 → SVD → 임베딩."""
        if not NUMPY_AVAILABLE:
            return False
        
        # 빈도 임계 통과 단어만. 큰 co-occurrence 그래프는 SVD 비용이 커지므로
        # 상위 빈도 단어로 제한한다. tie-break는 문자열 정렬이라 결정론적이다.
        words = [w for w, c in self.word_counts.items()
                 if c >= self.min_word_freq]
        words.sort(key=lambda w: (-self.word_counts[w], w))
        if self.max_train_vocab is not None and self.max_train_vocab > 0:
            words = words[: self.max_train_vocab]
        if len(words) < 5:
            return False
        
        word_idx = {w: i for i, w in enumerate(words)}
        n = len(words)
        
        # PMI 행렬 (Positive PMI - PPMI)
        pmi = np.zeros((n, n), dtype=np.float32)
        total = sum(self.cooc_counts.values()) or 1
        
        for (a, b), c in self.cooc_counts.items():
            if a not in word_idx or b not in word_idx:
                continue
            ia, ib = word_idx[a], word_idx[b]
            ca, cb = self.word_counts[a], self.word_counts[b]
            
            # PMI = log(P(a,b) / (P(a)*P(b)))
            p_ab = c / total
            p_a = ca / total
            p_b = cb / total
            denom = p_a * p_b
            if denom < 1e-12:
                continue
            pmi_val = np.log(p_ab / denom + 1e-12)
            # PPMI — 음수는 0으로
            ppmi = max(0.0, float(pmi_val))
            pmi[ia, ib] = ppmi
            pmi[ib, ia] = ppmi
        
        # SVD
        try:
            # k = min(dim, n-1) 안전
            k = min(self.dim, max(1, n - 1))
            U, S, Vt = np.linalg.svd(pmi, full_matrices=False)
            # 임베딩 = U[:, :k] * sqrt(S[:k])
            sqrt_s = np.sqrt(np.maximum(S[:k], 0))
            embeds = U[:, :k] * sqrt_s
            
            self.embeddings = {}
            for w, i in word_idx.items():
                vec = embeds[i]
                # NaN 체크
                if not np.isnan(vec).any():
                    self.embeddings[w] = vec
        except (np.linalg.LinAlgError, ValueError):
            return False
        
        self.last_train_obs = self.total_obs
        self.last_word_count = len(self.word_counts)
        self.train_count += 1
        return True

    # ===== 유사도 API =====

    def get_embedding(self, word: str):
        """단어 임베딩 반환 (학습 안 됐으면 None)."""
        return self.embeddings.get(word)

    def similarity(self, word_a: str, word_b: str) -> Optional[float]:
        """cosine similarity. 둘 다 학습됐을 때만."""
        if not NUMPY_AVAILABLE:
            return None
        ea = self.get_embedding(word_a)
        eb = self.get_embedding(word_b)
        if ea is None or eb is None:
            return None
        na = np.linalg.norm(ea)
        nb = np.linalg.norm(eb)
        if na < 1e-8 or nb < 1e-8:
            return None
        return float(np.dot(ea, eb) / (na * nb))

    def most_similar(self, word: str, top_n: int = 5,
                     min_sim: float = 0.0) -> list:
        """가장 유사한 단어들. 결정론적 (alphabet tiebreak)."""
        if word not in self.embeddings:
            return []
        sims = []
        for w in self.embeddings:
            if w == word:
                continue
            s = self.similarity(word, w)
            if s is not None and s > min_sim:
                sims.append((w, s))
        # 결정론 — 점수 내림차순, 같으면 alphabet
        sims.sort(key=lambda x: (-x[1], x[0]))
        return sims[:top_n]

    # ===== 텍스트 → 평균 임베딩 =====

    def text_embedding(self, text: str):
        """텍스트의 평균 임베딩 (BoW)."""
        if not NUMPY_AVAILABLE:
            return None
        tokens = simple_tokenize(text)
        vecs = [self.embeddings[t] for t in tokens if t in self.embeddings]
        if not vecs:
            return None
        return np.mean(vecs, axis=0)

    def text_similarity(self, text_a: str, text_b: str) -> Optional[float]:
        """두 텍스트의 의미 유사도 (평균 임베딩 cosine)."""
        if not NUMPY_AVAILABLE:
            return None
        va = self.text_embedding(text_a)
        vb = self.text_embedding(text_b)
        if va is None or vb is None:
            return None
        na = np.linalg.norm(va)
        nb = np.linalg.norm(vb)
        if na < 1e-8 or nb < 1e-8:
            return None
        return float(np.dot(va, vb) / (na * nb))

    # ===== 통계 =====

    def stats(self) -> dict:
        return {
            "total_obs": self.total_obs,
            "vocab_size": len(self.word_counts),
            "trained_size": len(self.embeddings),
            "cooc_pairs": len(self.cooc_counts),
            "train_count": self.train_count,
            "last_train_obs": self.last_train_obs,
            "bootstrap_attempted": self._bootstrap_attempted,
        }
