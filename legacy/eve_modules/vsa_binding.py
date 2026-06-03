"""
EVE v35 - E5: VSABinding (Vector Symbolic Architecture)
=========================================================

목표: 의미 합성 → 창의력 폭증.

기존 한계:
- 카테고리 페어 연결만 가능. 합성 표현 X.
- "빨간 사과" + "위" + "파란 새" 같은 결합 관계 표현 불가.
- 창의는 단순 연결만으로 한계.

E5 해결:
- 각 카테고리에 결정론적 bit vector 부여 (sha256 hash 기반, no random)
- bind(role, filler) = role ⊗ filler → bound vector (composition)
- unbind(bound, role) = bound ⊗ role → filler 복원
- compose(**role_filler) = 여러 binding의 superposition
- 합성 결과는 SA 그래프에 새 카테고리로 등록 (EVE 의미 단위 원칙 유지)

EVE 절대 원칙 부합:
- 결정론적 (sha256 hash, no random/sample)
- 카테고리 string 유지 (vector는 *내부 표현*, 외부 인터페이스는 카테고리 이름)
- SA 그래프에 새 카테고리 등록 (다른 모듈은 평소처럼 사용)
- 호르몬 변조 (선택, 합성 시 도파민↑ 창의 신호)

학계 기반:
- Smolensky 1990 — Tensor Product Representations
- Plate 1995 — Holographic Reduced Representations (HRR)
- Kanerva 2009 — Hyperdimensional Computing (HDC)
- Kleyko 2022 — VSA Survey, deterministic binding
- LARS-VSA 2024 — relational binding for AI
- Hofstadter 2001 — Combinatorial Creativity (창의의 본질)

핵심 함수:
1. get_vector(category)                  → 결정론적 bipolar vector
2. bind(role, filler, register=True)     → 합성 카테고리 + vector
3. unbind(bound, role)                   → filler 복원
4. compose(**role_filler)                → 다수 binding superposition
5. similarity(a, b)                      → cosine 유사도
6. nearest_category(vec, k=3)            → 가장 비슷한 카테고리
7. tick(dt)                              → no-op (외부 자극 모듈)

Author: 김민석 + Claude v35
"""

import hashlib
import numpy as np
from typing import Dict, List, Set, Tuple, Optional, Any, Union


class VSABinding:
    """
    Vector Symbolic Architecture - 의미 합성 모듈.

    의존성:
        - SpreadingActivation (필수, 합성 카테고리 등록)
        - HormoneSystem (선택, 합성 시 호르몬 자극)

    설계:
        - vector_dim 차원 bipolar (-1/+1) vector
        - sha256 결정론적 카테고리→vector 매핑
        - bind = element-wise multiply (XOR for bipolar)
        - unbind = bind와 동일 (involution: a*b*b = a)
        - compose = sum + sign() (superposition + cleanup)
        - 합성 카테고리는 SA 그래프에 등록되어 다른 모듈도 사용 가능
    """

    def __init__(self,
                 spreading_activation,
                 hormone_system=None,
                 vector_dim: int = 512,
                 hash_seed: str = 'eve_v35_vsa',
                 register_link_strength: float = 0.4,
                 use_hormone_modulation: bool = True):
        """
        Args:
            spreading_activation: SA (필수, 합성 카테고리 그래프 등록)
            hormone_system: HS (선택)
            vector_dim: bit vector 차원 (512 권장, 8 multiple)
            hash_seed: hash salt (결정론, 변경 시 모든 vector 달라짐)
            register_link_strength: bind 결과를 SA에 등록할 때 link 강도
            use_hormone_modulation: bind 시 호르몬 자극
        """
        if vector_dim % 8 != 0:
            raise ValueError("vector_dim must be multiple of 8")

        self.sa = spreading_activation
        self.hs = hormone_system
        self.dim = int(vector_dim)
        self.hash_seed = str(hash_seed)
        self.register_link_strength = float(register_link_strength)
        self.use_hormone_modulation = bool(use_hormone_modulation)

        # 카테고리 → bipolar vector 캐시 (-1/+1 ndarray)
        self.vectors: Dict[str, np.ndarray] = {}

        # 합성 카테고리 메타 (디버그/unbind 추적)
        # bound_name → {'role': str, 'filler': str, 'parts': dict}
        self.bound_meta: Dict[str, Dict[str, Any]] = {}

        # 통계
        self.bind_count = 0
        self.unbind_count = 0
        self.compose_count = 0
        self.similarity_count = 0

    # ============= 1. Vector 생성 (결정론) =============
    def get_vector(self, category: str) -> np.ndarray:
        """
        카테고리 → 결정론적 bipolar vector (-1/+1).

        sha256(seed + category)를 dim bits로 확장.
        같은 카테고리는 항상 같은 vector.

        반환: shape (dim,), dtype int8, 값 -1 또는 +1
        """
        if not category:
            raise ValueError("category cannot be empty")

        if category in self.vectors:
            return self.vectors[category]

        # sha256 → 256 bits. dim이 더 크면 반복.
        seed_bytes = f"{self.hash_seed}:{category}".encode('utf-8')
        all_bytes = b''
        cur = hashlib.sha256(seed_bytes).digest()  # 32 bytes = 256 bits
        bytes_needed = self.dim // 8
        all_bytes += cur
        # 더 필요하면 chained hash (결정론)
        while len(all_bytes) < bytes_needed:
            cur = hashlib.sha256(cur).digest()
            all_bytes += cur

        all_bytes = all_bytes[:bytes_needed]
        bits = np.unpackbits(np.frombuffer(all_bytes, dtype=np.uint8))
        # 0/1 → -1/+1 (bipolar)
        bipolar = (bits.astype(np.int8) * 2) - 1
        bipolar = bipolar[:self.dim].copy()
        self.vectors[category] = bipolar
        return bipolar

    # ============= 2. Bind =============
    def bind(self, role: str, filler: str,
             register: bool = True,
             register_in_sa: bool = True) -> Dict[str, Any]:
        """
        role + filler → bound 카테고리 + vector.

        bipolar XOR = element-wise multiply (commutative, self-inverse).
        합성 카테고리 이름 = "filler@role" (결정론, 동일 입력 동일 결과).

        Args:
            role: 역할 (예: '_MODIFIER', '_LOCATION')
            filler: 채울 값 (예: '빨간', '위')
            register: 합성 vector를 캐시에 저장
            register_in_sa: SA 그래프에 새 카테고리로 등록

        Returns:
            {'bound_name': str, 'vector': ndarray, 'role': str, 'filler': str}
        """
        if not role or not filler:
            raise ValueError("role and filler must be non-empty")

        v_role = self.get_vector(role)
        v_filler = self.get_vector(filler)
        v_bound = (v_role * v_filler).astype(np.int8)

        # 결정론적 이름
        bound_name = f"{filler}@{role}"

        if register:
            self.vectors[bound_name] = v_bound
            self.bound_meta[bound_name] = {
                'role': role,
                'filler': filler,
                'kind': 'bind',
            }

        if register_in_sa:
            # SA 그래프에 새 카테고리 등록 (다른 모듈도 사용 가능)
            self.sa.discover_category(
                bound_name,
                context_categories={role, filler},
                initial_strength=0.3,
                link_strength=self.register_link_strength,
            )

        # 호르몬: 합성 = 창의 신호 → DA 약하게
        if self.use_hormone_modulation and self.hs is not None:
            self.hs.hormones['dopamine'].stimulate(0.02)
            self.hs.hormones['acetylcholine'].stimulate(0.02)

        self.bind_count += 1
        return {
            'bound_name': bound_name,
            'vector': v_bound,
            'role': role,
            'filler': filler,
        }

    # ============= 3. Unbind (복원) =============
    def unbind(self,
              bound: Union[str, np.ndarray],
              role: str,
              top_k: int = 3) -> Dict[str, Any]:
        """
        bound + role → filler 복원.

        v_bound * v_role = v_filler_noisy
        v_filler_noisy와 가장 비슷한 카테고리 찾기.

        Args:
            bound: 합성 카테고리 이름 또는 vector
            role: 추출할 role
            top_k: 상위 k 후보 반환

        Returns:
            {'recovered': str, 'similarity': float, 'top_k': List[(cat, sim)]}
        """
        if isinstance(bound, str):
            v_bound = self.get_vector(bound)
        else:
            v_bound = np.asarray(bound, dtype=np.int8)

        v_role = self.get_vector(role)
        v_filler_recovered = (v_bound * v_role).astype(np.int8)

        # vectors 캐시에서 가장 비슷한 카테고리 찾기
        # (bound 자체와 role은 제외)
        candidates: List[Tuple[str, float]] = []
        for cat, vec in self.vectors.items():
            if isinstance(bound, str) and cat == bound:
                continue
            if cat == role:
                continue
            sim = self._cosine(v_filler_recovered, vec)
            candidates.append((cat, sim))

        # 결정론 정렬 (sim ↓, cat ↑)
        candidates.sort(key=lambda x: (-x[1], x[0]))
        top = candidates[:top_k]

        self.unbind_count += 1
        if top:
            return {
                'recovered': top[0][0],
                'similarity': float(top[0][1]),
                'top_k': [(c, float(s)) for c, s in top],
            }
        return {
            'recovered': None,
            'similarity': 0.0,
            'top_k': [],
        }

    # ============= 4. Compose (다중 binding) =============
    def compose(self,
               role_filler: Dict[str, str],
               name: Optional[str] = None,
               register_in_sa: bool = True) -> Dict[str, Any]:
        """
        여러 role-filler 쌍 → 합성 카테고리 (superposition).

        Σ bind(role_i, filler_i) → sign() → bipolar
        예: compose({'_MOD': '빨간', '_HEAD': '사과'})
            = sign(bind(_MOD,빨간) + bind(_HEAD,사과))
            = 합성 vector + 카테고리 이름 '빨간+사과'

        Args:
            role_filler: {role: filler, ...}
            name: 합성 카테고리 이름 (None이면 자동 생성)
            register_in_sa: SA 그래프 등록 여부

        Returns:
            {'composite_name': str, 'vector': ndarray, 'parts': dict}
        """
        if not role_filler:
            raise ValueError("role_filler dict cannot be empty")

        # 각 페어 bind (register X — superposition만 위해)
        bound_vecs = []
        # 결정론적 정렬 (key 알파벳 순)
        for role in sorted(role_filler.keys()):
            filler = role_filler[role]
            r = self.bind(role, filler, register=True, register_in_sa=False)
            bound_vecs.append(r['vector'])

        # Superposition: sum + sign() cleanup
        # sign(0) → +1 (결정론)
        composite_sum = np.sum(bound_vecs, axis=0)
        composite = np.sign(composite_sum).astype(np.int8)
        composite[composite == 0] = 1

        # 이름 자동 생성 (filler들 + 정렬, 결정론)
        if name is None:
            fillers_sorted = sorted(set(role_filler.values()))
            name = '+'.join(fillers_sorted)

        self.vectors[name] = composite
        self.bound_meta[name] = {
            'kind': 'compose',
            'parts': dict(role_filler),
            'fillers': sorted(role_filler.values()),
        }

        if register_in_sa:
            # SA 그래프에 등록
            ctx = set(role_filler.values()) | set(role_filler.keys())
            self.sa.discover_category(
                name,
                context_categories=ctx,
                initial_strength=0.4,
                link_strength=self.register_link_strength,
            )

        if self.use_hormone_modulation and self.hs is not None:
            # compose = 더 강한 합성 → DA 더 강하게 (창의 보상)
            self.hs.hormones['dopamine'].stimulate(0.04)
            self.hs.hormones['acetylcholine'].stimulate(0.03)

        self.compose_count += 1
        return {
            'composite_name': name,
            'vector': composite,
            'parts': dict(role_filler),
        }

    # ============= 5. 유사도 / 검색 =============
    def similarity(self, a: Union[str, np.ndarray],
                  b: Union[str, np.ndarray]) -> float:
        """
        두 카테고리/vector cosine 유사도.
        bipolar vector라 dot/dim과 같음.

        Returns: -1.0 ~ 1.0
        """
        v_a = self.get_vector(a) if isinstance(a, str) else np.asarray(a, dtype=np.int8)
        v_b = self.get_vector(b) if isinstance(b, str) else np.asarray(b, dtype=np.int8)
        self.similarity_count += 1
        return float(self._cosine(v_a, v_b))

    def _cosine(self, v_a: np.ndarray, v_b: np.ndarray) -> float:
        """bipolar vector 간 정규화 dot product (int8 overflow 방지 위해 int32로 캐스팅)"""
        # int8 * int8 dot은 overflow! → int32로 elevate
        return float(np.dot(v_a.astype(np.int32), v_b.astype(np.int32)) / self.dim)

    def nearest_category(self,
                        target: Union[str, np.ndarray],
                        k: int = 3,
                        exclude: Optional[Set[str]] = None) -> List[Tuple[str, float]]:
        """
        target과 가장 비슷한 카테고리 top-k.

        Args:
            target: 카테고리 이름 또는 vector
            k: top k
            exclude: 제외할 카테고리 set
        """
        v_target = (self.get_vector(target) if isinstance(target, str)
                   else np.asarray(target, dtype=np.int8))
        exclude = exclude or set()
        if isinstance(target, str):
            exclude = exclude | {target}

        candidates: List[Tuple[str, float]] = []
        for cat, vec in self.vectors.items():
            if cat in exclude:
                continue
            sim = self._cosine(v_target, vec)
            candidates.append((cat, float(sim)))

        # 결정론 정렬
        candidates.sort(key=lambda x: (-x[1], x[0]))
        return candidates[:k]

    # ============= 6. 통계 / tick =============
    def tick(self, dt: float = 0.5):
        """no-op — 외부 자극 모듈"""
        pass

    def stats(self) -> Dict[str, Any]:
        return {
            'vector_dim': self.dim,
            'cached_vectors': len(self.vectors),
            'bound_categories': len(self.bound_meta),
            'bind_count': self.bind_count,
            'unbind_count': self.unbind_count,
            'compose_count': self.compose_count,
            'similarity_count': self.similarity_count,
        }

    def __repr__(self):
        s = self.stats()
        return (f"VSABinding(dim={s['vector_dim']}, "
                f"vectors={s['cached_vectors']}, "
                f"binds={s['bind_count']}, "
                f"composes={s['compose_count']})")
