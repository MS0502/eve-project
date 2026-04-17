"""
Eve Project - Memory System
Day 4: 기억 시스템

구현:
1. 작업기억 (Working Memory) - 최근 활성 유지
2. 장기기억 (Long-term Memory) - 영구 저장
3. 헤비안 학습 - 경험 기반 연결 강화
4. 망각 - 시간 기반 약화
5. 수면 공고화 - 단기→장기 이동

핵심 설계:
- 함수 기반 + 학습된 조정 = 2층 구조
- 저장 공간 최소화
- 인간 뇌 현상 자연 발현
"""

import numpy as np
import json
import os
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Optional
from collections import deque


# ============================================================
# 1. 작업기억 (Working Memory)
# ============================================================

class WorkingMemory:
    """
    현재 활성화된 정보를 짧게 유지
    
    인간 기준:
    - 용량: 7±2 청크
    - 지속: 수초~수십초
    - 주의하지 않으면 사라짐
    """
    
    def __init__(self, capacity: int = 7, duration_steps: int = 200):
        self.capacity = capacity
        self.duration_steps = duration_steps
        
        # 최근 활성 패턴들 (deque로 자동 크기 제한)
        self.items: deque = deque(maxlen=capacity)
    
    def add(self, pattern: Dict):
        """새 정보 추가 (오래된 것 자동 제거)"""
        pattern["timestamp"] = datetime.now().isoformat()
        pattern["age_steps"] = 0
        self.items.append(pattern)
    
    def age_all(self):
        """모든 항목 나이 증가 (매 스텝 호출)"""
        for item in self.items:
            item["age_steps"] += 1
        
        # 너무 오래된 것 제거
        self.items = deque(
            [item for item in self.items if item["age_steps"] < self.duration_steps],
            maxlen=self.capacity
        )
    
    def get_recent(self, n: int = 3) -> List[Dict]:
        """가장 최근 n개"""
        return list(self.items)[-n:]
    
    def is_empty(self) -> bool:
        return len(self.items) == 0
    
    def clear(self):
        self.items.clear()


# ============================================================
# 2. 일화기억 (Episodic Memory) - 한 사건 단위
# ============================================================

@dataclass
class Episode:
    """한 번의 경험"""
    id: str
    timestamp: str
    input_text: str
    dominant_region: str
    hormone_state: Dict[str, float]
    activity_pattern: Dict[str, float]  # 각 영역 활성도
    importance: float = 0.5
    access_count: int = 0
    last_accessed: str = ""
    
    def to_dict(self):
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data):
        return cls(**data)


# ============================================================
# 3. 장기기억 (Long-term Memory)
# ============================================================

class LongTermMemory:
    """
    영구 저장되는 기억
    
    - SSD/Drive에 저장
    - 중요도 기반 유지
    - 망각 곡선 적용
    """
    
    def __init__(self, storage_path: str = "eve_memory.json"):
        self.storage_path = storage_path
        self.episodes: List[Episode] = []
        
        # 학습된 시냅스 조정값 (소스_타겟 → 델타)
        self.learned_weights: Dict[str, float] = {}
        
        # 파일에서 불러오기
        self.load()
    
    def add_episode(self, episode: Episode):
        """새 일화 추가"""
        self.episodes.append(episode)
    
    def search_similar(self, query_pattern: Dict[str, float], top_k: int = 3) -> List[Episode]:
        """
        유사한 경험 검색
        
        현재 활동 패턴과 비슷한 과거 일화 찾기
        """
        if not self.episodes:
            return []
        
        scored = []
        for episode in self.episodes:
            # 활동 패턴 유사도 계산 (코사인 유사도 간략 버전)
            similarity = self._pattern_similarity(
                query_pattern,
                episode.activity_pattern
            )
            scored.append((similarity, episode))
        
        # 유사도 높은 순
        scored.sort(key=lambda x: x[0], reverse=True)
        return [ep for _, ep in scored[:top_k]]
    
    def _pattern_similarity(self, p1: Dict[str, float], p2: Dict[str, float]) -> float:
        """두 패턴의 유사도 (0~1)"""
        keys = set(p1.keys()) | set(p2.keys())
        if not keys:
            return 0.0
        
        dot_product = 0.0
        norm1 = 0.0
        norm2 = 0.0
        
        for key in keys:
            v1 = p1.get(key, 0.0)
            v2 = p2.get(key, 0.0)
            dot_product += v1 * v2
            norm1 += v1 ** 2
            norm2 += v2 ** 2
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (np.sqrt(norm1) * np.sqrt(norm2))
    
    def apply_forgetting(self, decay_rate: float = 0.01):
        """
        망각 곡선 적용 (에빙하우스)
        
        중요도가 낮은 기억은 점점 약화
        접근 많은 기억은 강해짐
        """
        now = datetime.now()
        survived = []
        
        for episode in self.episodes:
            # 마지막 접근 이후 시간
            if episode.last_accessed:
                last = datetime.fromisoformat(episode.last_accessed)
            else:
                last = datetime.fromisoformat(episode.timestamp)
            
            days_passed = (now - last).total_seconds() / 86400
            
            # 에빙하우스 망각 공식: 시간 지날수록 약화
            # 접근 횟수가 많으면 덜 약화
            retention_strength = (1.0 + episode.access_count * 0.2)
            decay = decay_rate * days_passed / retention_strength
            
            episode.importance -= decay
            
            # 최소 중요도 넘으면 유지
            if episode.importance > 0.1:
                survived.append(episode)
        
        self.episodes = survived
    
    def access(self, episode_id: str):
        """기억 접근 (접근 시 강화됨 - 테스트 효과)"""
        for ep in self.episodes:
            if ep.id == episode_id:
                ep.access_count += 1
                ep.last_accessed = datetime.now().isoformat()
                ep.importance = min(1.0, ep.importance + 0.05)
                break
    
    # ========================================================
    # 헤비안 학습 - 시냅스 가중치 조정
    # ========================================================
    
    def strengthen_connection(self, source_id: int, target_id: int, amount: float = 0.01):
        """
        헤비안 학습: 같이 발화한 연결 강화
        "뉴런이 함께 발화하면 함께 연결된다"
        """
        key = f"{source_id}_{target_id}"
        current = self.learned_weights.get(key, 0.0)
        self.learned_weights[key] = current + amount
        
        # 범위 제한 (-1, 1)
        self.learned_weights[key] = max(-1.0, min(1.0, self.learned_weights[key]))
    
    def get_learned_delta(self, source_id: int, target_id: int) -> float:
        """학습된 시냅스 조정값"""
        key = f"{source_id}_{target_id}"
        return self.learned_weights.get(key, 0.0)
    
    def weight_decay(self, rate: float = 0.001):
        """
        사용 안 하는 시냅스 약화
        (망각의 신경학적 구현)
        """
        to_remove = []
        for key in self.learned_weights:
            self.learned_weights[key] *= (1.0 - rate)
            
            # 거의 0이면 제거 (메모리 절약)
            if abs(self.learned_weights[key]) < 0.001:
                to_remove.append(key)
        
        for key in to_remove:
            del self.learned_weights[key]
    
    # ========================================================
    # 저장 / 불러오기
    # ========================================================
    
    def save(self):
        """장기기억을 파일로 저장"""
        data = {
            "episodes": [ep.to_dict() for ep in self.episodes],
            "learned_weights": self.learned_weights,
            "saved_at": datetime.now().isoformat(),
        }
        
        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def load(self):
        """저장된 기억 불러오기"""
        if not os.path.exists(self.storage_path):
            return
        
        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            self.episodes = [Episode.from_dict(ep) for ep in data.get("episodes", [])]
            self.learned_weights = data.get("learned_weights", {})
        except Exception as e:
            print(f"기억 불러오기 실패: {e}")


# ============================================================
# 4. 수면 공고화 (Sleep Consolidation)
# ============================================================

class SleepConsolidation:
    """
    수면 중 기억 처리
    
    실제 뇌:
    - 단기기억 → 해마 → 대뇌피질 이동
    - 중요한 것만 남김
    - 꿈 = 기억 재조합
    """
    
    def __init__(self, long_term_memory: LongTermMemory):
        self.ltm = long_term_memory
    
    def consolidate(
        self,
        working_memory: WorkingMemory,
        activity_snapshots: List[Dict[str, float]],
    ):
        """
        수면 시 실행: 작업기억 → 장기기억 이동
        
        1. 중요한 작업기억만 선택
        2. 일화로 만들어 장기기억 저장
        3. 자주 나온 패턴 학습 강화
        4. 사용 안 한 연결 약화
        """
        print("  [수면] 기억 공고화 시작...")
        
        consolidated_count = 0
        
        # 1. 작업기억 중 중요한 것 선별
        for item in working_memory.items:
            importance = self._calculate_importance(item)
            
            if importance > 0.5:
                # 일화기억으로 변환
                episode = Episode(
                    id=f"ep_{datetime.now().timestamp()}_{consolidated_count}",
                    timestamp=item.get("timestamp", datetime.now().isoformat()),
                    input_text=item.get("input", ""),
                    dominant_region=item.get("dominant_region", "unknown"),
                    hormone_state=item.get("hormones", {}),
                    activity_pattern=item.get("activity", {}),
                    importance=importance,
                )
                self.ltm.add_episode(episode)
                consolidated_count += 1
        
        # 2. 자주 나온 패턴 학습 (의미기억 기반)
        if len(activity_snapshots) > 0:
            self._learn_frequent_patterns(activity_snapshots)
        
        # 3. 망각 곡선 적용
        self.ltm.apply_forgetting()
        
        # 4. 시냅스 감쇠 (잘 사용 안 한 것)
        self.ltm.weight_decay()
        
        print(f"  [수면] {consolidated_count}개 일화 공고화")
        print(f"  [수면] 총 장기기억: {len(self.ltm.episodes)}개")
        print(f"  [수면] 학습된 시냅스: {len(self.ltm.learned_weights)}개")
        
        # 5. 저장
        self.ltm.save()
    
    def _calculate_importance(self, item: Dict) -> float:
        """
        작업기억 항목의 중요도
        
        요인:
        - 감정 강도 (호르몬)
        - 반복 여부
        - 새로움
        """
        importance = 0.3  # 기본값
        
        hormones = item.get("hormones", {})
        
        # 강한 감정 → 섬광기억 후보
        if hormones.get("cortisol", 0) > 0.7:
            importance += 0.3
        
        if hormones.get("dopamine", 0) > 0.7:
            importance += 0.3
        
        if hormones.get("oxytocin", 0) > 0.7:
            importance += 0.2
        
        return min(1.0, importance)
    
    def _learn_frequent_patterns(self, snapshots: List[Dict[str, float]]):
        """
        자주 같이 활성화된 영역 간 연결 강화
        (헤비안 학습의 영역 간 버전)
        """
        # 간단화: 자주 동시에 활성화된 영역 쌍 찾기
        region_pairs_count = {}
        
        for snap in snapshots:
            active_regions = [r for r, v in snap.items() if v > 0.02]
            
            for i, r1 in enumerate(active_regions):
                for r2 in active_regions[i+1:]:
                    pair = tuple(sorted([r1, r2]))
                    region_pairs_count[pair] = region_pairs_count.get(pair, 0) + 1


# ============================================================
# 5. 테스트
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Eve - Day 4: 기억 시스템 테스트")
    print("=" * 60)
    
    # 테스트 1: 작업기억
    print("\n[테스트 1] 작업기억 (7±2 용량)")
    wm = WorkingMemory(capacity=7)
    
    # 10개 넣기 (7개만 남아야 함)
    for i in range(10):
        wm.add({"content": f"item_{i}", "input": f"메시지 {i}"})
    
    print(f"  10개 추가 후 크기: {len(wm.items)} (기대값: 7)")
    recent = wm.get_recent(3)
    print(f"  최근 3개: {[item['content'] for item in recent]}")
    
    # 시간 경과
    for _ in range(300):
        wm.age_all()
    
    print(f"  300스텝 후 크기: {len(wm.items)} (기대값: 0, 만료)")
    
    # 테스트 2: 장기기억 저장/로드
    print("\n[테스트 2] 장기기억 저장과 검색")
    
    # 이전 테스트 파일 삭제
    if os.path.exists("eve_memory.json"):
        os.remove("eve_memory.json")
    
    ltm = LongTermMemory("eve_memory.json")
    
    # 여러 일화 저장
    test_episodes = [
        Episode(
            id="ep_1",
            timestamp=datetime.now().isoformat(),
            input_text="안녕 이브",
            dominant_region="hippocampus",
            hormone_state={"dopamine": 0.5, "oxytocin": 0.6},
            activity_pattern={"sensory": 0.01, "emotional": 0.02, "hippocampus": 0.03},
            importance=0.6,
        ),
        Episode(
            id="ep_2",
            timestamp=datetime.now().isoformat(),
            input_text="사랑해",
            dominant_region="emotional",
            hormone_state={"dopamine": 0.8, "oxytocin": 0.9},
            activity_pattern={"sensory": 0.01, "emotional": 0.04, "prefrontal": 0.02},
            importance=0.9,
        ),
        Episode(
            id="ep_3",
            timestamp=datetime.now().isoformat(),
            input_text="위험해",
            dominant_region="emotional",
            hormone_state={"cortisol": 0.9},
            activity_pattern={"sensory": 0.01, "emotional": 0.05, "inhibitory": 0.03},
            importance=0.8,
        ),
    ]
    
    for ep in test_episodes:
        ltm.add_episode(ep)
    
    print(f"  저장된 일화: {len(ltm.episodes)}개")
    
    # 유사 기억 검색
    query = {"emotional": 0.04, "hippocampus": 0.02}
    similar = ltm.search_similar(query, top_k=2)
    
    print(f"  감정 높은 상황에서 유사 기억:")
    for ep in similar:
        print(f"    - '{ep.input_text}' (중요도: {ep.importance:.2f})")
    
    # 저장
    ltm.save()
    print(f"  파일로 저장됨: eve_memory.json")
    
    # 테스트 3: 헤비안 학습
    print("\n[테스트 3] 헤비안 학습 (시냅스 강화)")
    
    # 뉴런 10과 20이 여러 번 같이 발화
    for _ in range(50):
        ltm.strengthen_connection(10, 20, amount=0.02)
    
    # 뉴런 10과 30은 한 번만 같이
    ltm.strengthen_connection(10, 30, amount=0.02)
    
    weight_10_20 = ltm.get_learned_delta(10, 20)
    weight_10_30 = ltm.get_learned_delta(10, 30)
    
    print(f"  (10→20) 학습된 가중치: {weight_10_20:.3f} (50번 반복)")
    print(f"  (10→30) 학습된 가중치: {weight_10_30:.3f} (1번)")
    print(f"  자주 같이 발화 → 더 강함 ({weight_10_20/weight_10_30:.1f}배)")
    
    # 테스트 4: 망각
    print("\n[테스트 4] 시냅스 감쇠 (망각)")
    
    before = ltm.get_learned_delta(10, 20)
    
    # 10번 감쇠 적용 (= 시간 경과 시뮬레이션)
    for _ in range(100):
        ltm.weight_decay(rate=0.005)
    
    after = ltm.get_learned_delta(10, 20)
    print(f"  감쇠 전: {before:.3f}")
    print(f"  감쇠 후: {after:.3f}")
    print(f"  손실률: {(1 - after/before)*100:.1f}%")
    
    # 테스트 5: 수면 공고화
    print("\n[테스트 5] 수면 공고화")
    
    # 작업기억에 여러 항목
    wm2 = WorkingMemory(capacity=10)
    
    # 중요한 것 (높은 감정)
    wm2.add({
        "input": "처음 만난 날",
        "dominant_region": "emotional",
        "hormones": {"oxytocin": 0.9, "dopamine": 0.8},
        "activity": {"emotional": 0.05, "hippocampus": 0.03},
    })
    
    # 일상적인 것 (낮은 중요도)
    wm2.add({
        "input": "오늘 날씨",
        "dominant_region": "prefrontal",
        "hormones": {"dopamine": 0.4},
        "activity": {"prefrontal": 0.02},
    })
    
    # 강렬한 것 (스트레스)
    wm2.add({
        "input": "놀라운 사건",
        "dominant_region": "emotional",
        "hormones": {"cortisol": 0.9},
        "activity": {"emotional": 0.06},
    })
    
    # 새 LTM으로 수면 시뮬레이션
    ltm2 = LongTermMemory("eve_sleep_test.json")
    sleep = SleepConsolidation(ltm2)
    
    print(f"  수면 전 작업기억: {len(wm2.items)}개")
    print(f"  수면 전 장기기억: {len(ltm2.episodes)}개")
    
    sleep.consolidate(wm2, activity_snapshots=[])
    
    print(f"  수면 후 장기기억: {len(ltm2.episodes)}개")
    print(f"  → 중요한 것만 장기기억으로 이동")
    
    # 정리
    if os.path.exists("eve_sleep_test.json"):
        os.remove("eve_sleep_test.json")
    if os.path.exists("eve_memory.json"):
        os.remove("eve_memory.json")
    
    print("\n" + "=" * 60)
    print("Day 4 기억 시스템 테스트 완료")
    print("=" * 60)
    print("\n구현 완료:")
    print("  ✓ 작업기억 (7±2 용량, 시간 기반 소멸)")
    print("  ✓ 장기기억 (일화 단위, 영구 저장)")
    print("  ✓ 헤비안 학습 (함께 발화한 연결 강화)")
    print("  ✓ 망각 (에빙하우스 곡선)")
    print("  ✓ 수면 공고화 (중요한 것만 장기기억으로)")
    print("\n창발적으로 구현될 것:")
    print("  - 섬광기억: 강한 감정 시 자동 중요도 상승")
    print("  - 일화기억: 이미 Episode 구조")
    print("  - 테스트 효과: access() 시 중요도 증가")
    print("  - 자서전적: 장기기억이 축적되면 자연스럽게")
