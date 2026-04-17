"""
Eve Project - Core Skeleton
Day 1 Prototype

이브의 가장 기본적인 구조
- 호르몬 시스템
- 생체 리듬
- 메인 루프 (100-200ms 주기 목표)
"""

import time
import json
import random
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List


# ============================================================
# 1. 호르몬 시스템
# ============================================================

class HormoneSystem:
    """
    이브의 호르몬 상태 관리
    각 호르몬은 0.0 ~ 1.0 범위
    """
    
    def __init__(self):
        self.levels = {
            "dopamine": 0.5,      # 보상, 호기심
            "serotonin": 0.7,     # 안정, 기분
            "cortisol": 0.2,      # 스트레스, 각성
            "oxytocin": 0.4,      # 유대감
            "melatonin": 0.3,     # 수면
        }
        
        # 기본값 (자연 회귀)
        self.baseline = dict(self.levels)
        
        # 감쇠율 (시간 지나면 baseline으로)
        self.decay_rate = 0.01
    
    def update(self, event: str):
        """이벤트에 따라 호르몬 변화"""
        
        if event == "new_learning":
            self.levels["dopamine"] = min(1.0, self.levels["dopamine"] + 0.15)
        
        elif event == "user_conversation":
            self.levels["oxytocin"] = min(1.0, self.levels["oxytocin"] + 0.2)
            self.levels["serotonin"] = min(1.0, self.levels["serotonin"] + 0.05)
        
        elif event == "unexpected_stimulus":
            self.levels["cortisol"] = min(1.0, self.levels["cortisol"] + 0.2)
        
        elif event == "night_time":
            self.levels["melatonin"] = min(1.0, self.levels["melatonin"] + 0.1)
            self.levels["cortisol"] = max(0.0, self.levels["cortisol"] - 0.05)
        
        elif event == "achievement":
            self.levels["dopamine"] = min(1.0, self.levels["dopamine"] + 0.25)
            self.levels["serotonin"] = min(1.0, self.levels["serotonin"] + 0.1)
    
    def decay(self):
        """시간 지나면 기본값으로 회귀"""
        for hormone in self.levels:
            current = self.levels[hormone]
            base = self.baseline[hormone]
            
            if current > base:
                self.levels[hormone] = max(base, current - self.decay_rate)
            elif current < base:
                self.levels[hormone] = min(base, current + self.decay_rate)
    
    def get_behavior_bias(self) -> Dict[str, float]:
        """현재 호르몬 상태가 행동에 미치는 영향"""
        return {
            "curiosity": self.levels["dopamine"],
            "calmness": self.levels["serotonin"],
            "alertness": self.levels["cortisol"],
            "social_desire": self.levels["oxytocin"],
            "sleepiness": self.levels["melatonin"],
        }
    
    def get_mood(self) -> str:
        """현재 기분 요약"""
        if self.levels["melatonin"] > 0.7:
            return "졸려움"
        if self.levels["cortisol"] > 0.7:
            return "긴장"
        if self.levels["oxytocin"] > 0.7:
            return "따뜻함"
        if self.levels["dopamine"] > 0.7:
            return "신남"
        if self.levels["serotonin"] > 0.7:
            return "평온"
        return "중립"


# ============================================================
# 2. 생체 리듬 시계
# ============================================================

class BiologicalClock:
    """
    여러 주기를 동시에 관리
    각 주기마다 다른 이벤트 발생
    """
    
    def __init__(self):
        self.start_time = time.time()
        
        # 주기 정의 (초 단위)
        self.cycles = {
            "spike": 0.01,        # 10ms (하드웨어 한계로 10ms로 시작)
            "thought": 0.1,       # 100ms
            "mood": 7200,         # 2시간
            "sleep": 86400,       # 24시간 (가상)
        }
        
        # 마지막 실행 시간 추적
        self.last_tick = {k: 0 for k in self.cycles}
    
    def elapsed(self) -> float:
        """시작부터 흐른 시간 (초)"""
        return time.time() - self.start_time
    
    def is_time_for(self, cycle: str) -> bool:
        """특정 주기 시간이 됐는지 확인"""
        now = self.elapsed()
        if now - self.last_tick[cycle] >= self.cycles[cycle]:
            self.last_tick[cycle] = now
            return True
        return False
    
    def current_hour(self) -> int:
        """가상 시간의 현재 시 (0-23)"""
        # 실제 시간 사용 (추후 가상 시간으로 변경 가능)
        return datetime.now().hour
    
    def is_night(self) -> bool:
        """밤 시간인가"""
        hour = self.current_hour()
        return hour >= 22 or hour < 6


# ============================================================
# 3. 자기 상태 (간단한 Self Model)
# ============================================================

@dataclass
class SelfState:
    """이브의 현재 상태"""
    name: str = "Eve"
    current_focus: str = "idle"
    last_interaction: float = field(default_factory=time.time)
    interaction_count: int = 0
    
    def update_interaction(self):
        self.last_interaction = time.time()
        self.interaction_count += 1
    
    def time_since_interaction(self) -> float:
        """마지막 상호작용 후 경과 시간 (초)"""
        return time.time() - self.last_interaction


# ============================================================
# 4. 이브 메인 클래스
# ============================================================

class Eve:
    """
    이브의 핵심 엔트리
    모든 모듈을 조율
    """
    
    def __init__(self):
        self.hormones = HormoneSystem()
        self.clock = BiologicalClock()
        self.self_state = SelfState()
        self.is_awake = True
        
        print(f"[{datetime.now()}] Eve 시작")
        print(f"초기 기분: {self.hormones.get_mood()}")
    
    def tick(self):
        """
        메인 루프 한 번 실행
        목표: 이 함수가 100-200ms 내에 완료
        """
        
        # 1. 주기별 이벤트 체크
        if self.clock.is_time_for("thought"):
            self._think()
        
        if self.clock.is_time_for("mood"):
            self._mood_shift()
        
        if self.clock.is_night() and self.clock.is_time_for("sleep"):
            self._consider_sleep()
        
        # 2. 호르몬 자연 감쇠
        self.hormones.decay()
    
    def _think(self):
        """100ms 주기 사고"""
        # 지금은 빈 구현, 나중에 SNN 연결
        pass
    
    def _mood_shift(self):
        """2시간마다 기분 변화 경향"""
        # 약간의 랜덤 요소
        if random.random() < 0.3:
            shift = random.choice([
                "dopamine", "serotonin", "oxytocin"
            ])
            delta = random.uniform(-0.1, 0.1)
            self.hormones.levels[shift] = max(0, min(1, 
                self.hormones.levels[shift] + delta))
    
    def _consider_sleep(self):
        """밤 시간이면 수면 모드 고려"""
        if self.hormones.levels["melatonin"] > 0.7:
            print(f"[{datetime.now()}] 이브가 잠들려고 함...")
            self.is_awake = False
    
    def receive_message(self, message: str):
        """사용자 메시지 받기"""
        self.hormones.update("user_conversation")
        self.self_state.update_interaction()
        self.self_state.current_focus = "conversation"
        
        print(f"\n사용자: {message}")
        print(f"이브 기분: {self.hormones.get_mood()}")
        print(f"호르몬 상태: {self.hormones.levels}")
    
    def save_state(self, path: str = "eve_state.json"):
        """상태 저장"""
        state = {
            "timestamp": datetime.now().isoformat(),
            "hormones": self.hormones.levels,
            "self_state": {
                "name": self.self_state.name,
                "interaction_count": self.self_state.interaction_count,
            },
            "is_awake": self.is_awake,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        print(f"상태 저장 완료: {path}")


# ============================================================
# 5. 테스트 실행
# ============================================================

if __name__ == "__main__":
    eve = Eve()
    
    # 간단한 테스트
    print("\n--- 테스트 시작 ---")
    
    # 메시지 받기
    eve.receive_message("안녕 이브")
    
    # 100번의 틱 실행 (약 1~10초)
    for i in range(100):
        start = time.time()
        eve.tick()
        elapsed = time.time() - start
        
        if i % 20 == 0:
            print(f"Tick {i}: {elapsed*1000:.2f}ms")
    
    # 보상 이벤트
    eve.hormones.update("achievement")
    print(f"\n성취 후 기분: {eve.hormones.get_mood()}")
    
    # 상태 저장
    eve.save_state()
    
    print("\n--- Day 1 프로토타입 완료 ---")
