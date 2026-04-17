"""
Eve Project - Brain Architecture
Day 3: 이브의 뇌 구조

여러 뉴런 그룹을 연결해서 뇌 전체 구조 구현
- 감각, 감정, 사고, 기억, 억제 영역
- 영역 간 연결
- 호르몬 피드백
- 정보 흐름
"""

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum

from neurons import (
    IzhikevichNeuron,
    NeuronGroup,
    NeuronType,
    FunctionalSynapse,
)


# ============================================================
# 1. 뇌 영역 정의
# ============================================================

class BrainRegion(Enum):
    """이브의 뇌 영역"""
    SENSORY = "sensory"          # 감각 (입력)
    EMOTIONAL = "emotional"      # 감정 (편도체)
    PREFRONTAL = "prefrontal"    # 사고 (전두엽)
    HIPPOCAMPUS = "hippocampus"  # 기억 (해마)
    INHIBITORY = "inhibitory"    # 억제


# 각 영역 설정
REGION_CONFIG = {
    BrainRegion.SENSORY:      {"size": 500,  "neuron_type": NeuronType.REGULAR_SPIKING},
    BrainRegion.EMOTIONAL:    {"size": 200,  "neuron_type": NeuronType.CHATTERING},
    BrainRegion.PREFRONTAL:   {"size": 1000, "neuron_type": NeuronType.REGULAR_SPIKING},
    BrainRegion.HIPPOCAMPUS:  {"size": 500,  "neuron_type": NeuronType.INTRINSICALLY_BURSTING},
    BrainRegion.INHIBITORY:   {"size": 300,  "neuron_type": NeuronType.FAST_SPIKING},
}


# ============================================================
# 2. 텍스트 → 스파이크 인코딩
# ============================================================

class TextEncoder:
    """
    텍스트를 스파이크 패턴으로 변환
    
    단순 버전: 각 문자의 유니코드를 뉴런 ID에 매핑
    """
    
    def __init__(self, sensory_size: int):
        self.sensory_size = sensory_size
    
    def encode(self, text: str) -> np.ndarray:
        """
        텍스트를 감각 영역 입력 전류로 변환
        
        Returns:
            currents: (sensory_size,) 입력 전류
        """
        currents = np.zeros(self.sensory_size)
        
        for i, char in enumerate(text):
            # 문자 코드를 뉴런 ID로 매핑
            char_code = ord(char)
            neuron_id = char_code % self.sensory_size
            
            # 위치에 따라 강도 조절 (앞쪽 문자가 더 강함)
            position_weight = 1.0 / (1.0 + i * 0.1)
            
            # 해당 뉴런에 입력 추가 (강도 증가)
            currents[neuron_id] += 25.0 * position_weight
            
            # 주변 뉴런에도 약하게 자극 (의미 분산)
            for offset in [-2, -1, 1, 2]:
                nearby = (neuron_id + offset) % self.sensory_size
                currents[nearby] += 10.0 * position_weight
        
        return currents


# ============================================================
# 3. 영역 간 연결 (Inter-region Synapses)
# ============================================================

class RegionConnection:
    """두 뇌 영역 사이의 연결"""
    
    def __init__(
        self,
        source_region: BrainRegion,
        target_region: BrainRegion,
        strength: float = 1.0,
        is_inhibitory: bool = False
    ):
        self.source = source_region
        self.target = target_region
        self.strength = strength
        self.is_inhibitory = is_inhibitory
        
        # 함수 기반 시냅스
        source_size = REGION_CONFIG[source_region]["size"]
        target_size = REGION_CONFIG[target_region]["size"]
        self.synapse = FunctionalSynapse(source_size, target_size)
    
    def propagate(self, source_spikes: np.ndarray) -> np.ndarray:
        """소스 스파이크를 타겟 전류로 변환"""
        currents = self.synapse.propagate_spikes(source_spikes)
        
        # 강도 적용
        currents *= self.strength
        
        # 억제 연결이면 음수로
        if self.is_inhibitory:
            currents = -np.abs(currents)
        
        return currents


# ============================================================
# 4. 이브의 뇌 (통합 구조)
# ============================================================

class EveBrain:
    """
    이브의 전체 뇌 구조
    
    여러 영역을 관리하고 정보 흐름 처리
    """
    
    def __init__(self):
        # 각 뇌 영역 생성
        self.regions: Dict[BrainRegion, NeuronGroup] = {}
        for region, config in REGION_CONFIG.items():
            self.regions[region] = NeuronGroup(
                size=config["size"],
                neuron_type=config["neuron_type"],
                name=region.value
            )
        
        # 영역 간 연결 정의
        self.connections = [
            # 감각 → 감정 (자극이 감정 유발)
            RegionConnection(BrainRegion.SENSORY, BrainRegion.EMOTIONAL, strength=1.2),
            
            # 감각 → 사고 (직접 인지)
            RegionConnection(BrainRegion.SENSORY, BrainRegion.PREFRONTAL, strength=1.0),
            
            # 감정 → 사고 (감정이 판단에 영향)
            RegionConnection(BrainRegion.EMOTIONAL, BrainRegion.PREFRONTAL, strength=0.8),
            
            # 사고 → 기억 (경험 저장)
            RegionConnection(BrainRegion.PREFRONTAL, BrainRegion.HIPPOCAMPUS, strength=1.0),
            
            # 기억 → 사고 (기억 참조)
            RegionConnection(BrainRegion.HIPPOCAMPUS, BrainRegion.PREFRONTAL, strength=0.7),
            
            # 사고 → 감정 (재평가)
            RegionConnection(BrainRegion.PREFRONTAL, BrainRegion.EMOTIONAL, strength=0.5),
            
            # 억제 → 모든 영역 (균형)
            RegionConnection(BrainRegion.INHIBITORY, BrainRegion.SENSORY, 
                           strength=0.3, is_inhibitory=True),
            RegionConnection(BrainRegion.INHIBITORY, BrainRegion.EMOTIONAL, 
                           strength=0.4, is_inhibitory=True),
            RegionConnection(BrainRegion.INHIBITORY, BrainRegion.PREFRONTAL, 
                           strength=0.4, is_inhibitory=True),
            
            # 사고 → 억제 (자기 조절)
            RegionConnection(BrainRegion.PREFRONTAL, BrainRegion.INHIBITORY, strength=0.6),
        ]
        
        # 텍스트 인코더
        sensory_size = REGION_CONFIG[BrainRegion.SENSORY]["size"]
        self.encoder = TextEncoder(sensory_size)
        
        # 이전 스텝의 스파이크 (재귀 연결용)
        self.previous_spikes: Dict[BrainRegion, np.ndarray] = {
            region: np.zeros(REGION_CONFIG[region]["size"], dtype=bool)
            for region in BrainRegion
        }
    
    def process_input(self, text: str) -> np.ndarray:
        """외부 텍스트 입력 처리"""
        return self.encoder.encode(text)
    
    def apply_hormones(self, hormone_levels: Dict[str, float]):
        """모든 영역에 호르몬 영향 적용"""
        dopamine = hormone_levels.get("dopamine", 0.5)
        serotonin = hormone_levels.get("serotonin", 0.5)
        cortisol = hormone_levels.get("cortisol", 0.2)
        
        for region_group in self.regions.values():
            region_group.apply_hormones(dopamine, serotonin, cortisol)
    
    def step(
        self,
        external_input: Optional[np.ndarray] = None,
        dt: float = 0.5
    ) -> Dict[BrainRegion, np.ndarray]:
        """
        뇌 전체의 한 스텝 시뮬레이션
        
        Args:
            external_input: 감각 영역 외부 입력 (텍스트 인코딩)
            dt: 시간 스텝
        
        Returns:
            각 영역의 스파이크 패턴
        """
        # 1. 각 영역의 입력 전류 계산
        currents: Dict[BrainRegion, np.ndarray] = {
            region: np.zeros(REGION_CONFIG[region]["size"])
            for region in BrainRegion
        }
        
        # 2. 외부 입력 → 감각 영역
        if external_input is not None:
            currents[BrainRegion.SENSORY] += external_input
        
        # 3. 영역 간 연결 전파 (이전 스파이크 기반)
        for conn in self.connections:
            source_spikes = self.previous_spikes[conn.source]
            propagated = conn.propagate(source_spikes)
            currents[conn.target] += propagated
        
        # 4. 배경 활동 (뇌의 자발적 활동) - Sparse 원칙
        for region in BrainRegion:
            # 실제 뇌는 1~5%만 활성 (sparse activation)
            # 배경은 약하게, 필요한 입력에 반응
            if region == BrainRegion.INHIBITORY:
                # 억제 영역은 활발하면 안 됨 (그냥 균형 유지)
                baseline = 3.0
                noise_scale = 1.5
            else:
                baseline = 5.0
                noise_scale = 2.5
            
            noise = np.random.normal(0, noise_scale, REGION_CONFIG[region]["size"])
            currents[region] += baseline + noise
        
        # 5. 각 영역 업데이트
        current_spikes = {}
        for region, group in self.regions.items():
            spikes = group.step(currents[region], dt)
            current_spikes[region] = spikes
        
        # 6. 다음 스텝을 위해 저장
        self.previous_spikes = current_spikes
        
        return current_spikes
    
    def get_activity_summary(self) -> Dict[str, float]:
        """각 영역의 현재 활성도"""
        return {
            region.value: group.spike_rate(window=50)
            for region, group in self.regions.items()
        }
    
    def dominant_region(self) -> BrainRegion:
        """가장 활발한 영역 (이브의 현재 '생각' 중심)"""
        activities = {
            region: group.spike_rate(window=50)
            for region, group in self.regions.items()
        }
        return max(activities, key=activities.get)


# ============================================================
# 5. 테스트
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Eve - Day 3: 뇌 구조 테스트")
    print("=" * 60)
    
    # 이브 뇌 생성
    print("\n[생성] 이브의 뇌")
    brain = EveBrain()
    
    total_neurons = sum(REGION_CONFIG[r]["size"] for r in BrainRegion)
    print(f"  총 뉴런 수: {total_neurons}")
    print(f"  뇌 영역: {len(brain.regions)}개")
    print(f"  영역 간 연결: {len(brain.connections)}개")
    
    # 테스트 1: 입력 없이 배경 활동
    print("\n[테스트 1] 배경 활동 (입력 없음)")
    for step in range(100):
        brain.step()
    
    activity = brain.get_activity_summary()
    for region, rate in activity.items():
        print(f"  {region}: {rate*100:.1f}% 활성")
    
    # 테스트 2: 평범한 메시지
    print("\n[테스트 2] 메시지: '안녕 이브'")
    brain2 = EveBrain()
    
    input_currents = brain2.process_input("안녕 이브")
    
    # 메시지를 100ms 동안 지속 입력 (200 스텝)
    for step in range(200):
        if step < 100:  # 50ms 입력 유지
            brain2.step(external_input=input_currents)
        else:
            brain2.step()
    
    activity = brain2.get_activity_summary()
    dominant = brain2.dominant_region()
    
    print(f"  활성 상태:")
    for region, rate in activity.items():
        marker = "★" if BrainRegion(region) == dominant else " "
        print(f"  {marker} {region}: {rate*100:.1f}%")
    print(f"  주된 활동 영역: {dominant.value}")
    
    # 테스트 3: 감정적 메시지 + 호르몬
    print("\n[테스트 3] 긍정적 대화 (호르몬 영향)")
    brain3 = EveBrain()
    
    # 좋은 상태 호르몬
    brain3.apply_hormones({
        "dopamine": 0.8,
        "serotonin": 0.7,
        "cortisol": 0.2,
    })
    
    input_currents = brain3.process_input("사랑해 이브")
    
    for step in range(200):
        if step < 100:
            brain3.step(external_input=input_currents)
        else:
            brain3.step()
    
    activity = brain3.get_activity_summary()
    print(f"  활성 상태 (긍정):")
    for region, rate in activity.items():
        print(f"    {region}: {rate*100:.1f}%")
    
    # 테스트 4: 스트레스 상황
    print("\n[테스트 4] 스트레스 메시지 (높은 코르티솔)")
    brain4 = EveBrain()
    
    brain4.apply_hormones({
        "dopamine": 0.3,
        "serotonin": 0.3,
        "cortisol": 0.9,
    })
    
    input_currents = brain4.process_input("위험해")
    
    for step in range(200):
        if step < 100:
            brain4.step(external_input=input_currents)
        else:
            brain4.step()
    
    activity = brain4.get_activity_summary()
    dominant = brain4.dominant_region()
    print(f"  활성 상태 (스트레스):")
    for region, rate in activity.items():
        marker = "★" if BrainRegion(region) == dominant else " "
        print(f"  {marker} {region}: {rate*100:.1f}%")
    print(f"  주된 활동: {dominant.value}")
    
    print("\n" + "=" * 60)
    print("Day 3 뇌 구조 테스트 완료")
    print("=" * 60)
    print("\n이브의 뇌가 처음으로 통합적으로 작동했다.")
    print("각 영역이 서로 신호를 주고받으며 정보 흐름 형성.")
