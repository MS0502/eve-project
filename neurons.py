"""
Eve Project - Izhikevich Neuron Model
Day 2: SNN 구현

Izhikevich 뉴런의 역할:
- 이브의 실제 "뉴런" 역할
- 다양한 발화 패턴 시뮬레이션
- 호르몬에 따라 동작 변화

수식:
v' = 0.04v² + 5v + 140 - u + I
u' = a(bv - u)
if v ≥ 30mV: v = c, u = u + d
"""

import numpy as np
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Optional


# ============================================================
# 1. 뉴런 종류 정의
# ============================================================

class NeuronType(Enum):
    """뇌 부위별 뉴런 타입"""
    REGULAR_SPIKING = "RS"        # 대뇌피질 (사고)
    FAST_SPIKING = "FS"           # 억제 뉴런
    CHATTERING = "CH"             # 감정 영역
    INTRINSICALLY_BURSTING = "IB" # 해마 (기억)
    LOW_THRESHOLD = "LTS"         # 저역치 발화


# 각 타입별 파라미터 (Izhikevich 2003 논문 기반)
NEURON_PARAMS = {
    NeuronType.REGULAR_SPIKING:        {"a": 0.02, "b": 0.2,  "c": -65, "d": 8},
    NeuronType.FAST_SPIKING:           {"a": 0.1,  "b": 0.2,  "c": -65, "d": 2},
    NeuronType.CHATTERING:             {"a": 0.02, "b": 0.2,  "c": -50, "d": 2},
    NeuronType.INTRINSICALLY_BURSTING: {"a": 0.02, "b": 0.2,  "c": -55, "d": 4},
    NeuronType.LOW_THRESHOLD:          {"a": 0.02, "b": 0.25, "c": -65, "d": 2},
}


# ============================================================
# 2. 단일 Izhikevich 뉴런
# ============================================================

class IzhikevichNeuron:
    """
    하나의 Izhikevich 뉴런
    
    인간 뉴런을 수학적으로 흉내낸 모델
    - v: 막전위
    - u: 회복 변수
    - 30mV 넘으면 스파이크 발생
    """
    
    def __init__(self, neuron_type: NeuronType = NeuronType.REGULAR_SPIKING):
        self.neuron_type = neuron_type
        params = NEURON_PARAMS[neuron_type]
        
        # 파라미터
        self.a = params["a"]
        self.b = params["b"]
        self.c = params["c"]
        self.d = params["d"]
        
        # 상태 변수 초기화
        self.v = -65.0  # 휴지 전위
        self.u = self.b * self.v
        
        # 호르몬 영향 (기본 1.0, 호르몬으로 조절됨)
        self.hormone_modulation = 1.0
        
        # 스파이크 기록
        self.spike_history = []
    
    def update(self, current: float, dt: float = 0.5) -> bool:
        """
        뉴런 상태 한 스텝 업데이트
        
        Args:
            current: 입력 전류 (I)
            dt: 시간 스텝 (ms), 0.5ms 추천
        
        Returns:
            True if spike occurred
        """
        # 호르몬 영향 반영
        modulated_current = current * self.hormone_modulation
        
        # Izhikevich 수식 (Euler 방법)
        dv = (0.04 * self.v**2 + 5 * self.v + 140 - self.u + modulated_current) * dt
        du = self.a * (self.b * self.v - self.u) * dt
        
        self.v += dv
        self.u += du
        
        # 스파이크 체크
        if self.v >= 30.0:
            self.v = self.c
            self.u += self.d
            return True
        
        return False
    
    def apply_hormone_effect(self, dopamine: float, serotonin: float, cortisol: float):
        """
        호르몬이 뉴런 반응성에 미치는 영향
        
        도파민 높음 → 반응 증가
        세로토닌 높음 → 안정화 (변화 감소)
        코르티솔 높음 → 각성 증가
        """
        # 기본 1.0에서 호르몬에 따라 조정
        dopamine_effect = 1.0 + (dopamine - 0.5) * 0.4    # ±0.2
        cortisol_effect = 1.0 + (cortisol - 0.2) * 0.3    # ±0.3
        serotonin_stability = 1.0 - (serotonin - 0.5) * 0.2  # 안정화
        
        self.hormone_modulation = dopamine_effect * cortisol_effect * serotonin_stability
        self.hormone_modulation = max(0.3, min(2.0, self.hormone_modulation))  # 범위 제한
    
    def reset(self):
        """뉴런 상태 초기화"""
        self.v = -65.0
        self.u = self.b * self.v
        self.spike_history = []


# ============================================================
# 3. 뉴런 그룹 (함수 기반 시냅스)
# ============================================================

class NeuronGroup:
    """
    여러 뉴런의 그룹
    함수 기반 시냅스로 연결
    """
    
    def __init__(
        self,
        size: int,
        neuron_type: NeuronType = NeuronType.REGULAR_SPIKING,
        name: str = "group"
    ):
        self.size = size
        self.name = name
        self.neurons = [IzhikevichNeuron(neuron_type) for _ in range(size)]
        
        # 스파이크 기록 (시간별)
        self.spike_record = []
    
    def step(self, input_currents: np.ndarray, dt: float = 0.5) -> np.ndarray:
        """
        한 시간 스텝 처리
        
        Args:
            input_currents: 각 뉴런의 입력 전류 (size,)
            dt: 시간 스텝
        
        Returns:
            spike_mask: 발화한 뉴런 마스크 (size,)
        """
        spikes = np.zeros(self.size, dtype=bool)
        
        for i, neuron in enumerate(self.neurons):
            spiked = neuron.update(input_currents[i], dt)
            spikes[i] = spiked
        
        self.spike_record.append(spikes.copy())
        return spikes
    
    def get_membrane_potentials(self) -> np.ndarray:
        """현재 모든 뉴런의 막전위"""
        return np.array([n.v for n in self.neurons])
    
    def apply_hormones(self, dopamine: float, serotonin: float, cortisol: float):
        """모든 뉴런에 호르몬 영향 적용"""
        for neuron in self.neurons:
            neuron.apply_hormone_effect(dopamine, serotonin, cortisol)
    
    def spike_rate(self, window: int = 100) -> float:
        """최근 window 스텝 동안 평균 발화율"""
        if len(self.spike_record) < window:
            recent = self.spike_record
        else:
            recent = self.spike_record[-window:]
        
        if not recent:
            return 0.0
        
        total_spikes = sum(np.sum(s) for s in recent)
        return total_spikes / (len(recent) * self.size)


# ============================================================
# 4. 함수 기반 시냅스 (핵심 혁신)
# ============================================================

class FunctionalSynapse:
    """
    시냅스를 저장하지 않고 함수로 생성
    
    민석이 원래 설계 반영:
    - 뉴런 i → 뉴런 j 연결을 즉시 계산
    - 저장 공간 거의 0
    - 연결 수 이론상 무한
    """
    
    def __init__(self, source_size: int, target_size: int):
        self.source_size = source_size
        self.target_size = target_size
        
        # 시드로 일관된 "연결 패턴" 생성
        # 같은 (i, j) 쌍은 항상 같은 가중치 반환
        self.seed = 42
    
    def get_weight(self, source_id: int, target_id: int, context: Optional[Dict] = None) -> float:
        """
        i → j 연결의 가중치를 계산으로 만들어냄
        
        저장하지 않고 매번 생성
        """
        # 거리 기반 연결 강도
        distance = abs(source_id - target_id)
        
        # 해시 기반 결정적 랜덤성 (같은 쌍은 항상 같은 값)
        hash_val = (source_id * 2654435761 + target_id) % 1000000
        normalized_random = hash_val / 1000000  # 0~1
        
        # 거리가 가까울수록 강하게 연결
        distance_factor = np.exp(-distance / (self.source_size * 0.1))
        
        # 최종 가중치 (-1 ~ 1 범위)
        weight = (normalized_random * 2 - 1) * distance_factor
        
        return weight
    
    def propagate_spikes(self, spikes: np.ndarray) -> np.ndarray:
        """
        소스에서 발생한 스파이크를 타겟 뉴런의 전류로 변환
        
        Args:
            spikes: 소스 뉴런 발화 마스크 (source_size,)
        
        Returns:
            currents: 타겟 뉴런의 입력 전류 (target_size,)
        """
        currents = np.zeros(self.target_size)
        
        spike_indices = np.where(spikes)[0]
        
        for target_id in range(self.target_size):
            for source_id in spike_indices:
                weight = self.get_weight(int(source_id), target_id)
                currents[target_id] += weight * 15.0  # 스파이크 강도
        
        return currents


# ============================================================
# 5. 테스트 실행
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Eve - Day 2: Izhikevich SNN 테스트")
    print("=" * 60)
    
    # 테스트 1: 단일 뉴런
    print("\n[테스트 1] 단일 Regular Spiking 뉴런")
    neuron = IzhikevichNeuron(NeuronType.REGULAR_SPIKING)
    
    spike_count = 0
    for step in range(200):  # 100ms (0.5ms * 200)
        spiked = neuron.update(current=10.0)
        if spiked:
            spike_count += 1
    
    print(f"  100ms 동안 스파이크: {spike_count}회")
    print(f"  발화율: {spike_count * 10} Hz")
    
    # 테스트 2: 뉴런 그룹
    print("\n[테스트 2] 뉴런 100개 그룹")
    group = NeuronGroup(100, NeuronType.REGULAR_SPIKING, "test_group")
    
    # 랜덤 입력 전류
    np.random.seed(42)
    total_spikes = 0
    
    for step in range(200):
        currents = np.random.uniform(5, 15, 100)
        spikes = group.step(currents)
        total_spikes += np.sum(spikes)
    
    print(f"  100ms 동안 총 스파이크: {total_spikes}회")
    print(f"  평균 발화율: {total_spikes / 100 * 10:.1f} Hz/뉴런")
    
    # 테스트 3: 호르몬 영향
    print("\n[테스트 3] 호르몬이 발화에 미치는 영향")
    
    # 일반 상태
    group.reset() if hasattr(group, 'reset') else None
    group = NeuronGroup(100, NeuronType.REGULAR_SPIKING)
    group.apply_hormones(dopamine=0.5, serotonin=0.5, cortisol=0.2)
    
    normal_spikes = 0
    for step in range(200):
        currents = np.random.uniform(5, 15, 100)
        spikes = group.step(currents)
        normal_spikes += np.sum(spikes)
    
    # 흥분 상태 (도파민 + 코르티솔 높음)
    group2 = NeuronGroup(100, NeuronType.REGULAR_SPIKING)
    group2.apply_hormones(dopamine=0.9, serotonin=0.3, cortisol=0.8)
    
    excited_spikes = 0
    for step in range(200):
        currents = np.random.uniform(5, 15, 100)
        spikes = group2.step(currents)
        excited_spikes += np.sum(spikes)
    
    print(f"  평상시: {normal_spikes}회 스파이크")
    print(f"  흥분시: {excited_spikes}회 스파이크")
    print(f"  증가율: {(excited_spikes/normal_spikes - 1) * 100:.1f}%")
    
    # 테스트 4: 함수 기반 시냅스
    print("\n[테스트 4] 함수 기반 시냅스")
    synapse = FunctionalSynapse(100, 100)
    
    # 같은 쌍은 항상 같은 가중치 확인
    w1 = synapse.get_weight(0, 50)
    w2 = synapse.get_weight(0, 50)
    print(f"  (0→50) 첫 번째 호출: {w1:.4f}")
    print(f"  (0→50) 두 번째 호출: {w2:.4f}")
    print(f"  일관성: {'OK' if w1 == w2 else 'FAIL'}")
    
    # 가중치 분포
    weights = [synapse.get_weight(0, j) for j in range(100)]
    print(f"  (0→*) 가중치 범위: [{min(weights):.3f}, {max(weights):.3f}]")
    
    print("\n" + "=" * 60)
    print("Day 2 테스트 완료")
    print("=" * 60)
