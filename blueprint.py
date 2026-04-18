"""
Eve Blueprint Loader
====================
Blueprint JSON을 로드하고 쿼리하는 클래스.

Eve의 DNA를 메모리로 읽어서 네트워크 생성에 사용.
Adapter (Brian2, NEST 등)가 이 클래스를 사용해 실제 네트워크 구축.

Usage:
    bp = Blueprint('eve_blueprint_v0.7.json')
    
    # Population 조회
    pop = bp.get_population('L5_IB')
    print(pop['count'])  # 2250
    
    # 연결 확률
    p = bp.get_connection_probability('L2_RS', 'L5_IB')
    
    # 시냅스 강도
    g = bp.get_synapse_weight('L4_RS_stellate', 'L2_RS', receptor='AMPA')
    
    # 지연
    d = bp.get_delay('L2_RS', 'L5_RS')
    
    # 호르몬 조절
    lr = bp.modulate_learning_rate(base=1.0, hormones={'DA': 0.7, '5HT': 0.4, ...})

Author: MS0502 + Claude
Date: 2026-04-18 (Day 9)
"""

import json
from pathlib import Path
from typing import Dict, Optional, Union, List
import math


class Blueprint:
    """Eve의 뇌 DNA를 로드하고 쿼리하는 클래스."""
    
    # 레이어 관계 (거리 계산용)
    LAYER_ORDER = ['L1', 'L2', 'L3', 'L4', 'L5', 'L6']
    
    # E 타입들
    E_TYPES = {'RS', 'IB', 'CH'}
    I_TYPES = {'FS', 'LTS', 'VIP', 'NGF'}
    
    def __init__(self, json_path: Union[str, Path]):
        """Blueprint JSON 로드.
        
        Args:
            json_path: JSON 파일 경로
        """
        json_path = Path(json_path)
        if not json_path.exists():
            raise FileNotFoundError(f"Blueprint not found: {json_path}")
        
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        # 주요 섹션 참조
        self.metadata = self.data['metadata']
        self.scale = self.data['scale']
        self.layers = self.data['layers']
        self.neuron_types = self.data['neuron_types']
        self.populations = self.data['populations']
        self.synapse_types = self.data['synapse_types']
        self.connection_rules = self.data['connection_rules']
        self.synaptic_strengths = self.data['synaptic_strengths']
        self.delays_config = self.data['delays']
        self.hormones = self.data['hormones']
        self.simulation = self.data['simulation']
        self.validation = self.data['validation_criteria']
        
        # 총 뉴런 수 검증
        self._verify_neuron_count()
    
    def _verify_neuron_count(self):
        """populations 합계가 scale과 일치하는지 확인."""
        total = sum(p['count'] for p in self.populations.values())
        expected = self.scale['total_neurons']
        if total != expected:
            raise ValueError(
                f"Neuron count mismatch: populations sum = {total}, "
                f"expected = {expected}"
            )
    
    # ========================================================================
    # Population 쿼리
    # ========================================================================
    
    def get_population(self, pop_name: str) -> Dict:
        """Population 정보 조회."""
        if pop_name not in self.populations:
            raise KeyError(f"Unknown population: {pop_name}")
        return self.populations[pop_name]
    
    def get_population_names(self) -> List[str]:
        """모든 population 이름 반환."""
        return list(self.populations.keys())
    
    def get_populations_in_layer(self, layer: str) -> List[str]:
        """특정 레이어의 모든 population 반환."""
        return [name for name, p in self.populations.items() if p['layer'] == layer]
    
    def get_populations_by_type(self, neuron_type: str) -> List[str]:
        """특정 뉴런 타입의 모든 population 반환."""
        return [name for name, p in self.populations.items() if p['type'] == neuron_type]
    
    def is_excitatory(self, pop_name: str) -> bool:
        """Population이 흥분성인지 여부."""
        pop = self.get_population(pop_name)
        return self.neuron_types[pop['type']]['class'] == 'excitatory'
    
    def is_inhibitory(self, pop_name: str) -> bool:
        """Population이 억제성인지 여부."""
        return not self.is_excitatory(pop_name)
    
    # ========================================================================
    # Neuron 파라미터 쿼리
    # ========================================================================
    
    def get_izhikevich_params(self, neuron_type: str) -> Dict:
        """Izhikevich 2007 파라미터 반환 (C, k, vr, vt, vpeak, a, b, c, d)."""
        if neuron_type not in self.neuron_types:
            raise KeyError(f"Unknown neuron type: {neuron_type}")
        return self.neuron_types[neuron_type]['izhikevich_2007']
    
    def get_neuron_params(self, pop_name: str) -> Dict:
        """Population의 전체 뉴런 파라미터 반환."""
        pop = self.get_population(pop_name)
        return self.neuron_types[pop['type']]
    
    # ========================================================================
    # 내부 헬퍼
    # ========================================================================
    
    def _pop_to_layer_class(self, pop_name: str) -> str:
        """Population → 'L2e', 'L5i' 같은 레이어 클래스로 변환."""
        pop = self.get_population(pop_name)
        layer = pop['layer']
        
        if layer == 'L1':
            return 'L1i'  # L1은 항상 억제성
        
        suffix = 'e' if self.is_excitatory(pop_name) else 'i'
        return f"{layer}{suffix}"
    
    def _layer_distance(self, layer1: str, layer2: str) -> float:
        """두 레이어 간 거리 (mm)."""
        matrix = self.delays_config['layer_distance_matrix_mm']
        return matrix[layer1][layer2]
    
    def _layer_distance_category(self, layer1: str, layer2: str) -> str:
        """거리 카테고리 반환 ('same', 'adjacent', '2_layers', ...)."""
        if layer1 == layer2:
            return 'same'
        
        idx1 = self.LAYER_ORDER.index(layer1)
        idx2 = self.LAYER_ORDER.index(layer2)
        diff = abs(idx1 - idx2)
        
        if diff == 1:
            return 'adjacent'
        elif diff == 2:
            return '2_layers'
        elif diff == 3:
            return '3_layers'
        else:
            return '4plus_layers'
    
    # ========================================================================
    # 연결 확률 계산
    # ========================================================================
    
    def get_connection_probability(self, src_pop: str, tgt_pop: str) -> float:
        """두 population 간 연결 확률 계산.
        
        계산 순서:
        1. 기본 EI 매트릭스에서 베이스 확률
        2. I 타입별 승수 (FS, LTS, VIP, NGF)
        3. E 타입별 E→E 승수 (RS, IB, CH)
        4. 층 간 거리 승수
        5. 특수 규칙 (IT→ET 단방향 등)
        """
        src = self.get_population(src_pop)
        tgt = self.get_population(tgt_pop)
        src_type = src['type']
        tgt_type = tgt['type']
        src_layer = src['layer']
        tgt_layer = tgt['layer']
        
        src_class = self._pop_to_layer_class(src_pop)
        tgt_class = self._pop_to_layer_class(tgt_pop)
        
        # 1. 기본 EI 매트릭스
        # 주의: base_probability_matrix_EI[target][source] 구조
        base_matrix = self.connection_rules['base_probability_matrix_EI']
        if tgt_class not in base_matrix or src_class not in base_matrix[tgt_class]:
            return 0.0
        p = base_matrix[tgt_class][src_class]
        
        if p == 0:
            return 0.0
        
        # 2. I→E 또는 I→I: I 타입별 승수
        if self.is_inhibitory(src_pop):
            i_prefs = self.connection_rules['I_to_E_type_preferences']
            if self.is_excitatory(tgt_pop):
                # I→E: 승수 적용
                mult = i_prefs['multipliers_on_base_IE'].get(src_type, 1.0)
                p *= mult
            else:
                # I→I: 같은 층 매트릭스 사용 (다른 층은 승수로)
                i_to_i = self.connection_rules['I_to_I_matrix_same_layer']
                if src_layer == tgt_layer:
                    # 같은 층: 매트릭스 직접 사용
                    if src_type in i_to_i and tgt_type in i_to_i[src_type]:
                        p = i_to_i[src_type][tgt_type]
                else:
                    # 다른 층: 같은 층 값 × 거리 승수
                    if src_type in i_to_i and tgt_type in i_to_i[src_type]:
                        p = i_to_i[src_type][tgt_type]
        
        # 3. E→I: E 타입별 승수
        if self.is_excitatory(src_pop) and self.is_inhibitory(tgt_pop):
            e_prefs = self.connection_rules['E_to_I_type_preferences']
            if src_type in e_prefs:
                mult = e_prefs[src_type]['multipliers'].get(tgt_type, 1.0)
                p *= mult
        
        # 4. E→E: 타입별 승수 (단방향 포함)
        if self.is_excitatory(src_pop) and self.is_excitatory(tgt_pop):
            e_e = self.connection_rules['E_to_E_type_multipliers']
            if src_type in e_e and tgt_type in e_e[src_type]:
                mult = e_e[src_type][tgt_type]
                p *= mult  # IB→RS의 경우 × 0 → 0
        
        # 5. 층 간 거리 승수
        if src_layer != tgt_layer and self.is_inhibitory(src_pop):
            cross = self.connection_rules['cross_layer_multipliers_by_type']
            if src_type in cross:
                mults = cross[src_type]
                category = self._layer_distance_category(src_layer, tgt_layer)
                
                # 타입별 특수 규칙
                if src_type == 'LTS' and tgt_layer == 'L1':
                    p *= mults.get('L1_target', 1.0)
                elif src_type == 'NGF':
                    if src_layer == 'L1' and tgt_layer in ('L2', 'L3'):
                        p *= mults.get('L1_to_L2_3', 1.0)
                    elif src_layer == 'L1':
                        p *= mults.get('L1_to_L4_6', 0.2)
                    else:
                        p *= mults.get('other_local', 0.3)
                elif src_type == 'VIP':
                    groups = mults.get('groups', {})
                    src_group = 'upper' if src_layer in groups.get('upper', []) else 'deeper'
                    tgt_group = 'upper' if tgt_layer in groups.get('upper', []) else 'deeper'
                    if src_group == tgt_group:
                        p *= mults.get('same_group', 0.4)
                    else:
                        p *= mults.get('cross_group', 0.1)
                else:
                    # FS 및 일반 케이스
                    if category == 'adjacent':
                        p *= mults.get('adjacent', 0.5)
                    elif category in ('2_layers', '3_layers', '4plus_layers'):
                        p *= mults.get('far', 0.15)
        
        return max(0.0, min(1.0, p))
    
    # ========================================================================
    # 시냅스 강도 (g_max) 계산
    # ========================================================================
    
    def get_synapse_weight(self, src_pop: str, tgt_pop: str, 
                           receptor: str = 'AMPA') -> float:
        """두 population 간 시냅스 g_max 계산 (nS).
        
        Args:
            src_pop: 시작 population
            tgt_pop: 도착 population
            receptor: 'AMPA', 'NMDA', 'GABA_A', 'GABA_slow'
        """
        src = self.get_population(src_pop)
        tgt = self.get_population(tgt_pop)
        src_type = src['type']
        tgt_type = tgt['type']
        src_layer = src['layer']
        tgt_layer = tgt['layer']
        
        strengths = self.synaptic_strengths
        base = strengths['bases_nS'].get(receptor)
        if base is None:
            raise ValueError(f"Unknown receptor: {receptor}")
        
        g = base
        
        # E→I, E→E
        if self.is_excitatory(src_pop):
            if receptor not in ('AMPA', 'NMDA'):
                return 0.0  # E는 억제성 방출 안 함
            
            if self.is_inhibitory(tgt_pop):
                # E→I AMPA 승수
                e_to_i = strengths['E_to_I_AMPA_multipliers']
                g *= e_to_i.get(tgt_type, 1.0)
            else:
                # E→E 타입별 승수
                e_to_e = strengths['E_to_E_g_multipliers']
                key = f"{src_type}_{tgt_type}"
                g *= e_to_e.get(key, 1.0)
            
            # NMDA/AMPA 비율
            if receptor == 'NMDA':
                ratios = strengths['NMDA_AMPA_ratio_by_target']
                g *= ratios.get(tgt_type, 1.0)
        
        # I→E, I→I
        elif self.is_inhibitory(src_pop):
            # NGF만 GABA_slow, 나머지는 GABA_A
            if src_type == 'NGF':
                if receptor != 'GABA_slow':
                    return 0.0
            else:
                if receptor != 'GABA_A':
                    return 0.0
            
            if self.is_excitatory(tgt_pop):
                # I→E 승수
                i_to_e = strengths['I_to_E_GABA_multipliers']
                g *= i_to_e.get(src_type, 1.0)
            else:
                # I→I 승수
                i_to_i = strengths['I_to_I_g_multipliers']
                key = f"{src_type}_{tgt_type}"
                g *= i_to_i.get(key, 1.0)
        
        # 층 간 거리 감쇠
        category = self._layer_distance_category(src_layer, tgt_layer)
        attenuation = strengths['layer_distance_attenuation'].get(category, 1.0)
        g *= attenuation
        
        # 특수 경로
        special = strengths['special_paths']
        if src_pop == 'L4_RS' and tgt_layer in ('L2', 'L3') and self.is_excitatory(tgt_pop):
            g *= special.get('L4_pyramidal_to_L2_3_E', 1.0)
        elif src_pop == 'L4_RS_stellate' and tgt_layer in ('L2', 'L3') and self.is_excitatory(tgt_pop):
            g *= special.get('L4_stellate_to_L2_3_E', 1.0)
        elif src_pop == 'L1_NGF' and tgt_pop == 'L5_IB':
            g *= special.get('L1_NGF_to_L5_IB_apical', 1.0)
        elif src_pop == 'L1_LTS' and tgt_pop == 'L5_IB':
            g *= special.get('L1_LTS_to_L5_IB_apical', 1.0)
        
        return max(0.0, g)
    
    # ========================================================================
    # 지연 계산
    # ========================================================================
    
    def get_delay(self, src_pop: str, tgt_pop: str) -> float:
        """두 population 간 시냅스 지연 계산 (ms).
        
        delay = base × pre_mult + distance / speed
        """
        src = self.get_population(src_pop)
        tgt = self.get_population(tgt_pop)
        src_type = src['type']
        src_layer = src['layer']
        tgt_layer = tgt['layer']
        
        delays = self.delays_config
        
        # 기본 지연 선택
        src_e = self.is_excitatory(src_pop)
        tgt_e = self.is_excitatory(tgt_pop)
        
        if src_e and tgt_e:
            base = delays['base_ms']['E_to_E']
        elif src_e and not tgt_e:
            base = delays['base_ms']['E_to_I']
        elif not src_e and tgt_e:
            base = delays['base_ms']['I_to_E']
        else:
            base = delays['base_ms']['I_to_I']
        
        # Pre 타입 승수
        pre_mult = delays['pre_type_multipliers'].get(src_type, 1.0)
        
        # 축삭 지연 (거리 / 속도)
        distance_mm = self._layer_distance(src_layer, tgt_layer)
        distance_m = distance_mm * 0.001
        
        if src_type == 'FS':
            speed = delays['propagation_speed_m_per_s']['myelinated_FS']
        else:
            speed = delays['propagation_speed_m_per_s']['unmyelinated_default']
        
        axon_delay_ms = (distance_m / speed) * 1000  # seconds → ms
        
        total = base * pre_mult + axon_delay_ms
        return total
    
    # ========================================================================
    # 호르몬 조절
    # ========================================================================
    
    def modulate_learning_rate(self, base: float, hormones: Dict[str, float], 
                                social_context: bool = False) -> float:
        """마스터 학습률 공식 적용.
        
        Args:
            base: 기본 학습률
            hormones: {'DA': 0.7, '5HT': 0.4, 'ACh': 0.6, ...}
            social_context: 사회적 상황 여부
        """
        lr = base
        
        # Tier A
        lr *= (1 + hormones.get('DA', 0.3) * 1.0)
        lr *= max(0.1, 1 - hormones.get('5HT', 0.5))
        lr *= (1 + hormones.get('ACh', 0.3) * 0.7)
        lr *= (1 - hormones.get('Adenosine', 0.3) * 0.5)
        
        # NE Yerkes-Dodson (역 U자)
        ne = hormones.get('NE', 0.3)
        lr *= (1 + ne * 2 * (1 - ne))
        
        # Cortisol curve (급성 vs 만성)
        cort = hormones.get('Cortisol', 0.4)
        if cort < 0.4:
            lr *= (1 + cort * 0.8)
        elif cort > 0.7:
            lr *= max(0.0, 1 - (cort - 0.7) * 2)
        
        # Tier B
        lr *= (1 + hormones.get('BDNF', 0.3) * 0.3)
        lr *= (1 - hormones.get('Melatonin', 0.1) * 0.8)
        
        if social_context:
            lr *= (1 + hormones.get('Oxytocin', 0.2) * 2.0)
        
        # Thyroid 대사
        thyroid = hormones.get('Thyroid', 0.5)
        lr *= thyroid / 0.5
        
        # Insulin 기본
        if hormones.get('Insulin', 0.5) < 0.3:
            lr *= 0.5
        
        # Clip
        return max(0.01, min(2.0, lr))
    
    def modulate_vip_threshold(self, base: float, hormones: Dict[str, float]) -> float:
        """VIP 임계값 조절 (탈억제 회로 게이트)."""
        threshold = base
        threshold *= (1 - hormones.get('DA', 0.3) * 0.4)
        threshold *= (1 - hormones.get('NE', 0.3) * 0.5)
        threshold *= (1 - hormones.get('ACh', 0.3) * 0.5)
        return max(0.01, threshold)
    
    def modulate_ampa_g_max(self, base: float, hormones: Dict[str, float]) -> float:
        """AMPA g_max 조절."""
        g = base
        g *= (1 + hormones.get('ACh', 0.3) * 0.5)
        g *= (1 - hormones.get('Adenosine', 0.3) * 0.3)
        g *= (1 + hormones.get('BDNF', 0.3) * 0.3)
        g *= (1 + hormones.get('Estrogen', 0.3) * 0.15)
        return max(0.0, g)
    
    def modulate_gaba_a_g_max(self, base: float, hormones: Dict[str, float]) -> float:
        """GABA_A g_max 조절."""
        g = base
        g *= (1 + hormones.get('Progesterone', 0.2) * 0.4)
        return max(0.0, g)
    
    def modulate_stdp(self, A_plus_base: float, A_minus_base: float,
                      hormones: Dict[str, float]) -> tuple:
        """STDP 파라미터 조절."""
        A_plus = A_plus_base
        A_plus *= (1 + hormones.get('ACh', 0.3) * 1.0)
        A_plus *= (1 + hormones.get('DA', 0.3) * 0.5)
        A_plus *= max(0.1, 1 - hormones.get('5HT', 0.5) * 0.3)
        
        A_minus = A_minus_base
        A_minus *= (1 + hormones.get('ACh', 0.3) * 0.5)
        A_minus *= max(0.1, 1 - hormones.get('DA', 0.3) * 0.3)
        
        return A_plus, A_minus
    
    # ========================================================================
    # 요약 정보
    # ========================================================================
    
    def summary(self) -> str:
        """Blueprint 요약 출력."""
        lines = [
            f"Eve Blueprint v{self.metadata['version']}",
            f"=" * 50,
            f"Total neurons: {self.scale['total_neurons']:,}",
            f"Cortical area: {self.scale['cortical_area_mm2']} mm²",
            f"Depth: {self.scale['cortical_depth_mm']} mm",
            f"",
            f"Layers: {len(self.layers)}",
            f"Populations: {len(self.populations)}",
            f"Neuron types: {len(self.neuron_types)}",
            f"Synapse types: {len(self.synapse_types)}",
            f"Hormones: {len(self.hormones['definitions'])}",
            f"Active in Phase 1: {len(self.hormones['phase1_active'])}",
            f"",
            f"Validation tests: {len(self.validation['phase2_mandatory_tests'])}",
        ]
        return "\n".join(lines)


# ============================================================================
# Quick test
# ============================================================================

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        json_path = sys.argv[1]
    else:
        # 기본 경로
        json_path = Path(__file__).parent.parent / 'eve_blueprint_v0.7.json'
    
    print(f"Loading: {json_path}")
    bp = Blueprint(json_path)
    
    print()
    print(bp.summary())
    
    print()
    print("Tests:")
    print("-" * 50)
    
    # 연결 확률 테스트
    tests = [
        ('L2_RS', 'L2_RS', 'recurrent E'),
        ('L4_RS_stellate', 'L2_RS', 'L4 → L2/3'),
        ('L5_RS', 'L5_IB', 'IT → ET (should exist)'),
        ('L5_IB', 'L5_RS', 'ET → IT (should be 0)'),
        ('L2_VIP', 'L2_LTS', 'VIP → LTS (disinhibition)'),
        ('L2_FS', 'L2_RS', 'FS → E (peri-somatic)'),
        ('L2_LTS', 'L2_RS', 'LTS → E (dendritic)'),
    ]
    
    for src, tgt, desc in tests:
        p = bp.get_connection_probability(src, tgt)
        print(f"  p({src} → {tgt}) = {p:.4f}  [{desc}]")
    
    print()
    print("Synaptic weights:")
    print("-" * 50)
    
    weight_tests = [
        ('L2_RS', 'L2_RS', 'AMPA'),
        ('L2_RS', 'L2_RS', 'NMDA'),
        ('L2_FS', 'L2_RS', 'GABA_A'),
        ('L2_LTS', 'L2_RS', 'GABA_A'),
        ('L2_VIP', 'L2_LTS', 'GABA_A'),
        ('L1_NGF', 'L2_RS', 'GABA_slow'),
    ]
    
    for src, tgt, rec in weight_tests:
        g = bp.get_synapse_weight(src, tgt, rec)
        print(f"  g_{rec}({src} → {tgt}) = {g:.3f} nS")
    
    print()
    print("Delays:")
    print("-" * 50)
    
    delay_tests = [
        ('L2_RS', 'L2_RS', 'local E→E'),
        ('L2_FS', 'L2_RS', 'local FS→E (myelinated)'),
        ('L4_RS_stellate', 'L2_RS', 'L4→L2/3'),
        ('L5_RS', 'L1_NGF', 'L5→L1 (apical)'),
    ]
    
    for src, tgt, desc in delay_tests:
        d = bp.get_delay(src, tgt)
        print(f"  delay({src} → {tgt}) = {d:.2f} ms  [{desc}]")
    
    print()
    print("Hormone modulation:")
    print("-" * 50)
    
    # 쉬고 있음
    resting = {'DA': 0.3, '5HT': 0.5, 'NE': 0.3, 'ACh': 0.3, 'Adenosine': 0.3,
               'Cortisol': 0.4, 'BDNF': 0.3, 'Melatonin': 0.1, 'Thyroid': 0.5}
    
    # 학습 중
    learning = {'DA': 0.7, '5HT': 0.4, 'NE': 0.5, 'ACh': 0.7, 'Adenosine': 0.2,
                'Cortisol': 0.4, 'BDNF': 0.4, 'Melatonin': 0.1, 'Thyroid': 0.5}
    
    # 수면
    sleep = {'DA': 0.2, '5HT': 0.2, 'NE': 0.1, 'ACh': 0.1, 'Adenosine': 0.9,
             'Cortisol': 0.2, 'BDNF': 0.3, 'Melatonin': 0.7, 'Thyroid': 0.5}
    
    for name, h in [('Resting', resting), ('Learning', learning), ('Sleep', sleep)]:
        lr = bp.modulate_learning_rate(1.0, h)
        vip = bp.modulate_vip_threshold(1.0, h)
        print(f"  [{name}] lr = {lr:.3f}, VIP threshold = {vip:.3f}")
    
    print()
    print("✅ Blueprint loader working correctly.")
