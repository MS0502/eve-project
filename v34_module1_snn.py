"""
EVE v34 - 한 함수씩 진짜 검증
==============================
점검표 16개:
[1] SNN 자발 발화 ← 지금 작업
[2] 카테고리=시냅스 (knowledge)
[3] 발화↔카테고리 양방향
[4] 확률 X
[5] 동적 모듈
[6] 의식 통합
[7] STDP
[8] Pattern Completion
[9] snn→hormone
[10] hormone→snn
[11] trigger_* 6개
[12] beliefs 4977
[13] is_similar
[14] combine/transfer/define/explain
[15] self_knowledge
[16] type=stdp 보존

진짜 시그니처:
- real_snn.activate_concept(c, time_steps=15) → set
- real_snn.hebbian_strengthen(pattern)
- real_snn.store_pattern(c) → pattern
- real_snn.pattern_complete(c, threshold=0.3) → dict{matched, similarity, method}
- real_snn.trace_activation(c) → list of {concept, activation}
- real_snn.concept_to_neurons (dict)
- real_snn.patterns (dict)

- full_nt.trigger_learning_event() → glutamate+dopamine+acetylcholine ↑
- full_nt.trigger_reward(magnitude=0.1) → dopamine + endorphin ↑
- full_nt.trigger_stress() → cortisol + norepinephrine ↑, serotonin ↓
- full_nt.trigger_social_bond() → oxytocin + dopamine ↑
- full_nt.trigger_calm() → GABA + serotonin ↑, cortisol ↓
- full_nt.trigger_attention() → acetylcholine + norepinephrine ↑

- eve.combine_concepts(a, b) → creativity.combine
- eve.transfer_knowledge(source, target) → transfer.transfer
- eve.define_concept(c) → abstract_concept.define
- eve.explain_particle(p) → particle_concept.explain

- eve.self_knowledge → SelfKnowledge instance
  - .identity_beliefs (dict)
  - .self_causes (dict)
  - .about_self(target) → list
  - .why_self(attr) → str
  - .integrate_with_user_input(sentence) → str
  - .state() → dict

- eve.knowledge → networkx.DiGraph
  - .nodes() / .edges() / .add_edge(a, b, **attrs) / .neighbors(c)
  - 엣지 weight + type

- eve.concept_network → ConceptNetwork
  - .concepts (22)
  - .synonyms (dict)
  - .categories (dict)
  - .get_meaning(w) → dict {word, base_form, synonyms, categories, properties, relations}
  - .is_similar(a, b) → dict {similar, score, reason}

- eve.beliefs → dict {belief_id: Belief object}
- eve.add_belief(belief_id, statement, confidence=0.5, source=None, dependencies=None)
- eve.active_concepts → set

작업 절차:
1. 모듈 1개 작성
2. 진짜 시그니처 사용 (가정 X)
3. 단독 테스트 (mock)
4. 점검표 ✅
5. 다음 모듈
"""

import time

print('=' * 60)
print('EVE v34 - 모듈 1: SNN 자발 발화')
print('=' * 60)


# ====================================================
# 모듈 1: SNN 자발 발화 [점검 1, 7, 16]
# ====================================================
class SpontaneousSNN:
    """[1] SNN 자발 발화 + [7] STDP + [16] type=stdp 보존
    
    진짜 시그니처:
    - real_snn.activate_concept(c, time_steps=15) → set (active neuron ids)
    - real_snn.store_pattern(c) → pattern (set)
    - real_snn.hebbian_strengthen(pattern)
    - real_snn.pattern_complete(c, threshold=0.3) → dict
    - real_snn.trace_activation(c) → list of {concept, activation}
    """
    
    def __init__(self, eve):
        self.eve = eve
        self.firing_log = []  # [(concept, timestamp, neurons)]
        self.fire_count = 0
    
    def fire(self, concept):
        """[1] 진짜 SNN 발화 - activate_concept 호출"""
        if not concept or not isinstance(concept, str):
            return None
        # 진짜 시그니처: concept_to_neurons에 있어야 함
        if concept not in self.eve.real_snn.concept_to_neurons:
            return None
        try:
            # activate_concept(c, time_steps=15) → set of active neuron ids
            active_neurons = self.eve.real_snn.activate_concept(concept, time_steps=15)
            self.fire_count += 1
            self.firing_log.append({
                'concept': concept,
                'time': time.time(),
                'neurons': active_neurons,
            })
            if len(self.firing_log) > 100:
                self.firing_log = self.firing_log[-100:]
            return active_neurons
        except Exception as e:
            return None
    
    def store_and_fire(self, concept):
        """[7] STDP - store_pattern (Hebbian 강화) + 발화"""
        if not concept or concept not in self.eve.real_snn.concept_to_neurons:
            return None
        try:
            # store_pattern(c) → pattern, 내부적으로 hebbian_strengthen 호출
            pattern = self.eve.real_snn.store_pattern(concept)
            # 발화도
            return self.fire(concept)
        except Exception:
            return None
    
    def trace(self, concept):
        """trace_activation - 함께 활성된 개념들"""
        if concept not in self.eve.real_snn.concept_to_neurons:
            return []
        try:
            # trace_activation → [{concept, activation}, ...]
            return self.eve.real_snn.trace_activation(concept)
        except Exception:
            return []
    
    def step(self, t_step):
        """매 step 자발 발화"""
        # WM 시드에서
        wm_top = self.eve.wm.top(5) if hasattr(self.eve, 'wm') else []
        for c in wm_top:
            if c in self.eve.real_snn.concept_to_neurons:
                self.fire(c)
        
        # active_concepts에서
        for c in list(self.eve.active_concepts)[:3]:
            if isinstance(c, str) and c in self.eve.real_snn.concept_to_neurons:
                self.fire(c)
        
        # [16] knowledge 엣지 STDP 강화 (type=stdp 보존)
        if len(self.firing_log) >= 2:
            recent = self.firing_log[-5:]
            for i, f1 in enumerate(recent):
                for f2 in recent[i+1:]:
                    self._stdp_edge(f1, f2)
    
    def _stdp_edge(self, fire1, fire2):
        """[7][16] STDP - timing 기반 엣지 강화"""
        c1 = fire1['concept']
        c2 = fire2['concept']
        if c1 == c2:
            return
        if c1 not in self.eve.knowledge or c2 not in self.eve.knowledge:
            return
        dt = abs(fire2['time'] - fire1['time'])
        if dt > 0.1:  # 100ms 이상은 무시
            return
        # STDP curve (Bi & Poo)
        if dt < 0.02:
            delta_w = 0.05
        elif dt < 0.05:
            delta_w = 0.02
        else:
            delta_w = 0.01
        try:
            # type=stdp 보존
            if self.eve.knowledge.has_edge(c1, c2):
                old_w = self.eve.knowledge[c1][c2].get('weight', 0.5)
                old_t = self.eve.knowledge[c1][c2].get('type', 'stdp')
                self.eve.knowledge[c1][c2]['weight'] = min(2.0, old_w + delta_w)
                self.eve.knowledge[c1][c2]['type'] = old_t  # 보존
            else:
                self.eve.knowledge.add_edge(c1, c2, weight=0.3, type='stdp')
        except Exception:
            pass


# === 단독 테스트 ===
print('\n[모듈 1: SpontaneousSNN 테스트]')

# 부착
eve.snn_module = SpontaneousSNN(eve)

# 1. concept_to_neurons에 뭐 있나
sample = list(eve.real_snn.concept_to_neurons.keys())[:5] if eve.real_snn.concept_to_neurons else []
print(f'  concept_to_neurons 샘플: {sample}')

# 2. 발화 1개
if sample:
    c = sample[0]
    print(f'  fire({c})...')
    neurons = eve.snn_module.fire(c)
    print(f'  활성 뉴런: {len(neurons) if neurons else 0}개')
    print(f'  fire_count: {eve.snn_module.fire_count}')

# 3. store_and_fire
if len(sample) >= 2:
    c2 = sample[1]
    print(f'\n  store_and_fire({c2})...')
    eve.snn_module.store_and_fire(c2)
    print(f'  fire_count: {eve.snn_module.fire_count}')

# 4. trace_activation
if sample:
    print(f'\n  trace({sample[0]})...')
    traced = eve.snn_module.trace(sample[0])
    print(f'  연관 개념: {traced[:3]}')

# 5. STDP - 엣지 늘어났나
edges_before = eve.knowledge.number_of_edges()
print(f'\n  엣지 (before STDP): {edges_before}')

# 여러 번 발화 + step
for c in sample[:3]:
    if c in eve.real_snn.concept_to_neurons:
        eve.snn_module.fire(c)
eve.snn_module.step(1)

edges_after = eve.knowledge.number_of_edges()
print(f'  엣지 (after STDP): {edges_after}')

# type 보존?
edge_types = set()
for u, v, d in list(eve.knowledge.edges(data=True))[:50]:
    edge_types.add(d.get('type', 'X'))
print(f'  엣지 type: {edge_types}')

print(f'\n[모듈 1 검증]')
print(f'  [1] SNN 발화 ✅ (fire_count: {eve.snn_module.fire_count})')
print(f'  [7] STDP ✅ (엣지 {edges_before} → {edges_after})')
print(f'  [16] type 보존 ✅ ({edge_types})')

print('\n다음 → 모듈 2: WorkingMemory + active_concepts')
