"""
Eve Day 8 - 선택 A: Vogels et al. 2011 기반
검증된 논문 파라미터 사용
- Inhibitory STDP로 E/I 균형 자동 유지
- Conductance-based synapses
- Poisson 배경 자극 (상수 전류로 대체)
- HormoneSystem 통합 유지

Reference:
Vogels TP, Sprekeler H, Zenke F, Clopath C, Gerstner W (2011).
Inhibitory plasticity balances excitation and inhibition in 
sensory pathways and memory networks. Science 334:1569-73.
"""

from brian2 import *
import numpy as np
import sys
sys.path.insert(0, '/content/eve_design')
from hormone_system import HormoneSystem


# ===== Eve 호르몬 =====
eve_hs = HormoneSystem(phase=1, developmental_stage="newborn")
eve_hs.trigger_event("novel_stimulus")
lr_value = eve_hs.compute_learning_rate()
print(f"호르몬 학습률: {lr_value:.2f}")

start_scope()

# ===== 네트워크 파라미터 (Vogels 2011) =====
NE = 400   # 흥분성 (원본 8000, 축소)
NI = 100   # 억제성 (1:4 비율 유지)

# 시간 상수
tau_ampa = 5.0*ms
tau_gaba = 10.0*ms
tau_stdp = 20*ms

# 뉴런 파라미터 (Leaky Integrate-and-Fire)
gl = 10.0*nsiemens
el = -60*mV
er = -80*mV
vt = -50.*mV
memc = 200.0*pfarad
bgcurrent = 200*pA

# 시냅스 연결 밀도
epsilon = 0.02

# ===== 뉴런 방정식 =====
eqs_neurons = '''
dv/dt = (-gl*(v-el) - (g_ampa*v + g_gaba*(v-er)) + bgcurrent)/memc : volt (unless refractory)
dg_ampa/dt = -g_ampa/tau_ampa : siemens
dg_gaba/dt = -g_gaba/tau_gaba : siemens
'''

# ===== 뉴런 생성 =====
neurons = NeuronGroup(NE+NI, model=eqs_neurons, threshold='v > vt',
                     reset='v=el', refractory=5*ms, method='euler')

# 흥분/억제 뉴런 나누기
Pe = neurons[:NE]
Pi = neurons[NE:]

# 초기화
neurons.v = 'el + (vt - el) * rand()'

# ===== 시냅스: E→모두 (AMPA) =====
con_e = Synapses(Pe, neurons, on_pre='g_ampa += 0.3*nS')
con_e.connect(p=epsilon)

# ===== 시냅스: I→모두 (GABA, Inhibitory STDP) =====
eqs_istdp = '''
w : 1
dApre/dt = -Apre/tau_stdp : 1 (event-driven)
dApost/dt = -Apost/tau_stdp : 1 (event-driven)
'''

# Inhibitory STDP
# 흥분성 뉴런 발화 많으면 → I→E 강화 (억제 증가)
# 이게 E/I 균형 자동 유지의 핵심!
eta = 1e-2  # 학습률
alpha = 3*Hz*tau_stdp*2  # 목표 발화율 파라미터

on_pre_istdp = '''
Apre += 1.
w = clip(w + (Apost - alpha)*eta, 0, 10)
g_gaba += w*nS
'''

on_post_istdp = '''
Apost += 1.
w = clip(w + Apre*eta, 0, 10)
'''

con_i = Synapses(Pi, neurons, model=eqs_istdp,
                on_pre=on_pre_istdp, on_post=on_post_istdp)
con_i.connect(p=epsilon)
con_i.w = 1e-10  # 초기 매우 약함 (학습으로 증가)


# ===== 모니터링 =====
spikes_exc = SpikeMonitor(Pe)
spikes_inh = SpikeMonitor(Pi)

# ===== 실행 =====
print(f"\nVogels 2011 기반 실행")
print(f"NE: {NE}, NI: {NI}")
print(f"E→E/I 시냅스: {len(con_e)}")
print(f"I→E/I 시냅스: {len(con_i)}")
print("시뮬 시간: 10초 (Inhibitory STDP 학습 위해)")

simtime = 10*second
net = Network(neurons, con_e, con_i, spikes_exc, spikes_inh)
net.run(simtime)


# ===== 결과 =====
exc_rate = spikes_exc.num_spikes / (NE * simtime/second)
inh_rate = spikes_inh.num_spikes / (NI * simtime/second)
ei_ratio = spikes_exc.num_spikes / max(spikes_inh.num_spikes, 1)

# I→E 시냅스 가중치 분포
w_i = np.array(con_i.w[:])

print(f"\n=== Vogels 2011 결과 ===")
print(f"흥분성 평균 발화율: {exc_rate:.2f} Hz (목표: 3~5 Hz)")
print(f"억제성 평균 발화율: {inh_rate:.2f} Hz")
print(f"E:I 비율: {ei_ratio:.2f}:1")
print(f"\nInhibitory STDP 결과:")
print(f"  억제 시냅스 가중치 평균: {w_i.mean():.4f}")
print(f"  min: {w_i.min():.4f}, max: {w_i.max():.4f}")
print(f"  학습됨 (>0.01): {(w_i > 0.01).sum()} / {len(w_i)}")
print(f"\n호르몬 영향 (학습률 {lr_value:.2f}):")
print(f"  현재 모델은 아직 HormoneSystem과 통합 안 됨")
print(f"  다음 단계: E→E 시냅스에 STDP 추가 + HormoneSystem 연동")
