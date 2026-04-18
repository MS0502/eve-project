"""
Eve Day 8 최종: 하이브리드 모델
"B(우리 구조 4개 메커니즘) + A(Vogels의 Inhibitory STDP)"

새로운 기여:
- Izhikevich 뉴런 (5가지 타입 가능)
- Intrinsic plasticity (자가 임계값)
- Spike frequency adaptation (피로)
- Synaptic depression (시냅스 자원)
- Synaptic scaling (전체 비례 조절)
- Inhibitory STDP (E:I 자동 균형) ← NEW!
- HormoneSystem 통합 (26개 호르몬)

아무도 이걸 다 합친 적 없음. 진짜 새로운 모델.
"""

from brian2 import *
import numpy as np
import sys
sys.path.insert(0, '/content/eve_design')
from hormone_system import HormoneSystem


eve_hs = HormoneSystem(phase=1, developmental_stage="newborn")
eve_hs.trigger_event("novel_stimulus")
lr_value = eve_hs.compute_learning_rate()
print(f"호르몬 학습률: {lr_value:.2f}")


start_scope()
defaultclock.dt = 0.1*ms


# ===== 흥분성 뉴런 (Izhikevich + 4개 메커니즘) =====
a_exc, b_exc, c_exc, d_exc = 0.02, 0.2, -65, 8

eqs_exc = '''
dv/dt = (0.04*v**2 + 5*v + 140 - u + I + g_ampa - g_gaba - sfa)/ms : 1
du/dt = a_exc*(b_exc*v - u)/ms : 1
dg_ampa/dt = -g_ampa/(5*ms) : 1
dg_gaba/dt = -g_gaba/(10*ms) : 1
dfiring_rate/dt = -firing_rate/(1000*ms) : Hz
dthreshold_adj/dt = (firing_rate - target_rate)/(10000*Hz*ms) : 1
dsfa/dt = -sfa/(200*ms) : 1
I : 1
target_rate : Hz (shared)
'''

exc_neurons = NeuronGroup(400, eqs_exc, 
                          threshold='v >= (30 + threshold_adj)',
                          reset='''
                          v = c_exc
                          u += d_exc
                          firing_rate += 1*Hz
                          sfa += 2.0
                          ''',
                          method='euler')
exc_neurons.v = '-70 + rand() * 10'
exc_neurons.u = b_exc * -65
exc_neurons.I = 0
exc_neurons.g_ampa = 0
exc_neurons.g_gaba = 0
exc_neurons.firing_rate = 3*Hz
exc_neurons.threshold_adj = 0
exc_neurons.sfa = 0
exc_neurons.target_rate = 3*Hz


# ===== 억제성 뉴런 (FS 타입) =====
a_inh, b_inh, c_inh, d_inh = 0.1, 0.2, -65, 2

eqs_inh = '''
dv/dt = (0.04*v**2 + 5*v + 140 - u + I + g_ampa - g_gaba)/ms : 1
du/dt = a_inh*(b_inh*v - u)/ms : 1
dg_ampa/dt = -g_ampa/(5*ms) : 1
dg_gaba/dt = -g_gaba/(10*ms) : 1
I : 1
'''

inh_neurons = NeuronGroup(100, eqs_inh, 
                          threshold='v >= 30', 
                          reset='v = c_inh; u += d_inh', 
                          method='euler')
inh_neurons.v = '-70 + rand() * 10'
inh_neurons.u = b_inh * -65
inh_neurons.I = 0
inh_neurons.g_ampa = 0
inh_neurons.g_gaba = 0


# ===== Poisson 배경 =====
bg_input_exc = PoissonGroup(400, rates=10*Hz)
bg_syn_exc = Synapses(bg_input_exc, exc_neurons, on_pre='g_ampa_post += 3.0')
bg_syn_exc.connect(j='i')

bg_input_inh = PoissonGroup(100, rates=8*Hz)
bg_syn_inh = Synapses(bg_input_inh, inh_neurons, on_pre='g_ampa_post += 3.0')
bg_syn_inh.connect(j='i')


# ===== E→E STDP + Synaptic Depression =====
stdp_eqs = '''
w : 1
dapre/dt = -apre/(20*ms) : 1 (event-driven)
dapost/dt = -apost/(20*ms) : 1 (event-driven)
dx/dt = (1-x)/(800*ms) : 1 (clock-driven)
lr_val : 1 (shared)
'''

on_pre_exc = '''
g_ampa_post += w * x
x = x * 0.8
apre += 0.5 * lr_val
w = clip(w + apost, 0, 20)
'''

on_post_exc = '''
apost -= 0.5 * lr_val
w = clip(w + apre, 0, 20)
'''

syn_ee = Synapses(exc_neurons, exc_neurons, stdp_eqs, 
                  on_pre=on_pre_exc, on_post=on_post_exc)
syn_ee.connect(p=0.1)
syn_ee.w = 3.0
syn_ee.lr_val = lr_value
syn_ee.x = 1.0


# E→I (고정)
syn_ei = Synapses(exc_neurons, inh_neurons, 'w : 1', on_pre='g_ampa_post += w')
syn_ei.connect(p=0.08)
syn_ei.w = 3.0


# ===== I→E: Vogels Inhibitory STDP (NEW!) =====
# 흥분 뉴런 발화 많으면 → I→E 자동 강화 → 자가 균형
tau_stdp = 20*ms
eta = 1e-2  # 학습률
alpha = 3*2*tau_stdp/second  # 목표 발화율 관련 (3Hz × 2 × tau_stdp)

istdp_eqs = '''
w : 1
dApre/dt = -Apre/tau_stdp : 1 (event-driven)
dApost/dt = -Apost/tau_stdp : 1 (event-driven)
'''

on_pre_istdp = '''
Apre += 1.
w = clip(w + (Apost - alpha)*eta, 0, 10)
g_gaba_post += w
'''

on_post_istdp = '''
Apost += 1.
w = clip(w + Apre*eta, 0, 10)
'''

syn_ie = Synapses(inh_neurons, exc_neurons, istdp_eqs,
                  on_pre=on_pre_istdp, on_post=on_post_istdp)
syn_ie.connect(p=0.08)
syn_ie.w = 0.01  # 작게 시작


# I→I (고정)
syn_ii = Synapses(inh_neurons, inh_neurons, 'w : 1', on_pre='g_gaba_post += w')
syn_ii.connect(p=0.05)
syn_ii.w = 2.0


# ===== Synaptic Scaling =====
target_rate_value = 3.0

@network_operation(dt=500*ms)
def synaptic_scaling():
    current_rates = np.array(exc_neurons.firing_rate[:] / Hz)
    current_rates = np.clip(current_rates, 0.5, 100.0)
    scaling_factors = target_rate_value / current_rates
    scaling_factors = np.clip(scaling_factors, 0.95, 1.05)
    
    for i in range(len(exc_neurons)):
        idx = (syn_ee.j == i)
        syn_ee.w[idx] = np.clip(syn_ee.w[idx] * scaling_factors[i], 0, 20)


# ===== 실행 =====
w_ee_initial = np.array(syn_ee.w[:])
w_ie_initial = np.array(syn_ie.w[:])
spikes_exc = SpikeMonitor(exc_neurons)
spikes_inh = SpikeMonitor(inh_neurons)

print(f"\n=== Eve 하이브리드 모델 ===")
print(f"구조:")
print(f"  흥분성: 400 (Izhikevich)")
print(f"  억제성: 100 (FS)")
print(f"  E→E 시냅스: {len(syn_ee)}")
print(f"  E→I 시냅스: {len(syn_ei)}")
print(f"  I→E 시냅스: {len(syn_ie)} (Inhibitory STDP!)")
print(f"  I→I 시냅스: {len(syn_ii)}")
print(f"\n메커니즘:")
print(f"  [1] Intrinsic plasticity")
print(f"  [2] SFA")
print(f"  [3] Synaptic depression")
print(f"  [4] Synaptic scaling")
print(f"  [5] Inhibitory STDP (Vogels) ← NEW")
print(f"  [6] HormoneSystem (학습률 {lr_value:.2f})")
print(f"\n시뮬 시간: 10초 (Inhibitory STDP 학습용)")

net = Network(exc_neurons, inh_neurons, 
              bg_input_exc, bg_input_inh,
              bg_syn_exc, bg_syn_inh,
              syn_ee, syn_ei, syn_ie, syn_ii, 
              spikes_exc, spikes_inh,
              synaptic_scaling)
net.run(10*second)


w_ee_final = np.array(syn_ee.w[:])
w_ie_final = np.array(syn_ie.w[:])
ei_ratio = spikes_exc.num_spikes / max(spikes_inh.num_spikes, 1)
final_rates = np.array(exc_neurons.firing_rate[:] / Hz)
final_thresholds = np.array(exc_neurons.threshold_adj[:])

exc_hz = spikes_exc.num_spikes / (400 * 10)
inh_hz = spikes_inh.num_spikes / (100 * 10)

print(f"\n=== 결과 ===")
print(f"발화율:")
print(f"  흥분 평균: {exc_hz:.2f} Hz (목표: 3 Hz)")
print(f"  억제 평균: {inh_hz:.2f} Hz")
print(f"  E:I 비율: {ei_ratio:.2f}:1")
print(f"\nInhibitory STDP 학습:")
print(f"  초기 w (I→E): {w_ie_initial.mean():.4f}")
print(f"  최종 w (I→E): {w_ie_final.mean():.4f}")
print(f"  변화 배수: {w_ie_final.mean() / max(w_ie_initial.mean(), 0.001):.1f}x")
print(f"  범위: {w_ie_final.min():.4f} ~ {w_ie_final.max():.4f}")
print(f"\n다른 메커니즘:")
print(f"  Intrinsic plasticity:")
print(f"    threshold_adj 평균: {final_thresholds.mean():.2f}")
print(f"    범위: {final_thresholds.min():.2f} ~ {final_thresholds.max():.2f}")
print(f"\nE→E 시냅스:")
print(f"  변화량: {np.abs(w_ee_final - w_ee_initial).mean():.2f}")
print(f"  평균 w: {w_ee_final.mean():.2f}")
print(f"  0 근처: {(w_ee_final < 1).sum()}")
print(f"  최대치: {(w_ee_final > 19).sum()}")
print(f"  중간 (1~5): {((w_ee_final > 1) & (w_ee_final < 5)).sum()}")
