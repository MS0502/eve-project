"""
Eve Day 8 - 선택 B v3: Hz 단위 유지하면서 차원 맞춤
firing_rate는 Hz로 유지 (생물학적 의미)
threshold_adj와의 차원만 맞춤
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


# ===== 흥분성 뉴런 =====
a_exc, b_exc, c_exc, d_exc = 0.02, 0.2, -65, 8

# firing_rate: Hz (생물학적 의미)
# target_rate: Hz
# threshold_adj: 단위 없음 (v에 더해지므로)
# 차원 맞춤: (Hz - Hz) / (Hz*ms) = 1/ms → /dt 하면 1 ✓
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


# ===== 억제성 뉴런 =====
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


# ===== E→E (STDP + Depression) =====
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


syn_ei = Synapses(exc_neurons, inh_neurons, 'w : 1', on_pre='g_ampa_post += w')
syn_ei.connect(p=0.08)
syn_ei.w = 3.0


syn_ie = Synapses(inh_neurons, exc_neurons, 'w : 1', on_pre='g_gaba_post += w')
syn_ie.connect(p=0.08)
syn_ie.w = 4.0


syn_ii = Synapses(inh_neurons, inh_neurons, 'w : 1', on_pre='g_gaba_post += w')
syn_ii.connect(p=0.05)
syn_ii.w = 2.0


# ===== Synaptic Scaling =====
target_rate_value = 3.0  # Hz

@network_operation(dt=500*ms)
def synaptic_scaling():
    # Hz → 숫자 변환
    current_rates = np.array(exc_neurons.firing_rate[:] / Hz)
    current_rates = np.clip(current_rates, 0.5, 100.0)
    scaling_factors = target_rate_value / current_rates
    scaling_factors = np.clip(scaling_factors, 0.95, 1.05)
    
    for i in range(len(exc_neurons)):
        idx = (syn_ee.j == i)
        syn_ee.w[idx] = np.clip(syn_ee.w[idx] * scaling_factors[i], 0, 20)


# ===== 실행 =====
w_initial = np.array(syn_ee.w[:])
spikes_exc = SpikeMonitor(exc_neurons)
spikes_inh = SpikeMonitor(inh_neurons)

print(f"\n구조:")
print(f"  흥분성: 400 (Izhikevich + 4 메커니즘)")
print(f"  억제성: 100")
print(f"  E→E 시냅스: {len(syn_ee)}")
print(f"\n시뮬 시간: 5초")

net = Network(exc_neurons, inh_neurons, 
              bg_input_exc, bg_input_inh,
              bg_syn_exc, bg_syn_inh,
              syn_ee, syn_ei, syn_ie, syn_ii, 
              spikes_exc, spikes_inh,
              synaptic_scaling)
net.run(5*second)


w_final = np.array(syn_ee.w[:])
ei_ratio = spikes_exc.num_spikes / max(spikes_inh.num_spikes, 1)
final_rates = np.array(exc_neurons.firing_rate[:] / Hz)
final_thresholds = np.array(exc_neurons.threshold_adj[:])

exc_hz = spikes_exc.num_spikes / (400 * 5)
inh_hz = spikes_inh.num_spikes / (100 * 5)

print(f"\n=== 선택 B 결과 (4개 메커니즘, Hz 유지) ===")
print(f"흥분성 평균 발화율: {exc_hz:.2f} Hz (목표: 3 Hz)")
print(f"억제성 평균 발화율: {inh_hz:.2f} Hz")
print(f"E:I 비율: {ei_ratio:.2f}:1")
print(f"\n메커니즘 작동 확인:")
print(f"  [1] Intrinsic plasticity:")
print(f"      평균 threshold_adj: {final_thresholds.mean():.2f}")
print(f"      범위: {final_thresholds.min():.2f} ~ {final_thresholds.max():.2f}")
print(f"  [2] SFA: reset에서 sfa+=2.0")
print(f"  [3] Synaptic depression: 매 spike마다 x*=0.8")
print(f"  [4] Synaptic scaling: 500ms마다")
print(f"\n뉴런 내부 발화율 추적:")
print(f"  평균: {final_rates.mean():.2f} Hz")
print(f"  범위: {final_rates.min():.2f} ~ {final_rates.max():.2f}")
print(f"\n시냅스 (E→E):")
print(f"  변화량: {np.abs(w_final - w_initial).mean():.2f}")
print(f"  평균 w: {w_final.mean():.2f}")
print(f"  min: {w_final.min():.2f}, max: {w_final.max():.2f}")
print(f"  0 근처 (<1): {(w_final < 1).sum()}")
print(f"  최대치 (>19): {(w_final > 19).sum()}")
print(f"  중간 (1~5): {((w_final > 1) & (w_final < 5)).sum()}")
