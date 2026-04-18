"""
Eve Day 8 - Layer 1 v3: Conductance-based 시냅스
AMPA/GABA 시간 상수 추가
실제 뇌 시냅스처럼 자극이 시간에 걸쳐 지속
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


# 흥분성 뉴런 - Conductance-based
a_exc, b_exc, c_exc, d_exc = 0.02, 0.2, -65, 8

eqs_exc = '''
dv/dt = (0.04*v**2 + 5*v + 140 - u + I + g_ampa - g_gaba)/ms : 1
du/dt = a_exc*(b_exc*v - u)/ms : 1
dg_ampa/dt = -g_ampa/(5*ms) : 1
dg_gaba/dt = -g_gaba/(10*ms) : 1
dfiring_rate/dt = -firing_rate/(500*ms) : Hz
I : 1
'''

exc_neurons = NeuronGroup(400, eqs_exc, 
                          threshold='v >= 30', 
                          reset='v = c_exc; u += d_exc; firing_rate += 1*Hz', 
                          method='euler')
exc_neurons.v = '-70 + rand() * 10'
exc_neurons.u = b_exc * -65
exc_neurons.I = 0
exc_neurons.g_ampa = 0
exc_neurons.g_gaba = 0
exc_neurons.firing_rate = 5*Hz


# 억제성 뉴런 - Conductance-based
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


# Poisson 배경 (AMPA 수용체에 전달)
bg_input_exc = PoissonGroup(400, rates=10*Hz)
bg_syn_exc = Synapses(bg_input_exc, exc_neurons, on_pre='g_ampa_post += 3.0')
bg_syn_exc.connect(j='i')

bg_input_inh = PoissonGroup(100, rates=8*Hz)
bg_syn_inh = Synapses(bg_input_inh, inh_neurons, on_pre='g_ampa_post += 3.0')
bg_syn_inh.connect(j='i')


# E→E STDP (AMPA)
stdp_eqs = '''
w : 1
dapre/dt = -apre/(20*ms) : 1 (event-driven)
dapost/dt = -apost/(20*ms) : 1 (event-driven)
lr_val : 1 (shared)
'''

on_pre_exc = '''
g_ampa_post += w
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


# E→I (AMPA)
syn_ei = Synapses(exc_neurons, inh_neurons, 'w : 1', on_pre='g_ampa_post += w')
syn_ei.connect(p=0.08)
syn_ei.w = 3.0


# I→E (GABA)
syn_ie = Synapses(inh_neurons, exc_neurons, 'w : 1', on_pre='g_gaba_post += w')
syn_ie.connect(p=0.08)
syn_ie.w = 4.0


# I→I (GABA)
syn_ii = Synapses(inh_neurons, inh_neurons, 'w : 1', on_pre='g_gaba_post += w')
syn_ii.connect(p=0.05)
syn_ii.w = 2.0


# 모니터링
w_initial = np.array(syn_ee.w[:])
spikes_exc = SpikeMonitor(exc_neurons)
spikes_inh = SpikeMonitor(inh_neurons)

print(f"E→E 시냅스: {len(syn_ee)}")

net = Network(exc_neurons, inh_neurons, 
              bg_input_exc, bg_input_inh,
              bg_syn_exc, bg_syn_inh,
              syn_ee, syn_ei, syn_ie, syn_ii, 
              spikes_exc, spikes_inh)
net.run(1000*ms)


# 결과
w_final = np.array(syn_ee.w[:])
ei_ratio = spikes_exc.num_spikes / max(spikes_inh.num_spikes, 1)
final_rates = np.array(exc_neurons.firing_rate[:] / Hz)

print(f"\n=== Conductance-based 결과 ===")
print(f"흥분성 스파이크: {spikes_exc.num_spikes}")
print(f"억제성 스파이크: {spikes_inh.num_spikes}")
print(f"E:I 비율: {ei_ratio:.1f}:1")
print(f"\n뉴런 발화율:")
print(f"  평균: {final_rates.mean():.1f} Hz (목표: 1~10 Hz)")
print(f"  min: {final_rates.min():.1f} Hz")
print(f"  max: {final_rates.max():.1f} Hz")
print(f"\n시냅스 (E→E):")
print(f"  변화량: {np.abs(w_final - w_initial).mean():.2f}")
print(f"  평균 w: {w_final.mean():.2f}")
print(f"  min: {w_final.min():.2f}, max: {w_final.max():.2f}")
print(f"  0 근처 (<1): {(w_final < 1).sum()}")
print(f"  최대치 (>19): {(w_final > 19).sum()}")
print(f"  중간 (1~5): {((w_final > 1) & (w_final < 5)).sum()}")
