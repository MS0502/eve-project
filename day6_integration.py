"""
Eve Day 6: Brian2 + HormoneSystem 통합 실험
3가지 감정 상태에서 학습 비교
"""

import sys
sys.path.insert(0, '/content/eve_design')
from brian2 import *
from hormone_system import HormoneSystem
import numpy as np


def run_experiment(event_name):
    """한 이벤트 상황에서 Eve 실험"""
    # 호르몬 시스템 초기화
    eve_hs = HormoneSystem(phase=1, developmental_stage="newborn")
    eve_hs.trigger_event(event_name)
    lr = eve_hs.compute_learning_rate()
    
    # Brian2 시뮬 설정
    start_scope()
    defaultclock.dt = 0.1*ms
    a, b, c, d = 0.02, 0.2, -65, 8
    
    eqs = '''
    dv/dt = (0.04*v**2 + 5*v + 140 - u + I)/ms : 1
    du/dt = a*(b*v - u)/ms : 1
    I : 1
    '''
    
    neurons = NeuronGroup(100, eqs, threshold='v >= 30', 
                         reset='v = c; u += d', method='euler')
    neurons.v = '-70 + rand() * 10'
    neurons.u = b * -65
    neurons.I = 'rand() * 3 + 2'
    
    stdp_eqs = '''
    w : 1
    dapre/dt = -apre/(20*ms) : 1 (event-driven)
    dapost/dt = -apost/(20*ms) : 1 (event-driven)
    lr : 1 (shared)
    '''
    
    on_pre = '''
    v_post += w
    apre += 0.5 * lr
    w = clip(w + apost, 0, 20)
    '''
    
    on_post = '''
    apost -= 0.5 * lr
    w = clip(w + apre, 0, 20)
    '''
    
    syn = Synapses(neurons, neurons, stdp_eqs, 
                  on_pre=on_pre, on_post=on_post)
    syn.connect(p=0.2)
    syn.w = 5.0
    syn.lr = lr
    
    # 초기 가중치 저장
    w_initial = np.array(syn.w[:])
    
    # 모니터링
    spikes = SpikeMonitor(neurons)
    
    # 시뮬 실행
    net = Network(neurons, syn, spikes)
    net.run(1000*ms)
    
    # 결과
    w_final = np.array(syn.w[:])
    changes = w_final - w_initial
    mood = eve_hs.compute_mood()
    
    return {
        'event': event_name,
        'lr': lr,
        'change': np.abs(changes).mean(),
        'spikes': spikes.num_spikes,
        'valence': mood['valence'],
        'arousal': mood['arousal'],
        'hormones': {name: eve_hs.hormones[name].level 
                    for name in eve_hs.active_hormones}
    }


# 3가지 감정 상태에서 실험
print("=" * 60)
print("Day 6: 감정 상태별 학습 비교")
print("=" * 60)

events_to_test = ["reward", "stress", "threat", "social_contact", "learning_success"]

all_results = []
for event in events_to_test:
    print(f"\n--- {event} ---")
    result = run_experiment(event)
    all_results.append(result)
    print(f"학습률: {result['lr']:.2f}")
    print(f"변화량: {result['change']:.2f}")
    print(f"스파이크: {result['spikes']}")
    print(f"기분: valence={result['valence']:+.2f}, arousal={result['arousal']:.2f}")


# 요약
print()
print("=" * 60)
print("최종 비교")
print("=" * 60)
print(f"{'이벤트':<20} {'학습률':>8} {'변화량':>8} {'valence':>10}")
print("-" * 60)
for r in all_results:
    print(f"{r['event']:<20} {r['lr']:>8.2f} {r['change']:>8.2f} {r['valence']:>+10.2f}")

print()
print("=" * 60)
print("해석")
print("=" * 60)

# 가장 빠른 학습
fastest = max(all_results, key=lambda x: x['change'])
slowest = min(all_results, key=lambda x: x['change'])
print(f"가장 빠른 학습: {fastest['event']} (변화량 {fastest['change']:.2f})")
print(f"가장 느린 학습: {slowest['event']} (변화량 {slowest['change']:.2f})")
print(f"비율: {fastest['change']/slowest['change']:.1f}배")
