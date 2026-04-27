"""
EVE 튜닝 v3 - 안정적 (변동 ↓)
================================
v2 문제: 검증 변동 ±20-30Hz
v3 해결:
1. 더 긴 시뮬 (5초)
2. Power-weighted mean (peak 대신)
3. 검증 시 더 많은 반복
4. 더 큰 뉴런 (3000)

목표: 67% → 안정적 80%+
"""

import numpy as np
import time
import json

# Optuna
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)


def simulate_stable(params, n=3000, duration_ms=5000, n_repeats=3):
    """안정적 시뮬 - 더 길게, 가중 평균."""
    freqs = []
    
    for _ in range(n_repeats):
        mask = np.random.rand(n) < 0.8
        a = np.where(mask, params['a_excit'], 0.10).astype(np.float64)
        b = np.full(n, params['b'], dtype=np.float64)
        c = np.full(n, params['c'], dtype=np.float64)
        d = np.where(mask, params['d'], 2.0).astype(np.float64)
        
        V = -65.0 * np.ones(n, dtype=np.float64)
        u = -13.0 * np.ones(n, dtype=np.float64)
        W = (np.random.randn(n, 20) * params['coupling']).astype(np.float64)
        
        spike_counts = np.zeros(duration_ms, dtype=np.int32)
        
        for t in range(duration_ms):
            V += 0.5 * (0.04 * V**2 + 5*V + 140 - u)
            V += 0.5 * (0.04 * V**2 + 5*V + 140 - u)
            u += a * (b * V - u)
            V += params['noise'] * np.random.randn(n)
            
            fired_idx = np.where(V >= 30.0)[0]
            spike_counts[t] = len(fired_idx)
            
            if len(fired_idx) > 0:
                for idx in fired_idx[:30]:
                    target = (idx + np.arange(20) * 7) % n
                    V[target] += W[idx]
                
                V[fired_idx] = c[fired_idx]
                u[fired_idx] += d[fired_idx]
        
        # FFT 분석
        rates = np.array(spike_counts, dtype=np.float64)
        if rates.sum() == 0:
            freqs.append(0)
            continue
        
        # 처음 500ms는 transient (제외)
        rates = rates[500:]
        
        fft = np.fft.fft(rates)
        ff = np.fft.fftfreq(len(rates), d=0.001)
        amps = np.abs(fft)
        
        # === 핵심: Power-weighted mean ===
        band = (ff > 1) & (ff < 100)
        if not band.any():
            freqs.append(0)
            continue
        
        band_freqs = ff[band]
        band_amps = amps[band]
        
        # 가중 평균 (peak보다 안정)
        if band_amps.sum() > 0:
            weighted_freq = np.sum(band_freqs * band_amps) / band_amps.sum()
        else:
            weighted_freq = 0
        
        freqs.append(weighted_freq)
    
    return np.mean(freqs)


# 감정 목표
EMOTION_TARGETS = {
    '평온': {'target_freq': 10.5, 'range': (8, 13), 'wave': '알파'},
    '집중': {'target_freq': 21.0, 'range': (13, 30), 'wave': '베타'},
    '기쁨': {'target_freq': 45.0, 'range': (30, 60), 'wave': '감마'},
    '두려움': {'target_freq': 35.0, 'range': (20, 50), 'wave': '베타+감마'},
    '슬픔': {'target_freq': 5.5, 'range': (3, 8), 'wave': '세타↓'},
    '유대감': {'target_freq': 12.5, 'range': (10, 15), 'wave': '알파/베타'},
}


PARAM_RANGES_V3 = {
    '평온': {
        'a_excit': (0.015, 0.025),
        'b': (0.18, 0.22),
        'c': (-68, -62),
        'd': (6, 10),
        'noise': (2.0, 5.0),
        'coupling': (0.10, 0.30),
    },
    '집중': {
        'a_excit': (0.020, 0.030),
        'b': (0.18, 0.22),
        'c': (-68, -62),
        'd': (5, 9),
        'noise': (4.0, 7.0),
        'coupling': (0.20, 0.40),
    },
    '기쁨': {
        'a_excit': (0.035, 0.060),
        'b': (0.20, 0.28),
        'c': (-65, -55),
        'd': (3, 7),
        'noise': (6.0, 10.0),
        'coupling': (0.40, 0.80),
    },
    '두려움': {
        'a_excit': (0.025, 0.045),
        'b': (0.18, 0.24),
        'c': (-68, -60),
        'd': (4, 8),
        'noise': (5.0, 9.0),
        'coupling': (0.30, 0.55),
    },
    '슬픔': {
        'a_excit': (0.012, 0.020),
        'b': (0.18, 0.22),
        'c': (-72, -66),
        'd': (8, 12),
        'noise': (2.5, 4.0),
        'coupling': (0.10, 0.20),
    },
    '유대감': {
        'a_excit': (0.020, 0.026),
        'b': (0.19, 0.21),
        'c': (-66, -63),
        'd': (7, 9),
        'noise': (4.0, 5.5),
        'coupling': (0.18, 0.28),
    },
}


def make_objective_stable(emotion):
    target = EMOTION_TARGETS[emotion]['target_freq']
    ranges = PARAM_RANGES_V3[emotion]
    
    def objective(trial):
        params = {
            k: trial.suggest_float(k, lo, hi)
            for k, (lo, hi) in ranges.items()
        }
        
        # 안정적 시뮬 (5초, 3회)
        actual_freq = simulate_stable(
            params, n=3000, duration_ms=5000, n_repeats=3
        )
        
        return abs(actual_freq - target)
    
    return objective


def tune_emotion_v3(emotion, n_trials=50):
    """안정적 튜닝 (변동 ↓)."""
    target = EMOTION_TARGETS[emotion]
    print(f"\n[{emotion}] 안정 튜닝 (n={n_trials})")
    print(f"  목표: {target['wave']} {target['target_freq']}Hz")
    
    study = optuna.create_study(direction='minimize')
    
    start = time.time()
    study.optimize(make_objective_stable(emotion), n_trials=n_trials, 
                   show_progress_bar=False)
    elapsed = time.time() - start
    
    print(f"  완료: {elapsed:.0f}초")
    print(f"  최적 거리: {study.best_value:.2f}Hz")
    
    return study.best_params


def tune_all_v3(n_trials=50):
    """전체 안정 튜닝."""
    print("=" * 60)
    print(f"전체 안정 튜닝 v3 ({n_trials} trial × 6)")
    print("=" * 60)
    print("⚠️ 더 정밀 = 더 느림 (5분 × 6 = 30분)")
    
    total_start = time.time()
    
    results = {}
    for emotion in EMOTION_TARGETS.keys():
        results[emotion] = tune_emotion_v3(emotion, n_trials)
    
    total_elapsed = time.time() - total_start
    
    # 검증
    print("\n" + "=" * 60)
    print("검증 - 안정적 (10회 평균, 5초 시뮬)")
    print("=" * 60)
    print(f"\n{'감정':<8} {'목표':<18} {'EVE 평균':<14} {'일치':<6}")
    print("-" * 55)
    
    matches = 0
    for emotion, params in results.items():
        target = EMOTION_TARGETS[emotion]
        target_min, target_max = target['range']
        
        freqs = []
        for _ in range(10):
            f = simulate_stable(params, n=3000, duration_ms=5000, n_repeats=1)
            freqs.append(f)
        
        avg = np.mean(freqs)
        std = np.std(freqs)
        
        match = "✅" if target_min <= avg <= target_max else "⚠️"
        if target_min <= avg <= target_max:
            matches += 1
        
        target_str = f"{target['wave']} {target_min}-{target_max}"
        print(f"{emotion:<8} {target_str:<18} {avg:<6.1f}±{std:<3.1f}      {match}")
    
    print(f"\n=== 최종 ===")
    print(f"정확도: {matches}/6 = {matches/6*100:.0f}%")
    print(f"총 시간: {total_elapsed:.0f}초 ({total_elapsed/60:.1f}분)")
    
    # 저장
    try:
        with open('/content/drive/MyDrive/eve_emotion_params_v3.json', 'w') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n💾 저장: eve_emotion_params_v3.json")
    except:
        pass
    
    return results


print("""
=========================================
EVE 튜닝 v3 - 안정적 (변동 ↓)
=========================================

핵심 개선:
✅ 더 긴 시뮬 (5초)
✅ Power-weighted mean (peak 대신)
✅ 더 큰 뉴런 (3000)
✅ Transient 제거 (처음 500ms)

사용법:

# 전체 안정 튜닝 (30분)
results = tune_all_v3(50)

# 또는 빠른 (15분)
results = tune_all_v3(30)

# 단일
tune_emotion_v3('슬픔', n_trials=50)

목표: 변동 ↓, 80%+ 일치
""")
