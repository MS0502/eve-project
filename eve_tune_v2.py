"""
EVE 자동 튜닝 v2 - cache 완전 제거
=========================================
이전 버그 fix:
- Numba cache 완전 제거
- 모든 jit cache=False

사용:
  !pip install optuna numba
  exec(open('/content/drive/MyDrive/eve_tune_v2.py').read())
  tune_emotion('평온', n_trials=30)
"""

import numpy as np
import time
import json

# Numba (cache 완전 X)
try:
    from numba import njit
    @njit
    def _step_izhikevich(V, u, a, b, c, d, W, noise_strength, n, n_synapses):
        """1 tick - Numba 가속, cache X."""
        # Izhikevich 공식
        for i in range(n):
            V[i] += 0.5 * (0.04 * V[i]**2 + 5*V[i] + 140 - u[i])
            V[i] += 0.5 * (0.04 * V[i]**2 + 5*V[i] + 140 - u[i])
            u[i] += a[i] * (b[i] * V[i] - u[i])
            V[i] += noise_strength * np.random.randn()
        
        n_fired = 0
        for i in range(n):
            if V[i] >= 30.0:
                # 시냅스 전송
                for j in range(n_synapses):
                    target = (i + j * 7) % n
                    V[target] += W[i, j]
                
                # Reset
                V[i] = c[i]
                u[i] += d[i]
                n_fired += 1
        
        return n_fired
    
    HAS_NUMBA = True
    print("✅ Numba 사용 (cache X, 5-10x 가속)")
    
except ImportError:
    HAS_NUMBA = False
    print("⚠️ Numba 없음")
    
    def _step_izhikevich(V, u, a, b, c, d, W, noise_strength, n, n_synapses):
        """순수 NumPy 버전."""
        V += 0.5 * (0.04 * V**2 + 5*V + 140 - u)
        V += 0.5 * (0.04 * V**2 + 5*V + 140 - u)
        u += a * (b * V - u)
        V += noise_strength * np.random.randn(n)
        
        fired_idx = np.where(V >= 30.0)[0]
        n_fired = len(fired_idx)
        
        if n_fired > 0:
            for idx in fired_idx[:30]:
                target = np.random.choice(n, n_synapses, replace=False)
                V[target] += W[idx]
            
            V[fired_idx] = c[fired_idx]
            u[fired_idx] += d[fired_idx]
        
        return n_fired


# Optuna
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
    print("✅ Optuna OK")
except ImportError:
    HAS_OPTUNA = False
    print("⚠️ pip install optuna 필요")


def simulate_emotion(params, n=1500, duration_ms=1000, n_repeats=3):
    """감정 시뮬 - 평균 주파수."""
    freqs = []
    
    for _ in range(n_repeats):
        # 파라미터
        mask = np.random.rand(n) < 0.8
        a = np.where(mask, params['a_excit'], 0.10).astype(np.float64)
        b = np.full(n, params['b'], dtype=np.float64)
        c = np.full(n, params['c'], dtype=np.float64)
        d = np.where(mask, params['d'], 2.0).astype(np.float64)
        
        V = -65.0 * np.ones(n, dtype=np.float64)
        u = -13.0 * np.ones(n, dtype=np.float64)
        
        n_synapses = 20
        W = (np.random.randn(n, n_synapses) * params['coupling']).astype(np.float64)
        
        # 시뮬
        spike_counts = np.zeros(duration_ms, dtype=np.int32)
        for t in range(duration_ms):
            spike_counts[t] = _step_izhikevich(
                V, u, a, b, c, d, W, params['noise'], n, n_synapses
            )
        
        # FFT
        rates = np.array(spike_counts, dtype=np.float64)
        if rates.sum() == 0:
            freqs.append(0)
            continue
        
        fft = np.fft.fft(rates)
        ff = np.fft.fftfreq(len(rates), d=0.001)
        amps = np.abs(fft)
        
        band = (ff > 1) & (ff < 100)
        if not band.any():
            freqs.append(0)
            continue
        
        peak_idx = np.argmax(amps[band]) + np.where(band)[0][0]
        freqs.append(ff[peak_idx])
    
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


# 파라미터 범위 (감정별)
PARAM_RANGES = {
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
        'a_excit': (0.005, 0.018),
        'b': (0.15, 0.22),
        'c': (-75, -65),
        'd': (8, 14),
        'noise': (0.5, 3.0),
        'coupling': (0.05, 0.18),
    },
    '유대감': {
        'a_excit': (0.018, 0.028),
        'b': (0.18, 0.22),
        'c': (-68, -62),
        'd': (6, 10),
        'noise': (3.0, 6.0),
        'coupling': (0.15, 0.30),
    },
}


def make_objective(emotion):
    """Optuna objective."""
    target = EMOTION_TARGETS[emotion]['target_freq']
    ranges = PARAM_RANGES[emotion]
    
    def objective(trial):
        params = {
            k: trial.suggest_float(k, lo, hi)
            for k, (lo, hi) in ranges.items()
        }
        
        actual_freq = simulate_emotion(
            params, n=1500, duration_ms=1000, n_repeats=3
        )
        
        return abs(actual_freq - target)
    
    return objective


def tune_emotion(emotion, n_trials=100):
    """감정 1개 튜닝."""
    if not HAS_OPTUNA:
        print("⚠️ Optuna 필요!")
        return None
    
    target = EMOTION_TARGETS[emotion]
    print(f"\n[{emotion}] 튜닝 시작")
    print(f"  목표: {target['wave']} {target['target_freq']}Hz "
          f"(범위 {target['range'][0]}-{target['range'][1]})")
    
    study = optuna.create_study(direction='minimize')
    
    start = time.time()
    study.optimize(make_objective(emotion), n_trials=n_trials, 
                   show_progress_bar=False)
    elapsed = time.time() - start
    
    print(f"  완료: {elapsed:.0f}초 ({elapsed/60:.1f}분)")
    print(f"  최적 거리: {study.best_value:.2f}Hz")
    print(f"  최적 파라미터:")
    for k, v in study.best_params.items():
        print(f"    {k}: {v:.4f}")
    
    return study.best_params


def tune_all_emotions(n_trials_per_emotion=100, save_path=None):
    """모든 감정 튜닝."""
    if not HAS_OPTUNA:
        print("⚠️ Optuna 필요!")
        return None
    
    print("=" * 60)
    print(f"전체 자동 튜닝 ({n_trials_per_emotion} trial × 6 감정)")
    print("=" * 60)
    
    total_start = time.time()
    
    results = {}
    for emotion in EMOTION_TARGETS.keys():
        results[emotion] = tune_emotion(emotion, n_trials_per_emotion)
    
    total_elapsed = time.time() - total_start
    
    # 검증
    print("\n" + "=" * 60)
    print("검증 - 최적 파라미터로 5회 평균")
    print("=" * 60)
    print(f"\n{'감정':<8} {'목표':<18} {'EVE 평균':<14} {'일치':<6}")
    print("-" * 50)
    
    matches = 0
    for emotion, params in results.items():
        if params is None:
            continue
        
        target = EMOTION_TARGETS[emotion]
        target_min, target_max = target['range']
        
        freqs = []
        for _ in range(5):
            f = simulate_emotion(params, n=2000, duration_ms=1500, n_repeats=1)
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
    if save_path is None:
        save_path = '/content/drive/MyDrive/eve_emotion_params.json'
    
    try:
        save_data = {k: v for k, v in results.items() if v is not None}
        with open(save_path, 'w') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        print(f"\n💾 저장: {save_path}")
    except Exception as e:
        print(f"\n⚠️ 저장 실패: {e}")
    
    return results


print("\n" + "=" * 60)
print("EVE 자동 튜닝 v2 (cache X)")
print("=" * 60)
print("""
사용법:

# 1) 빠른 테스트 (5분)
tune_emotion('평온', n_trials=30)

# 2) 적당히 (30-60분)
results = tune_all_emotions(100)

# 3) 정밀 (1-2시간)
results = tune_all_emotions(200)

준비됨!
""")
