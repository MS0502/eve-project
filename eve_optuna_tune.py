"""
EVE 호르몬 → 뇌파 자동 튜닝
================================
Optuna 자동 튜닝 (CPU만 사용!)
GPU 필요 X

목표: 인간 뇌파와 일치
- 평온: 알파 (8-13Hz)
- 집중: 베타 (13-30Hz)
- 기쁨: 감마 (30-60Hz)
- 두려움: 베타+감마 (20-50Hz)
- 슬픔: 세타↓ (3-8Hz)
- 유대감: 알파/베타 (10-15Hz)

설치:
  !pip install optuna numba

실행 (Colab):
  exec(open('/content/drive/MyDrive/eve_optuna_tune.py').read())
  results = tune_all_emotions()
  
시간:
  - Numba 없이: 1-3시간
  - Numba 사용: 30분-1시간
"""

import numpy as np
import time

# Numba 옵션 (10배 빠름)
try:
    from numba import jit
    HAS_NUMBA = True
    print("✅ Numba 사용 (10배 가속)")
except ImportError:
    HAS_NUMBA = False
    print("⚠️ Numba 없음 (NumPy만 사용)")

# Optuna 필수
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    print("⚠️ pip install optuna 필요!")


def _simulate_core(V, u, a, b, c, d, W, noise, n, duration_ms):
    """SNN 시뮬 코어 (Numba 가속 가능)."""
    spike_counts = np.zeros(duration_ms, dtype=np.int32)
    
    for t in range(duration_ms):
        # Izhikevich
        V += 0.5 * (0.04 * V**2 + 5*V + 140 - u)
        V += 0.5 * (0.04 * V**2 + 5*V + 140 - u)
        u += a * (b * V - u)
        V += noise * np.random.randn(n)
        
        # Fire
        n_fired = 0
        for i in range(n):
            if V[i] >= 30.0:
                # Reset
                V[i] = c[i]
                u[i] += d[i]
                
                # Synaptic transmission (random subset)
                for j in range(min(20, n)):
                    target = (i + j * 7) % n  # Pseudo-random
                    V[target] += W[i, j]
                
                n_fired += 1
        
        spike_counts[t] = n_fired
    
    return spike_counts


# Numba JIT (있으면)
if HAS_NUMBA:
    _simulate_core = jit(nopython=True, cache=True)(_simulate_core)


def simulate_emotion(params, n=2000, duration_ms=1500, n_repeats=3):
    """감정 시뮬 - 평균 주파수 반환."""
    freqs = []
    
    for _ in range(n_repeats):
        # 파라미터 셋업
        mask = np.random.rand(n) < 0.8
        a = np.where(mask, params['a_excit'], 0.10).astype(np.float64)
        b = np.full(n, params['b'], dtype=np.float64)
        c = np.full(n, params['c'], dtype=np.float64)
        d = np.where(mask, params['d'], 2.0).astype(np.float64)
        
        V = -65.0 * np.ones(n, dtype=np.float64)
        u = -13.0 * np.ones(n, dtype=np.float64)
        
        n_synapses = 20
        W = (np.random.randn(n, n_synapses) * params['coupling']).astype(np.float64)
        
        spike_counts = _simulate_core(
            V, u, a, b, c, d, W, params['noise'], n, duration_ms
        )
        
        # FFT 분석
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


# 감정별 목표
EMOTION_TARGETS = {
    '평온': {'target_freq': 10.5, 'range': (8, 13), 'wave': '알파'},
    '집중': {'target_freq': 21, 'range': (13, 30), 'wave': '베타'},
    '기쁨': {'target_freq': 45, 'range': (30, 60), 'wave': '감마'},
    '두려움': {'target_freq': 35, 'range': (20, 50), 'wave': '베타+감마'},
    '슬픔': {'target_freq': 5.5, 'range': (3, 8), 'wave': '세타↓'},
    '유대감': {'target_freq': 12.5, 'range': (10, 15), 'wave': '알파/베타'},
}


def make_objective(emotion):
    """감정별 Optuna objective 함수."""
    target = EMOTION_TARGETS[emotion]['target_freq']
    
    def objective(trial):
        # 감정별 파라미터 범위
        if emotion == '평온':
            params = {
                'a_excit': trial.suggest_float('a_excit', 0.015, 0.025),
                'b': trial.suggest_float('b', 0.18, 0.22),
                'c': trial.suggest_float('c', -68, -62),
                'd': trial.suggest_float('d', 6, 10),
                'noise': trial.suggest_float('noise', 2.0, 5.0),
                'coupling': trial.suggest_float('coupling', 0.10, 0.30),
            }
        elif emotion == '집중':
            params = {
                'a_excit': trial.suggest_float('a_excit', 0.020, 0.030),
                'b': trial.suggest_float('b', 0.18, 0.22),
                'c': trial.suggest_float('c', -68, -62),
                'd': trial.suggest_float('d', 5, 9),
                'noise': trial.suggest_float('noise', 4.0, 7.0),
                'coupling': trial.suggest_float('coupling', 0.20, 0.40),
            }
        elif emotion == '기쁨':
            params = {
                'a_excit': trial.suggest_float('a_excit', 0.035, 0.060),
                'b': trial.suggest_float('b', 0.20, 0.28),
                'c': trial.suggest_float('c', -65, -55),
                'd': trial.suggest_float('d', 3, 7),
                'noise': trial.suggest_float('noise', 6.0, 10.0),
                'coupling': trial.suggest_float('coupling', 0.40, 0.80),
            }
        elif emotion == '두려움':
            params = {
                'a_excit': trial.suggest_float('a_excit', 0.025, 0.045),
                'b': trial.suggest_float('b', 0.18, 0.24),
                'c': trial.suggest_float('c', -68, -60),
                'd': trial.suggest_float('d', 4, 8),
                'noise': trial.suggest_float('noise', 5.0, 9.0),
                'coupling': trial.suggest_float('coupling', 0.30, 0.55),
            }
        elif emotion == '슬픔':
            params = {
                'a_excit': trial.suggest_float('a_excit', 0.005, 0.018),
                'b': trial.suggest_float('b', 0.15, 0.22),
                'c': trial.suggest_float('c', -75, -65),
                'd': trial.suggest_float('d', 8, 14),
                'noise': trial.suggest_float('noise', 0.5, 3.0),
                'coupling': trial.suggest_float('coupling', 0.05, 0.18),
            }
        else:  # 유대감
            params = {
                'a_excit': trial.suggest_float('a_excit', 0.018, 0.028),
                'b': trial.suggest_float('b', 0.18, 0.22),
                'c': trial.suggest_float('c', -68, -62),
                'd': trial.suggest_float('d', 6, 10),
                'noise': trial.suggest_float('noise', 3.0, 6.0),
                'coupling': trial.suggest_float('coupling', 0.15, 0.30),
            }
        
        # 시뮬
        actual_freq = simulate_emotion(
            params, 
            n=1500,  # 작게 (속도)
            duration_ms=1000,  # 1초
            n_repeats=3,  # 3회 평균
        )
        
        # 거리 (절대값)
        dist = abs(actual_freq - target)
        return dist
    
    return objective


def tune_emotion(emotion, n_trials=100):
    """감정 1개 자동 튜닝."""
    print(f"\n[{emotion}] 튜닝 시작 (목표 {EMOTION_TARGETS[emotion]['target_freq']}Hz)")
    
    if not HAS_OPTUNA:
        print("⚠️ Optuna 필요!")
        return None
    
    study = optuna.create_study(direction='minimize')
    
    start = time.time()
    study.optimize(make_objective(emotion), n_trials=n_trials, show_progress_bar=False)
    elapsed = time.time() - start
    
    print(f"  완료: {elapsed:.0f}초")
    print(f"  최적 거리: {study.best_value:.2f}Hz")
    print(f"  최적 파라미터:")
    for k, v in study.best_params.items():
        print(f"    {k}: {v:.4f}")
    
    return study.best_params


def tune_all_emotions(n_trials_per_emotion=100):
    """모든 감정 자동 튜닝."""
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
    
    print(f"\n{'감정':<8} {'목표':<14} {'EVE 평균':<12} {'일치':<6}")
    print("-" * 45)
    
    matches = 0
    for emotion, params in results.items():
        if params is None:
            continue
        
        target = EMOTION_TARGETS[emotion]
        target_min, target_max = target['range']
        
        # 5회 평균
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
        print(f"{emotion:<8} {target_str:<14} {avg:<6.1f}±{std:<3.1f}    {match}")
    
    print(f"\n=== 최종 ===")
    print(f"정확도: {matches}/6 = {matches/6*100:.0f}%")
    print(f"총 시간: {total_elapsed:.0f}초 = {total_elapsed/60:.1f}분")
    
    # JSON 저장
    import json
    with open('/content/drive/MyDrive/eve_emotion_params.json', 'w') as f:
        json.dump({k: v for k, v in results.items() if v is not None}, 
                  f, ensure_ascii=False, indent=2)
    print(f"\n💾 저장: /content/drive/MyDrive/eve_emotion_params.json")
    
    return results


# 실행 가이드
if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("EVE 자동 튜닝")
    print("=" * 60)
    print("""
사용법:

1) Colab에서:
   !pip install optuna numba
   exec(open('/content/drive/MyDrive/eve_optuna_tune.py').read())
   
   # 빠른 테스트 (감정 1개, 50 trial, 5분)
   tune_emotion('평온', n_trials=50)
   
   # 전체 (6 감정, 100 trial 각각, 30분-1시간)
   results = tune_all_emotions(100)
   
   # 정밀 (200 trial, 1-2시간)
   results = tune_all_emotions(200)

2) 결과:
   - 정확도 50% → 80%+ 가능
   - JSON 저장 (/content/drive/MyDrive/eve_emotion_params.json)
   - EVE에 로드해서 사용
""")
    
    # 빠른 테스트
    print("\n[빠른 검증 - 평온만 30 trial]:")
    if HAS_OPTUNA:
        tune_emotion('평온', n_trials=30)
