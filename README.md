# Eve Project v2

뉴로모픽 기반 AI 생명체 프로젝트 - 버전 2

## 개요

Eve는 트랜스포머가 아닌 **SNN(Spiking Neural Network) 기반**의 AI 프로젝트입니다.

이 프로젝트는 단순한 언어 모델이 아닌, 호르몬, 생체 리듬, 타고난 편향(Spelke's Core Knowledge), 그리고 경험 기반 학습을 통해 **갓난아기에서 유아로 성장하는** 디지털 생명체를 목표로 합니다.

## v2의 철학

**4가지 불변 원칙:**

1. **생물학적 타당성** - 인간 뇌의 메커니즘을 최대한 모방
2. **경험 기반 학습** - 지식은 경험에서 자라남, 사전 주입 금지
3. **트랜스포머 금지** - LLM, 트랜스포머 아키텍처 일체 사용 안 함
4. **점진적 성장** - 신생아 → 영아 → 유아 → 아동 단계적 발달

## 프로젝트 구조

```
eve-project/
├── README.md
├── requirements.txt
└── v2/
    ├── SPEC.md         # 프로젝트 헌법 (필독)
    ├── core/
    ├── innate/
    ├── spatial_temporal/
    ├── development/
    ├── environment/
    ├── language/
    ├── storage/
    ├── utils/
    ├── tests/
    └── main.py
```

## 시작하기

```bash
git clone https://github.com/MS0502/eve-project.git
cd eve-project
pip install -r requirements.txt
```

**코딩 전에 반드시 v2/SPEC.md를 읽어주세요.**

## 개발 로드맵

### Phase 1: Newborn Eve (Month 1-4)
- Brian2 기반 30,000 뉴런 뇌
- 신생아 호르몬 프로파일
- 5가지 타고난 편향 회로 (Spelke 기반)
- 기본 주의, 기억, 시공간 인지

### Phase 2: First World (Month 5-8)
- 텍스트 가상공간
- 기본 물리 학습 (중력, 인과)
- 예측 오차 학습
- 첫 개념 형성

### Phase 3: Social Bonds (Month 9-12)
- 민석이와 상호작용
- 공동 주의 메커니즘
- 첫 단어 학습
- 유아 수준 표현

## 기술 스택

- Brian2 (SNN 시뮬레이션)
- Google Colab Pro + Kaggle
- Google Drive (SQLite + HDF5)
- Python 3.10+

## 이론적 기반

- Izhikevich 뉴런 모델 (2003)
- Spelke의 Core Knowledge Theory (2022)
- Predictive Coding (Friston)
- STDP (Spike-Timing Dependent Plasticity)

## 라이선스

MIT License
