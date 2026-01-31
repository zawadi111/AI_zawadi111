# FHMM 파이썬 구현 사용 가이드 (4개 파일 버전)

## 📁 파일 구조 및 데이터 경로

```
FHMM구현/
├── 1_data_preprocessing.py           # 1단계: 데이터 전처리
├── 2_sse_analysis.py                 # 2단계: SSE 플롯 및 최적 k 찾기
├── 3_gmm_parameter_learning.py       # 3단계: GMM 파라미터 학습
├── 4_fhmm_model.py                   # 4단계: FHMM 클래스 및 디코딩
├── test_FHMM.py                      # 테스트 코드
├── FHMM_unified.py                   # (선택) 통합 버전
├── data/                             # 원본 CSV 파일들 (필수)
│   ├── Electricity_CDE.csv
│   ├── Electricity_CWE.csv
│   ├── Electricity_DWE.csv
│   ├── Electricity_HPE.csv
│   └── Electricity_WOE.csv
└── modified data/                    # 처리된 파일들 (자동 생성)
    ├── modified CDE.csv
    ├── modified CWE.csv
    ├── modified DWE.csv
    ├── modified HPE.csv
    ├── modified WOE.csv
    ├── Total_Sumed.csv
    ├── elbow_plot.png
    └── gmm_params_summary.txt
```

**중요:** `data/` 폴더와 `Electricity_*.csv` 파일들은 **반드시 존재**해야 합니다.

---

## 🚀 빠른 시작 (순차 실행)

### 터미널에서 순서대로 실행하기

```bash
# 1단계: 데이터 전처리 (CSV에서 P열 추출 및 샘플링)
python 1_data_preprocessing.py

# 2단계: SSE 플롯 (Elbow Method로 최적 k 결정)
python 2_sse_analysis.py

# 3단계: GMM 파라미터 학습 (평균, 표준편차, 초기확률, 전이확률)
python 3_gmm_parameter_learning.py

# 4단계: FHMM 모델 생성 및 디코딩 테스트
python 4_fhmm_model.py
```

---

## 📚 파일별 상세 설명

### 1️⃣ 1_data_preprocessing.py - 데이터 전처리

**기능:**
- 원본 CSV 파일에서 P 컬럼(유효전력) 추출
- 원하는 샘플 수만큼 잘라내기 (기본: 262,080개 = 반년치)
- 모든 가전 데이터 합산

**사용 방법:**

```python
from 1_data_preprocessing import DataPreprocessor

preprocessor = DataPreprocessor(
    data_dir='./data',           # 원본 CSV 폴더
    output_dir='./modified data' # 출력 폴더
)

# 모든 가전 전처리
preprocessor.preprocess_all(num_samples=262080)

# 모든 가전 전력 합산
result_df = preprocessor.aggregate_data()
```

**생성 파일:**
- `modified data/modified CDE.csv`
- `modified data/modified CWE.csv`
- `modified data/modified DWE.csv`
- `modified data/modified HPE.csv`
- `modified data/modified WOE.csv`
- `modified data/Total_Sumed.csv` (모든 가전의 전력 합산)

---

### 2️⃣ 2_sse_analysis.py - SSE 플롯 및 최적 k 찾기

**기능:**
- OFF 상태(P < 10W)와 활성 상태 분리
- k=1~8에 대해 GMM 학습 및 SSE 계산
- Elbow Method를 통해 최적 k 결정
- SSE 그래프 시각화

**사용 방법:**

```python
from 2_sse_analysis import ElbowMethodAnalyzer

analyzer = ElbowMethodAnalyzer(
    data_dir='./modified data',
    output_dir='./modified data'
)

# 데이터 로드 및 OFF 분리
analyzer.load_and_separate_data()

# SSE 계산
analyzer.calculate_sse()

# 그래프 시각화
analyzer.plot_elbow(save_path='./modified data/elbow_plot.png')

# 요약 출력
analyzer.print_summary()
```

**생성 파일:**
- `modified data/elbow_plot.png` (2x3 서브플롯)

**출력 결과:**
- 각 k에 대한 SSE 값 출력
- 그래프에서 꺾이는 지점(Elbow) 확인

**권장 최적 k:**
- CWE: 3, DWE: 2, CDE: 2, WOE: 2, HPE: 2
- (OFF 상태 포함하면 각각 +1)

---

### 3️⃣ 3_gmm_parameter_learning.py - GMM 파라미터 학습

**기능:**
- 각 가전의 상태별 평균 전력 학습 (means)
- 각 상태의 표준편차 학습 (std)
- 각 상태의 초기 확률 학습 (initial probabilities)
- 실제 데이터에서 상태 전이 확률 추정 (transition matrices)

**사용 방법:**

```python
from 3_gmm_parameter_learning import GMMParameterLearner

learner = GMMParameterLearner(
    data_dir='./modified data',
    output_dir='./modified data'
)

# 데이터 로드 및 OFF 분리
learner.load_and_separate_data()

# GMM 학습
learner.learn_parameters()

# 전이 확률 추정
learner.estimate_transition_matrices()

# 파라미터 저장
learner.save_summary()
learner.print_summary()

# 학습된 파라미터 직접 접근
print(learner.gmm_params['CWE']['means'])
print(learner.trans_matrices['CWE'])
```

**생성 파일:**
- `modified data/gmm_params_summary.txt` (파라미터 요약)

**포함 정보:**
```
learner.gmm_params[appliance_code] = {
    'n_states': 3,                           # 총 상태 수
    'optimal_k': 2,                          # 최적 k
    'means': np.array([[...], [...], ...]),  # 각 상태의 평균 전력
    'covariances': np.array([...]),          # 각 상태의 분산
    'weights': np.array([...])               # 각 상태의 초기 확률
}

learner.trans_matrices[appliance_code] = np.array([[...], [...]])  # n_states x n_states 행렬
```

---

### 4️⃣ 4_fhmm_model.py - FHMM 클래스 및 디코딩

**기능:**
- FHMM 모델 클래스 구현
- Viterbi Beam Search 디코딩 알고리즘
- 관측 전력값에서 각 가전의 상태 추론

**FHMM 클래스 사용:**

```python
from 4_fhmm_model import FHMM
import numpy as np

# 파라미터 정의
means = [
    [0, 100, 500],      # 가전1: OFF, 낮음, 높음
    [0, 50],            # 가전2: OFF, ON
    [0, 200]            # 가전3: OFF, ON
]

trans_matrices = [
    np.array([[0.9, 0.05, 0.05],
              [0.1, 0.8, 0.1],
              [0.05, 0.05, 0.9]]),
    np.array([[0.9, 0.1], [0.1, 0.9]]),
    np.array([[0.95, 0.05], [0.1, 0.9]])
]

initial_probs = [
    np.array([0.8, 0.15, 0.05]),
    np.array([0.9, 0.1]),
    np.array([0.95, 0.05])
]

# 모델 생성
model = FHMM(
    app_num=3,
    means=means,
    initial_probs=initial_probs,
    trans_matrices=trans_matrices,
    std=30  # 노이즈 표준편차
)

# 디코딩
observations = [0, 50, 100, 150, 200, 250, 350, 250, 100, 0]
result_path = model.decode(
    observations,
    beam_width=50,  # 빔 폭
    margin=30       # 탐색 범위
)

# 결과 확인
for t, state in enumerate(result_path):
    print(f"t={t}: {state}")
```

**학습된 파라미터로 모델 생성:**

```python
from 3_gmm_parameter_learning import GMMParameterLearner
from 4_fhmm_model import load_fhmm_from_learner

learner = GMMParameterLearner()
learner.load_and_separate_data()
learner.learn_parameters()
learner.estimate_transition_matrices()

# FHMM 모델 자동 생성
model = load_fhmm_from_learner(learner)

# 디코딩
observations = [...]  # 관측 데이터
result = model.decode(observations, beam_width=50, margin=30)
```

---

## 🎯 주요 파라미터 설정

### beam_width (빔 서치 폭)
각 시점에서 유지할 최고 점수 경로의 개수

```python
# 적은 값: 빠르지만 부정확
result = model.decode(obs, beam_width=10)

# 중간값: 균형 (권장)
result = model.decode(obs, beam_width=50)

# 큰 값: 느리지만 정확
result = model.decode(obs, beam_width=100)
```

### margin (탐색 범위)
관측값 주변에서 후보 상태 조합을 찾는 범위 (단위: W)

```python
# 작은 값: 빠르지만 후보 부족
result = model.decode(obs, margin=10)

# 중간값: 균형 (권장)
result = model.decode(obs, margin=30)

# 큰 값: 느리지만 후보 많음
result = model.decode(obs, margin=50)
```

### std (노이즈 표준편차)
관측값의 노이즈 정도

```python
# 노이즈 적음
model = FHMM(..., std=10)

# 노이즈 중간 (권장)
model = FHMM(..., std=30)

# 노이즈 많음
model = FHMM(..., std=50)
```

---

## 📋 완전한 워크플로우 예제

```python
# 1단계: 데이터 전처리
from 1_data_preprocessing import DataPreprocessor

preprocessor = DataPreprocessor()
preprocessor.preprocess_all(num_samples=262080)
preprocessor.aggregate_data()

# 2단계: SSE 분석
from 2_sse_analysis import ElbowMethodAnalyzer

analyzer = ElbowMethodAnalyzer()
analyzer.load_and_separate_data()
analyzer.calculate_sse()
analyzer.plot_elbow()

# 3단계: 파라미터 학습
from 3_gmm_parameter_learning import GMMParameterLearner

learner = GMMParameterLearner()
learner.load_and_separate_data()
learner.learn_parameters()
learner.estimate_transition_matrices()
learner.save_summary()

# 4단계: FHMM 모델 생성 및 디코딩
from 4_fhmm_model import load_fhmm_from_learner
import pandas as pd

model = load_fhmm_from_learner(learner)

df = pd.read_csv('./modified data/Total_Sumed.csv')
observations = df['Total_P'].values[:1000]

result = model.decode(observations, beam_width=50, margin=30)

if result:
    print(f"✓ 디코딩 성공! 추론 경로 길이: {len(result)}")
    for t in range(min(10, len(result))):
        print(f"t={t}: {result[t]}")
```

---

## 💾 데이터 경로 정리

| 용도 | 경로 | 자동 생성 | 필수 |
|------|------|---------|------|
| 원본 데이터 | `./data/Electricity_*.csv` | ❌ | ✅ |
| 전처리 결과 | `./modified data/modified *.csv` | ✅ | - |
| 합산 데이터 | `./modified data/Total_Sumed.csv` | ✅ | - |
| SSE 그래프 | `./modified data/elbow_plot.png` | ✅ | - |
| 파라미터 요약 | `./modified data/gmm_params_summary.txt` | ✅ | - |

---

## ⚠️ 주의사항

### 1. 원본 데이터 필수
반드시 다음 파일들이 필요합니다:
```
data/Electricity_CDE.csv
data/Electricity_CWE.csv
data/Electricity_DWE.csv
data/Electricity_HPE.csv
data/Electricity_WOE.csv
```

### 2. 폴더 자동 생성
- `./data/` 폴더와 CSV 파일: 사용자가 준비
- `./modified data/` 폴더: 자동 생성

### 3. 메모리 사용
- 대량 데이터 처리 시 메모리 확인
- beam_width 크면 메모리 증가

### 4. 디코딩 실패 시
```python
# margin 값 증가
result = model.decode(obs, margin=50)  # 30 → 50

# 또는 beam_width 증가
result = model.decode(obs, beam_width=100)  # 50 → 100
```

---

## 🧪 테스트하기

```bash
python test_FHMM.py
```

또는 Python REPL:
```python
python
>>> from 1_data_preprocessing import DataPreprocessor
>>> preprocessor = DataPreprocessor()
>>> preprocessor.preprocess_all()
```

---

## 📞 문제 해결

### "FileNotFoundError: ./data/Electricity_CDE.csv"
→ `data/` 폴더를 만들고 CSV 파일 복사

### "No survivors at t=X" 경고
→ `margin` 값을 증가시키기 (30 → 50)

### 느린 속도
→ `beam_width` 감소 (50 → 30) 또는 데이터 샘플 감소

### 부정확한 결과
→ `beam_width` 증가 (50 → 100)

---

이제 사용할 준비가 완료되었습니다! 🚀
