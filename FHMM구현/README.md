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

## 🚀 빠른 시작 (4단계 실행)

### 방법 1: 순차 실행 (권장)

각 파일을 순서대로 실행합니다:

```bash
# 1단계: 데이터 전처리
python 1_data_preprocessing.py

# 2단계: SSE 플롯 및 최적 k 찾기
python 2_sse_analysis.py

# 3단계: GMM 파라미터 학습
python 3_gmm_parameter_learning.py

# 4단계: FHMM 모델 생성 및 디코딩
python 4_fhmm_model.py
```

각 단계의 결과는 `modified data/` 폴더에 저장됩니다.

---

## 📚 각 파일별 상세 사용법

### 1️⃣ 1_data_preprocessing.py (데이터 전처리)

```python
from FHMM_unified import DataPreprocessor

# 초기화
preprocessor = DataPreprocessor(
    data_dir='./data',           # 원본 CSV 폴더
    output_dir='./modified data' # 출력 폴더
)

# 메서드 1: 모든 가전 전처리
preprocessor.preprocess_all(
    num_samples=262080  # 저장할 샘플 개수 (기본: 반년치)
)
# 결과: modified data/modified CDE.csv, modified CWE.csv 등 생성

# 메서드 2: 모든 가전 전력 합산
result_df = preprocessor.aggregate_data()
# 결과: modified data/Total_Sumed.csv 생성
# result_df: pandas DataFrame (Total_P 컬럼 포함)
```

**자동 동작:**
- `data/` 폴더에서 `Electricity_*.csv` 파일 자동 인식
- `modified data/` 폴더 없으면 자동 생성
- P 컬럼만 추출
- CSV 파일로 자동 저장

---

### 2️⃣ ElbowMethodAnalyzer (SSE 플롯)

```python
from FHMM_unified import ElbowMethodAnalyzer

# 초기화
analyzer = ElbowMethodAnalyzer(
    data_dir='./modified data',
    output_dir='./modified data'
)

# 메서드 1: 데이터 로드 및 OFF 상태 분리
analyzer.load_and_separate_data()
# 결과: analyzer.data_stats에 통계 저장

# 메서드 2: SSE 계산
analyzer.calculate_sse()
# 결과: analyzer.sse_results에 SSE 값들 저장

# 메서드 3: 그래프 저장
fig, axes = analyzer.plot_elbow(
    save_path='./modified data/elbow_plot.png'
)
# 결과: 2x3 서브플롯 (각 가전별 SSE 그래프)
```

**자동 동작:**
- `modified data/` 폴더에서 CSV 자동 읽기
- OFF 상태 (P < 10W) 자동 분리
- k=1~8 범위에서 SSE 계산
- 그래프 자동 생성 및 저장

---

### 3️⃣ GMMParameterLearner (파라미터 학습)

```python
from FHMM_unified import GMMParameterLearner

# 초기화
learner = GMMParameterLearner(
    data_dir='./modified data',
    output_dir='./modified data'
)

# 메서드 1: GMM 학습
learner.learn_parameters()
# 결과: learner.gmm_params에 저장
# 포함 정보: n_states, means, covariances, weights

# 메서드 2: 전이 확률 추정
learner.estimate_transition_matrices()
# 결과: learner.trans_matrices에 저장

# 메서드 3: 파라미터 요약 저장
learner.save_summary(
    filepath='./modified data/gmm_params_summary.txt'
)

# 메서드 4: FHMM 모델 자동 생성
fhmm_model = learner.get_fhmm_model()
# 반환: FHMM 클래스 인스턴스 (디코딩 준비 완료)

# 학습된 파라미터 직접 접근
print(learner.gmm_params['CWE']['means'])
print(learner.trans_matrices['CWE'])
```

**자동 동작:**
- 최적 k: CWE=3, DWE=2, CDE=2, WOE=2, HPE=2 (고정)
- OFF 상태 자동 추가 (n_states = k+1)
- 전이 확률 데이터에서 추정
- 파라미터 자동 정규화

---

### 4️⃣ FHMM (디코딩)

```python
from FHMM_unified import FHMM

# 초기화 (파라미터 필수)
model = FHMM(
    app_num=5,                          # 가전 개수
    means=[[...], [...], ...],          # 각 가전의 상태별 평균 전력
    initial_probs=[np.array([...]), ...], # 각 가전의 초기확률
    trans_matrices=[np.array([[...]]), ...], # 각 가전의 전이확률 행렬
    std=30                              # 노이즈 표준편차 (기본값: 30)
)

# 메서드: 디코딩 (Viterbi Beam Search)
result_path = model.decode(
    observations=[100, 120, 110, 200, ...],  # 관측 시계열
    beam_width=50,                           # 빔 폭 (클수록 정확)
    margin=30                                # 탐색 범위 (W)
)

# 반환값
# - result_path: 각 시점의 추론 상태 조합 리스트
#   예: [(0, 1, 0, 2, 1), (0, 1, 1, 2, 1), ...]
# - None: 디코딩 실패 (margin 증가 필요)

# 결과 해석
if result_path:
    for t, state_tuple in enumerate(result_path):
        # state_tuple: (기기1_상태, 기기2_상태, 기기3_상태, ...)
        print(f"Time {t}: {state_tuple}")
```

**내부 동작:**
- 조합 사전계산: 모든 상태 조합 생성 및 정렬 (초기화 시 1회)
- 이진 탐색: 관측값 근처 후보 빠르게 검색
- 빔 서치: K개 최고 경로만 유지하며 계산
- 역추적: 최고 경로 복원

---

## 🎯 주요 파라미터 설정

### beam_width (빔 서치 폭)
```python
# 적은 값: 빠르지만 부정확
result = model.decode(obs, beam_width=10)

# 중간값: 균형
result = model.decode(obs, beam_width=50)  # 권장

# 큰 값: 느리지만 정확
result = model.decode(obs, beam_width=100)
```

### margin (탐색 범위)
```python
# 작은 값: 빠르지만 후보 부족
result = model.decode(obs, margin=10)

# 중간값: 균형
result = model.decode(obs, margin=30)  # 권장

# 큰 값: 느리지만 후보 많음
result = model.decode(obs, margin=50)
```

### std (노이즈 표준편차)
```python
# 노이즈 적음: std 작음
model = FHMM(..., std=10)

# 노이즈 중간: std 중간
model = FHMM(..., std=30)  # 권장

# 노이즈 많음: std 큼
model = FHMM(..., std=50)
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

## 📋 완전한 워크플로우 예제

```python
from FHMM_unified import (
    DataPreprocessor, 
    ElbowMethodAnalyzer, 
    GMMParameterLearner
)
import pandas as pd

# ===== 1단계: 데이터 전처리 =====
print("[1] 데이터 전처리 중...")
preprocessor = DataPreprocessor()
preprocessor.preprocess_all(num_samples=262080)
preprocessor.aggregate_data()

# ===== 2단계: SSE 분석 =====
print("[2] SSE 분석 중...")
analyzer = ElbowMethodAnalyzer()
analyzer.load_and_separate_data()
analyzer.calculate_sse()
analyzer.plot_elbow()

# ===== 3단계: 파라미터 학습 =====
print("[3] 파라미터 학습 중...")
learner = GMMParameterLearner()
learner.learn_parameters()
learner.estimate_transition_matrices()
learner.save_summary()

# ===== 4단계: 모델 생성 및 디코딩 =====
print("[4] 디코딩 중...")
model = learner.get_fhmm_model()

# 데이터 로드
df = pd.read_csv('./modified data/Total_Sumed.csv')
observations = df['Total_P'].values[:1000]  # 처음 1000개만

# 디코딩
result = model.decode(observations, beam_width=50, margin=30)

# 결과 분석
if result:
    print(f"✓ 디코딩 성공! 추론 경로 길이: {len(result)}")
    print(f"첫 10개 상태: {result[:10]}")
else:
    print("✗ 디코딩 실패")
```

---

## ⚠️ 주의사항

1. **반드시 원본 데이터 필요**
   ```
   data/Electricity_CDE.csv
   data/Electricity_CWE.csv
   data/Electricity_DWE.csv
   data/Electricity_HPE.csv
   data/Electricity_WOE.csv
   ```

2. **폴더 생성은 자동**
   - `./data/` 폴더와 CSV 파일: 사용자가 준비
   - `./modified data/` 폴더: 자동 생성

3. **메모리 주의**
   - 대량 데이터 처리 시 메모리 사용량 확인
   - beam_width 크면 메모리 증가

4. **디코딩 실패 시**
   - margin 값 증가 시도
   - beam_width 값 조정
   - 데이터 범위 확인

---

## 🧪 테스트하기

```bash
# 전체 테스트 실행
python test_FHMM.py

# 또는 Python REPL
python
>>> from FHMM_unified import main
>>> model, learner = main()
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
