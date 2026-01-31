# test.py 사용 가이드

## 📁 파일 구조

```
FHMM구현/
├── a_data_preprocessing.py       # 단계 A: 데이터 전처리
├── b_sse_analysis.py             # 단계 B: SSE 플롯
├── c_gmm_parameter_learning.py   # 단계 C: GMM 파라미터 학습
├── d_fhmm_model.py               # 단계 D: FHMM 모델 생성
├── test.py                       # 통합 테스트 파일 (메인)
├── fhmm_model.pkl                # 저장된 모델 (자동 생성)
├── data/                         # 원본 CSV 파일들
└── modified data/                # 처리된 파일들
```

---

## 🚀 사용 방법

### 방법 1️⃣: 모델 구축 (처음 한 번만)

모든 단계를 순서대로 실행하여 모델 구축:

```bash
python test.py
```

**실행 순서:**
1. 단계 A: 데이터 전처리 (`a_data_preprocessing.py`)
2. 단계 B: SSE 분석 (`b_sse_analysis.py`)
3. 단계 C: 파라미터 학습 (`c_gmm_parameter_learning.py`)
4. 단계 D: 모델 생성 (`d_fhmm_model.py`)
5. 모델 저장 (`fhmm_model.pkl`)
6. 디코딩 테스트 실행

---

### 방법 2️⃣: 저장한 모델로 디코딩 (빠름)

두 번째부터는 이미 저장한 모델을 재사용:

```python
# Python REPL 또는 스크립트
from test import load_model, run_decoding

# 저장한 모델 로드
model = load_model('./fhmm_model.pkl')

# 디코딩 실행
result = run_decoding(
    model,
    start_idx=870,
    end_idx=950,
    beam_width=50,
    margin=30
)
```

---

### 방법 3️⃣: 개별 단계 실행

필요한 단계만 선택해서 실행:

```python
from test import (
    step_a_preprocessing,
    step_b_sse_analysis,
    step_c_gmm_learning,
    step_d_fhmm_model
)

# 단계 A만 실행
step_a_preprocessing()

# 단계 B만 실행
step_b_sse_analysis()

# 단계 C만 실행
learner = step_c_gmm_learning()

# 단계 D만 실행
model = step_d_fhmm_model(learner)
```

---

### 방법 4️⃣: 조건 변경하여 디코딩

다양한 파라미터 조합으로 디코딩 테스트:

```python
from test import load_model, run_decoding

model = load_model('./fhmm_model.pkl')

print("\n조건 1: beam_width=30, margin=20")
result1 = run_decoding(model, beam_width=30, margin=20)

print("\n조건 2: beam_width=50, margin=30")
result2 = run_decoding(model, beam_width=50, margin=30)

print("\n조건 3: beam_width=100, margin=50")
result3 = run_decoding(model, beam_width=100, margin=50)
```

---

### 방법 5️⃣: 모든 단계를 선택적으로 실행

필요한 단계만 선택해서 모델 구축:

```python
from test import build_model

# B, C, D만 실행 (A는 스킵)
model = build_model(
    run_a=False,  # 데이터 전처리 스킵
    run_b=True,   # SSE 분석 실행
    run_c=True,   # 파라미터 학습 실행
    run_d=True,   # 모델 생성 실행
    save=True     # 모델 저장
)

# 디코딩
from test import run_decoding
run_decoding(model, beam_width=50, margin=30)
```

---

## 📊 함수 레퍼런스

### 모델 저장/로드

```python
from test import save_model, load_model

# 모델 저장
save_model(model, filepath='./fhmm_model.pkl')

# 모델 로드
model = load_model(filepath='./fhmm_model.pkl')
```

---

### 단계별 실행 함수

```python
from test import (
    step_a_preprocessing,    # 데이터 전처리
    step_b_sse_analysis,     # SSE 분석
    step_c_gmm_learning,     # GMM 학습
    step_d_fhmm_model        # 모델 생성
)

# 각 함수 호출
preprocessor = step_a_preprocessing()
analyzer = step_b_sse_analysis()
learner = step_c_gmm_learning()
model = step_d_fhmm_model(learner)
```

---

### 모델 구축

```python
from test import build_model

# 전체 실행 (권장)
model = build_model(
    run_a=True,   # A단계 실행
    run_b=True,   # B단계 실행
    run_c=True,   # C단계 실행
    run_d=True,   # D단계 실행
    save=True     # 모델 저장
)

# 일부만 실행
model = build_model(run_a=False, run_b=False)
```

---

### 디코딩 실행

```python
from test import run_decoding

result = run_decoding(
    model,              # FHMM 모델
    start_idx=870,      # 테스트 시작 인덱스
    end_idx=950,        # 테스트 종료 인덱스
    beam_width=50,      # 빔 폭 (클수록 정확하지만 느림)
    margin=30           # 탐색 범위 (W)
)

# 반환값
# result: 각 시점의 추론 상태 조합 리스트
# 또는 None (디코딩 실패 시)
```

---

## 💡 실제 사용 예제

### 예제 1: 모델 구축 후 디코딩

```bash
# 터미널
python test.py
```

```
======================================================================
FHMM 통합 테스트
======================================================================

======================================================================
단계 A: 데이터 전처리
======================================================================
...
✓ 단계 A 완료

======================================================================
단계 B: SSE 플롯 및 최적 k 찾기
======================================================================
...
✓ 단계 B 완료

======================================================================
단계 C: GMM 파라미터 학습
======================================================================
...
✓ 단계 C 완료

======================================================================
단계 D: FHMM 모델 생성
======================================================================
✓ 1440개 조합 계산 완료
✓ 모델 생성 완료

======================================================================
디코딩 실행 (beam_width=50, margin=30)
======================================================================
✓ 디코딩 성공!
```

---

### 예제 2: 저장된 모델로 빠른 디코딩

```python
# Python REPL
from test import load_model, run_decoding

# 모델 로드 (매우 빠름)
model = load_model()

# 디코딩 조건 변경
run_decoding(model, start_idx=0, end_idx=100, beam_width=50, margin=30)
run_decoding(model, start_idx=870, end_idx=950, beam_width=100, margin=50)
```

---

### 예제 3: 개별 단계 재실행

```python
from test import step_b_sse_analysis, step_c_gmm_learning, step_d_fhmm_model

# B단계 다시 실행
step_b_sse_analysis()

# C단계 다시 실행
learner = step_c_gmm_learning()

# D단계 다시 실행
model = step_d_fhmm_model(learner)
```

---

## ⚙️ 파라미터 설정

### beam_width (빔 서치 폭)

```python
# 빠른 계산 (부정확할 수 있음)
run_decoding(model, beam_width=10)

# 균형 (권장)
run_decoding(model, beam_width=50)

# 정확한 계산 (느림)
run_decoding(model, beam_width=100)
```

### margin (탐색 범위)

```python
# 좁은 범위 (빠름, 후보 부족 위험)
run_decoding(model, margin=10)

# 적절한 범위 (권장)
run_decoding(model, margin=30)

# 넓은 범위 (느림, 후보 많음)
run_decoding(model, margin=50)
```

---

## 📝 참고 사항

1. **모델 저장**: 첫 번째 실행 시 자동으로 `fhmm_model.pkl` 생성
2. **재사용**: 두 번째부터는 저장된 모델 로드하면 빠름
3. **개별 실행**: 각 함수는 독립적으로 호출 가능
4. **디버깅**: 필요하면 각 단계별로 따로 실행하며 확인 가능

---

이제 원하는 방식으로 유연하게 사용할 수 있습니다! 🚀
