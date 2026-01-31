"""
검정 데이터 (Validation Data)를 이용한 모델 성능 평가
학습 데이터: t=0 ~ 262079 (262,080개 샘플)
검정 데이터: t=262080 이후의 데이터
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from test import load_model


# ============================================================================
# Helper functions for F-score and NDE
# ============================================================================

def calculate_fscore(actual, predicted, threshold=None):
    """
    Calculate F-score for load disaggregation
    threshold: ON/OFF threshold (if None, use mean of actual values)
    """
    if threshold is None:
        threshold = np.mean(actual)
    
    # Binary classification: actual >= threshold
    actual_binary = (actual >= threshold).astype(int)
    predicted_binary = (predicted >= threshold).astype(int)
    
    # TP, FP, FN calculation
    tp = np.sum((actual_binary == 1) & (predicted_binary == 1))
    fp = np.sum((actual_binary == 0) & (predicted_binary == 1))
    fn = np.sum((actual_binary == 1) & (predicted_binary == 0))
    
    # Precision and Recall
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    # F-score
    fscore = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return fscore

def calculate_nde(actual, predicted):
    """
    Calculate Normalized Disaggregation Error (NDE)
    NDE = sqrt(sum((predicted - actual)^2)) / sqrt(sum(actual^2))
    """
    numerator = np.sqrt(np.sum((predicted - actual) ** 2))
    denominator = np.sqrt(np.sum(actual ** 2))
    
    nde = numerator / denominator if denominator > 0 else 0
    return nde

# ============================================================================
# 설정값 (자유롭게 조정 가능)
# ============================================================================

# 학습 데이터 범위
TRAIN_END = 262080

# 검정 데이터 범위 (START_IDX는 TRAIN_END 이후여야 함)
VALIDATION_START = 262080      # 262080 ~ 302079 (10,000개) 추천
VALIDATION_END = 283080        # 또는 원하는 범위로 조정

# 그래프 상한값 설정
Y_LIMITS = {
    'CDE': 6000,
    'CWE': 1000,
    'DWE': 1000,
    'WOE': 4000,
    'HPE': 3000
}

# 가전 순서 (a_data_preprocessing.py와 일치)
APPLIANCES = ['CDE', 'CWE', 'DWE', 'WOE', 'HPE']

# 디코딩 파라미터
BEAM_WIDTH = 50
MARGIN = 200


# ============================================================================
# 검정 데이터 디코딩 및 시각화
# ============================================================================

print("\n" + "="*70)
print("Validation Data Performance Evaluation")
print("="*70 + "\n")

# 1. Load Model
print("[1] Loading model...")
try:
    model = load_model('./fhmm_model.pkl')
except FileNotFoundError:
    print("ERROR: Model file not found.")
    print("Please run 'python test.py' first to build the model.")
    exit(1)

# 2. Load Raw Data
print("[2] Loading raw data...")
raw_data = {}
for code in APPLIANCES:
    df = pd.read_csv(f'./data/Electricity_{code}.csv')
    raw_data[code] = df['P'].values

# 3. Calculate Total Power
print("[3] Calculating total power...")
# Total_Sumed.csv only has 262,080 samples, so calculate from raw data
observations_all = np.zeros(len(raw_data[APPLIANCES[0]]))
for code in APPLIANCES:
    observations_all += raw_data[code]

# 4. Extract Validation Data
print(f"[4] Extracting validation data ({VALIDATION_START}~{VALIDATION_END})...")
test_obs = observations_all[VALIDATION_START:VALIDATION_END].tolist()
test_size = len(test_obs)

print(f"    Validation data size: {test_size:,} samples")
print(f"    Observation range: {min(test_obs):.1f}W ~ {max(test_obs):.1f}W\n")

# 5. Run Decoding
print(f"[5] Running decoding (beam_width={BEAM_WIDTH}, margin={MARGIN})...")
result_path = model.decode(test_obs, beam_width=BEAM_WIDTH, margin=MARGIN)

if result_path is None:
    print("ERROR: Decoding failed")
    exit(1)

print(f"Success! Decoded {len(result_path)} time steps\n")

# 6. Calculate Predicted Power
print("[6] Calculating predicted power...")

# 각 가전의 평균 전력 추출
means = [model.means[i] for i in range(model.app_num)]

# 각 시점의 예측 전력값 계산
predicted_powers = {code: [] for code in APPLIANCES}

for t, state_combo in enumerate(result_path):
    for app_idx, code in enumerate(APPLIANCES):
        state_idx = state_combo[app_idx]
        predicted_power = means[app_idx][state_idx]
        predicted_powers[code].append(predicted_power)

# 7. Create Graph
print("[7] Creating graphs...\n")

fig, axes = plt.subplots(5, 1, figsize=(14, 12))
fig.suptitle(
    f'FHMM Validation Data Performance\n'
    f'Train: 0~{TRAIN_END:,} | Validation: {VALIDATION_START:,}~{VALIDATION_END:,}',
    fontsize=14,
    fontweight='bold'
)

all_fscore = {}
all_nde = {}

for app_idx, code in enumerate(APPLIANCES):
    ax = axes[app_idx]
    
    # Extract actual and predicted values
    actual = raw_data[code][VALIDATION_START:VALIDATION_END]
    predicted = np.array(predicted_powers[code])
    
    # Calculate performance metrics
    fscore = calculate_fscore(actual, predicted)
    nde = calculate_nde(actual, predicted)
    all_fscore[code] = fscore
    all_nde[code] = nde
    
    # 그래프 그리기
    x = np.arange(len(actual))
    
    ax.plot(x, actual, 'r--', label='Actual', linewidth=2, alpha=0.7)
    ax.plot(x, predicted, 'b-', label='Predicted', linewidth=1.5, alpha=0.8)
    
    ax.set_ylabel(f'{code} Power (W)', fontsize=10, fontweight='bold')
    ax.set_ylim([0, Y_LIMITS[code]])
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    ax.set_title(
        f'{code} | F-score: {fscore:.4f} | NDE: {nde:.4f}',
        fontsize=11
    )

plt.tight_layout()

# 8. Save Graph
filename = f'validation_{VALIDATION_START}_{VALIDATION_END}.png'
plt.savefig(filename, dpi=100, bbox_inches='tight')
print(f"Graph saved: ./{filename}\n")
plt.show()

# 9. Print Summary Statistics
print("="*70)
print("Validation Data Performance Summary")
print("="*70 + "\n")

for code in APPLIANCES:
    actual = raw_data[code][VALIDATION_START:VALIDATION_END]
    print(f"{code}:")
    print(f"  Actual range: {np.min(actual):.2f}W ~ {np.max(actual):.2f}W")
    print(f"  F-score: {all_fscore[code]:.4f}")
    print(f"  NDE:     {all_nde[code]:.4f}")
    print()
