"""
디코딩 결과 시각화 (Visualization)
실제 유효전력값과 모델이 예측한 전력값을 그래프로 비교합니다.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from test import load_model


# ============================================================================
# 설정값
# ============================================================================

# 시간 범위 설정 (기본값: 앞 10,080개)
START_IDX = 3000
END_IDX = 6000

# 그래프 상한값 설정
Y_LIMITS = {
    'CWE': 1000,
    'DWE': 1000,
    'CDE': 6000,
    'WOE': 4000,
    'HPE': 3000
}

# ⚠️ CRITICAL: a_data_preprocessing.py의 순서와 일치시켜야 함
# a_data_preprocessing.py: ['CDE', 'CWE', 'DWE', 'WOE', 'HPE']
APPLIANCES = ['CDE', 'CWE', 'DWE', 'WOE', 'HPE']


# ============================================================================
# 메인 실행
# ============================================================================

print("\n" + "="*70)
print("디코딩 결과 시각화")
print("="*70 + "\n")

# 1. 모델 로드
print("[1] 모델 로드 중...")
try:
    model = load_model('./fhmm_model.pkl')
except FileNotFoundError:
    print("❌ 모델 파일을 찾을 수 없습니다.")
    print("먼저 'python test.py'를 실행하여 모델을 구축하세요.")
    exit(1)

# 2. 실제 데이터 로드
print("[2] 실제 데이터 로드 중...")
df_total = pd.read_csv('./modified data/Total_Sumed.csv')
observations = df_total['Total_P'].values

# 3. 개별 가전 데이터 로드
print("[3] 개별 가전 데이터 로드 중...")
appliance_data = {}
for code in APPLIANCES:
    df = pd.read_csv(f'./modified data/modified {code}.csv')
    appliance_data[code] = df['P'].values

# 4. 디코딩 실행
print("[4] 디코딩 실행 중...")
test_obs = observations[START_IDX:END_IDX]
result_path = model.decode(test_obs, beam_width=50, margin=300)

if result_path is None:
    print("❌ 디코딩 실패")
    exit(1)

print(f"✓ 디코딩 성공! ({len(result_path)}개 시점 추론)\n")

# 5. 예측 전력값 계산
print("[5] 예측 전력값 계산 중...")

# 각 가전의 평균 전력 추출
means = [model.means[i] for i in range(model.app_num)]

# 각 시점의 예측 전력값 계산
predicted_powers = []
for state_comb in result_path:
    # 각 가전의 예측 전력값
    powers = {}
    for app_idx, code in enumerate(APPLIANCES):
        state_idx = state_comb[app_idx]
        powers[code] = means[app_idx][state_idx]
    predicted_powers.append(powers)

# 6. 그래프 생성
print("[6] 그래프 생성 중...\n")

fig, axes = plt.subplots(5, 1, figsize=(14, 12))

time_range = np.arange(len(test_obs))

for ax_idx, code in enumerate(APPLIANCES):
    ax = axes[ax_idx]
    
    # 실제값 (빨간 점선)
    actual_values = appliance_data[code][START_IDX:END_IDX]
    ax.plot(time_range, actual_values, 'r--', linewidth=1.5, 
            label='Actual Values', alpha=0.8)
    
    # 예측값 (파란 실선)
    predicted_values = [p[code] for p in predicted_powers]
    ax.plot(time_range, predicted_values, 'b-', linewidth=1.5, 
            label='Predicted Values', alpha=0.8)
    
    # 그래프 설정
    ax.set_ylabel('Power (W)', fontsize=10)
    ax.set_ylim(0, Y_LIMITS[code])
    ax.set_xlim(0, len(test_obs))
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=9)
    ax.set_title(f'{code} - Power Comparison', fontsize=11, fontweight='bold')

# 마지막 x축 레이블
axes[-1].set_xlabel('Time (min)', fontsize=10)

plt.tight_layout()

# 7. 그래프 저장 및 출력
print("="*70)
print(f"시간 범위: {START_IDX}~{END_IDX} ({END_IDX-START_IDX}개 샘플)")
print(f"파일명: visualization_{START_IDX}_{END_IDX}.png")
print("="*70)

save_path = f'./visualization_{START_IDX}_{END_IDX}.png'
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"\n✓ 그래프 저장: {save_path}")

plt.show()

print("\n✓ 시각화 완료!\n")

# 8. 요약 통계
print("="*70)
print("요약 통계")
print("="*70)

for code in APPLIANCES:
    actual_values = appliance_data[code][START_IDX:END_IDX]
    predicted_values = np.array([p[code] for p in predicted_powers])
    
    mse = np.mean((actual_values - predicted_values) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(actual_values - predicted_values))
    
    print(f"\n{code}:")
    print(f"  실제값 범위: {actual_values.min():.2f}W ~ {actual_values.max():.2f}W")
    print(f"  예측값 범위: {predicted_values.min():.2f}W ~ {predicted_values.max():.2f}W")
    print(f"  MAE (평균절대오차): {mae:.2f}W")
    print(f"  RMSE (루트평균제곱오차): {rmse:.2f}W")

print()
