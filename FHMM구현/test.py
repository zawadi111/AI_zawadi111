"""
통합 테스트 파일 (test.py)
a, b, c, d 단계를 개별적으로 실행하고, 모델 구축과 디코딩을 별개로 수행합니다.
"""

import pickle
import pandas as pd
from a_data_preprocessing import DataPreprocessor
from b_sse_analysis import ElbowMethodAnalyzer
from c_gmm_parameter_learning import GMMParameterLearner
from d_fhmm_model import load_fhmm_from_learner, FHMM


# ============================================================================
# 모델 저장/로드 함수
# ============================================================================

def save_model(model, filepath='./fhmm_model.pkl'):
    """학습한 FHMM 모델 저장"""
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print(f"[OK] 모델 저장: {filepath}\n")


def load_model(filepath='./fhmm_model.pkl'):
    """저장한 FHMM 모델 로드"""
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    print(f"[OK] 모델 로드: {filepath}\n")
    return model


# ============================================================================
# 단계별 실행 함수
# ============================================================================

def step_a_preprocessing():
    """단계 A: 데이터 전처리"""
    print("\n" + "="*70)
    print("단계 A: 데이터 전처리")
    print("="*70 + "\n")
    
    preprocessor = DataPreprocessor(
        data_dir='./data',
        output_dir='./modified data'
    )
    
    preprocessor.preprocess_all(num_samples=262080)
    result_df = preprocessor.aggregate_data()
    
    print("✓ 단계 A 완료\n")
    return preprocessor


def step_b_sse_analysis():
    """단계 B: SSE 플롯 및 최적 k 찾기"""
    print("\n" + "="*70)
    print("단계 B: SSE 플롯 및 최적 k 찾기")
    print("="*70 + "\n")
    
    analyzer = ElbowMethodAnalyzer(
        data_dir='./modified data',
        output_dir='./modified data'
    )
    
    analyzer.load_and_separate_data()
    analyzer.calculate_sse()
    analyzer.plot_elbow()
    analyzer.print_summary()
    
    print("✓ 단계 B 완료\n")
    return analyzer


def step_c_gmm_learning():
    """단계 C: GMM 파라미터 학습"""
    print("\n" + "="*70)
    print("단계 C: GMM 파라미터 학습")
    print("="*70 + "\n")
    
    learner = GMMParameterLearner(
        data_dir='./modified data',
        output_dir='./modified data'
    )
    
    learner.load_and_separate_data()
    learner.learn_parameters()
    learner.estimate_transition_matrices()
    learner.save_summary()
    learner.print_summary()
    
    print("✓ 단계 C 완료\n")
    return learner


def step_d_fhmm_model(learner):
    """단계 D: FHMM 모델 생성"""
    print("\n" + "="*70)
    print("단계 D: FHMM 모델 생성")
    print("="*70 + "\n")
    
    model = load_fhmm_from_learner(learner)
    
    print("✓ 단계 D 완료\n")
    return model


# ============================================================================
# 모델 구축 (모든 단계 실행)
# ============================================================================

def build_model(run_a=True, run_b=True, run_c=True, run_d=True, save=True):
    """
    모든 단계를 실행하여 모델 구축
    
    Args:
        run_a: A단계 실행 여부
        run_b: B단계 실행 여부
        run_c: C단계 실행 여부
        run_d: D단계 실행 여부
        save: 모델 저장 여부
    
    Returns:
        학습한 FHMM 모델
    """
    print("\n" + "="*80)
    print("모델 구축 시작")
    print("="*80)
    
    # 단계 A: 데이터 전처리
    if run_a:
        step_a_preprocessing()
    
    # 단계 B: SSE 분석
    if run_b:
        step_b_sse_analysis()
    
    # 단계 C: 파라미터 학습
    if run_c:
        learner = step_c_gmm_learning()
    else:
        raise ValueError("C단계는 필수입니다 (파라미터 필요)")
    
    # 단계 D: 모델 생성
    if run_d:
        model = step_d_fhmm_model(learner)
    else:
        raise ValueError("D단계는 필수입니다 (모델 필요)")
    
    # 모델 저장
    if save:
        save_model(model)
    
    print("="*80)
    print("✓ 모델 구축 완료!")
    print("="*80 + "\n")
    
    return model


# ============================================================================
# 디코딩 실행
# ============================================================================

def run_decoding(model, start_idx=870, end_idx=950, beam_width=50, margin=150):
    """
    저장한 모델로 디코딩 실행
    
    Args:
        model: FHMM 모델
        start_idx: 테스트 시작 인덱스
        end_idx: 테스트 종료 인덱스
        beam_width: 빔 서치 폭
        margin: 탐색 범위
    """
    print("\n" + "="*70)
    print(f"디코딩 실행 (beam_width={beam_width}, margin={margin})")
    print("="*70 + "\n")
    
    # 테스트 데이터 로드
    df_total = pd.read_csv('./modified data/Total_Sumed.csv')
    observations = df_total['Total_P'].tolist()
    
    test_obs = observations[start_idx:end_idx]
    print(f"테스트 구간: {start_idx}~{end_idx} ({len(test_obs)}개 샘플)")
    print(f"관측값 범위: {min(test_obs):.1f}W ~ {max(test_obs):.1f}W\n")
    
    # 디코딩 실행
    print("디코딩 실행 중...")
    result_path = model.decode(test_obs, beam_width=beam_width, margin=margin)
    
    if result_path:
        print(f"✓ 디코딩 성공!\n")
        
        print("="*70)
        print(f"추론 결과 (처음 20개 시점)")
        print("="*70)
        print(f"{'시간':<5} {'관측값':<10} {'상태 조합':<40}")
        print("-"*70)
        
        for t in range(min(20, len(result_path))):
            print(f"{t:<5} {test_obs[t]:<10.1f} {result_path[t]}")
        
        print()
        return result_path
    else:
        print("✗ 디코딩 실패\n")
        return None


# ============================================================================
# 사용 예제
# ============================================================================

def example_usage():
    """사용 예제"""
    
    # --- 방법 1: 모델 구축 후 디코딩 ---
    # model = build_model(run_a=True, run_b=True, run_c=True, run_d=True, save=True)
    # result = run_decoding(model, beam_width=50, margin=30)
    
    # --- 방법 2: 저장한 모델 로드 후 디코딩 (두 번째 실행 시) ---
    # model = load_model('./fhmm_model.pkl')
    # result = run_decoding(model, beam_width=50, margin=30)
    
    # --- 방법 3: 개별 단계만 실행 ---
    # step_a_preprocessing()
    # step_b_sse_analysis()
    # learner = step_c_gmm_learning()
    # model = step_d_fhmm_model(learner)
    
    # --- 방법 4: 여러 조건으로 디코딩 비교 ---
    # model = load_model('./fhmm_model.pkl')
    # result1 = run_decoding(model, beam_width=30, margin=20)
    # result2 = run_decoding(model, beam_width=50, margin=30)
    # result3 = run_decoding(model, beam_width=100, margin=50)
    
    pass


if __name__ == "__main__":
    """
    사용 방법:
    
    1. 모델 구축 (처음 한 번만):
       python test.py
       (자동으로 A, B, C, D 단계 모두 실행 후 모델 저장)
    
    2. 저장한 모델로 디코딩만 실행 (빠름):
       python test.py
       (저장된 모델이 있으면 로드 후 디코딩)
    
    3. 개별 단계 실행:
       >>> from test import step_a_preprocessing, step_b_sse_analysis
       >>> step_a_preprocessing()
       >>> step_b_sse_analysis()
    
    4. 다양한 조건으로 디코딩:
       >>> from test import load_model, run_decoding
       >>> model = load_model()
       >>> run_decoding(model, beam_width=50, margin=30)
       >>> run_decoding(model, beam_width=100, margin=50)
    """
    
    # 모델 구축 (모든 단계)
    print("\n" + "="*80)
    print("FHMM 통합 테스트")
    print("="*80)
    
    model = build_model(
        run_a=True,
        run_b=True,
        run_c=True,
        run_d=True,
        save=True
    )
    
    # 디코딩 실행
    result = run_decoding(
        model =load_model('./fhmm_model.pkl'),
        start_idx=870,
        end_idx=950,
        beam_width=50,
        margin=150
    )
