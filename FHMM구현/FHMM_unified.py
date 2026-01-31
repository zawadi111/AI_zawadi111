"""
FHMM(Factorial Hidden Markov Model) 통합 구현
기능:
1. 데이터 전처리: CSV 파일에서 필요한 부분 자르기 (P열, 원하는 시간대)
2. SSE 플롯 계산 및 출력: Elbow Method로 최적 k 찾기
3. FHMM 클래스: 디코딩 기능을 가진 FHMM 모델 구현
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from bisect import bisect_left, bisect_right
from sklearn.mixture import GaussianMixture
import os

# ============================================================================
# 1. 데이터 전처리 (Data Preprocessing)
# ============================================================================

class DataPreprocessor:
    """CSV 파일에서 필요한 부분을 자르고 처리하는 클래스"""
    
    def __init__(self, data_dir='./data', output_dir='./modified data'):
        """
        Args:
            data_dir: 원본 데이터 폴더 경로
            output_dir: 처리된 데이터 저장 폴더 경로
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.names = ['CDE', 'CWE', 'DWE', 'WOE', 'HPE']
        
        # 출력 폴더 생성
        os.makedirs(output_dir, exist_ok=True)
    
    def preprocess_all(self, num_samples=262080):
        """
        모든 가전 데이터를 전처리하고 저장
        
        Args:
            num_samples: 저장할 샘플 개수 (기본: 반년치=262080)
        """
        print("="*60)
        print("데이터 전처리 시작")
        print("="*60)
        
        for name in self.names:
            # 파일 경로 설정
            input_path = os.path.join(self.data_dir, f'Electricity_{name}.csv')
            output_path = os.path.join(self.output_dir, f'modified {name}.csv')
            
            if not os.path.exists(input_path):
                print(f"⚠️  파일 없음: {input_path}")
                continue
            
            # CSV 읽기
            df = pd.read_csv(input_path)
            
            # P열만 추출 (유효전력 Active Power)
            df = df[['P']]
            
            # 원하는 시간대만 자르기
            df = df.head(num_samples)
            
            # 저장
            df.to_csv(output_path, index=False)
            print(f"✓ {name}: {len(df):,}개 샘플 저장 -> {output_path}")
        
        print("\n✓ 데이터 전처리 완료!\n")
    
    def aggregate_data(self):
        """
        모든 가전의 전력을 합산하여 Total_Sumed.csv 생성
        """
        print("="*60)
        print("데이터 합산 시작")
        print("="*60)
        
        total_data = 0
        
        for name in self.names:
            path = os.path.join(self.output_dir, f'modified {name}.csv')
            
            if not os.path.exists(path):
                print(f"⚠️  파일 없음: {path}")
                continue
            
            df = pd.read_csv(path)
            total_data += df['P']
            print(f"✓ {name} 로드 및 합산")
        
        # 결과 저장
        result_df = pd.DataFrame({'Total_P': total_data})
        output_path = os.path.join(self.output_dir, 'Total_Sumed.csv')
        result_df.to_csv(output_path, index=False)
        print(f"\n✓ 전체 합산 데이터 저장 -> {output_path}\n")
        
        return result_df


# ============================================================================
# 2. SSE 플롯 및 최적 k 찾기 (Find Proper k)
# ============================================================================

class ElbowMethodAnalyzer:
    """SSE를 이용한 Elbow Method로 최적 k를 찾는 클래스"""
    
    def __init__(self, data_dir='./modified data', output_dir='./modified data'):
        """
        Args:
            data_dir: 전처리된 데이터 폴더
            output_dir: 결과 저장 폴더
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.appliances = {
            'CWE': 'Clothes Washer',
            'DWE': 'Dishwasher',
            'CDE': 'Clothes Dryer',
            'WOE': 'Wall Oven',
            'HPE': 'Heat Pump'
        }
        self.OFF_THRESHOLD = 10
        self.MAX_K = 8
        self.data_stats = {}
        self.sse_results = {}
    
    def load_and_separate_data(self):
        """데이터 로드 및 OFF 상태 분리"""
        print("="*60)
        print("데이터 로드 및 OFF 상태 분리")
        print("="*60)
        
        # 데이터 로드
        data = {}
        for code, name in self.appliances.items():
            df = pd.read_csv(os.path.join(self.data_dir, f'modified {code}.csv'))
            data[code] = df
            print(f"✓ {name}: {len(df):,}개 샘플")
        
        # OFF 상태 분리
        for code, name in self.appliances.items():
            df = data[code]
            
            df_off = df[df['P'] < self.OFF_THRESHOLD]
            df_active = df[df['P'] >= self.OFF_THRESHOLD]
            
            stats = {
                'df': df,
                'df_off': df_off,
                'df_active': df_active,
                'off_ratio': len(df_off) / len(df),
                'off_mean': df_off['P'].mean() if len(df_off) > 0 else 0,
                'off_std': df_off['P'].std() if len(df_off) > 0 else 0,
            }
            
            self.data_stats[code] = stats
            
            print(f"\n{name}:")
            print(f"  OFF: {len(df_off):,}개 ({stats['off_ratio']*100:.1f}%)")
            print(f"  활성: {len(df_active):,}개")
    
    def calculate_sse(self):
        """GMM 학습 및 SSE 계산"""
        print("\n" + "="*60)
        print("GMM 학습 및 SSE 계산")
        print("="*60 + "\n")
        
        for code, name in self.appliances.items():
            print(f"{name}:")
            
            df_active = self.data_stats[code]['df_active']
            X = df_active[['P']].values
            
            sse_list = []
            
            for k in range(1, self.MAX_K + 1):
                gmm = GaussianMixture(
                    n_components=k,
                    covariance_type='full',
                    max_iter=100,
                    n_init=10,
                    random_state=42
                )
                gmm.fit(X)
                
                # SSE 계산
                labels = gmm.predict(X)
                sse = 0
                for i in range(k):
                    cluster_samples = X[labels == i]
                    if len(cluster_samples) > 0:
                        sse += np.sum((cluster_samples - gmm.means_[i]) ** 2)
                
                sse_list.append(sse)
                print(f"  k={k}: SSE={sse:.2e}")
            
            self.sse_results[code] = sse_list
            print()
    
    def plot_elbow(self, save_path=None):
        """SSE 그래프 시각화"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, (code, name) in enumerate(self.appliances.items()):
            sse_list = self.sse_results[code]
            k_range = range(1, self.MAX_K + 1)
            
            axes[idx].plot(k_range, sse_list, 'bo-', linewidth=2, markersize=8)
            axes[idx].set_xlabel('Number of Clusters (k)', fontsize=10)
            axes[idx].set_ylabel('SSE', fontsize=10)
            axes[idx].set_title(f'{name}', fontsize=11, fontweight='bold')
            axes[idx].grid(True, alpha=0.3)
        
        # 마지막 subplot 숨기기
        axes[-1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ 그래프 저장: {save_path}")
        
        plt.show()
        return fig, axes


# ============================================================================
# 3. FHMM 클래스 (Factorial Hidden Markov Model)
# ============================================================================

class BeamNode:
    """빔 서치에서 살아남은 노드 정보를 저장하는 클래스"""
    
    def __init__(self, state_comb, score, parent_idx):
        """
        Args:
            state_comb: 현재 상태 조합 (예: (1, 0, 1))
            score: 현재까지 누적된 로그 확률
            parent_idx: 이전 시점 트렐리스 리스트에서의 인덱스
        """
        self.state = state_comb
        self.score = score
        self.parent_idx = parent_idx


class FHMM:
    """디코딩 기능을 가진 Factorial Hidden Markov Model"""
    
    def __init__(self, app_num, means, initial_probs, trans_matrices, std=30):
        """
        FHMM 모델 초기화
        
        Args:
            app_num: 가전 개수
            means: 각 가전의 상태별 평균 전력
            trans_matrices: 각 가전의 전이확률 행렬 리스트
            initial_probs: 각 가전의 초기확률 리스트
            std: 관측 노이즈 표준편차 (sigma)
        """
        self.app_num = app_num
        self.means = means
        self.initial_probs = initial_probs
        self.trans_matrices = trans_matrices
        
        # 확률을 로그로 미리 계산 (속도 최적화)
        self.log_trans = [np.log(np.maximum(mat, 1e-10)) for mat in trans_matrices]
        self.log_initial = [np.log(np.maximum(p, 1e-10)) for p in initial_probs]
        self.var_term = 2 * (std**2)
        
        # 모든 조합 사전 계산 및 정렬
        self.sorted_power_sums = []
        self.sorted_combinations = []
        self._precompute_combinations()
    
    def _precompute_combinations(self):
        """모든 가전 상태의 조합을 사전 계산하고 전력값 기준으로 정렬"""
        state_indices = [range(len(m)) for m in self.means]
        all_combos = list(itertools.product(*state_indices))
        
        temp_list = []
        for combo in all_combos:
            total_power = sum(self.means[i][state] for i, state in enumerate(combo))
            temp_list.append((total_power, combo))
        
        temp_list.sort(key=lambda x: x[0])
        
        self.sorted_combinations = temp_list
        self.sorted_power_values = [x[0] for x in temp_list]
        
        print(f"✓ {len(all_combos)}개 조합 계산 완료")
    
    def _get_candidates(self, observation, margin=30):
        """관측값 근처의 후보들을 이진 탐색으로 찾음"""
        candidates_range_left = bisect_left(self.sorted_power_values, observation - margin)
        candidates_range_right = bisect_right(self.sorted_power_values, observation + margin)
        
        candidates = self.sorted_combinations[candidates_range_left:candidates_range_right]
        
        if not candidates:
            idx = bisect_left(self.sorted_power_values, observation)
            if idx >= len(self.sorted_power_values):
                idx = len(self.sorted_power_values) - 1
            candidates = [self.sorted_combinations[idx]]
        
        return candidates
    
    def _calc_emission_log(self, observation, power_sum):
        """방출 확률 (가우시안 로그 확률) 계산"""
        diff = observation - power_sum
        return -(diff ** 2) / self.var_term
    
    def _calc_transition_log(self, prev_combo, curr_combo):
        """전이 확률 (로그) 계산"""
        log_prob = 0.0
        for i in range(self.app_num):
            p_prev = prev_combo[i]
            p_curr = curr_combo[i]
            log_prob += self.log_trans[i][p_prev][p_curr]
        return log_prob
    
    def decode(self, observations, beam_width=50, margin=30):
        """
        Viterbi Beam Search를 이용한 디코딩
        
        Args:
            observations: 관측 시계열 데이터
            beam_width: 빔 서치 폭
            margin: 관측값 주변 탐색 범위
            
        Returns:
            추론된 상태 조합의 시계열
        """
        T = len(observations)
        psi = [[] for _ in range(T)]
        
        # 초기화 (t=0)
        first_candidates = self._get_candidates(observations[0], margin)
        
        for power, state_comb in first_candidates:
            init_score = 0
            for i in range(self.app_num):
                init_score += self.log_initial[i][state_comb[i]]
            
            emit_score = self._calc_emission_log(observations[0], power)
            total_score = init_score + emit_score
            
            psi[0].append(BeamNode(state_comb, total_score, -1))
        
        psi[0].sort(key=lambda x: x.score, reverse=True)
        psi[0] = psi[0][:beam_width]
        
        # Forward pass (t=1 ~ T-1)
        for t in range(1, T):
            obs = observations[t]
            current_candidates = self._get_candidates(obs, margin)
            prev_survivors = psi[t-1]
            next_survivors = []
            
            for cand_power, cand_state in current_candidates:
                emission_score = self._calc_emission_log(obs, cand_power)
                
                max_score_for_this_cand = -np.inf
                best_parent_idx = -1
                
                for p_idx, parent_node in enumerate(prev_survivors):
                    trans_score = self._calc_transition_log(parent_node.state, cand_state)
                    score = parent_node.score + trans_score + emission_score
                    
                    if score > max_score_for_this_cand:
                        max_score_for_this_cand = score
                        best_parent_idx = p_idx
                
                if best_parent_idx != -1:
                    next_survivors.append(BeamNode(cand_state, max_score_for_this_cand, best_parent_idx))
            
            next_survivors.sort(key=lambda x: x.score, reverse=True)
            psi[t] = next_survivors[:beam_width]
            
            if not psi[t]:
                print(f"⚠️  Warning: No survivors at t={t}. Increasing margin might help.")
                break
        
        # Backward pass (역추적)
        best_path = []
        
        if not psi[T-1]:
            return None
        
        curr_node = psi[T-1][0]
        best_path.append(curr_node.state)
        
        for t in range(T-1, 0, -1):
            parent_idx = curr_node.parent_idx
            curr_node = psi[t-1][parent_idx]
            best_path.append(curr_node.state)
        
        best_path.reverse()
        
        return best_path


# ============================================================================
# 4. GMM 파라미터 학습 및 전이확률 추정
# ============================================================================

class GMMParameterLearner:
    """GMM을 이용한 파라미터 학습"""
    
    def __init__(self, data_dir='./modified data', output_dir='./modified data'):
        """
        Args:
            data_dir: 전처리된 데이터 폴더
            output_dir: 결과 저장 폴더
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.appliances = {
            'CWE': 'Clothes Washer',
            'DWE': 'Dishwasher',
            'CDE': 'Clothes Dryer',
            'WOE': 'Wall Oven',
            'HPE': 'Heat Pump'
        }
        self.OFF_THRESHOLD = 10
        self.optimal_k = {
            'CWE': 3,
            'DWE': 2,
            'CDE': 2,
            'WOE': 2,
            'HPE': 2
        }
        self.gmm_params = {}
        self.trans_matrices = {}
    
    def learn_parameters(self):
        """GMM을 이용하여 파라미터 학습"""
        print("="*60)
        print("GMM 파라미터 학습")
        print("="*60 + "\n")
        
        # 데이터 로드
        data = {}
        for code, name in self.appliances.items():
            df = pd.read_csv(os.path.join(self.data_dir, f'modified {code}.csv'))
            data[code] = df
        
        # OFF 상태 분리
        data_stats = {}
        for code, name in self.appliances.items():
            df = data[code]
            df_off = df[df['P'] < self.OFF_THRESHOLD]
            df_active = df[df['P'] >= self.OFF_THRESHOLD]
            
            data_stats[code] = {
                'df_off': df_off,
                'df_active': df_active,
                'off_ratio': len(df_off) / len(df),
                'off_mean': df_off['P'].mean() if len(df_off) > 0 else 0,
                'off_std': df_off['P'].std() if len(df_off) > 0 else 0,
            }
        
        # GMM 학습
        for code, name in self.appliances.items():
            print(f"{name}:")
            
            k = self.optimal_k[code]
            df_active = data_stats[code]['df_active']
            X = df_active[['P']].values
            
            gmm = GaussianMixture(
                n_components=k,
                covariance_type='full',
                max_iter=200,
                n_init=20,
                random_state=42
            )
            gmm.fit(X)
            
            # 파라미터 추출
            means = gmm.means_
            covariances = gmm.covariances_
            weights = gmm.weights_
            
            # 전력 순 정렬
            sorted_indices = np.argsort(means.flatten())
            means = means[sorted_indices]
            covariances = covariances[sorted_indices]
            weights = weights[sorted_indices]
            
            # OFF 상태 추가
            off_mean = data_stats[code]['off_mean']
            off_std = data_stats[code]['off_std']
            off_ratio = data_stats[code]['off_ratio']
            
            off_mean_arr = np.array([[off_mean]])
            off_cov_arr = np.array([[[off_std**2]]])
            off_weight_arr = np.array([off_ratio])
            
            all_means = np.vstack([off_mean_arr, means])
            all_covariances = np.concatenate([off_cov_arr, covariances])
            all_weights = np.concatenate([off_weight_arr, weights * (1 - off_ratio)])
            
            self.gmm_params[code] = {
                'n_states': k + 1,
                'optimal_k': k,
                'means': all_means,
                'covariances': all_covariances,
                'weights': all_weights
            }
            
            # 결과 출력
            print(f"  총 상태: {k + 1}개 (OFF 포함)")
            for i in range(k + 1):
                mean_val = all_means[i, 0]
                std_val = np.sqrt(all_covariances[i, 0, 0])
                weight_val = all_weights[i]
                state_name = "OFF" if i == 0 else f"State {i}"
                print(f"  {state_name}: 평균 {mean_val:7.2f}W, 표준편차 {std_val:6.2f}W, 비율 {weight_val*100:5.2f}%")
            print()
    
    def estimate_transition_matrices(self):
        """실제 데이터에서 전이 확률 추정"""
        print("="*60)
        print("전이 확률 추정")
        print("="*60 + "\n")
        
        data = {}
        for code in self.appliances.keys():
            df = pd.read_csv(os.path.join(self.data_dir, f'modified {code}.csv'))
            data[code] = df
        
        for code, name in self.appliances.items():
            df = data[code]
            power = df['P'].values
            means = np.array([self.gmm_params[code]['means'][:, 0]])
            n_states = self.gmm_params[code]['n_states']
            
            # 각 시점을 가장 가까운 상태로 할당
            states = []
            for p in power:
                distances = np.abs(means.flatten() - p)
                states.append(np.argmin(distances))
            
            # 전이 카운트
            trans_count = np.zeros((n_states, n_states))
            for i in range(len(states)-1):
                trans_count[states[i], states[i+1]] += 1
            
            # 정규화
            trans_prob = trans_count / (trans_count.sum(axis=1, keepdims=True) + 1e-10)
            self.trans_matrices[code] = trans_prob
            
            print(f"{name} 전이 확률:")
            print(trans_prob.round(3))
            print()
    
    def save_summary(self, filepath=None):
        """파라미터 요약 저장"""
        if filepath is None:
            filepath = os.path.join(self.output_dir, 'gmm_params_summary.txt')
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("GMM 파라미터 요약\n\n")
            
            for code, name in self.appliances.items():
                params = self.gmm_params[code]
                f.write(f"{name} ({code})\n")
                f.write(f"총 상태 수: {params['n_states']}\n\n")
                
                for i in range(params['n_states']):
                    mean_val = params['means'][i, 0]
                    std_val = np.sqrt(params['covariances'][i, 0, 0])
                    weight_val = params['weights'][i]
                    
                    state_name = "OFF" if i == 0 else f"State {i}"
                    f.write(f"  {state_name}:\n")
                    f.write(f"    평균: {mean_val:.2f} W\n")
                    f.write(f"    표준편차: {std_val:.2f} W\n")
                    f.write(f"    비율: {weight_val*100:.2f}%\n\n")
                
                f.write("-"*50 + "\n\n")
        
        print(f"✓ 파라미터 요약 저장: {filepath}")
    
    def get_fhmm_model(self):
        """학습된 파라미터로 FHMM 모델 생성"""
        means = [self.gmm_params[code]['means'][:, 0].tolist() for code in self.appliances.keys()]
        initial_probs = [self.gmm_params[code]['weights'] for code in self.appliances.keys()]
        trans_matrices = [self.trans_matrices[code] for code in self.appliances.keys()]
        
        model = FHMM(
            app_num=len(self.appliances),
            means=means,
            initial_probs=initial_probs,
            trans_matrices=trans_matrices
        )
        
        return model


# ============================================================================
# 5. 메인 실행 함수
# ============================================================================

def main():
    """전체 FHMM 파이프라인 실행"""
    
    print("\n" + "="*60)
    print("FHMM 통합 구현 - 전체 파이프라인")
    print("="*60 + "\n")
    
    # 1. 데이터 전처리
    print("[1단계] 데이터 전처리\n")
    preprocessor = DataPreprocessor(data_dir='./data', output_dir='./modified data')
    preprocessor.preprocess_all(num_samples=262080)
    preprocessor.aggregate_data()
    
    # 2. SSE 플롯 및 최적 k 찾기
    print("\n[2단계] SSE 플롯 및 최적 k 찾기\n")
    elbow_analyzer = ElbowMethodAnalyzer(data_dir='./modified data', output_dir='./modified data')
    elbow_analyzer.load_and_separate_data()
    elbow_analyzer.calculate_sse()
    elbow_analyzer.plot_elbow(save_path='./modified data/elbow_plot.png')
    
    # 3. GMM 파라미터 학습
    print("\n[3단계] GMM 파라미터 학습\n")
    param_learner = GMMParameterLearner(data_dir='./modified data', output_dir='./modified data')
    param_learner.learn_parameters()
    param_learner.estimate_transition_matrices()
    param_learner.save_summary()
    
    # 4. FHMM 모델 생성 및 테스트
    print("\n[4단계] FHMM 모델 생성 및 테스트\n")
    fhmm_model = param_learner.get_fhmm_model()
    
    # 테스트 데이터 로드
    df_total = pd.read_csv('./modified data/Total_Sumed.csv')
    observations = df_total['Total_P'].tolist()
    
    # 일부 데이터로 디코딩 테스트
    test_obs = observations[870:950]
    print("="*60)
    print("디코딩 테스트 (870:950 구간)")
    print("="*60 + "\n")
    
    result_path = fhmm_model.decode(test_obs, beam_width=50, margin=30)
    
    if result_path:
        print("✓ 추론 성공!\n")
        print("처음 20개 시점의 추론 결과:")
        print("-" * 60)
        for t in range(min(20, len(result_path))):
            print(f"Time {t}: 관측 {test_obs[t]:.1f}W -> 추론 상태 {result_path[t]}")
    else:
        print("✗ 추론 실패 (생존자 없음)")
    
    print("\n" + "="*60)
    print("✓ 전체 파이프라인 완료!")
    print("="*60)
    
    return fhmm_model, param_learner


if __name__ == "__main__":
    fhmm_model, param_learner = main()
