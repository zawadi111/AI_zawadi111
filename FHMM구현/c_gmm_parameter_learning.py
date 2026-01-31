"""
3. GMM 파라미터 학습 (GMM Parameter Learning)
각 가전의 상태별 평균, 표준편차, 초기확률을 학습하고 전이확률을 추정합니다.
"""

import pandas as pd
import numpy as np
import os
import warnings
from sklearn.mixture import GaussianMixture

# Windows MKL 경고 무시
warnings.filterwarnings('ignore', category=UserWarning)


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
        
        # 최적 k (Elbow Method 결과, OFF 상태 제외)
        self.optimal_k = {
            'CWE': 3,
            'DWE': 3,
            'CDE': 2,
            'WOE': 2,
            'HPE': 2
        }
        
        self.gmm_params = {}
        self.trans_matrices = {}
        self.data_stats = {}
    
    def load_and_separate_data(self):
        """데이터 로드 및 OFF 상태 분리"""
        print("="*70)
        print("3단계: GMM 파라미터 학습")
        print("="*70)
        print("\n[데이터 로드 및 OFF 상태 분리]")
        
        data = {}
        for code, name in self.appliances.items():
            filepath = os.path.join(self.data_dir, f'modified {code}.csv')
            df = pd.read_csv(filepath)
            data[code] = df
            print(f"✓ {name}: {len(df):,}개 샘플")
        
        # OFF 상태 분리
        print()
        for code, name in self.appliances.items():
            df = data[code]
            
            df_off = df[df['P'] < self.OFF_THRESHOLD]
            df_active = df[df['P'] >= self.OFF_THRESHOLD]
            
            self.data_stats[code] = {
                'df_off': df_off,
                'df_active': df_active,
                'off_ratio': len(df_off) / len(df),
                'off_mean': df_off['P'].mean() if len(df_off) > 0 else 0,
                'off_std': df_off['P'].std() if len(df_off) > 0 else 0,
            }
    
    def learn_parameters(self):
        """GMM을 이용하여 파라미터 학습"""
        print("\n" + "="*70)
        print("GMM 학습 및 파라미터 추출")
        print("="*70 + "\n")
        
        for code, name in self.appliances.items():
            print(f"{name}:")
            
            k = self.optimal_k[code]
            df_active = self.data_stats[code]['df_active']
            X = df_active[['P']].values
            
            # GMM 학습
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
            off_mean = self.data_stats[code]['off_mean']
            off_std = self.data_stats[code]['off_std']
            off_ratio = self.data_stats[code]['off_ratio']
            
            off_mean_arr = np.array([[off_mean]])
            off_cov_arr = np.array([[[off_std**2]]])
            off_weight_arr = np.array([off_ratio])
            
            all_means = np.vstack([off_mean_arr, means])
            all_covariances = np.concatenate([off_cov_arr, covariances])
            all_weights = np.concatenate([off_weight_arr, weights * (1 - off_ratio)])
            
            # 표준편차 추출 (상태별 std)
            all_stds = np.sqrt(all_covariances.flatten())
            
            self.gmm_params[code] = {
                'n_states': k + 1,
                'optimal_k': k,
                'means': all_means,
                'covariances': all_covariances,
                'stds': all_stds,  # ⭐ 상태별 표준편차 추가
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
        print("="*70)
        print("전이 확률 추정")
        print("="*70 + "\n")
        
        data = {}
        for code in self.appliances.keys():
            filepath = os.path.join(self.data_dir, f'modified {code}.csv')
            df = pd.read_csv(filepath)
            data[code] = df
        
        for code, name in self.appliances.items():
            df = data[code]
            
            # P 컬럼 추출
            if 'P' in df.columns:
                power = df['P'].values
            else:
                power = df.iloc[:, 0].values
            
            means = self.gmm_params[code]['means'][:, 0]
            n_states = self.gmm_params[code]['n_states']
            
            # 각 시점을 가장 가까운 상태로 할당
            states = []
            for p in power:
                distances = np.abs(means - p)
                states.append(np.argmin(distances))
            
            # 전이 카운트
            trans_count = np.zeros((n_states, n_states))
            for i in range(len(states) - 1):
                trans_count[states[i], states[i+1]] += 1
            
            # 정규화
            trans_prob = trans_count / (trans_count.sum(axis=1, keepdims=True) + 1e-10)
            self.trans_matrices[code] = trans_prob
            
            print(f"{name} 전이 확률:")
            print(trans_prob.round(3))
            print()
    
    def save_summary(self, filepath=None):
        """파라미터 요약을 텍스트 파일로 저장"""
        if filepath is None:
            filepath = os.path.join(self.output_dir, 'gmm_params_summary.txt')
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("GMM 파라미터 요약\n")
            f.write("="*70 + "\n\n")
            
            for code, name in self.appliances.items():
                params = self.gmm_params[code]
                f.write(f"{name} ({code})\n")
                f.write(f"총 상태 수: {params['n_states']}\n")
                f.write(f"최적 k: {params['optimal_k']}\n\n")
                
                for i in range(params['n_states']):
                    mean_val = params['means'][i, 0]
                    std_val = np.sqrt(params['covariances'][i, 0, 0])
                    weight_val = params['weights'][i]
                    
                    state_name = "OFF" if i == 0 else f"State {i}"
                    f.write(f"  {state_name}:\n")
                    f.write(f"    평균: {mean_val:.2f} W\n")
                    f.write(f"    표준편차: {std_val:.2f} W\n")
                    f.write(f"    초기확률: {weight_val*100:.2f}%\n\n")
                
                f.write("-"*70 + "\n\n")
        
        print("="*70)
        print("파라미터 요약 저장")
        print("="*70)
        print(f"✓ {filepath}\n")
    
    def print_summary(self):
        """파라미터 요약 콘솔 출력"""
        print("\n" + "="*70)
        print("요약")
        print("="*70)
        
        for code, name in self.appliances.items():
            params = self.gmm_params[code]
            print(f"{name}: k={params['optimal_k']}, 총 {params['n_states']}개 상태")
        print()


def main():
    """메인 실행 함수"""
    learner = GMMParameterLearner(
        data_dir='./modified data',
        output_dir='./modified data'
    )
    
    # 데이터 로드 및 OFF 상태 분리
    learner.load_and_separate_data()
    
    # GMM 학습
    learner.learn_parameters()
    
    # 전이 확률 추정
    learner.estimate_transition_matrices()
    
    # 요약 저장 및 출력
    learner.save_summary()
    learner.print_summary()
    
    return learner


if __name__ == "__main__":
    learner = main()
