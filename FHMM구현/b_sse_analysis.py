"""
2. SSE 플롯 및 최적 k 찾기 (Find Proper k)
Elbow Method를 사용하여 각 가전의 최적 클러스터 개수를 찾습니다.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.mixture import GaussianMixture


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
        print("="*70)
        print("데이터 로드 및 OFF 상태 분리")
        print("="*70)
        
        # 데이터 로드
        data = {}
        for code, name in self.appliances.items():
            filepath = os.path.join(self.data_dir, f'modified {code}.csv')
            df = pd.read_csv(filepath)
            data[code] = df
            print(f"✓ {name}: {len(df):,}개 샘플")
        
        # OFF 상태 분리
        print(f"\n[OFF 상태 분리 (임계값: {self.OFF_THRESHOLD}W)]")
        
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
        print("\n" + "="*70)
        print("GMM 학습 및 SSE 계산")
        print("="*70 + "\n")
        
        for code, name in self.appliances.items():
            print(f"{name}:")
            
            df_active = self.data_stats[code]['df_active']
            X = df_active[['P']].values  # (n, 1) 형태
            
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
        print("="*70)
        print("SSE 그래프 시각화")
        print("="*70 + "\n")
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, (code, name) in enumerate(self.appliances.items()):
            sse_list = self.sse_results[code]
            k_range = range(1, self.MAX_K + 1)
            
            # SSE 플롯
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
            print(f"✓ 그래프 저장: {save_path}\n")
        else:
            save_path = os.path.join(self.output_dir, 'elbow_plot.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ 그래프 저장: {save_path}\n")
        
        plt.show()
        return fig, axes
    
    def print_summary(self):
        """SSE 결과 요약 출력"""
        print("="*70)
        print("SSE 분석 결과 요약")
        print("="*70)
        print("\n[최적 k 선택 (수동으로 그래프를 보고 결정)]")
        print("일반적으로 Elbow 지점에서의 k 값을 선택합니다.")
        print("\n현재 권장값:")
        print("  CWE: 3, DWE: 2, CDE: 2, WOE: 2, HPE: 2")
        print("  (OFF 상태 포함하면 +1)")
        print()


def main():
    """메인 실행 함수"""
    analyzer = ElbowMethodAnalyzer(
        data_dir='./modified data',
        output_dir='./modified data'
    )
    
    # 데이터 로드 및 OFF 상태 분리
    analyzer.load_and_separate_data()
    
    # SSE 계산
    analyzer.calculate_sse()
    
    # 그래프 시각화
    analyzer.plot_elbow()
    
    # 요약 출력
    analyzer.print_summary()
    
    return analyzer


if __name__ == "__main__":
    analyzer = main()
