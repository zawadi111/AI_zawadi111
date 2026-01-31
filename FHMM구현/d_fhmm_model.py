"""
4. FHMM 클래스 및 디코딩 (FHMM Model and Decoding)
Factorial Hidden Markov Model 구현 및 Viterbi Beam Search 디코딩
"""

import numpy as np
import itertools
import pandas as pd
import os
from bisect import bisect_left, bisect_right


class BeamNode:
    """빔 서치에서 살아남은 노드 정보를 저장하는 클래스"""
    
    def __init__(self, state_comb, score, parent_idx):
        """
        Args:
            state_comb: 현재 상태 조합 (예: (1, 0, 1))
            score: 현재까지 누적된 로그 확률 (클수록 좋음)
            parent_idx: 이전 시점 트렐리스 리스트에서의 인덱스 (역추적용)
        """
        self.state = state_comb
        self.score = score
        self.parent_idx = parent_idx


class FHMM:
    """디코딩 기능을 가진 Factorial Hidden Markov Model"""
    
    def __init__(self, app_num, means, initial_probs, trans_matrices, stds=None, std=30):
        """
        FHMM 모델 초기화
        
        Args:
            app_num: 가전 개수 N
            means: 각 가전의 상태별 평균 전력
                   형식: [[off, on1, on2], [off, on], ...]
            initial_probs: 각 가전의 초기확률 리스트
                          형식: [np.array([...]), np.array([...]), ...]
            trans_matrices: 각 가전의 전이확률 행렬 리스트
                           형식: [np.array([...]), np.array([...]), ...]
            stds: 각 가전의 상태별 표준편차 리스트 (상태별 적용)
                  형식: [np.array([...]), np.array([...]), ...]
            std: 관측 노이즈 표준편차 (sigma) - stds가 없을 때 일괄 적용
        """
        self.app_num = app_num
        self.means = means
        self.initial_probs = initial_probs
        self.trans_matrices = trans_matrices
        self.stds = stds if stds is not None else [np.array([std] * len(m)) for m in means]
        
        # 확률은 미리 로그를 취해둔다. (매번 log 계산하면 느림)
        self.log_trans = [np.log(np.maximum(mat, 1e-10)) for mat in trans_matrices]
        self.log_initial = [np.log(np.maximum(p, 1e-10)) for p in initial_probs]
        
        # 상태별 분산 테이블 사전 계산 (성능 최적화)
        self.combo_stds_lookup = {}
        
        # 모든 조합 사전 계산 및 정렬
        self.sorted_power_values = []
        self.sorted_combinations = []
        self._precompute_combinations()
    
    # =========================================================================
    # 사전 준비 단계 (Pre-computation)
    # =========================================================================
    
    def _precompute_combinations(self):
        """
        모든 가전 상태의 조합을 만들고 합계 전력순으로 정렬합니다.
        이 과정은 모델 초기화 시 단 1회만 실행됩니다.
        """
        print("="*70)
        print("FHMM 모델 초기화")
        print("="*70)
        
        # 1. 각 가전의 가능한 상태 인덱스 생성
        state_indices = [range(len(m)) for m in self.means]
        
        # 2. 모든 조합 생성 (itertools.product 활용)
        all_combos = list(itertools.product(*state_indices))
        
        # 3. 각 조합의 전력 합계 계산
        temp_list = []
        for combo in all_combos:
            total_power = sum(self.means[i][state] for i, state in enumerate(combo))
            temp_list.append((total_power, combo))
        
        # 4. 전력 합계 기준으로 정렬 (이진 탐색용)
        temp_list.sort(key=lambda x: x[0])
        
        # 5. 상태별 표준편차 미리 계산 (방출 확률 계산 최적화)
        for power, combo in temp_list:
            stds_tuple = tuple(self.stds[i][combo[i]] for i in range(self.app_num))
            self.combo_stds_lookup[combo] = stds_tuple
        
        # 6. 저장
        self.sorted_combinations = temp_list
        self.sorted_power_values = [x[0] for x in temp_list]
        
        print(f"✓ {len(all_combos)}개 조합 계산 완료")
        print(f"✓ 전력값 범위: {self.sorted_power_values[0]:.1f}W ~ {self.sorted_power_values[-1]:.1f}W")
        print(f"✓ 상태별 표준편차 테이블 생성 완료\n")
    
    # =========================================================================
    # 헬퍼 함수들 (계산 로직)
    # =========================================================================
    
    def _get_candidates(self, observation, margin=50):
        """
        관측값 근처의 후보들을 이진 탐색으로 찾습니다.
        
        Args:
            observation: 현재 관측값
            margin: 관측값 주변 탐색 범위 (W)
            
        Returns:
            해당 범위의 (전력, 상태조합) 리스트
        """
        candidates_range_left = bisect_left(self.sorted_power_values, observation - margin)
        candidates_range_right = bisect_right(self.sorted_power_values, observation + margin)
        
        candidates = self.sorted_combinations[candidates_range_left:candidates_range_right]
        
        # 예외처리: 범위 내 후보가 없으면 가장 가까운 조합 추가
        if not candidates:
            idx = bisect_left(self.sorted_power_values, observation)
            if idx >= len(self.sorted_power_values):
                idx = len(self.sorted_power_values) - 1
            candidates = [self.sorted_combinations[idx]]
        
        return candidates
    
    def _calc_emission_log(self, observation, power_sum, state_combo):
        """
        방출 확률 (가우시안 로그 확률) 계산
        각 가전의 상태별 표준편차를 적용하여 계산
        log P(observation | state) ~ -(observation - power_sum)^2 / (2*Sigma_combined^2)
        
        Args:
            observation: 관측값
            power_sum: 상태 조합에 따른 예상 전력값
            state_combo: 상태 조합 (표준편차 조회용)
            
        Returns:
            로그 확률
        """
        diff = observation - power_sum
        
        # 상태별 표준편차를 이용한 결합 분산 계산
        # 독립 가우시안 변수의 합: Var(X1+X2+...) = Var(X1) + Var(X2) + ...
        stds_tuple = self.combo_stds_lookup[state_combo]
        var_sum = sum(std**2 for std in stds_tuple)
        var_term = 2 * var_sum
        
        return -(diff ** 2) / var_term
    
    def _calc_transition_log(self, prev_combo, curr_combo):
        """
        전이 확률 (로그) 계산
        log P(current | previous) = Sum of log P(각 기기의 전이)
        
        Args:
            prev_combo: 이전 상태 조합
            curr_combo: 현재 상태 조합
            
        Returns:
            로그 확률의 합
        """
        log_prob = 0.0
        for i in range(self.app_num):
            p_prev = prev_combo[i]
            p_curr = curr_combo[i]
            log_prob += self.log_trans[i][p_prev][p_curr]
        return log_prob
    
    # =========================================================================
    # 메인 알고리즘 (Viterbi Beam Search)
    # =========================================================================
    
    def decode(self, observations, beam_width=50, margin=150):
        """
        Viterbi Beam Search를 이용한 디코딩
        
        Args:
            observations: 관측 시계열 데이터
            beam_width: 빔 서치 폭 (각 시점에서 유지할 경로 수)
            margin: 관측값 주변 탐색 범위 (W)
            
        Returns:
            각 시점의 추론 상태 조합 리스트, 실패 시 None
        """
        T = len(observations)
        psi = [[] for _ in range(T)]  # 각 시점의 생존자 리스트
        
        # --- t=0 초기화 ---
        first_candidates = self._get_candidates(observations[0], margin)
        
        for power, state_comb in first_candidates:
            # 초기확률 + 방출확률
            init_score = 0
            for i in range(self.app_num):
                init_score += self.log_initial[i][state_comb[i]]
            
            emit_score = self._calc_emission_log(observations[0], power, state_comb)
            total_score = init_score + emit_score
            
            psi[0].append(BeamNode(state_comb, total_score, -1))
        
        # 상위 K개만 남기기
        psi[0].sort(key=lambda x: x.score, reverse=True)
        psi[0] = psi[0][:beam_width]
        
        # --- t=1 ~ T-1 루프 (Forward Pass) ---
        for t in range(1, T):
            obs = observations[t]
            
            # 1. 이번 턴의 후보들 찾기
            current_candidates = self._get_candidates(obs, margin)
            
            # 2. 저번 턴의 생존자들 (K개)
            prev_survivors = psi[t-1]
            
            next_survivors = []
            
            # 3. 모든 후보와 생존자 조합 검증 (M x K)
            for cand_power, cand_state in current_candidates:
                
                # 방출 확률 미리 계산
                emission_score = self._calc_emission_log(obs, cand_power, cand_state)
                
                # 가장 좋은 부모 찾기
                max_score_for_this_cand = -np.inf
                best_parent_idx = -1
                
                for p_idx, parent_node in enumerate(prev_survivors):
                    # 전이 확률 계산
                    trans_score = self._calc_transition_log(parent_node.state, cand_state)
                    
                    # 점수 합산: 부모누적점수 + 전이 + 방출
                    score = parent_node.score + trans_score + emission_score
                    
                    # 최고 점수 갱신
                    if score > max_score_for_this_cand:
                        max_score_for_this_cand = score
                        best_parent_idx = p_idx
                
                # 유효한 부모가 있으면 생존자 후보에 추가
                if best_parent_idx != -1:
                    next_survivors.append(BeamNode(cand_state, max_score_for_this_cand, best_parent_idx))
            
            # 4. 상위 K개 선택 (Pruning)
            next_survivors.sort(key=lambda x: x.score, reverse=True)
            psi[t] = next_survivors[:beam_width]
            
            # 생존자가 없으면 알고리즘 실패
            if not psi[t]:
                print(f"⚠️  Warning: No survivors at t={t}. Increase margin or beam_width.")
                return None
        
        # --- 역추적 (Backward Pass) ---
        best_path = []
        
        if not psi[T-1]:
            return None
        
        # 마지막 시점의 최고 점수 노드에서 시작
        curr_node = psi[T-1][0]
        best_path.append(curr_node.state)
        
        # 부모를 따라가며 역추적
        for t in range(T-1, 0, -1):
            parent_idx = curr_node.parent_idx
            curr_node = psi[t-1][parent_idx]
            best_path.append(curr_node.state)
        
        # 역순이므로 뒤집기
        best_path.reverse()
        
        return best_path


def load_fhmm_from_learner(learner):
    """
    GMMParameterLearner에서 학습한 파라미터로 FHMM 모델 생성
    
    Args:
        learner: GMMParameterLearner 인스턴스
        
    Returns:
        FHMM 모델 인스턴스
    """
    # ⚠️ CRITICAL: a_data_preprocessing.py의 순서와 일치시켜야 함
    # a_data_preprocessing.py: ['CDE', 'CWE', 'DWE', 'WOE', 'HPE']
    appliances = ['CDE', 'CWE', 'DWE', 'WOE', 'HPE']
    
    means = [learner.gmm_params[code]['means'][:, 0].tolist() for code in appliances]
    initial_probs = [learner.gmm_params[code]['weights'] for code in appliances]
    trans_matrices = [learner.trans_matrices[code] for code in appliances]
    stds = [learner.gmm_params[code]['stds'] for code in appliances]  # ⭐ 상태별 std 추가
    
    model = FHMM(
        app_num=len(appliances),
        means=means,
        initial_probs=initial_probs,
        trans_matrices=trans_matrices,
        stds=stds  # ⭐ stds 파라미터 전달
    )
    
    return model


def main():
    """메인 실행 함수"""
    print("\n" + "="*70)
    print("4단계: FHMM 모델 생성")
    print("="*70 + "\n")
    
    from c_gmm_parameter_learning import GMMParameterLearner
    
    # 파라미터 학습
    learner = GMMParameterLearner()
    learner.load_and_separate_data()
    learner.learn_parameters()
    learner.estimate_transition_matrices()
    
    # FHMM 모델 생성
    model = load_fhmm_from_learner(learner)
    
    return model


if __name__ == "__main__":
    model = main()
