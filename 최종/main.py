import sim_core
from sim_core import FactoryConfig
import kpi
from config import global_variable
from visualization import animate_from_amr_runs
import random

# A(산화), B(노광), C(식각), D(증착), E(계측)


if __name__ == "__main__":
    # ==========================================
    # 1. 최적화된 파라미터 설정
    # ==========================================
    # 설비 조합 (Total 29대)
    machine_counts = {
        "A": 5,
        "B": 8,
        "C": 6,
        "D": 5,
        "E": 5,
    }
    
    amr_count = 9        # 최적 AMR 개수
    wip_limit = 51         # 최적 WIP Limit
    dispatch_policy = "eta"  # 최적 운영 정책
    
    sim_time = 1296000.0   # 15일
    seed = 42              # 결과 재현을 위한 고정 시드

    # ==========================================
    # 2. 설비 좌표: 직접 지정한 배치 사용
    # ==========================================
    machine_positions = {
        "A": [
            (14,17), (14,15), (14,13), (14,7), (14,5),
        ],
        "B": [
            (22,17), (22,15), (22,13), (22,7), (22,5), (22,3),
            (30,17), (30,15),
        ],
        "C": [
            (30,13), (30,7), (30,5), (30,3),
            (38,17), (38,15),
        ],
        "D": [
            (38,13), (38,7), (38,5), (38,3),
            (46,17),
        ],
        "E": [
            (46,15), (46,13), (46,7), (46,5), (46,3),
        ],
    }
    # 사용하지 않는 슬롯: (14,3)

    # 좌표 개수와 설비 개수 일치 여부 간단 체크 (선택 사항)
    for stage, count in machine_counts.items():
        if stage not in machine_positions:
            raise ValueError(f"{stage} 공정의 좌표가 machine_positions에 없습니다.")
        if len(machine_positions[stage]) != count:
            raise ValueError(
                f"{stage} 공정: machine_counts({count})와 좌표 개수({len(machine_positions[stage])})가 다릅니다."
            )

    # ==========================================
    # 3. 시뮬레이션 설정 및 실행
    # ==========================================
    print(f"=== 최적화 시뮬레이션 시작 ===")
    print(f"설비: {machine_counts} (총 {sum(machine_counts.values())}대)")
    print(f"AMR: {amr_count}대")
    print(f"WIP Limit: {wip_limit}")
    print(f"Policy: {dispatch_policy}")

    cfg = FactoryConfig(
        sim_time=sim_time,
        seed=seed,
        feed_sequence=("ProdA", "ProdB"),
        machine_counts=machine_counts,
        machine_positions=machine_positions,  # ← 여기!
        amr_count=amr_count,
    )

    # [중요] WIP Limit과 Policy 적용을 위해 수동 초기화 수행
    sim_core.reset_sim()
    global_variable.CURRENT_CFG = cfg
    global_variable.SIM_END = float(cfg.sim_time)
    
    # *** 핵심 파라미터 주입 ***
    global_variable.WIP_LIMIT = wip_limit
    global_variable.DISPATCH_POLICY = dispatch_policy

    sim_core.build_factory(cfg)
    random.seed(cfg.seed)
    global_variable.FEED_SEQ = list(cfg.feed_sequence)
    global_variable.FEED_IDX = 0
    
    # 실행
    sim_core.schedule(0.0, sim_core.bootstrap_start)
    sim_core.run()

    # ==========================================
    # 4. 결과 출력 (KPI & Visualization)
    # ==========================================
    kpi.information()
    kpi.profit(amr_count=amr_count, machine_counts=machine_counts)
    kpi.print_amr_utilization()
    kpi.print_stage_bottleneck()

    # 애니메이션 실행
    animate_from_amr_runs(
        global_variable.amr_runs,
        interval_ms=50,  # 속도 조절
        frames=1296000,     # 프레임 수 조절
        trail=True,
        machine_positions=machine_positions,  # ← 여기도 새 배치 사용
    )