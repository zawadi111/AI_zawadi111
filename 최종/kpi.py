from config import global_variable
from typing import Dict


def information():
    # 출력 형식은 그대로 유지
    print("=== 생산 요약 ===")
    print(f"Stocker K | Out제품 수: {global_variable.FEED_COUNT}")
    print(f"  - ProdA: {global_variable.FEED_COUNT_A}")
    print(f"  - ProdB: {global_variable.FEED_COUNT_B}")
    print("A 총 보관 수:", len(global_variable.STOCKERS["STK-01"].list_jobs_A()))
    print("B 총 보관 수:", len(global_variable.STOCKERS["STK-01"].list_jobs_B()))


def profit(amr_count: int, machine_counts: Dict[str, int]) -> None:
    """
    Profit = ( [100 * Min(OutputA, OutputB)] - 5 * [InputA + InputB] )
             / ( fac + 0.011 * AMR수 ) * 100000
      - fac = Σ(stage별 설비 수 × stage별 계수)
      - 계수: A=4, B=9, C=8, D=8, E=5.5
    """

    # 1) 설비 투자비 계수
    parameter = {
        "A": 4.0,
        "B": 9.0,
        "C": 8.0,
        "D": 8.0,
        "E": 5.5,
    }

    # fac = Σ(stage별 설비 수 × stage별 계수)
    t = 0.0
    for stage, count in machine_counts.items():
        # 정의되지 않은 stage에 대해서는 0.0으로 처리
        coef = parameter.get(stage, 0.0)
        t += count * coef

    # 2) 산출 / 투입 개수
    stk = global_variable.STOCKERS.get("STK-01")
    if stk is None:
        print("profit: Stocker(STK-01)가 없어 Profit을 계산할 수 없습니다.")
        return

    output_A = len(stk.list_jobs_A())
    output_B = len(stk.list_jobs_B())

    input_A = global_variable.FEED_COUNT_A
    input_B = global_variable.FEED_COUNT_B

    # 3) 수익/비용 부분 (분자)
    p = (
        100.0 * min(output_A, output_B)
        - 5.0 * (input_A + input_B)
    )

    # 4) 분모: fac + 0.011 * AMR수
    denominator = t + 0.011 * amr_count
    if denominator <= 0:
        print("profit: 분모(fac + 0.011*AMR 수)가 0 이하라 Profit을 계산할 수 없습니다.")
        return

    raw_profit = p / denominator
    scaled_profit = round(raw_profit, 2) * 100000.0

    # 최종 출력 형식
    print("profit:", f"{scaled_profit:,.2f}", "원")


# ===== 여기부터 AMR Utilization 관련 =====

def _get_sim_time():
    """실제 사용 시뮬레이션 시간(설정시간 vs 실제 종료시간 중 작은 값)"""
    cfg_time = (
        global_variable.SIM_END
        if global_variable.SIM_END != float("inf")
        else global_variable.CURRENT_CFG.sim_time
    )
    # run()이 global_variable.now 를 마지막 이벤트 시각으로 유지한다고 가정
    now_time = getattr(global_variable, "now", cfg_time)
    return min(cfg_time, now_time)


def calculate_amr_utilization() -> Dict[str, float]:
    if not global_variable.CURRENT_CFG:
        print("Warning: Simulation configuration is not valid.")
        return {}

    sim_time = _get_sim_time()
    if sim_time <= 0:
        print("Warning: Simulation time is not valid.")
        return {}

    utilizations: Dict[str, float] = {}

    for amr_name in global_variable.amr_runs.keys():
        working_time = 0.0

        # 적재 상태로 이동한 시간
        runs = global_variable.amr_runs.get(amr_name, [])
        for s, e, job_id, frm, to, loaded in runs:
            if loaded:
                working_time += (e - s)

        # 로드/언로드 동안 대기(실제 작업 시간)
        waits = global_variable.amr_waits.get(amr_name, [])
        for arrive_time, depart_time, job_id, wait_type, xy in waits:
            working_time += (depart_time - arrive_time)

        utilization = working_time / sim_time
        utilizations[amr_name] = utilization

    return utilizations


def print_amr_utilization():
    utilizations = calculate_amr_utilization()

    if not utilizations:
        print("AMR utilization no data.")
        return

    print("\n=== AMR Utilization ===")

    for amr_name in sorted(utilizations.keys()):
        util = utilizations[amr_name]
        util_percent = util * 100
        print(f"{amr_name}: {util_percent:.2f}% ({util:.4f})")

    avg_util = sum(utilizations.values()) / len(utilizations)
    avg_percent = avg_util * 100
    print(f"\nAverage Utilization: {avg_percent:.2f}% ({avg_util:.4f})")

    print("\n=== Detail Information ===")
    sim_time = _get_sim_time()

    for amr_name in sorted(utilizations.keys()):
        runs = global_variable.amr_runs.get(amr_name, [])
        waits = global_variable.amr_waits.get(amr_name, [])

        loaded_move_time = sum(e - s for s, e, _, _, _, loaded in runs if loaded)
        empty_move_time = sum(e - s for s, e, _, _, _, loaded in runs if not loaded)
        load_unload_time = sum(depart - arrive for arrive, depart, _, _, _ in waits)

        total_working_time = loaded_move_time + load_unload_time

        print(f"\n{amr_name}:")
        print(f"  Total Simulation Time: {sim_time:.2f} seconds")
        print(f"  Loaded Move Time: {loaded_move_time:.2f} seconds")
        print(f"  Empty Move Time: {empty_move_time:.2f} seconds")
        print(f"  Load/Unload Time: {load_unload_time:.2f} seconds")
        print(f"  Total Working Time: {total_working_time:.2f} seconds")
        print(f"  Utilization: {utilizations[amr_name] * 100:.2f}%")
        print(f"  Working Count: {len([r for r in runs if r[5]])} times (Loaded Move)")


# ============================================
# 공정(Stage) 별 설비 가동률 / 병목 확인용 KPI
# ============================================

def calc_stage_utilization():
    """
    공정(stage)별 설비 가동률을 계산한다.

    반환값:
      stage_info: 각 stage 별 집계 정보 딕셔너리
      machine_busy: 설비별 busy time (초 단위) 딕셔너리
      sim_time: 시뮬레이션 총 시간(초)
    """
    # 1) 시뮬레이션 총 시간 가져오기
    if global_variable.SIM_END != float("inf"):
        sim_time = global_variable.SIM_END
    elif global_variable.CURRENT_CFG is not None:
        sim_time = global_variable.CURRENT_CFG.sim_time
    else:
        print("[StageKPI] 시뮬레이션 시간이 설정되지 않았습니다.")
        return {}, {}, 0.0

    if sim_time <= 0:
        print("[StageKPI] sim_time 이 0 이하입니다.")
        return {}, {}, sim_time

    # 2) 설비별 busy time 집계 (machine_runs 기반)
    #    machine_runs: { "A-01": [(start, end, job_id, stage), ...], ... }
    machine_busy = {}

    for mname, runs in global_variable.machine_runs.items():
        busy = 0.0
        for s, e, job_id, stage in runs:
            busy += (e - s)
        machine_busy[mname] = busy

    # 3) 한 번도 돌지 않은 설비도 0초로 채워넣기
    for stage, machines in global_variable.MACHINES.items():
        for m in machines:
            if m.name not in machine_busy:
                machine_busy[m.name] = 0.0

    # 4) stage 별 집계
    stage_info = {}
    for stage, machines in global_variable.MACHINES.items():
        n = len(machines)
        if n == 0:
            continue

        total_busy = 0.0
        for m in machines:
            total_busy += machine_busy.get(m.name, 0.0)

        # 전체 용량 = (설비 수) × (시뮬레이션 시간)
        capacity = sim_time * n
        util = total_busy / capacity if capacity > 0 else 0.0  # 0~1 사이 값
        avg_busy = total_busy / n

        stage_info[stage] = {
            "machine_count": n,
            "total_busy": total_busy,
            "avg_busy": avg_busy,
            "util": util,
        }

    return stage_info, machine_busy, sim_time


def print_stage_bottleneck():
    """
    공정별 가동률 요약 + 병목 공정 출력.
    - 각 Stage(A~E)의 평균 가동률
    - 가장 가동률이 높은 Stage를 병목 후보로 표시
    - 그 Stage 안의 설비별 가동률도 출력
    """
    stage_info, machine_busy, sim_time = calc_stage_utilization()

    if not stage_info:
        print("\n[StageKPI] 집계할 stage 정보가 없습니다.")
        return

    # 1) 공정별 평균 가동률 출력
    print("\n=== 공정(Stage) 별 평균 가동률 ===")
    for stage in sorted(stage_info.keys()):
        info = stage_info[stage]
        util_pct = info["util"] * 100.0
        print(
            f"Stage {stage}: "
            f"설비수={info['machine_count']}, "
            f"평균 가동률={util_pct:5.2f}%, "
            f"설비당 평균 Busy Time={info['avg_busy']:.1f}s"
        )

    # 2) 평균 가동률이 가장 높은 Stage = 병목 후보
    bottleneck_stage = max(stage_info.keys(), key=lambda s: stage_info[s]["util"])
    bn_info = stage_info[bottleneck_stage]

    print("\n=== 병목 공정(Bottleneck Stage) 후보 ===")
    print(
        f"Stage {bottleneck_stage}: "
        f"평균 가동률={bn_info['util']*100:.2f}% "
        f"(설비수={bn_info['machine_count']})"
    )

    # 3) 그 Stage 안 설비별 가동률 상세
    print(f"\n[Stage {bottleneck_stage}] 설비별 가동률")
    for m in global_variable.MACHINES.get(bottleneck_stage, []):
        busy = machine_busy.get(m.name, 0.0)
        util = busy / sim_time if sim_time > 0 else 0.0
        print(
            f"  - {m.name}: "
            f"{util*100:5.2f}% (busy={busy:.1f}s / total={sim_time:.1f}s)"
        )