from typing import Optional,Callable,Tuple
import heapq, random
from data_structures import AMR,Machine,Job,Stocker,FactoryConfig,Warehouse,GlobalVariable

global_variable  = GlobalVariable()

def reset_sim():
    '''시나리오 재실행을 위한 전역 상태 완전 초기화'''
    global_variable.reset()

def load_time_for(stage: str) -> float:
    '''amr load time 반환'''
    cfg = global_variable.CURRENT_CFG
    if cfg is None:
        return global_variable.DEFAULT_AMR_LOAD
    base = getattr(cfg, "amr_load_time", global_variable.DEFAULT_AMR_LOAD)
    stage_map = getattr(cfg, "amr_load_time_by_stage", None)
    if stage_map and stage in stage_map:
        return stage_map[stage]
    return base

def unload_time_for(stage_or_stk: str) -> float:
    ''''amr unload time 반환'''
    cfg = global_variable.CURRENT_CFG
    if cfg is None:
        return global_variable.DEFAULT_AMR_UNLOAD
    base = getattr(cfg, "amr_unload_time", global_variable.DEFAULT_AMR_UNLOAD)
    stage_map = getattr(cfg, "amr_unload_time_by_stage", None)
    if stage_map and stage_or_stk in stage_map:
        return stage_map[stage_or_stk]
    return base

def next_stage_for(job: Job, current_stage: str) -> Optional[str]:
    if getattr(job, "pending_stage", None):
        return job.pending_stage
    try:
        i = global_variable.ROUTE.index(current_stage)
        
    except ValueError:
        return None
    
    if i + 1 < len(global_variable.ROUTE):
        job.pending_stage = global_variable.ROUTE[i + 1]
        return job.pending_stage
    
    if job.cycle_idx + 1 < job.max_cycles:
        job.cycle_idx += 1
        job.pending_stage = global_variable.ROUTE[0]
        return job.pending_stage
    
    else:
        return None
    
def process_time_for(stage: str, job: Job, m: Optional[Machine] = None) -> float:
    '''주어진 제품(job)의 사이클(cycle)과 스테이지(stage)에 맞는 공정시간(process time)을 설정값에서 찾아 반환하는 함수'''
    cfg = global_variable.CURRENT_CFG
    try:
        if cfg and getattr(cfg, "process_times_by_product_cycle", None):
            per_cycle = cfg.process_times_by_product_cycle.get(job.product, {})
            if job.cycle_idx in per_cycle and stage in per_cycle[job.cycle_idx]:
                return per_cycle[job.cycle_idx][stage]
            if per_cycle:
                cand = [(k, d[stage]) for k, d in per_cycle.items()
                        if k <= job.cycle_idx and stage in d]
                if cand:
                    cand.sort(key=lambda x: x[0], reverse=True)
                    return cand[0][1]
    except:
        print("ERROR") 



def build_factory(cfg: FactoryConfig):
    ''' 공장 초기 설정 생성 '''
    random.seed(cfg.seed)
    global_variable.MACHINES.clear()
    for stage in global_variable.ROUTE:
        n = cfg.machine_counts.get(stage, 0)
        pt = cfg.process_times.get(stage, 1.0)
        
        if not cfg.machine_positions or stage not in cfg.machine_positions:
            raise ValueError(f"[build_factory] '{stage}' 스테이지의 수동 좌표가 누락되었습니다. (필요 개수: {n})")
        manual_positions = cfg.machine_positions[stage]
        if len(manual_positions) < n:
            raise ValueError(
                f"[build_factory] '{stage}' 스테이지의 수동 좌표가 부족합니다. "
                f"필요: {n}, 제공: {len(manual_positions)}"
            )
        elif len(manual_positions) > n:
            log(f"[build_factory] '{stage}': 좌표 {len(manual_positions)}개 중 앞의 {n}개만 사용합니다.")
            manual_positions = manual_positions[:n]

        for i, xy in enumerate(manual_positions):
            if not isinstance(xy, tuple) or len(xy) != 2:
                raise ValueError(f"[build_factory] '{stage}' 좌표 #{i+1} 형식 오류: {xy} (예: (x, y))")
        if len(set(manual_positions)) != len(manual_positions):
            raise ValueError(f"[build_factory] '{stage}' 수동 좌표에 중복이 있습니다: {manual_positions}")

        manual_names = []
        if cfg.machine_names and stage in cfg.machine_names:
            manual_names = cfg.machine_names[stage]
            if len(manual_names) < n:
                raise ValueError(
                    f"[build_factory] '{stage}' 설비 이름이 부족합니다. 필요: {n}, 제공: {len(manual_names)}"
                )
            elif len(manual_names) > n:
                log(f"[build_factory] '{stage}': 이름 {len(manual_names)}개 중 앞의 {n}개만 사용합니다.")
                manual_names = manual_names[:n]

        global_variable.MACHINES[stage] = []
        for idx in range(n):
            mname = manual_names[idx] if manual_names else f"{stage}-{idx+1}"
            xy = manual_positions[idx]
            global_variable.MACHINES[stage].append(Machine(mname, stage, xy, pt))
                        
    global_variable.AMRS.clear()
    
    if cfg.amr_positions and len(cfg.amr_positions) >= cfg.amr_count:
        positions = cfg.amr_positions[:cfg.amr_count]
    else:
        positions = [(4.0, i * 0.1) for i in range(cfg.amr_count)]

    for i in range(cfg.amr_count):
        global_variable.AMRS.append(AMR(f"AMR-{i+1:02d}", positions[i], cfg.amr_speed))

    if len(global_variable.AMRS) == 0:
        log("AMR가 0대입니다. 이 상태에선 공정 간 이송이 불가합니다.")
    
    # 원자재 창고와 stocker 생성
    global_variable.STOCKERS["STK-01"] = Stocker(name="STK-01", xy=cfg.stocker_xy) 
    global_variable.WAREHOUSE = Warehouse(name="WH-01", xy=cfg.warehouse_xy)       


def log(msg: str):
    '''시간별 mmsg 출력'''
    print(f"[t={global_variable.now:6.2f}s] {msg}")
    

def record_machine_run(m: Machine, job: Job, s: float, e: float):
    '''설비 기록 저장'''
    global_variable.machine_runs.setdefault(m.name, []).append((s, e, job.job_id, m.stage))
    global_variable.job_runs.setdefault(job.job_id, []).append((m.stage, s, e, m.name))

def record_amr_run(a: AMR, job: Job, s: float, e: float,frm: Tuple[float,float], to: Tuple[float,float],loaded: bool):
    '''AMR 기록 저장'''
    global_variable.amr_runs.setdefault(a.name, []).append((s, e, job.job_id, frm, to, loaded))

def schedule(at: float, fn: Callable[[], None]):
    '''스케쥴 신청'''
    global_variable._seq += 1
    heapq.heappush(global_variable.pq, (at, global_variable._seq, fn))

def run():
    '''t=0 -> t=END 실행 함수'''
    while global_variable.pq:
        at, _, fn = heapq.heappop(global_variable.pq)
        if at > global_variable.SIM_END:
            global_variable.now = global_variable.SIM_END
            break
        global_variable.now = at
        fn()