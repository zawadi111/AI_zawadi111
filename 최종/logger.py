from typing import Optional
from data_structures import *
      
def _amr_push_task(amr: AMR, *, job_id: Optional[str], pick_xy, drop_xy,
                   depart_at, arrive_pick, depart_pick, arrive_drop, depart_drop):
    """reserve_amr 직후 예약 타임라인을 AMR에 가시적으로 적재"""
    amr.tasks.append({
        "job_id": job_id,
        "pick_xy": pick_xy,
        "drop_xy": drop_xy,
        "depart_at": depart_at,
        "arrive_pick": arrive_pick,
        "depart_pick": depart_pick,
        "arrive_drop": arrive_drop,
        "depart_drop": depart_drop,
    })
    

def _amr_pop_task(amr: AMR, *, job_id: Optional[str], depart_drop: float, tol: float = 1e-9):
    """드롭 완료 시점에 해당 예약 1건 제거 (job_id가 있으면 우선 매칭, 없으면 시간으로 매칭)"""
    idx = -1
    if job_id is not None:
        for i, t in enumerate(amr.tasks):
            if t.get("job_id") == job_id and abs(t.get("depart_drop", -1) - depart_drop) < tol:
                idx = i
                break
    if idx < 0:  # fallback: depart_drop만으로 매칭
        for i, t in enumerate(amr.tasks):
            if abs(t.get("depart_drop", -1) - depart_drop) < tol:
                idx = i
                break
    if idx >= 0:
        amr.tasks.pop(idx)
    
        