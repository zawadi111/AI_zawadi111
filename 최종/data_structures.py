from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Any

# @dataclass -> class 선언 
@dataclass
class Job:
    '''Job 정보'''
    job_id: str
    product: str
    reserved: bool = False    
    in_transit: bool = False  
    cycle_idx: int = 0        
    max_cycles: int = 1
    pending_stage: Optional[str] = None      

@dataclass
class Warehouse:
    '''warehouse 정보'''
    name: str
    xy: Tuple[float, float]
    inventory: List[Job] = field(default_factory=list)

    def put(self, job: Job):
        self.inventory.append(job)
        print(f"{self.name}: {job.job_id} 입고 (재고 {len(self.inventory)}개)")

    def pop(self) -> Optional[Job]:
        if not self.inventory:
            return None
        job = self.inventory.pop(0)
        print(f"{self.name}: {job.job_id} 출고 (재고 {len(self.inventory)}개)")
        return job
    
@dataclass
class Machine:
    '''Machine 정보'''
    name: str
    stage: str
    xy: Tuple[float, float]        
    process_time: float

    input_buf: List["Job"] = field(default_factory=list) 
    input_capacity: int = 1
    input_reserved: bool = False 
    
    processing_job: Optional["Job"] = None
    output_buf: Optional["Job"] = None
    waiting_done: Optional["Job"] = None

    # 포트 관련
    port_offset: int = 2         # 포트 간 거리 2 
    input_port: Tuple[float, float] = field(init=False)
    output_port: Tuple[float, float] = field(init=False)

    def __post_init__(self):
        """생성 시 자동으로 input/output 포트 설정"""
        x, y = self.xy
        self.input_port = (x - self.port_offset, y)
        self.output_port = (x + self.port_offset, y)

    

@dataclass
class AMR:
    '''AMR'''
    name: str
    xy: Tuple[float, float]
    speed: float
    free_time: float = 0.0
    planned_xy: Optional[Tuple[float, float]] = None
    tasks: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class Stocker: 
    '''stocker'''
    name: str
    xy: Tuple[float, float] # LOCATION
    stored_jobs_A: List[str] = field(default_factory=list)
    stored_jobs_B: List[str] = field(default_factory=list)
    
    def store(self, job_id: str):
        """제품을 보관 리스트에 추가"""
        print("Store에 저장되는 제품",job_id) # Store에 저장되는 제품 ProdA-0005
        productType = job_id.split("-")[0]
        if "A" in productType:
            self.stored_jobs_A.append(job_id)
        elif "B" in productType:
            self.stored_jobs_B.append(job_id)
        else:
            print("ERROR")
            
        print(f"보관 중 ProductA (총 {len(self.stored_jobs_A)}개)")
        print(f"보관 중 ProductB (총 {len(self.stored_jobs_B)}개)")

    def list_jobs_A(self):
        """현재 보관 중인 제품 목록"""
        return self.stored_jobs_A.copy()
    
    def list_jobs_B(self):
        """현재 보관 중인 제품 목록"""
        return self.stored_jobs_B.copy()

@dataclass
class FactoryConfig: # Factory 설정
    warehouse_xy: Tuple[float, float] = (4,10) # 고정
    stocker_xy : Tuple[float,float] = (56,10)  # 고정
    amr_speed: float = 1.0                     # 고정
    amr_positions: Optional[List[Tuple[float,float]]] = None  
    job_cycles: int = 2                        # 고정
    sim_time: float = 1296000                   # 시뮬레이션 시간, 입력 값
    feed_sequence: Tuple[str, ...] = ("ProdA","ProdB") 
    
    machine_counts: Dict[str, int] = field(default_factory=lambda: {"A":1,"B":1,"C":1,"D":1,"E":1})
    process_times: Dict[str, float] = field(default_factory=lambda: {"A":1,"B":1,"C":1,"D":1,"E":1})
    # 고정
    process_times_by_product_cycle: Dict[str, Dict[int, Dict[str, float]]] = field(
        default_factory=lambda: {
            "ProdA": {
                0: {"A": 15*60, "B": 15*60, "C": 15*60, "D": 15*60, "E": 15*60},
                1: {"A": 15*60, "B": 15*60, "C": 15*60, "D": 15*60, "E": 15*60},
            },
            "ProdB": {
                0: {"A": 5*60, "B": 40*60, "C": 25*60, "D": 2*60, "E": 5*60},
                1: {"A": 10*60, "B": 10*60, "C": 5*60, "D": 10*60, "E": 15*60},
            },
        }
    )
    machine_positions: Optional[Dict[str, List[Tuple[float, float]]]] = None
    machine_names: Optional[Dict[str, List[str]]] = None
    jobs_by_product: Dict[str, int] = field(default_factory=dict)
    shuffle_product_feeds: bool = True
    seed: int = 42
    amr_load_time: float = 10     # 고정
    amr_unload_time: float = 10   # 고정
    amr_count : int = 1             
    
    
class GlobalVariable:
    def __init__(self):
        self.init_all()

    # config.py 내부의 init_all 함수 수정
    def init_all(self):
        """모든 전역 변수 초기화"""
        self.now = 0.0
        self._seq = 0
        self.pq = []

        self.ROUTE = ["A", "B", "C", "D", "E"] # 공정 순서

        self.MACHINES: Dict[str, List[Machine]] = {}
        self.AMRS: List[AMR] = []
        self.STAGE_Q: Dict[str, List[Job]] = {s: [] for s in self.ROUTE}
        self.STOCKERS: Dict[str, Stocker] = {}
        self.WAREHOUSE: Optional[Warehouse] = None

        self.CURRENT_CFG: Optional[FactoryConfig] = None
        self.SIM_END: float = float("inf")

        self.ROUND_ROBIN_IDX: Dict[str, int] = {s: 0 for s in self.ROUTE}
        self.DEFAULT_AMR_LOAD = 10
        self.DEFAULT_AMR_UNLOAD = 10

        self.FEED_SEQ: List[str] = []
        self.FEED_IDX: int = 0
        self.FEED_COUNT = 0
        self.FEED_COUNT_A = 0
        self.FEED_COUNT_B = 0
        
        # [추가됨] WIP 제한 (기본값 무한대)
        self.WIP_LIMIT = float('inf')

        self.machine_runs: Dict[str, List[Tuple[float, float, str, str]]] = {}
        self.job_runs: Dict[str, List[Tuple[str, float, float, str]]] = {}
        self.amr_runs: Dict[str, List[Tuple[float, float, str, Tuple[float,float], Tuple[float,float], bool]]] = {}
        self.amr_waits: Dict[str, List[Tuple[float, float, str, str, Tuple[float,float]]]] = {}

    def reset(self):
        """시나리오 재실행 시 완전 초기화"""
        self.init_all()



PREV_OF = {
    "WH": ["A"],        
    "A":  ["E", "WH"],   
    "B": ["A"],
    "C": ["B"],
    "D": ["C"],
    "E": ["D"],
}