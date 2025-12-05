import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from dataclasses import dataclass
from typing import Dict, List, Tuple, Set
from config import global_variable
from sim_core import _build_obstacles
import math
import heapq  

Grid = Tuple[int, int]  

MACHINE_COLORS = {
    "A": "#f4cccc",  
    "B": "#c9daf8",  
    "C": "#d9ead3", 
    "D": "#fff2cc",  
    "E": "#ead1dc", 
}
@dataclass(frozen=True)
class Bounds:
    min_x: int; max_x: int
    min_y: int; max_y: int

EPS = 1e-6

def _segment_state(runs, t: float):
    in_seg = False
    last_e = -np.inf
    last_ld = False
    for s, e, *_rest, ld in runs:
        s = float(s); e = float(e)
        if s - EPS <= t <= e + EPS:
            in_seg = True
        if e <= t + EPS and e > last_e:
            last_e = e
            last_ld = bool(ld)
    return in_seg, last_e, last_ld


def build_blocks_by_unit(machine_positions: Dict[str, List[Tuple[int,int]]], radius: int = 1):
    units = []
    for group, centers in machine_positions.items():
        color = MACHINE_COLORS.get(group, "mistyrose")
        for i, (x, y) in enumerate(centers, start=1):
            cells = set()
            for xx in range(x - 2, x + 2):
                for yy in range(y - 1, y + 1):
                    cells.add((xx, yy))
            units.append({
                "group": group,
                "idx": i,
                "center": (x, y),
                "cells": cells,
                "color": color,
            })
    return units


def infer_bounds_from_runs_and_blocks(amr_runs, blocks: Set[Grid], margin: int = 2) -> Bounds:
    xs, ys = set(), set()
    for runs in amr_runs.values():
        for s, e, job_id, frm, to, loaded in runs:
            xs.update([int(frm[0]), int(to[0])])
            ys.update([int(frm[1]), int(to[1])])
    for (x, y) in blocks:
        xs.add(int(x)); ys.add(int(y))
    if not xs: xs = {0}
    if not ys: ys = {0}
    return Bounds(min(xs)-margin, max(xs)+margin, min(ys)-margin, max(ys)+margin)

def _unify_timeline_from_runs(amr_runs, frames=300):
    t0, t1 = float("inf"), float("-inf")
    for runs in amr_runs.values():
        for s, e, *_ in runs:
            t0 = min(t0, float(s))
            t1 = max(t1, float(e))
    if not np.isfinite(t0) or not np.isfinite(t1) or t1 <= t0:
        return np.array([])
    return np.linspace(t0, t1, num=frames)
def draw_blocks(ax, group_blocks):
    for group, blocks in group_blocks.items():
        for (x, y), color in blocks:
            ax.fill_between([x, x + 1], y, y + 1, color=color, edgecolor='k', linewidth=0.2)

def _get_dijkstra_path(start_xy, goal_xy):
    """다익스트라 알고리즘으로 경로를 반환 (시각화용)"""
    MIN_X, MAX_X = 0, 60
    MIN_Y, MAX_Y = 0, 20
    
    sx, sy = int(round(start_xy[0])), int(round(start_xy[1]))
    gx, gy = int(round(goal_xy[0])), int(round(goal_xy[1]))
    start = (sx, sy)
    goal = (gx, gy)
    
    if start == goal:
        return [start_xy, goal_xy]
    
    # 직선 경로가 장애물과 충돌하지 않으면 직선 사용
    blocked = _build_obstacles()
    path_clear = True
    steps = max(abs(gx - sx), abs(gy - sy), 1)
    for i in range(0, steps + 1, max(1, steps // 10)):
        t = i / steps if steps > 0 else 0
        x = int(sx * (1-t) + gx * t)
        y = int(sy * (1-t) + gy * t)
        if (x, y) in blocked:
            path_clear = False
            break
    
    if path_clear:
        return [start_xy, goal_xy]
    
    # 다익스트라로 경로 계산
    blocked = set(blocked)
    blocked.discard(start)
    blocked.discard(goal)
    
    INF = float("inf")
    dist_map = {}
    prev_map = {}
    hq = []
    dist_map[start] = 0.0
    prev_map[start] = None
    heapq.heappush(hq, (0.0, start))
    
    while hq:
        d, (x, y) = heapq.heappop(hq)
        
        if d > dist_map.get((x, y), INF):
            continue
        
        if (x, y) == goal:
            # 경로 역추적
            path = []
            current = goal
            while current is not None:
                path.append((float(current[0]), float(current[1])))
                current = prev_map.get(current)
            path.reverse()
            return path if len(path) > 1 else [start_xy, goal_xy]
        
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            v = (nx, ny)
            
            if not (MIN_X <= nx <= MAX_X and MIN_Y <= ny <= MAX_Y):
                continue
            if v in blocked:
                continue
            
            nd = d + 1.0
            
            if nd < dist_map.get(v, INF):
                dist_map[v] = nd
                prev_map[v] = (x, y)
                heapq.heappush(hq, (nd, v))
    
    # 경로를 찾지 못하면 직선 반환
    return [start_xy, goal_xy]
            
def _pos_loaded_at_t(runs, t):
    last_xy = (0.0, 0.0); last_loaded = False
    for s, e, _job, xy0, xy1, loaded in runs:
        s, e = float(s), float(e)
        if t < s:
            return last_xy, last_loaded
        if s <= t <= e:
            if e == s:
                return xy1, loaded
            # 다익스트라 경로를 따라 이동
            path = _get_dijkstra_path(xy0, xy1)
            if len(path) <= 2:
                # 직선 경로면 기존 방식 사용
                r = (t - s) / (e - s)
                x = xy0[0] + (xy1[0] - xy0[0]) * r
                y = xy0[1] + (xy1[1] - xy0[1]) * r
                return (x, y), loaded
            else:
                # 경로를 따라 이동
                total_dist = sum(math.hypot(path[i+1][0] - path[i][0], path[i+1][1] - path[i][1]) 
                                for i in range(len(path)-1))
                if total_dist == 0:
                    return xy1, loaded
                r = (t - s) / (e - s)
                target_dist = total_dist * r
                current_dist = 0.0
                for i in range(len(path) - 1):
                    seg_dist = math.hypot(path[i+1][0] - path[i][0], path[i+1][1] - path[i][1])
                    if current_dist + seg_dist >= target_dist:
                        seg_r = (target_dist - current_dist) / seg_dist if seg_dist > 0 else 0
                        x = path[i][0] + (path[i+1][0] - path[i][0]) * seg_r
                        y = path[i][1] + (path[i+1][1] - path[i][1]) * seg_r
                        return (x, y), loaded
                    current_dist += seg_dist
                return xy1, loaded
        last_xy, last_loaded = xy1, loaded
    return last_xy, last_loaded

def infer_bounds_from_runs_blocks_and_points(amr_runs, blocks, extra_points=(), margin=2):
    xs, ys = set(), set()
    for runs in amr_runs.values():
        for _s, _e, _job, frm, to, _ld in runs:
            xs.update([int(frm[0]), int(to[0])])
            ys.update([int(frm[1]), int(to[1])])
    for (x, y) in blocks:
        xs.add(int(x)); ys.add(int(y))
    for (x, y) in extra_points:
        xs.add(int(x)); ys.add(int(y))
    if not xs: xs = {0}
    if not ys: ys = {0}
    return Bounds(min(xs)-margin, max(xs)+margin, min(ys)-margin, max(ys)+margin)

def animate_from_amr_runs(amr_runs,
                          interval_ms: int = 400,
                          frames: int = 400,
                          trail: bool = True,
                          machine_positions: Dict[str, List[Grid]] | None = None):
    timeline = _unify_timeline_from_runs(amr_runs, frames=frames)
    if timeline.size == 0:
        print("빈 타임라인입니다.")
        return
    X_MIN, X_MAX = 0, 60
    Y_MIN, Y_MAX = 0, 20


    wh_xy  = getattr(global_variable.WAREHOUSE, "xy", None)
    stk_xy = getattr(global_variable.STOCKERS.get("STK-01", None), "xy", None) if hasattr(global_variable, "STOCKERS") else None

    fig, ax = plt.subplots(figsize=(12, 6))

    for x in range(X_MIN, X_MAX + 1):
        ax.plot([x, x], [Y_MIN, Y_MAX], linewidth=0.3, color="lightgray", zorder=1)
    for y in range(Y_MIN, Y_MAX + 1):
        ax.plot([X_MIN, X_MAX], [y, y], linewidth=0.3, color="lightgray", zorder=1)


    units = build_blocks_by_unit(machine_positions)
    draw_units(ax, units)                               
    draw_unit_outlines(ax, units, lw=2.0, color='k')    
    draw_unit_seams(ax, units, lw=1.2, color='dimgray') 
    
    if wh_xy:
        ax.scatter([wh_xy[0] - 1], [wh_xy[1] ], s=140, marker='s',
                   edgecolors='k', facecolors='white', zorder=5, label="Warehouse")
        ax.text(wh_xy[0] - 1, wh_xy[1] + 1, "WH", ha='center', va='bottom', fontsize=9, zorder=6)

    if stk_xy:
        ax.scatter([stk_xy[0] + 1], [stk_xy[1]], s=160, marker='s',
                   edgecolors='k', facecolors='white', zorder=5, label="STK-01")
        ax.text(stk_xy[0] + 1, stk_xy[1] + 1, "STOCKER", ha='center', va='bottom', fontsize=9, zorder=6)

    ax.set_aspect('equal', 'box')
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_xlabel("X"); ax.set_ylabel("Y")
    ax.set_title("AMR Runs (Loaded=Red, Empty=Blue)")

 
    artists = {}
    for amr in amr_runs.keys():
        (line,) = ax.plot([], [], linewidth=2, label=amr, zorder=3)
        marker = ax.scatter([], [], s=80, zorder=4) 
        artists[amr] = {"line": line, "marker": marker, "trail_x": [], "trail_y": []}

    ax.legend(loc="upper left")

    def init():
        dr = []
        for a in artists.values():
            a["line"].set_data([], [])
            a["marker"].set_offsets(np.empty((0, 2)))
            dr.extend([a["line"], a["marker"]])
        return dr

    def update(i):
        t = float(timeline[i])
        draw_list = []
        for amr, runs in amr_runs.items():
            (x, y), loaded = _pos_loaded_at_t(runs, t)

            in_seg, last_e, last_ld = _segment_state(runs, t)

          
            if in_seg:
                color = "red" if loaded else "blue"
            else:
               
                if last_ld and np.isfinite(last_e):
                    if t >= last_e + global_variable.CURRENT_CFG.amr_unload_time - EPS:
                        color = "blue"     
                    else:
                        color = "red"      
                else:
                    color = "blue" if not loaded else "red"

            if trail:
                artists[amr]["trail_x"].append(x)
                artists[amr]["trail_y"].append(y)
                artists[amr]["line"].set_data(artists[amr]["trail_x"], artists[amr]["trail_y"])

            artists[amr]["marker"].set_offsets(np.array([[x, y]]))
            artists[amr]["marker"].set_color(color)
            draw_list.extend([artists[amr]["line"], artists[amr]["marker"]])

        ax.set_title(f"AMR Runs — t={t:.2f} (Loaded=Red, Empty=Blue)")
        return draw_list

    ani = animation.FuncAnimation(fig, update, frames=len(timeline),
                                  init_func=init, blit=False,
                                  interval=interval_ms, repeat=False,
                                  cache_frame_data=False)
    
    animate_from_amr_runs._ani_ref = ani
    plt.show()

def _build_outline_segments(cells: Set[Tuple[int, int]]):
    from collections import Counter
    def _norm_edge(p0, p1):
        return tuple(sorted((p0, p1)))
    edges = Counter()
    for (x, y) in cells:
        e1 = _norm_edge((x,   y),   (x+1, y))
        e2 = _norm_edge((x+1, y),   (x+1, y+1))
        e3 = _norm_edge((x+1, y+1), (x,   y+1))
        e4 = _norm_edge((x,   y+1), (x,   y))
        edges.update([e1, e2, e3, e4])


    outline = [edge for edge, cnt in edges.items() if cnt == 1]
    return outline


def draw_group_outlines(ax, group_blocks, lw=2.0, color='k', z=6):
    for group, blocks in group_blocks.items():

        cells = {(x, y) for (x, y), _c in blocks}
        if not cells:
            continue

        outline = _build_outline_segments(cells)
        for (x0, y0), (x1, y1) in outline:
            ax.plot([x0, x1], [y0, y1],
                    linewidth=lw, color=color, zorder=z)
def draw_units(ax, units, edge_lw_fill=0.2):
    for u in units:
        for (x, y) in u["cells"]:
            ax.fill_between([x, x+1], y, y+1,
                            color=u["color"], edgecolor='k',
                            linewidth=edge_lw_fill, zorder=2)

def draw_unit_outlines(ax, units, lw=2.0, color='k', z=6):
    for u in units:
        outline = _build_outline_segments(u["cells"])
        for (x0, y0), (x1, y1) in outline:
            ax.plot([x0, x1], [y0, y1], linewidth=lw, color=color, zorder=z)

def _norm_edge(p0, p1):
    return tuple(sorted((p0, p1)))

def draw_unit_seams(ax, units, lw=1.2, color='dimgray', z=5):
    unit_edges = []
    for uid, u in enumerate(units):
        edges = set()
        for (x, y) in u["cells"]:
            e1 = _norm_edge((x,   y),   (x+1, y))
            e2 = _norm_edge((x+1, y),   (x+1, y+1))
            e3 = _norm_edge((x+1, y+1), (x,   y+1))
            e4 = _norm_edge((x,   y+1), (x,   y))
            edges.update([e1, e2, e3, e4])
        unit_edges.append(edges)

    n = len(units)
    for i in range(n):
        for j in range(i+1, n):
            shared = unit_edges[i].intersection(unit_edges[j])
            if not shared:
                continue
            for (p0, p1) in shared:
                (x0, y0), (x1, y1) = p0, p1
                ax.plot([x0, x1], [y0, y1], linewidth=lw, color=color, zorder=z)
