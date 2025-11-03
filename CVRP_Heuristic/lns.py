# lns.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Callable
import math
import random
import time
import logging
import numpy as np

from utils import (
    build_distance_matrix,
    route_cost,
    route_load,
)

# Optional: short intensification via your local search
try:
    from local_search import local_search
    HAS_LS = True
except Exception:
    HAS_LS = False


@dataclass
class LNSSolution:
    routes: List[List[int]]
    cost: float


# ---------------------------
# Small utilities
# ---------------------------

def _copy_routes(routes: List[List[int]]) -> List[List[int]]:
    return [r[:] for r in routes]

def _loads(demand: np.ndarray, routes: List[List[int]]) -> List[int]:
    return [int(route_load(demand, r)) for r in routes]

def _cost(D: np.ndarray, routes: List[List[int]]) -> float:
    return float(sum(route_cost(D, r) for r in routes))

def _insertion_delta(D: np.ndarray, route: List[int], cust: int, pos: int) -> float:
    if not route:
        return D[0, cust] + D[cust, 0]
    u = 0 if pos == 0 else route[pos - 1]
    v = 0 if pos == len(route) else route[pos]
    return -D[u, v] + D[u, cust] + D[cust, v]

def _best_pos(D: np.ndarray, route: List[int], cust: int) -> Tuple[int, float]:
    best_pos, best_delta = 0, float("inf")
    for j in range(len(route) + 1):
        d = _insertion_delta(D, route, cust, j)
        if d < best_delta:
            best_delta, best_pos = d, j
    return best_pos, float(best_delta)

def _routes_equal(a: List[List[int]], b: List[List[int]]) -> bool:
    if len(a) != len(b): return False
    for r1, r2 in zip(a, b):
        if r1 != r2: return False
    return True


# ---------------------------
# Ruin (destroy) operators
# ---------------------------

def ruin_random(rng: random.Random, routes: List[List[int]], q: int) -> Tuple[List[List[int]], List[int]]:
    all_customers = [v for r in routes for v in r]
    if not all_customers: return _copy_routes(routes), []
    q = min(q, len(all_customers))
    removed = rng.sample(all_customers, q)
    remset = set(removed)
    new_routes = [[v for v in r if v not in remset] for r in routes]
    return new_routes, removed

def ruin_worst(D: np.ndarray, rng: random.Random, routes: List[List[int]], q: int) -> Tuple[List[List[int]], List[int]]:
    contribs: List[Tuple[float, int]] = []
    for r in routes:
        if not r: continue
        path = [0] + r + [0]
        for i in range(1, len(path) - 1):
            u, v, w = path[i - 1], path[i], path[i + 1]
            contribs.append((D[u, v] + D[v, w] - D[u, w], v))
    if not contribs: return _copy_routes(routes), []
    contribs.sort(reverse=True)
    top = contribs[:min(q * 3, len(contribs))]
    rng.shuffle(top)
    removed = [v for _, v in top[:q]]
    remset = set(removed)
    new_routes = [[v for v in r if v not in remset] for r in routes]
    return new_routes, removed

def ruin_shaw(
    D: np.ndarray, demand: np.ndarray, rng: random.Random, routes: List[List[int]], q: int,
    relatedness_weights: Tuple[float, float, float] = (1.0, 0.1, 0.1)
) -> Tuple[List[List[int]], List[int]]:
    all_customers = [v for r in routes for v in r]
    if not all_customers: return _copy_routes(routes), []
    seed = rng.choice(all_customers)
    route_of: Dict[int, int] = {}
    for ri, r in enumerate(routes):
        for v in r: route_of[v] = ri
    w_d, w_q, w_r = relatedness_weights
    maxD = float(np.max(D)) if np.max(D) > 0 else 1.0
    mean_dem = float(np.mean(demand[1:])) if len(demand) > 1 else 1.0
    def related(a: int, b: int) -> float:
        rd = D[a, b] / maxD
        rq = abs(int(demand[a]) - int(demand[b])) / (mean_dem if mean_dem > 0 else 1.0)
        rr = 1.0 if route_of.get(a, -1) == route_of.get(b, -1) else 0.0
        return w_d * rd + w_q * rq - w_r * rr
    removed = [seed]
    remaining = set(all_customers) - {seed}
    while len(removed) < min(q, len(all_customers)):
        best_b, best_s = None, float("inf")
        for b in remaining:
            s = min(related(a, b) for a in removed)
            if s < best_s:
                best_s, best_b = s, b
        if best_b is None: break
        removed.append(best_b); remaining.remove(best_b)
    remset = set(removed)
    new_routes = [[v for v in r if v not in remset] for r in routes]
    return new_routes, removed

def ruin_route(rng: random.Random, routes: List[List[int]], q: int) -> Tuple[List[List[int]], List[int]]:
    if not routes: return _copy_routes(routes), []
    order = list(range(len(routes))); rng.shuffle(order)
    removed: List[int] = []
    kept: List[List[int]] = []
    for idx in order:
        if len(removed) < q: removed.extend(routes[idx])
        else: kept.append(routes[idx][:])
    kept = [r for r in kept if r]
    return kept, removed

def ruin_clustered_routes(D: np.ndarray, rng: random.Random, routes: List[List[int]], q: int) -> Tuple[List[List[int]], List[int]]:
    non_empty = [r for r in routes if r]
    if not non_empty: return _copy_routes(routes), []
    def medoid(route: List[int]) -> int:
        if not route: return 0
        vals = [(sum(D[v, u] for u in route), v) for v in route]
        vals.sort()
        return vals[0][1]
    seed_r = rng.choice(non_empty)
    seed_m = medoid(seed_r)
    scored = [(D[seed_m, medoid(r)], r) for r in non_empty]
    scored.sort(key=lambda x: x[0])
    removed: List[int] = []
    kept: List[List[int]] = []
    for _, r in scored:
        if len(removed) < q: removed.extend(r)
        else: kept.append(r[:])
    return kept, removed


# ---------------------------
# Recreate (repair) operators
# ---------------------------

def repair_greedy(D: np.ndarray, demand: np.ndarray, Q: int, rng: random.Random,
                  routes: List[List[int]], removed: List[int]) -> List[List[int]]:
    routes = _copy_routes(routes)
    loads = _loads(demand, routes)
    order = removed[:]; rng.shuffle(order)
    for v in order:
        best = None
        for ri, r in enumerate(routes):
            if loads[ri] + int(demand[v]) > Q: continue
            pos, delta = _best_pos(D, r, v)
            if (best is None) or (delta < best[0]): best = (delta, ri, pos)
        if best is None:
            routes.append([v]); loads.append(int(demand[v]))
        else:
            _, ri, pos = best
            routes[ri].insert(pos, v); loads[ri] += int(demand[v])
    return routes

def repair_regret(D: np.ndarray, demand: np.ndarray, Q: int, rng: random.Random,
                  routes: List[List[int]], removed: List[int], k: int = 2) -> List[List[int]]:
    routes = _copy_routes(routes)
    loads = _loads(demand, routes)
    unplaced = removed[:]
    while unplaced:
        best_choice = None  # (regret, delta1, v, ri, pos)
        for v in unplaced:
            cands: List[Tuple[float, int, int]] = []
            for ri, r in enumerate(routes):
                if loads[ri] + int(demand[v]) > Q: continue
                pos, delta = _best_pos(D, r, v)
                cands.append((delta, ri, pos))
            if not cands:
                cands = [(D[0, v] + D[v, 0], len(routes), 0)]
            cands.sort(key=lambda x: x[0])
            delta1 = cands[0][0]
            regret = sum(cands[m][0] - delta1 for m in range(1, min(k, len(cands))))
            if (best_choice is None) or (regret > best_choice[0]) or (abs(regret - best_choice[0]) < 1e-9 and delta1 < best_choice[1]):
                best_choice = (regret, delta1, v, cands[0][1], cands[0][2])
        _, _, v, ri, pos = best_choice
        if ri == len(routes):
            routes.append([v]); loads.append(int(demand[v]))
        else:
            routes[ri].insert(pos, v); loads[ri] += int(demand[v])
        unplaced.remove(v)
    return routes


# ---------------------------
# LNS core
# ---------------------------

def lns(
    coords: np.ndarray,
    demand: np.ndarray,
    Q: int,
    routes_init: List[List[int]],
    edge_weight_type: str = "EUC_2D",
    round_euclidean: bool = False,
    # Intensity
    remove_fraction: Tuple[float, float] = (0.12, 0.25),
    # Acceptance (choose one style)
    use_sa: bool = True,
    start_temperature: float = 800.0,
    end_temperature: float = 5.0,
    cooling_rate: float = 0.996,
    # If not SA: greedy with occasional random uphill acceptance
    p_accept_worse: float = 0.0,   # e.g., 0.02 for light randomization
    # Limits & logging
    time_limit_sec: Optional[float] = None,
    max_iters: int = 200000,
    seed: Optional[int] = None,
    logger: Optional[logging.Logger] = None,
    verbose: bool = False,
    log_every: int = 5,
    log_detail: bool = True,
    # Operators
    use_ruin: Tuple[str, ...] = ("route", "clustered_routes", "shaw", "worst", "random"),
    use_repair: Tuple[str, ...] = ("regret3", "regret2", "greedy"),
    shaw_weights: Tuple[float, float, float] = (1.0, 0.1, 0.1),
    # Intensification
    use_local_search: bool = True,
    ls_time_limit_sec: float = 1.0,
    ls_max_passes_without_gain: int = 1,
) -> LNSSolution:
    """
    Large Neighborhood Search (ruin & recreate) for CVRP.
    - Capacity-feasible via repairs; new routes created as needed.
    - Either SA acceptance (recommended) or greedy with optional random uphill.
    - Optional short LS after repair.
    """

    if logger is None and verbose:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%H:%M:%S",
        )
        logger = logging.getLogger("LNS")

    def log(msg: str):
        if logger is not None:
            logger.info(msg)

    rng = random.Random(seed)
    start_time = time.perf_counter()
    def time_up() -> bool:
        return (time_limit_sec is not None) and ((time.perf_counter() - start_time) >= time_limit_sec)

    # Distances
    D = build_distance_matrix(coords, edge_weight_type, round_euclidean=round_euclidean)

    # Init solution
    curr_routes = _copy_routes(routes_init)
    curr_cost = _cost(D, curr_routes)
    best_routes = _copy_routes(curr_routes)
    best_cost = curr_cost

    n = coords.shape[0] - 1
    min_remove = max(1, int(remove_fraction[0] * n))
    max_remove = max(min_remove, int(remove_fraction[1] * n))

    # Operator registries
    ruin_ops: Dict[str, Callable[..., Tuple[List[List[int]], List[int]]]] = {}
    if "random" in use_ruin:
        ruin_ops["random"] = lambda routes, q: ruin_random(rng, routes, q)
    if "worst" in use_ruin:
        ruin_ops["worst"]  = lambda routes, q: ruin_worst(D, rng, routes, q)
    if "shaw" in use_ruin:
        ruin_ops["shaw"]   = lambda routes, q: ruin_shaw(D, demand, rng, routes, q, shaw_weights)
    if "route" in use_ruin:
        ruin_ops["route"]  = lambda routes, q: ruin_route(rng, routes, q)
    if "clustered_routes" in use_ruin:
        ruin_ops["clustered_routes"] = lambda routes, q: ruin_clustered_routes(D, rng, routes, q)

    repair_ops: Dict[str, Callable[..., List[List[int]]]] = {}
    if "greedy" in use_repair:
        repair_ops["greedy"]  = lambda routes, removed: repair_greedy(D, demand, Q, rng, routes, removed)
    if "regret2" in use_repair:
        repair_ops["regret2"] = lambda routes, removed: repair_regret(D, demand, Q, rng, routes, removed, k=2)
    if "regret3" in use_repair:
        repair_ops["regret3"] = lambda routes, removed: repair_regret(D, demand, Q, rng, routes, removed, k=3)

    ruin_list = list(ruin_ops.keys())
    repair_list = list(repair_ops.keys())

    # SA temp
    T = float(start_temperature)
    cool = float(cooling_rate)
    eps = 1e-9

    log(f"LNS start | routes={len(curr_routes)} cost={curr_cost:.2f} "
        f"| remove=[{min_remove},{max_remove}] | ops R={ruin_list}, C={repair_list} | seed={seed} | SA={use_sa}")

    it = 0
    while it < max_iters and not time_up():
        it += 1
        iter_start = time.perf_counter()

        # Pick operators uniformly (LNS = no adaptation)
        r_key = rng.choice(ruin_list)
        c_key = rng.choice(repair_list)

        # Ruin
        q = rng.randint(min_remove, max_remove)
        routes_before = len(curr_routes)
        partial_routes, removed = ruin_ops[r_key](curr_routes, q)
        removed_cnt = len(removed)

        # Recreate
        cand_routes = repair_ops[c_key](partial_routes, removed)

        # Optional LS
        ls_delta = 0.0
        if use_local_search and HAS_LS and not time_up():
            try:
                before_cost = _cost(D, cand_routes)
                ls = local_search(
                    coords=coords, demand=demand, Q=Q,
                    routes=[r[:] for r in cand_routes],
                    edge_weight_type=edge_weight_type,
                    round_euclidean=round_euclidean,
                    k_nearest=0,
                    max_passes_without_gain=ls_max_passes_without_gain,
                    time_limit_sec=max(0.05, min(ls_time_limit_sec, (time_limit_sec or 9e9) - (time.perf_counter() - start_time))),
                    verbose=False,
                )
                cand_routes = ls.routes
                after_cost = _cost(D, cand_routes)
                ls_delta = after_cost - before_cost
            except Exception:
                pass

        cand_cost = _cost(D, cand_routes)
        delta = cand_cost - curr_cost

        # Acceptance
        accept = False
        reason = ""
        prob = 1.0

        if use_sa:
            if delta <= 0:
                accept = True; reason = "improve"
            else:
                prob = math.exp(-(delta) / max(T, 1e-9))
                u = random.random()
                accept = (u < prob); reason = f"uphill p={prob:.3f} u={u:.3f}"
        else:
            if delta <= 0:
                accept = True; reason = "improve"
            else:
                u = random.random()
                accept = (u < p_accept_worse)
                reason = f"rand_worse p={p_accept_worse:.3f} u={u:.3f}"

        if accept:
            curr_routes = cand_routes
            curr_cost   = cand_cost

        improved_best = (curr_cost + eps < best_cost)
        if improved_best:
            best_routes = _copy_routes(curr_routes)
            best_cost   = curr_cost

        # Logging
        iter_time = (time.perf_counter() - iter_start)
        if verbose and (it % log_every == 0 or improved_best):
            if log_detail:
                log(
                    f"[LNS] it={it} "
                    f"{'SA ' if use_sa else 'GR '}"
                    f"T={T:.2f} R={r_key} C={c_key} q={q} "
                    f"routes_bef={routes_before} removed={removed_cnt} "
                    f"routes_cand={len(cand_routes)} cand={cand_cost:.2f} "
                    f"Δ={delta:+.2f} accept={accept} ({reason}) "
                    f"LSΔ={ls_delta:+.2f} curr={curr_cost:.2f} best={best_cost:.2f} "
                    f"iter_ms={iter_time*1000:.1f}"
                )
            else:
                log(f"[LNS] it={it} Δ={delta:+.2f} curr={curr_cost:.2f} best={best_cost:.2f} R={r_key} C={c_key} q={q}")

        # Cool
        if use_sa:
            T = max(end_temperature, T * cool)

    return LNSSolution(routes=best_routes, cost=best_cost)
