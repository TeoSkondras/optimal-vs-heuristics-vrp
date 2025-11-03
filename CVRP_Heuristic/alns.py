# alns.py
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
    total_cost,
    route_cost,
    route_load,
)

# Optional: short intensification via your local search (import only if present)
try:
    from local_search import local_search
    HAS_LS = True
except Exception:
    HAS_LS = False


@dataclass
class ALNSSolution:
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
# Destroy operators
# ---------------------------

def destroy_random(rng: random.Random, routes: List[List[int]], q: int) -> Tuple[List[List[int]], List[int]]:
    """Uniform random customer removal."""
    all_customers = [v for r in routes for v in r]
    if not all_customers: return _copy_routes(routes), []
    q = min(q, len(all_customers))
    removed = rng.sample(all_customers, q)
    remset = set(removed)
    new_routes = [[v for v in r if v not in remset] for r in routes]
    return new_routes, removed

def destroy_worst_distance(D: np.ndarray, rng: random.Random, routes: List[List[int]], q: int) -> Tuple[List[List[int]], List[int]]:
    """Remove customers with largest marginal contribution to route cost."""
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

def destroy_shaw(
    D: np.ndarray, demand: np.ndarray, rng: random.Random, routes: List[List[int]], q: int,
    relatedness_weights: Tuple[float, float, float] = (1.0, 0.1, 0.1)
) -> Tuple[List[List[int]], List[int]]:
    """Shaw removal based on spatial/demand/same-route relatedness."""
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
        # pick b with smallest relatedness to any removed (closest cluster)
        best_b, best_s = None, float("inf")
        for b in remaining:
            s = min(related(a, b) for a in removed)
            if s < best_s:
                best_s, best_b = s, b
        if best_b is None: break
        removed.append(best_b)
        remaining.remove(best_b)

    remset = set(removed)
    new_routes = [[v for v in r if v not in remset] for r in routes]
    return new_routes, removed

def destroy_route_removal(rng: random.Random, routes: List[List[int]], q: int) -> Tuple[List[List[int]], List[int]]:
    """Remove entire routes until >= q customers removed (diversification)."""
    if not routes: return _copy_routes(routes), []
    order = list(range(len(routes)))
    rng.shuffle(order)
    removed: List[int] = []
    kept_routes: List[List[int]] = []
    for idx in order:
        if len(removed) < q:
            removed.extend(routes[idx])
        else:
            kept_routes.append(routes[idx][:])
    # append remaining routes not processed
    for idx in order[len(kept_routes) + (1 if removed else 0):]:
        if idx < len(routes) and routes[idx] not in kept_routes:
            kept_routes.append(routes[idx][:])
    # Clean and keep only non-empty
    kept_routes = [r for r in kept_routes if r]
    return kept_routes, removed

def destroy_clustered_routes(
    D: np.ndarray, rng: random.Random, routes: List[List[int]], q: int
) -> Tuple[List[List[int]], List[int]]:
    """Remove a seed route and then routes whose centroids are closest to the seed (clustered removal)."""
    non_empty = [r for r in routes if r]
    if not non_empty: return _copy_routes(routes), []
    # compute centroids
    def centroid(route: List[int]) -> Tuple[float, float]:
        # We'll use average of customer coords via indices; we don't have coords here,
        # but we can approximate with distance matrix by picking a medoid: the node with min sum D to others
        if not route: return (0.0, 0.0)
        s = [(sum(D[v, u] for u in route), v) for v in route]
        s.sort()
        return (float(s[0][1]), 0.0)  # proxy by medoid id
    cents = [centroid(r) for r in non_empty]
    seed_idx = rng.randrange(len(non_empty))
    seed_route = non_empty[seed_idx]
    # similarity by medoid distance (in D space)
    def medoid(route: List[int]) -> int:
        if not route: return 0
        vals = [(sum(D[v, u] for u in route), v) for v in route]
        vals.sort()
        return vals[0][1]
    seed_med = medoid(seed_route)
    scored = []
    for r in non_empty:
        m = medoid(r)
        scored.append((D[seed_med, m], r))
    scored.sort(key=lambda x: x[0])
    removed: List[int] = []
    kept: List[List[int]] = []
    for _, r in scored:
        if len(removed) < q:
            removed.extend(r)
        else:
            kept.append(r[:])
    return kept, removed


# ---------------------------
# Repair operators
# ---------------------------

def repair_greedy(D: np.ndarray, demand: np.ndarray, Q: int, rng: random.Random,
                  routes: List[List[int]], removed: List[int]) -> List[List[int]]:
    routes = _copy_routes(routes)
    loads = _loads(demand, routes)
    order = removed[:]; rng.shuffle(order)
    for v in order:
        best = None
        for r_idx, r in enumerate(routes):
            if loads[r_idx] + int(demand[v]) > Q: continue
            pos, delta = _best_pos(D, r, v)
            if (best is None) or (delta < best[0]): best = (delta, r_idx, pos)
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
# ALNS core
# ---------------------------

def alns(
    coords: np.ndarray,
    demand: np.ndarray,
    Q: int,
    routes_init: List[List[int]],
    edge_weight_type: str = "EUC_2D",
    round_euclidean: bool = False,
    # Acceptance (SA)
    start_temperature: float = 200.0,
    end_temperature: float = 1.0,
    cooling_rate: float = 0.995,
    # Destroy/repair intensity
    remove_fraction: Tuple[float, float] = (0.06, 0.18),  # a bit lighter by default
    # Adaptive weights
    sigma1: float = 6.0, sigma2: float = 3.0, sigma3: float = 1.0, reaction: float = 0.2, segment_length: int = 50,
    # Limits and control
    time_limit_sec: Optional[float] = None,
    max_iters: int = 20000,
    seed: Optional[int] = None,
    logger: Optional[logging.Logger] = None,
    verbose: bool = False,
    log_every: int = 10,         # NEW: log progress every N iterations
    # Operator toggles / params
    use_destroy: Tuple[str, ...] = ("random", "worst", "shaw", "route", "clustered_routes"),
    use_repair: Tuple[str, ...] = ("greedy", "regret2", "regret3"),
    shaw_weights: Tuple[float, float, float] = (1.0, 0.1, 0.1),
    # Intensification
    use_local_search: bool = True,
    ls_time_limit_sec: float = 0.5,   # short LS burst per accepted candidate
    ls_max_passes_without_gain: int = 1,
) -> ALNSSolution:
    """
    Adaptive Large Neighborhood Search for CVRP.
    - Capacity-feasible at all times.
    - SA acceptance; adaptive operator weights.
    - Stronger destroy ops and optional short local search per iteration.
    - "No-change" guard with random kick if repair rebuilds the same solution.
    """
    # logging
    if logger is None and verbose:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%H:%M:%S",
        )
        logger = logging.getLogger("ALNS")

    def log(msg: str):
        if logger is not None:
            logger.info(msg)

    # RNG & time
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

    n = coords.shape[0] - 1  # customers
    min_remove = max(1, int(remove_fraction[0] * n))
    max_remove = max(min_remove, int(remove_fraction[1] * n))

    # Operator registries
    destroy_ops: Dict[str, Callable[..., Tuple[List[List[int]], List[int]]]] = {}
    if "random" in use_destroy:
        destroy_ops["random"] = lambda routes, q: destroy_random(rng, routes, q)
    if "worst" in use_destroy:
        destroy_ops["worst"]  = lambda routes, q: destroy_worst_distance(D, rng, routes, q)
    if "shaw" in use_destroy:
        destroy_ops["shaw"]   = lambda routes, q: destroy_shaw(D, demand, rng, routes, q, shaw_weights)
    if "route" in use_destroy:
        destroy_ops["route"]  = lambda routes, q: destroy_route_removal(rng, routes, q)
    if "clustered_routes" in use_destroy:
        destroy_ops["clustered_routes"] = lambda routes, q: destroy_clustered_routes(D, rng, routes, q)

    repair_ops: Dict[str, Callable[..., List[List[int]]]] = {}
    if "greedy" in use_repair:
        repair_ops["greedy"]  = lambda routes, removed: repair_greedy(D, demand, Q, rng, routes, removed)
    if "regret2" in use_repair:
        repair_ops["regret2"] = lambda routes, removed: repair_regret(D, demand, Q, rng, routes, removed, k=2)
    if "regret3" in use_repair:
        repair_ops["regret3"] = lambda routes, removed: repair_regret(D, demand, Q, rng, routes, removed, k=3)

    destroy_list = list(destroy_ops.keys())
    repair_list  = list(repair_ops.keys())

    # Adaptive weights and scores
    w_destroy = {k: 1.0 for k in destroy_list}
    w_repair  = {k: 1.0 for k in repair_list}
    score_destroy = {k: 0.0 for k in destroy_list}
    score_repair  = {k: 0.0 for k in repair_list}
    use_count_destroy = {k: 1e-9 for k in destroy_list}
    use_count_repair  = {k: 1e-9 for k in repair_list}

    # SA temperature
    T = float(start_temperature)
    cool = float(cooling_rate)
    eps = 1e-9

    log(f"ALNS start | routes={len(curr_routes)} cost={curr_cost:.2f} | "
        f"remove=[{min_remove},{max_remove}] | ops D={destroy_list}, R={repair_list} | seed={seed}")

    it = 0
    since_last_update = 0

    while it < max_iters and not time_up():
        it += 1
        since_last_update += 1

        # Roulette-wheel pick
        def _roulette_pick(weights: Dict[str, float]) -> str:
            tot = sum(max(0.0, w) for w in weights.values())
            r = rng.random() * tot if tot > 0 else 0.0
            acc = 0.0
            for k, w in weights.items():
                acc += max(0.0, w)
                if r <= acc: return k
            return list(weights.keys())[0]

        d_key = _roulette_pick(w_destroy)
        r_key = _roulette_pick(w_repair)
        use_count_destroy[d_key] += 1.0
        use_count_repair[r_key]  += 1.0

        # Destroy
        q = rng.randint(min_remove, max_remove)
        partial_routes, removed = destroy_ops[d_key](curr_routes, q)

        # Repair
        cand_routes = repair_ops[r_key](partial_routes, removed)

        # If nothing changed (rare but possible), kick once (remove one random route)
        if _routes_equal(cand_routes, curr_routes):
            # small kick to force change
            cand_routes, removed2 = destroy_route_removal(rng, cand_routes, max(1, len(removed)//4))
            cand_routes = repair_greedy(D, demand, Q, rng, cand_routes, removed2)

        # Optional intensification
        if use_local_search and HAS_LS and not time_up():
            try:
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
            except Exception:
                # if local_search import exists but something goes wrong, just ignore intensification
                pass

        cand_cost = _cost(D, cand_routes)
        delta = cand_cost - curr_cost

        # SA acceptance
        accept = False
        if delta <= 0.0:
            accept = True
        else:
            prob = math.exp(-(delta) / max(T, 1e-9))
            accept = (random.random() < prob)

        if accept:
            curr_routes = cand_routes
            curr_cost   = cand_cost

        # Best update
        improved_curr = delta < -1e-9
        if curr_cost + eps < best_cost:
            best_routes = _copy_routes(curr_routes)
            best_cost   = curr_cost

        # Score update
        if curr_cost + eps < best_cost + eps:  # already handled
            score = sigma1
        elif accept and improved_curr:
            score = sigma2
        elif accept:
            score = sigma3
        else:
            score = 0.0
        score_destroy[d_key] += score
        score_repair[r_key]  += score

        # Weights update
        if since_last_update >= segment_length:
            for k in destroy_list:
                avg = score_destroy[k] / max(use_count_destroy[k], 1e-9)
                w_destroy[k] = (1 - reaction) * w_destroy[k] + reaction * avg
                score_destroy[k] = 0.0; use_count_destroy[k] = 1e-9
            for k in repair_list:
                avg = score_repair[k] / max(use_count_repair[k], 1e-9)
                w_repair[k] = (1 - reaction) * w_repair[k] + reaction * avg
                score_repair[k] = 0.0; use_count_repair[k] = 1e-9
            since_last_update = 0

        # Cool
        T = max(end_temperature, T * cool)

        # Logging
        if verbose and ((it % log_every) == 0 or (cand_cost + eps < best_cost + eps)):
            logger.info(f"[ALNS] it={it} T={T:.2f} Δ={delta:+.2f} curr={curr_cost:.2f} best={best_cost:.2f} "
                        f"D={d_key} R={r_key} q={q} routes={len(curr_routes)}")

    return ALNSSolution(routes=best_routes, cost=best_cost)
