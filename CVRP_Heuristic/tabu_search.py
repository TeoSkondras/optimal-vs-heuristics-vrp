# tabu_search.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
import time
import random
import logging
import numpy as np

from utils import (
    build_distance_matrix,
    route_cost,
    total_cost,
    route_load,
    nearest_neighbors,
    # deltas
    delta_relocate_intra,
    delta_relocate_inter,
    delta_swap_inter,
    delta_2opt_intra,
    # apply
    apply_relocate_intra,
    apply_relocate_inter,
    apply_swap_inter,
    apply_2opt_intra,
)


@dataclass
class TSolution:
    routes: List[List[int]]
    cost: float


# ---------------------------
# Internal helpers
# ---------------------------

def _recompute_route_cost(D: np.ndarray, r: List[int]) -> float:
    return route_cost(D, r)

def _route_costs(D: np.ndarray, routes: List[List[int]]) -> List[float]:
    return [route_cost(D, r) for r in routes]

def _loads(demand: np.ndarray, routes: List[List[int]]) -> List[int]:
    return [route_load(demand, r) for r in routes]


# ---------------------------
# Tabu Search
# ---------------------------

def tabu_search(
    coords: np.ndarray,
    demand: np.ndarray,
    Q: int,
    routes_init: List[List[int]],
    edge_weight_type: str = "EUC_2D",
    round_euclidean: bool = False,
    # --- Tabu params ---
    tabu_tenure: int = 15,                 # base tenure
    tabu_tenure_range: Optional[Tuple[int,int]] = None,  # if set, random tenure in [a,b]
    aspiration: bool = True,               # allow tabu if it beats global best
    # --- Search control ---
    time_limit_sec: Optional[float] = None,
    max_iters: int = 10000,
    max_no_improve: int = 200,             # patience before stopping
    improvement_eps: float = 1e-6,         # minimum drop to count as improvement
    # --- Neighborhood control ---
    k_nearest: int = 20,                   # NN list length (for pruning inter-route candidates)
    max_intra_candidates_per_route: int = 64,
    max_inter_candidates_per_route: int = 128,
    # --- Logging/seed ---
    logger: Optional[logging.Logger] = None,
    verbose: bool = False,
    seed: Optional[int] = None,
) -> TSolution:
    """
    Tabu Search for CVRP using Relocate (intra/inter), Swap(1,1), and 2-opt (intra).
    - Keeps feasibility at all times (capacity).
    - Safe commit: recompute costs; revert if the move doesn't produce the evaluated improvement.
    - Attribute-based tabu: forbids moving certain nodes (and node pairs for swap) for some iterations.

    Returns:
        TSolution (best found).
    """
    # --- logging setup ---
    if logger is None and verbose:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%H:%M:%S",
        )
        logger = logging.getLogger("TabuSearch")

    def log(msg: str):
        if logger is not None:
            logger.info(msg)

    # --- RNG ---
    rng = random.Random(seed)

    # --- time control ---
    start_time = time.perf_counter()
    def time_up() -> bool:
        return (time_limit_sec is not None) and ((time.perf_counter() - start_time) >= time_limit_sec)

    # --- distances, initial state ---
    D = build_distance_matrix(coords, edge_weight_type, round_euclidean=round_euclidean)
    routes = [r[:] for r in routes_init]
    rcost = _route_costs(D, routes)
    loads = _loads(demand, routes)
    curr_cost = float(sum(rcost))

    best_routes = [r[:] for r in routes]
    best_cost = float(curr_cost)

    # --- nearest neighbors (for inter-route pruning) ---
    nn_lists = nearest_neighbors(D, k=k_nearest) if k_nearest and k_nearest > 0 else None

    # --- tabu list: dict[(tag, key) -> release_iter] ---
    # tags: "move_node", "swap_pair"
    tabu: Dict[Tuple[str, Any], int] = {}

    def is_tabu(move_tag: str, key: Any, iter_idx: int) -> bool:
        rel = tabu.get((move_tag, key), -1)
        return rel >= iter_idx  # still tabu if release iteration >= now

    def add_tabu(move_tag: str, key: Any, iter_idx: int):
        # pick tenure
        if tabu_tenure_range:
            t = rng.randint(tabu_tenure_range[0], tabu_tenure_range[1])
        else:
            t = tabu_tenure
        tabu[(move_tag, key)] = iter_idx + t

    log(f"TS start | routes={len(routes)} cost={curr_cost:.4f} | "
        f"time_limit={time_limit_sec}s | tenure={tabu_tenure if not tabu_tenure_range else tabu_tenure_range} | "
        f"patience={max_no_improve}")

    # --- selection helpers (safe commit) ---
    eps = 1e-9
    def accept_improvement(delta: float) -> bool:
        return delta < -eps

    def commit_intra(r_idx: int, delta_reported: float) -> Tuple[bool, float]:
        """Reprice route; return (kept, real_delta)."""
        old = rcost[r_idx]
        new = _recompute_route_cost(D, routes[r_idx])
        real = new - old
        if abs(real - delta_reported) > 1e-6:
            log(f"[WARN] Intra Δ mismatch | rep={delta_reported:.6f} real={real:.6f}")
        # capacity check
        if route_load(demand, routes[r_idx]) > Q:
            return False, real
        if not accept_improvement(real):
            return False, real
        rcost[r_idx] = new
        return True, real

    def commit_inter(ra: int, rb: int, delta_reported: float) -> Tuple[bool, float]:
        """Reprice two routes; return (kept, real_delta)."""
        oldA, oldB = rcost[ra], rcost[rb]
        newA = _recompute_route_cost(D, routes[ra])
        newB = _recompute_route_cost(D, routes[rb])
        real = (newA - oldA) + (newB - oldB)
        if abs(real - delta_reported) > 1e-6:
            log(f"[WARN] Inter Δ mismatch | rep={delta_reported:.6f} real={real:.6f}")
        la = route_load(demand, routes[ra])
        lb = route_load(demand, routes[rb])
        if la > Q or lb > Q:
            return False, real
        rcost[ra], rcost[rb] = newA, newB
        loads[ra], loads[rb] = la, lb
        if not accept_improvement(real):
            return False, real
        return True, real

    # ---------------------------
    # Main Tabu loop
    # ---------------------------
    it = 0
    no_improve = 0
    while it < max_iters and not time_up():
        it += 1

        best_move = None          # (delta, move_kind, details...)
        best_move_real_delta = None
        best_move_is_tabu = False

        # 1) Explore INTRA moves (2-opt, Or-opt1/2/3 optional → keep it simple with 2-opt + relocate(1))
        #    For scalability: cap candidates per route
        for r_idx, r in enumerate(routes):
            if time_up(): break
            n = len(r)
            if n >= 3:
                # 2-opt candidates: scan a limited window (first-improvement style but we pick best)
                cnt = 0
                for i in range(0, n - 1):
                    if time_up() or cnt >= max_intra_candidates_per_route: break
                    for k in range(i + 1, n):
                        delta = delta_2opt_intra(D, r, i, k)
                        cnt += 1
                        if not accept_improvement(delta): 
                            continue
                        # tabu check based on nodes touched (simple node attribute)
                        a, b = r[i], r[k]
                        tabu_key = tuple(sorted((a, b)))
                        is_t = is_tabu("move_node", a, it) or is_tabu("move_node", b, it) or is_tabu("swap_pair", tabu_key, it)
                        # aspiration
                        admissible = (not is_t) or (aspiration and (curr_cost + delta < best_cost - improvement_eps))
                        if not admissible:
                            continue
                        # choose best admissible
                        if (best_move is None) or (delta < best_move[0]):
                            best_move = (delta, "2opt_intra", (r_idx, i, k))
                            best_move_is_tabu = is_t
            # relocate(1) intra
            if n >= 2:
                cnt = 0
                for i in range(n):
                    if time_up() or cnt >= max_intra_candidates_per_route: break
                    for j in range(0, n + 1):
                        if j == i or j == i + 1: 
                            continue
                        delta = delta_relocate_intra(D, r, i, j)
                        cnt += 1
                        if not accept_improvement(delta): 
                            continue
                        v = r[i]
                        is_t = is_tabu("move_node", v, it)
                        admissible = (not is_t) or (aspiration and (curr_cost + delta < best_cost - improvement_eps))
                        if not admissible:
                            continue
                        if (best_move is None) or (delta < best_move[0]):
                            best_move = (delta, "reloc_intra", (r_idx, i, j))
                            best_move_is_tabu = is_t

        # 2) Explore INTER moves (relocate(1) and swap(1,1))
        for ra in range(len(routes)):
            if time_up(): break
            A = routes[ra]
            if not A: 
                continue
            # Build a small candidate target set using NN of nodes in A (fast heuristic)
            target_routes = range(len(routes))  # simple default; NN pruning can be added here
            # relocate(1) inter
            cnt = 0
            for ia in range(len(A)):
                if time_up(): break
                v = A[ia]
                dv = int(demand[v])
                for rb in target_routes:
                    if rb == ra: 
                        continue
                    if loads[rb] + dv > Q: 
                        continue
                    B = routes[rb]
                    for jb in range(0, len(B) + 1):
                        if time_up(): break
                        delta = delta_relocate_inter(D, A, ia, B, jb)
                        cnt += 1
                        if not accept_improvement(delta): 
                            continue
                        is_t = is_tabu("move_node", v, it)
                        admissible = (not is_t) or (aspiration and (curr_cost + delta < best_cost - improvement_eps))
                        if not admissible:
                            continue
                        if (best_move is None) or (delta < best_move[0]):
                            best_move = (delta, "reloc_inter", (ra, ia, rb, jb))
                            best_move_is_tabu = is_t
                        if cnt >= max_inter_candidates_per_route:
                            break
                    if time_up(): break
                if time_up(): break

            # swap(1,1)
            cnt = 0
            for ia in range(len(A)):
                if time_up(): break
                x = A[ia]; dx = int(demand[x])
                for rb in range(ra + 1, len(routes)):
                    if time_up(): break
                    B = routes[rb]
                    if not B: 
                        continue
                    for ib in range(len(B)):
                        y = B[ib]; dy = int(demand[y])
                        if loads[ra] - dx + dy > Q or loads[rb] - dy + dx > Q:
                            continue
                        delta = delta_swap_inter(D, A, ia, B, ib)
                        cnt += 1
                        if not accept_improvement(delta): 
                            continue
                        tabu_key = tuple(sorted((x, y)))
                        is_t = is_tabu("swap_pair", tabu_key, it) or is_tabu("move_node", x, it) or is_tabu("move_node", y, it)
                        admissible = (not is_t) or (aspiration and (curr_cost + delta < best_cost - improvement_eps))
                        if not admissible:
                            continue
                        if (best_move is None) or (delta < best_move[0]):
                            best_move = (delta, "swap_inter", (ra, ia, rb, ib))
                            best_move_is_tabu = is_t
                        if cnt >= max_inter_candidates_per_route:
                            break
                    if time_up(): break
                if time_up(): break

        # If we found no admissible move, stop
        if best_move is None:
            log(f"[ITER {it}] no admissible move | curr={curr_cost:.4f} | best={best_cost:.4f}")
            break

        delta_rep, kind, info = best_move
        kept = False
        real_delta = None

        # Apply tentatively, commit if real improvement
        if kind == "2opt_intra":
            r_idx, i, k = info
            before = routes[r_idx][:]
            apply_2opt_intra(routes[r_idx], i, k)
            ok, real = commit_intra(r_idx, delta_rep)
            if ok:
                kept = True; real_delta = real
                curr_cost += real
                # add tabu attributes
                a, b = before[i], before[k]
                add_tabu("move_node", a, it)
                add_tabu("move_node", b, it)
                add_tabu("swap_pair", tuple(sorted((a, b))), it)
            else:
                routes[r_idx] = before

        elif kind == "reloc_intra":
            r_idx, i, j = info
            before = routes[r_idx][:]
            apply_relocate_intra(routes[r_idx], i, j)
            ok, real = commit_intra(r_idx, delta_rep)
            if ok:
                kept = True; real_delta = real
                curr_cost += real
                v = before[i]
                add_tabu("move_node", v, it)
            else:
                routes[r_idx] = before

        elif kind == "reloc_inter":
            ra, ia, rb, jb = info
            A_before = routes[ra][:]
            B_before = routes[rb][:]
            apply_relocate_inter(routes[ra], ia, routes[rb], jb)
            ok, real = commit_inter(ra, rb, delta_rep)
            if ok:
                kept = True; real_delta = real
                curr_cost += real
                v = A_before[ia]
                add_tabu("move_node", v, it)
            else:
                routes[ra] = A_before; routes[rb] = B_before

        elif kind == "swap_inter":
            ra, ia, rb, ib = info
            A_before = routes[ra][:]
            B_before = routes[rb][:]
            x = A_before[ia]; y = B_before[ib]
            apply_swap_inter(routes[ra], ia, routes[rb], ib)
            ok, real = commit_inter(ra, rb, delta_rep)
            if ok:
                kept = True; real_delta = real
                curr_cost += real
                add_tabu("move_node", x, it)
                add_tabu("move_node", y, it)
                add_tabu("swap_pair", tuple(sorted((x, y))), it)
            else:
                routes[ra] = A_before; routes[rb] = B_before

        # Feedback & best update
        if kept:
            if verbose and logger is not None:
                logger.info(f"[ITER {it}] {kind} kept | Δreal={real_delta:.4f} | curr={curr_cost:.4f} | best={best_cost:.4f}")
            if curr_cost + improvement_eps < best_cost:
                best_cost = curr_cost
                best_routes = [r[:] for r in routes]
                log(f"  -> new BEST at iter {it} | best={best_cost:.4f} ({kind})")
                no_improve = 0
            else:
                no_improve += 1
        else:
            # No move could be committed (should be rare due to admissibility)
            no_improve += 1

        # stop conditions
        if no_improve >= max_no_improve:
            log(f"[STOP] patience reached at iter {it} | best={best_cost:.4f}")
            break

    return TSolution(routes=best_routes, cost=best_cost)
