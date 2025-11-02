# local_search.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple
import math
import time
import logging
import numpy as np

from utils import (
    build_distance_matrix,
    total_cost,
    route_load,
    nearest_neighbors,
    # deltas
    delta_relocate_intra,
    delta_relocate_inter,
    delta_swap_inter,
    delta_2opt_intra,
    delta_2opt_star,
    delta_oropt_intra,
    # apply
    apply_relocate_intra,
    apply_relocate_inter,
    apply_swap_inter,
    apply_2opt_intra,
    apply_2opt_star,
    apply_oropt_intra,
    route_cost,
)


@dataclass
class LSSolution:
    routes: List[List[int]]
    cost: float


def _recompute_cost(D: np.ndarray, routes: List[List[int]]) -> float:
    return total_cost(D, routes)


def _recompute_route_cost(D: np.ndarray, r: List[int]) -> float:
    return route_cost(D, r)


def _route_costs(D: np.ndarray, routes: List[List[int]]) -> List[float]:
    return [route_cost(D, r) for r in routes]


def _loads(demand: np.ndarray, routes: List[List[int]]) -> List[int]:
    return [route_load(demand, r) for r in routes]


# --- replace your local_search(...) with this version ---

def local_search(
    coords, demand, Q, routes,
    edge_weight_type="EUC_2D",
    round_euclidean=False,
    k_nearest=20,
    max_passes_without_gain=2,   # ← control this
    time_limit_sec=None,
    logger=None,
    verbose=False,
    check_capacity=True,
    max_intra_moves_per_pass=20000,
    max_inter_moves_per_pass=20000,
    improvement_eps=1e-6,        # ← new: minimum drop to count as “gain”
) -> LSSolution:
    # --- logging setup ---
    if logger is None and verbose:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%H:%M:%S",
        )
        logger = logging.getLogger("LocalSearch")

    def log(msg: str):
        if logger is not None:
            logger.info(msg)

    start_time = time.perf_counter()
    def time_up() -> bool:
        return (time_limit_sec is not None) and ((time.perf_counter() - start_time) >= time_limit_sec)

    eps = 1e-9

    D = build_distance_matrix(coords, edge_weight_type, round_euclidean=round_euclidean)

    loads = _loads(demand, routes)
    rcost = _route_costs(D, routes)
    total = float(sum(rcost))
    best_routes = [r[:] for r in routes]
    best_cost = float(total)

    no_gain_passes = 0
    pass_id = 0

    log(f"LS start | routes={len(routes)} cost={total:.4f} | "
            f"time_limit={time_limit_sec}s | no_gain_passes={max_passes_without_gain} | "
            f"impr_eps={improvement_eps}")
    
    def accept(real_delta: float) -> bool:
        # Only treat as improvement if it beats the epsilon
        return real_delta < -max(improvement_eps, 1e-9)

    # ---------- safe commit helpers ----------
    def commit_intra(r_idx: int, delta_reported: float, move_tag: str) -> bool:
        """Reprice route r_idx; capacity check; return True if real improvement kept, else revert."""
        nonlocal total, best_cost, best_routes
        old = rcost[r_idx]
        new = _recompute_route_cost(D, routes[r_idx])
        real_delta = new - old
        # Δ mismatch? log it (only if quite off to avoid spam)
        if abs(real_delta - delta_reported) > 1e-6:
            log(f"[WARN] {move_tag} Δ-mismatch | reported={delta_reported:.6f} real={real_delta:.6f}")
        if check_capacity:
            if route_load(demand, routes[r_idx]) > Q:
                return False  # caller must revert
        if not accept(real_delta):
            return False    # caller must revert
        rcost[r_idx] = new
        total = float(sum(rcost))
        if total + improvement_eps < best_cost:
            best_cost = total
            best_routes = [r[:] for r in routes]
            log(f"  -> new BEST after {move_tag} | best={best_cost:.4f}")
        return True

    def commit_inter(ra: int, rb: int, delta_reported: float, move_tag: str) -> bool:
        """Reprice both routes; capacity check; return True if real improvement kept, else revert."""
        nonlocal total, best_cost, best_routes, loads
        oldA, oldB = rcost[ra], rcost[rb]
        newA = _recompute_route_cost(D, routes[ra])
        newB = _recompute_route_cost(D, routes[rb])
        real_delta = (newA - oldA) + (newB - oldB)
        if abs(real_delta - delta_reported) > 1e-6:
            log(f"[WARN] {move_tag} Δ-mismatch | reported={delta_reported:.6f} real={real_delta:.6f}")
        if check_capacity:
            la = route_load(demand, routes[ra])
            lb = route_load(demand, routes[rb])
            if la > Q or lb > Q:
                return False
            loads[ra], loads[rb] = la, lb
        if not accept(real_delta):
            return False
        rcost[ra], rcost[rb] = newA, newB
        total = float(sum(rcost))
        if total + eps < best_cost:
            best_cost = total
            best_routes = [r[:] for r in routes]
            log(f"  -> new BEST after {move_tag} | best={best_cost:.4f}")
        return True

    # for quick revert without rebuilding: copy slices before we mutate
    def snap(r_idx: int) -> List[int]:
        return routes[r_idx][:]

    while True:
        pass_id += 1
        if time_up():
            log(f"[TIMEOUT] pass={pass_id} | best_cost={best_cost:.4f}")
            break

        improved = False
        intra_moves = 0
        inter_moves = 0

        # ===== 1) INTRA: 2-opt =====
        for r_idx, r in enumerate(routes):
            if time_up(): break
            if len(r) < 3: continue
            found = False
            for i in range(0, len(r) - 1):
                if time_up(): break
                for k in range(i + 1, len(r)):
                    if time_up(): break
                    delta_rep = delta_2opt_intra(D, r, i, k)
                    if delta_rep < -eps:
                        before = snap(r_idx)
                        apply_2opt_intra(r, i, k)
                        intra_moves += 1
                        if commit_intra(r_idx, delta_rep, f"2-opt r={r_idx} i={i} k={k}"):
                            log(f"[2-opt] r={r_idx} i={i} k={k} kept")
                            improved = True; found = True
                        else:
                            # revert
                            routes[r_idx] = before
                        if improved or time_up(): break
                if found or time_up(): break
            if improved or time_up(): break
            if intra_moves >= max_intra_moves_per_pass: break
        if improved:
            no_gain_passes = 0
            continue

        # ===== 2) INTRA: Or-opt (3,2,1) =====
        for r_idx, r in enumerate(routes):
            if time_up(): break
            if len(r) < 2: continue
            found = False
            for L in (3, 2, 1):
                if len(r) < L: continue
                for i in range(0, len(r) - L + 1):
                    if time_up(): break
                    for j in range(0, len(r) + 1):
                        if j >= i and j <= i + L: continue
                        delta_rep = delta_oropt_intra(D, r, i, L, j)
                        if delta_rep < -eps:
                            before = snap(r_idx)
                            apply_oropt_intra(r, i, L, j)
                            intra_moves += 1
                            if commit_intra(r_idx, delta_rep, f"Or-opt{L} r={r_idx} i={i} j={j}"):
                                log(f"[Or-opt{L}] r={r_idx} i={i} j={j} kept")
                                improved = True; found = True
                            else:
                                routes[r_idx] = before
                            if improved or time_up(): break
                if found or time_up(): break
            if improved or time_up(): break
            if intra_moves >= max_intra_moves_per_pass: break
        if improved:
            no_gain_passes = 0
            continue

        # ===== 3) INTRA: Relocate(1) =====
        for r_idx, r in enumerate(routes):
            if time_up(): break
            if len(r) < 2: continue
            found = False
            for i in range(len(r)):
                if time_up(): break
                for j in range(0, len(r) + 1):
                    if j == i or j == i + 1: continue
                    delta_rep = delta_relocate_intra(D, r, i, j)
                    if delta_rep < -eps:
                        before = snap(r_idx)
                        apply_relocate_intra(r, i, j)
                        intra_moves += 1
                        if commit_intra(r_idx, delta_rep, f"Reloc-intra r={r_idx} i={i} j={j}"):
                            log(f"[Reloc-intra] r={r_idx} i={i} j={j} kept")
                            improved = True; found = True
                        else:
                            routes[r_idx] = before
                        if improved or time_up(): break
                if found or time_up(): break
            if improved or time_up(): break
            if intra_moves >= max_intra_moves_per_pass: break
        if improved:
            no_gain_passes = 0
            continue

        # ===== 4) INTER: Relocate(1) =====
        for ra in range(len(routes)):
            if time_up(): break
            A = routes[ra]
            if not A: continue
            found = False
            for ia in range(len(A)):
                if time_up(): break
                v = A[ia]; dv = int(demand[v])
                for rb in range(len(routes)):
                    if rb == ra: continue
                    if loads[rb] + dv > Q: continue
                    B = routes[rb]
                    for jb in range(0, len(B) + 1):
                        if time_up(): break
                        delta_rep = delta_relocate_inter(D, A, ia, B, jb)
                        if delta_rep < -eps:
                            # mutate
                            savedA = A[:]; savedB = B[:]
                            apply_relocate_inter(A, ia, B, jb)
                            inter_moves += 1
                            if commit_inter(ra, rb, delta_rep, f"Reloc-inter A={ra} i={ia} -> B={rb} j={jb}"):
                                log(f"[Reloc-inter] A={ra} i={ia} -> B={rb} j={jb} kept")
                                improved = True; found = True
                            else:
                                routes[ra] = savedA; routes[rb] = savedB
                            if improved or time_up(): break
                    if found or time_up(): break
                if found or time_up(): break
            if improved or time_up(): break
            if inter_moves >= max_inter_moves_per_pass: break
        if improved:
            no_gain_passes = 0
            continue

        # ===== 5) INTER: Swap(1,1) =====
        for ra in range(len(routes)):
            if time_up(): break
            A = routes[ra]
            if not A: continue
            found = False
            for rb in range(ra + 1, len(routes)):
                if time_up(): break
                B = routes[rb]
                if not B: continue
                for ia in range(len(A)):
                    if time_up(): break
                    x = A[ia]; dx = int(demand[x])
                    for ib in range(len(B)):
                        if time_up(): break
                        y = B[ib]; dy = int(demand[y])
                        if loads[ra] - dx + dy > Q or loads[rb] - dy + dx > Q: continue
                        delta_rep = delta_swap_inter(D, A, ia, B, ib)
                        if delta_rep < -eps:
                            savedA = A[:]; savedB = B[:]
                            apply_swap_inter(A, ia, B, ib)
                            inter_moves += 1
                            if commit_inter(ra, rb, delta_rep, f"Swap A={ra} ia={ia} <-> B={rb} ib={ib}"):
                                log(f"[Swap] A={ra} ia={ia} <-> B={rb} ib={ib} kept")
                                improved = True; found = True
                            else:
                                routes[ra] = savedA; routes[rb] = savedB
                            if improved or time_up(): break
                    if found or time_up(): break
                if found or time_up(): break
            if improved or time_up(): break
            if inter_moves >= max_inter_moves_per_pass: break
        if improved:
            no_gain_passes = 0
            continue

        # ===== 6) INTER: 2-opt* =====
        for ra in range(len(routes)):
            if time_up(): break
            A = routes[ra]
            if not A: continue
            found = False
            for rb in range(ra + 1, len(routes)):
                if time_up(): break
                B = routes[rb]
                if not B: continue
                for ia in range(-1, len(A)):
                    if time_up(): break
                    for ib in range(-1, len(B)):
                        if time_up(): break
                        delta_rep = delta_2opt_star(D, A, ia, B, ib)
                        if delta_rep < -eps:
                            newA, newB = apply_2opt_star(A, ia, B, ib)
                            la = route_load(demand, newA); lb = route_load(demand, newB)
                            if la <= Q and lb <= Q:
                                savedA = A[:]; savedB = B[:]
                                routes[ra], routes[rb] = newA, newB
                                inter_moves += 1
                                if commit_inter(ra, rb, delta_rep, f"2-opt* A={ra} ia={ia} | B={rb} ib={ib}"):
                                    log(f"[2-opt*] A={ra} ia={ia} | B={rb} ib={ib} kept")
                                    improved = True; found = True
                                else:
                                    routes[ra], routes[rb] = savedA, savedB
                                if improved or time_up(): break
                    if found or time_up(): break
            if improved or time_up(): break
            if inter_moves >= max_inter_moves_per_pass: break
        if improved:
            no_gain_passes = 0
            continue

        # ===== PASS SUMMARY / STOP =====
        if not improved:
            no_gain_passes += 1
            log(f"[PASS {pass_id}] no improvement | cost={total:.4f} | no_gain_passes={no_gain_passes}")
            if no_gain_passes >= max_passes_without_gain or time_up():
                if time_up():
                    log(f"[TIMEOUT] exit | best_cost={best_cost:.4f}")
                else:
                    log(f"[STOP] local optimum | best_cost={best_cost:.4f}")
                break

    return LSSolution(routes=best_routes, cost=best_cost)
