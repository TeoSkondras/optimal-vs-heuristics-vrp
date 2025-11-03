# ils.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple, Callable
import time
import math
import logging
import numpy as np

from utils import (
    build_distance_matrix,
    total_cost,
    route_load,
)
from local_search import local_search, LSSolution


@dataclass
class ILSConfig:
    # Global time budget (seconds) for the whole ILS (including LS calls)
    time_limit_sec: float = 60.0
    # Per-iteration LS time slice (seconds); we’ll clamp to remaining time automatically
    ls_time_slice_sec: float = 2.0
    # How many ILS iterations at most (also bounded by time)
    max_iters: int = 10_000
    # Acceptance temperature schedule
    T_init_factor: float = 0.01   # T0 = T_init_factor * initial_cost
    T_cooling: float = 0.98       # T <- T * T_cooling each iteration (geometric cooling)
    # Logging
    verbose: bool = True
    # Local search knobs (forwarded)
    max_passes_without_gain: int = 2
    k_nearest: int = 20
    improvement_eps: float = 1e-6
    # Perturbation mix
    p_double_bridge: float = 0.4
    p_random_reloc: float = 0.4
    p_segment_swap: float = 0.2
    # Random inter-relocate attempts per perturbation call
    reloc_tries: int = 20
    # Random contiguous segment lengths for swap
    seg_len_min: int = 1
    seg_len_max: int = 3
    # Deterministic runs
    seed: Optional[int] = 0


@dataclass
class ILSSolution:
    routes: List[List[int]]
    cost: float
    iterations: int
    elapsed_sec: float


# -------------------------
# Helpers
# -------------------------

def _rng(seed: Optional[int]) -> np.random.Generator:
    return np.random.default_rng(seed if seed is not None else None)


def _choose_index_by_len(routes: List[List[int]], rng: np.random.Generator, min_len: int) -> Optional[int]:
    cand = [i for i, r in enumerate(routes) if len(r) >= min_len]
    if not cand:
        return None
    return int(rng.choice(cand))


def _compute_cost(coords: np.ndarray, routes: List[List[int]], edge_weight_type: str, round_euclidean: bool) -> float:
    D = build_distance_matrix(coords, edge_weight_type, round_euclidean=round_euclidean)
    return total_cost(D, routes)


def _copy_routes(routes: List[List[int]]) -> List[List[int]]:
    return [r[:] for r in routes]


# -------------------------
# Perturbations (feasible)
# -------------------------

def perturb_double_bridge(
    routes: List[List[int]],
    rng: np.random.Generator,
) -> bool:
    """
    Double-bridge on a single route (classic TSP-style kick), requires len>=8.
    Operates on one route only; preserves feasibility (load unchanged).
    """
    ridx = _choose_index_by_len(routes, rng, min_len=8)
    if ridx is None:
        return False
    r = routes[ridx]
    n = len(r)
    # choose 4 cut points (in order) ensuring separated segments
    cuts = sorted(rng.choice(np.arange(1, n), size=4, replace=False))
    a, b, c, d = cuts
    # segments: [0:a] [a:b] [b:c] [c:d] [d:n]
    new_r = r[0:a] + r[c:d] + r[b:c] + r[a:b] + r[d:n]
    routes[ridx] = new_r
    return True


def perturb_random_reloc(
    routes: List[List[int]],
    demand: np.ndarray,
    Q: int,
    rng: np.random.Generator,
    tries: int = 20,
) -> bool:
    """
    Try random single-customer relocations between routes while respecting capacity.
    Returns True if at least one relocation was applied.
    """
    applied = False
    m = len(routes)
    for _ in range(tries):
        if m < 2:
            break
        ra, rb = int(rng.integers(0, m)), int(rng.integers(0, m))
        if ra == rb or not routes[ra]:
            continue
        ia = int(rng.integers(0, len(routes[ra])))
        v = routes[ra][ia]
        dv = int(demand[v])
        # pick insert position in rb
        jb = int(rng.integers(0, len(routes[rb]) + 1))
        # capacity check
        load_rb = sum(int(demand[x]) for x in routes[rb])
        if load_rb + dv > Q:
            continue
        # apply
        node = routes[ra].pop(ia)
        routes[rb].insert(jb, node)
        applied = True
    return applied


def perturb_segment_swap(
    routes: List[List[int]],
    demand: np.ndarray,
    Q: int,
    rng: np.random.Generator,
    seg_len_min: int = 1,
    seg_len_max: int = 3,
) -> bool:
    """
    Swap two contiguous segments between two routes (no reversal), if capacities remain feasible.
    """
    if len(routes) < 2:
        return False
    # choose two distinct routes with enough length
    ridxs = [i for i, r in enumerate(routes) if len(r) >= seg_len_min]
    if len(ridxs) < 2:
        return False
    ra, rb = tuple(rng.choice(ridxs, size=2, replace=False))
    A, B = routes[ra], routes[rb]
    if not A or not B:
        return False

    # choose segment lengths
    La = int(rng.integers(seg_len_min, min(seg_len_max, len(A)) + 1))
    Lb = int(rng.integers(seg_len_min, min(seg_len_max, len(B)) + 1))
    ia = int(rng.integers(0, len(A) - La + 1))
    ib = int(rng.integers(0, len(B) - Lb + 1))

    segA = A[ia:ia+La]
    segB = B[ib:ib+Lb]

    # capacity check
    loadA = sum(int(demand[x]) for x in A)
    loadB = sum(int(demand[x]) for x in B)
    new_loadA = loadA - sum(int(demand[x]) for x in segA) + sum(int(demand[x]) for x in segB)
    new_loadB = loadB - sum(int(demand[x]) for x in segB) + sum(int(demand[x]) for x in segA)
    if new_loadA > Q or new_loadB > Q:
        return False

    # apply
    A[ia:ia+La] = segB
    B[ib:ib+Lb] = segA
    return True


def _apply_perturbation(
    routes: List[List[int]],
    demand: np.ndarray,
    Q: int,
    rng: np.random.Generator,
    cfg: ILSConfig,
) -> bool:
    p = rng.random()
    if p < cfg.p_double_bridge:
        return perturb_double_bridge(routes, rng)
    p -= cfg.p_double_bridge
    if p < cfg.p_random_reloc:
        return perturb_random_reloc(routes, demand, Q, rng, tries=cfg.reloc_tries)
    # else segment swap
    return perturb_segment_swap(routes, demand, Q, rng, cfg.seg_len_min, cfg.seg_len_max)


# -------------------------
# ILS main loop
# -------------------------

def run_ils(
    coords: np.ndarray,
    demand: np.ndarray,
    Q: int,
    routes_initial: List[List[int]],
    edge_weight_type: str = "EUC_2D",
    round_euclidean: bool = False,
    cfg: Optional[ILSConfig] = None,
    logger: Optional[logging.Logger] = None,
) -> ILSSolution:
    """
    Iterated Local Search wrapper around your local_search().
    - Starts from routes_initial
    - Repeats: perturb -> local_search -> accept?
    - Returns the best solution found within the time budget.
    """
    cfg = cfg or ILSConfig()
    rng = _rng(cfg.seed)

    # logging
    if logger is None and cfg.verbose:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%H:%M:%S",
        )
        logger = logging.getLogger("ILS")

    def log(msg: str):
        if logger is not None:
            logger.info(msg)

    start = time.perf_counter()

    # Distance matrix just for quick cost checks between LS runs
    D = build_distance_matrix(coords, edge_weight_type, round_euclidean=round_euclidean)

    # Initial LS (small slice) to polish the constructive (optional but recommended)
    remaining = cfg.time_limit_sec - (time.perf_counter() - start)
    initial_slice = max(0.2 * cfg.ls_time_slice_sec, 0.25)  # tiny polish
    initial_slice = min(initial_slice, max(0.1, remaining))
    base_ls = local_search(
        coords=coords, demand=demand, Q=Q,
        routes=[r[:] for r in routes_initial],
        edge_weight_type=edge_weight_type,
        round_euclidean=round_euclidean,
        k_nearest=cfg.k_nearest,
        max_passes_without_gain=cfg.max_passes_without_gain,
        time_limit_sec=initial_slice,
        verbose=False,
        check_capacity=True,
        improvement_eps=cfg.improvement_eps,
    )
    best_routes = [r[:] for r in base_ls.routes]
    best_cost = total_cost(D, best_routes)
    cur_routes = [r[:] for r in best_routes]
    cur_cost = float(best_cost)

    # SA temperature
    T = cfg.T_init_factor * max(best_cost, 1.0)

    it = 0
    log(f"ILS start | routes={len(cur_routes)} cost={cur_cost:.4f} | time_limit={cfg.time_limit_sec}s | T0={T:.4f}")

    while it < cfg.max_iters:
        it += 1
        elapsed = time.perf_counter() - start
        if elapsed >= cfg.time_limit_sec:
            log(f"[TIMEOUT] iter={it} | best={best_cost:.4f}")
            break

        # 1) Perturb a copy of the current solution
        cand_routes = _copy_routes(cur_routes)
        ok = _apply_perturbation(cand_routes, demand, Q, rng, cfg)
        if not ok:
            # fallback: try a different perturbation once
            ok = _apply_perturbation(cand_routes, demand, Q, rng, cfg)
            if not ok:
                # if nothing applies, skip iteration
                T *= cfg.T_cooling
                continue

        # 2) Run local search on the perturbed solution (bounded time slice)
        remaining = cfg.time_limit_sec - (time.perf_counter() - start)
        slice_sec = min(cfg.ls_time_slice_sec, max(0.05, remaining))
        ls_out: LSSolution = local_search(
            coords=coords, demand=demand, Q=Q,
            routes=cand_routes,
            edge_weight_type=edge_weight_type,
            round_euclidean=round_euclidean,
            k_nearest=cfg.k_nearest,
            max_passes_without_gain=cfg.max_passes_without_gain,
            time_limit_sec=slice_sec,
            verbose=False,
            check_capacity=True,
            improvement_eps=cfg.improvement_eps,
        )

        # 3) Evaluate candidate
        cand_routes = [r[:] for r in ls_out.routes]
        cand_cost = total_cost(D, cand_routes)

        # 4) Acceptance (SA-style)
        delta = cand_cost - cur_cost
        accept = False
        if cand_cost + cfg.improvement_eps < cur_cost:
            accept = True
        else:
            if T > 1e-12:
                prob = math.exp(-delta / max(T, 1e-12))
                if rng.random() < prob:
                    accept = True

        # 5) Update current / best
        if accept:
            cur_routes, cur_cost = cand_routes, cand_cost
            log(f"[ACC] iter={it} | cur={cur_cost:.4f} (Δ={delta:+.4f}) T={T:.4f}")
        else:
            log(f"[REJ] iter={it} | cand={cand_cost:.4f} worse by {delta:+.4f} | T={T:.4f}")

        if cur_cost + cfg.improvement_eps < best_cost:
            best_routes, best_cost = _copy_routes(cur_routes), cur_cost
            log(f"  -> new BEST | iter={it} best={best_cost:.4f}")

        # 6) Cool temperature
        T *= cfg.T_cooling

    total_elapsed = time.perf_counter() - start
    log(f"ILS done | iters={it} | best={best_cost:.4f} | elapsed={total_elapsed:.2f}s")
    return ILSSolution(routes=best_routes, cost=best_cost, iterations=it, elapsed_sec=total_elapsed)
