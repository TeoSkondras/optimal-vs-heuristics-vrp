# local_search.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Optional

import numpy as np

from vrptw_parser import VRPTWInstance
from utils import (
    build_distance_matrix,
    normalize_routes,
    solution_cost,
    route_cost,
    route_load,
    is_route_feasible,
    is_feasible,
    delta_cost_relocate_intra,
    delta_cost_swap_intra,
    delta_cost_2opt,
    delta_cost_relocate_inter,
    delta_cost_swap_inter,
    is_route_time_feasible_with_prefix,
    compute_route_schedule,
)
import time

EPS = 1e-6


@dataclass
class Move:
    kind: str          # "relocate_intra", "swap_intra", "2opt", "relocate_inter", "swap_inter"
    delta: float       # new_cost - old_cost (negative is improvement)
    r1: int            # primary route index
    i1: int            # primary index
    r2: Optional[int] = None  # secondary route index (for inter moves)
    i2: Optional[int] = None  # secondary index (for inter moves)
    j: Optional[int] = None   # insertion pos / 2-opt end index


# ---------------- Low-level route edit helpers ----------------

def _apply_relocate_intra(route: Sequence[int], i: int, j: int) -> List[int]:
    new_route = list(route)
    node = new_route.pop(i)
    # After removal at i, indices > i shift left
    if j > i:
        j -= 1
    new_route.insert(j, node)
    return new_route


def _apply_swap_intra(route: Sequence[int], i: int, j: int) -> List[int]:
    new_route = list(route)
    new_route[i], new_route[j] = new_route[j], new_route[i]
    return new_route


def _apply_2opt(route: Sequence[int], i: int, j: int) -> List[int]:
    new_route = list(route)
    new_route[i:j + 1] = reversed(new_route[i:j + 1])
    return new_route


def _apply_relocate_inter(
    routes: Sequence[Sequence[int]],
    r_from: int,
    i_from: int,
    r_to: int,
    j_to: int,
) -> List[List[int]]:
    """
    Move node at (r_from, i_from) into route r_to before position j_to.
    Assumes indices valid and depots not moved.
    """
    new_routes = [list(r) for r in routes]

    node = new_routes[r_from].pop(i_from)
    if j_to > len(new_routes[r_to]):
        j_to = len(new_routes[r_to])
    new_routes[r_to].insert(j_to, node)

    # Normalize & clean trivial routes
    new_routes = normalize_routes(new_routes, depot=0)
    new_routes = [r for r in new_routes if len(r) > 2]
    return new_routes


def _apply_swap_inter(
    routes: Sequence[Sequence[int]],
    r1: int,
    i1: int,
    r2: int,
    i2: int,
) -> List[List[int]]:
    """
    Swap customers at (r1, i1) and (r2, i2) across routes.
    """
    new_routes = [list(r) for r in routes]
    new_routes[r1][i1], new_routes[r2][i2] = new_routes[r2][i2], new_routes[r1][i1]
    return new_routes


# ---------------- Neighborhood search ----------------

def _find_best_move(
    inst: VRPTWInstance,
    routes: Sequence[Sequence[int]],
    D: np.ndarray,
    current_cost: float,
    use_inter_route: bool = True,
    loads: Optional[List[int]] = None,
    schedules: Optional[List[object]] = None,
) -> Optional[Move]:
    """
    Explore neighborhoods and return best improving move (if any):

      - Intra-route:
          * relocate i -> j
          * swap(i, j)
          * 2-opt(i, j)
      - Inter-route (if enabled):
          * relocate between routes
          * swap between routes

    All candidates are checked for VRPTW feasibility.
    """
    best_move: Optional[Move] = None
    n_routes = len(routes)
    # local bindings for speed
    D_local = D
    inst_demand = inst.demand
    EPS_local = EPS

    # ---- Intra-route moves ----
    for r_idx, route in enumerate(routes):
        m = len(route)
        if m <= 3:
            continue

        # Relocate within route
        for i in range(1, m - 1):  # internal (no depot)
            if route[i] == 0:
                continue
            for j in range(1, m):   # insert before j
                if j == i or j == i + 1:
                    continue
                if j <= 0 or j >= m:
                    continue

                delta = delta_cost_relocate_intra(route, i, j, D)
                if delta >= -EPS:
                    continue

                new_route = _apply_relocate_intra(route, i, j)
                if not is_route_feasible(new_route, inst, D):
                    continue

                if (best_move is None) or (delta < best_move.delta):
                    best_move = Move(
                        kind="relocate_intra",
                        delta=delta,
                        r1=r_idx,
                        i1=i,
                        j=j,
                    )

        # Swap within route
        for i in range(1, m - 1):
            if route[i] == 0:
                continue
            for j in range(i + 1, m - 1):
                if route[j] == 0:
                    continue

                delta = delta_cost_swap_intra(route, i, j, D)
                if delta >= -EPS:
                    continue

                new_route = _apply_swap_intra(route, i, j)
                if not is_route_feasible(new_route, inst, D):
                    continue

                if (best_move is None) or (delta < best_move.delta):
                    best_move = Move(
                        kind="swap_intra",
                        delta=delta,
                        r1=r_idx,
                        i1=i,
                        j=j,
                    )

        # 2-opt within route
        for i in range(1, m - 2):
            for j in range(i + 1, m - 1):
                delta = delta_cost_2opt(route, i, j, D)
                if delta >= -EPS:
                    continue

                new_route = _apply_2opt(route, i, j)
                if not is_route_feasible(new_route, inst, D):
                    continue

                if (best_move is None) or (delta < best_move.delta):
                    best_move = Move(
                        kind="2opt",
                        delta=delta,
                        r1=r_idx,
                        i1=i,
                        j=j,
                    )

    # ---- Inter-route moves ----
    if use_inter_route and n_routes > 1:
        # cache loads to avoid repeated summation
        if loads is None:
            loads = [route_load(r, inst) for r in routes]
        # Relocate between routes
        for r_from in range(n_routes):
            route_from = routes[r_from]
            m_from = len(route_from)
            if m_from <= 3:
                continue
            load_from = loads[r_from]

            for i in range(1, m_from - 1):
                node = route_from[i]
                if node == 0:
                    continue
                dem_node = inst.demand[node]

                for r_to in range(n_routes):
                    if r_to == r_from:
                        continue
                    route_to = routes[r_to]
                    m_to = len(route_to)
                    load_to = loads[r_to]

                    if load_to + dem_node > inst.capacity:
                        continue
                    if load_from - dem_node < 0:
                        continue

                    for j in range(1, m_to):  # insert before j
                        try:
                            delta = delta_cost_relocate_inter(route_from, i, route_to, j, D_local)
                        except ValueError:
                            continue

                        if delta >= -EPS_local:
                            continue

                        # Build the two affected routes and check their feasibility only
                        node = route_from[i]
                        new_from = list(route_from)
                        new_from.pop(i)
                        new_to = list(route_to)
                        if j > len(new_to):
                            jj = len(new_to)
                        else:
                            jj = j
                        new_to.insert(jj, node)

                        # capacity quick checks
                        if route_load(new_from, inst) > inst.capacity:
                            continue
                        if route_load(new_to, inst) > inst.capacity:
                            continue

                        # incremental time-window checks using cached schedules when available
                        sched_from = None
                        sched_to = None
                        if schedules is not None:
                            sched_from = schedules[r_from]
                            sched_to = schedules[r_to]

                        # For removal, earliest changed index is i-1 (keep prefix up to i-1)
                        pref_from = max(1, i)  # keep at least depot
                        # For insertion, earliest changed index is jj-1
                        pref_to = max(1, jj)

                        ok_from = is_route_time_feasible_with_prefix(route_from, sched_from, new_from, inst, D_local, pref_from) if sched_from is not None else is_route_feasible(new_from, inst, D_local)
                        if not ok_from:
                            continue

                        ok_to = is_route_time_feasible_with_prefix(route_to, sched_to, new_to, inst, D_local, pref_to) if sched_to is not None else is_route_feasible(new_to, inst, D_local)
                        if not ok_to:
                            continue

                        if (best_move is None) or (delta < best_move.delta):
                            best_move = Move(
                                kind="relocate_inter",
                                delta=delta,
                                r1=r_from,
                                i1=i,
                                r2=r_to,
                                j=j,
                            )

        # Swap between routes
        for r1 in range(n_routes):
            route1 = routes[r1]
            m1 = len(route1)
            if m1 <= 3:
                continue
            load1 = route_load(route1, inst)

            for r2 in range(r1 + 1, n_routes):
                route2 = routes[r2]
                m2 = len(route2)
                if m2 <= 3:
                    continue
                load2 = route_load(route2, inst)

                for i in range(1, m1 - 1):
                    n1 = route1[i]
                    if n1 == 0:
                        continue
                    d1 = inst.demand[n1]

                    for j in range(1, m2 - 1):
                        n2 = route2[j]
                        if n2 == 0:
                            continue
                        d2 = inst.demand[n2]

                        # Quick capacity check
                        if load1 - d1 + d2 > inst.capacity:
                            continue
                        if load2 - d2 + d1 > inst.capacity:
                            continue

                        try:
                            delta = delta_cost_swap_inter(route1, i, route2, j, D_local)
                        except ValueError:
                            continue

                        if delta >= -EPS_local:
                            continue

                        # build candidate routes and check feasibility of affected routes
                        new_r1 = list(route1)
                        new_r2 = list(route2)
                        new_r1[i], new_r2[j] = new_r2[j], new_r1[i]

                        # capacity quick checks
                        if route_load(new_r1, inst) > inst.capacity:
                            continue
                        if route_load(new_r2, inst) > inst.capacity:
                            continue

                        sched1 = schedules[r1] if schedules is not None else None
                        sched2 = schedules[r2] if schedules is not None else None

                        # earliest changed index is i-1 and j-1 respectively
                        pref1 = max(1, i)
                        pref2 = max(1, j)

                        ok1 = is_route_time_feasible_with_prefix(route1, sched1, new_r1, inst, D_local, pref1) if sched1 is not None else is_route_feasible(new_r1, inst, D_local)
                        if not ok1:
                            continue
                        ok2 = is_route_time_feasible_with_prefix(route2, sched2, new_r2, inst, D_local, pref2) if sched2 is not None else is_route_feasible(new_r2, inst, D_local)
                        if not ok2:
                            continue

                        if (best_move is None) or (delta < best_move.delta):
                            best_move = Move(
                                kind="swap_inter",
                                delta=delta,
                                r1=r1,
                                i1=i,
                                r2=r2,
                                i2=j,
                            )

    return best_move


# ---------------- Apply move & main driver ----------------

def _apply_move(
    inst: VRPTWInstance,
    routes: Sequence[Sequence[int]],
    move: Move,
) -> List[List[int]]:
    """
    Return a new set of routes with `move` applied.
    """
    if move.kind == "relocate_intra":
        new_routes = [list(r) for r in routes]
        r = move.r1
        new_routes[r] = _apply_relocate_intra(new_routes[r], move.i1, move.j)
        return new_routes

    if move.kind == "swap_intra":
        new_routes = [list(r) for r in routes]
        r = move.r1
        new_routes[r] = _apply_swap_intra(new_routes[r], move.i1, move.j)
        return new_routes

    if move.kind == "2opt":
        new_routes = [list(r) for r in routes]
        r = move.r1
        new_routes[r] = _apply_2opt(new_routes[r], move.i1, move.j)
        return new_routes

    if move.kind == "relocate_inter":
        return _apply_relocate_inter(routes, move.r1, move.i1, move.r2, move.j)

    if move.kind == "swap_inter":
        return _apply_swap_inter(routes, move.r1, move.i1, move.r2, move.i2)

    raise ValueError(f"Unknown move kind: {move.kind}")


def local_search_vrptw(
    inst: VRPTWInstance,
    routes: Sequence[Sequence[int]],
    D: Optional[np.ndarray] = None,
    max_iterations: int = 10_000,
    use_inter_route: bool = True,
    verbose: bool = False,
    time_limit_sec: Optional[float] = None,
) -> List[List[int]]:
    """
    VRPTW local search engine.

    - Input: feasible solution (e.g., from clarke_wright_vrptw).
    - Neighborhoods:
        * relocate_intra
        * swap_intra
        * 2opt (intra)
        * relocate_inter (optional)
        * swap_inter (optional)
    - Strategy: best-improvement; loop until no improving move.

    Returns:
      Locally optimal routes under selected neighborhoods.
    """
    if D is None:
        D = build_distance_matrix(inst)

    current_routes = normalize_routes(routes, depot=0)
    current_routes = [r for r in current_routes if len(r) > 2]

    if not is_feasible(current_routes, inst, D, require_all_customers=True):
        raise ValueError("Initial solution is infeasible for VRPTW.")

    current_cost = solution_cost(current_routes, D)
    if verbose:
        print(f"[LS] Initial cost: {current_cost:.4f}, routes: {len(current_routes)}")
    # Precompute per-route loads & schedules for incremental feasibility checks
    loads = [route_load(r, inst) for r in current_routes]
    schedules = [compute_route_schedule(r, inst, D) for r in current_routes]

    start_time = time.time()

    for it in range(max_iterations):
        if time_limit_sec is not None and (time.time() - start_time) > time_limit_sec:
            if verbose:
                print(f"[LS] Time limit {time_limit_sec}s reached at iter {it}. Stop.")
            break

        move = _find_best_move(
            inst, current_routes, D, current_cost, use_inter_route, loads=loads, schedules=schedules
        )

        if move is None or move.delta >= -EPS:
            if verbose:
                print(f"[LS] No improving move found at iter {it}. Stop.")
            break

        new_routes = _apply_move(inst, current_routes, move)
        new_routes = normalize_routes(new_routes, depot=0)
        new_routes = [r for r in new_routes if len(r) > 2]

        new_cost = solution_cost(new_routes, D)

        # Safety: accept only if consistent with predicted improvement
        if new_cost > current_cost + 1e-4:
            if verbose:
                print(
                    f"[LS][WARN] Move {move.kind} predicted delta={move.delta:.4f}, "
                    f"but cost increased {current_cost:.4f} -> {new_cost:.4f}. Abort."
                )
            break

        current_routes = new_routes
        current_cost = new_cost

        # update loads & schedules
        loads = [route_load(r, inst) for r in current_routes]
        schedules = [compute_route_schedule(r, inst, D) for r in current_routes]

        if verbose:
            print(
                f"[LS] Iter {it}: {move.kind}, "
                f"delta={move.delta:.4f}, cost={current_cost:.4f}, "
                f"routes={len(current_routes)}"
            )

    return current_routes
