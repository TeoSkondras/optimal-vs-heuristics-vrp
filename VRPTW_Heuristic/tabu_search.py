# tabu_search.py
from __future__ import annotations

import time
from typing import List, Sequence, Optional, Dict, Tuple

import numpy as np

from vrptw_parser import VRPTWInstance
from utils import (
    build_distance_matrix,
    normalize_routes,
    solution_cost,
    route_load,
    is_route_feasible,
    is_feasible,
    delta_cost_relocate_intra,
    delta_cost_swap_intra,
    delta_cost_2opt,
    delta_cost_relocate_inter,
    delta_cost_swap_inter,
    compute_route_schedule,
    is_route_time_feasible_with_prefix,
)
from local_search import (
    _apply_relocate_intra,
    _apply_swap_intra,
    _apply_2opt,
    _apply_relocate_inter,
    _apply_swap_inter,
    _apply_move,
    Move,
)

EPS = 1e-6


def _is_tabu(node_ids: Sequence[int], tabu_until: Dict[int, int], iteration: int) -> bool:
    """Return True if any of the given nodes is tabu at this iteration."""
    for nid in node_ids:
        if tabu_until.get(nid, -1) > iteration:
            return True
    return False


def _moved_nodes_for_move(move: Move, routes: Sequence[Sequence[int]]) -> List[int]:
    """Identify nodes affected by a move for tabu marking (simple node-based memory)."""
    kind = move.kind

    if kind == "relocate_intra":
        return [routes[move.r1][move.i1]]

    if kind == "swap_intra":
        r = routes[move.r1]
        return [r[move.i1], r[move.j]]

    if kind == "2opt":
        # Mark endpoints of the reversed segment
        r = routes[move.r1]
        return [r[move.i1], r[move.j]]

    if kind == "relocate_inter":
        return [routes[move.r1][move.i1]]

    if kind == "swap_inter":
        r1 = routes[move.r1]
        r2 = routes[move.r2]
        return [r1[move.i1], r2[move.i2]]

    return []


def _select_best_tabu_move(
    inst: VRPTWInstance,
    routes: Sequence[Sequence[int]],
    D: np.ndarray,
    current_cost: float,
    best_cost_global: float,
    tabu_until: Dict[int, int],
    iteration: int,
    use_intra_relocate: bool,
    use_intra_swap: bool,
    use_2opt: bool,
    use_inter_relocate: bool,
    use_inter_swap: bool,
) -> Tuple[Optional[Move], Optional[float]]:
    """
    Evaluate neighborhood and return the best admissible move under Tabu rules.

    - Considers both improving and non-improving moves.
    - Node-based tabu: nodes touched by a move are tabu for a number of iterations.
    - Aspiration: allow tabu move if it improves global best.

    Returns:
        (best_move, best_move_cost_after)
    """
    best_move: Optional[Move] = None
    best_cost_after: Optional[float] = None

    n_routes = len(routes)
    loads = [route_load(r, inst) for r in routes]  # for inter-route capacity checks
    # Precompute schedules for incremental TW checks
    schedules = [compute_route_schedule(r, inst, D) for r in routes]
    D_local = D
    EPS_local = EPS

    # ---------------- Intra-route moves ----------------
    for r_idx, route in enumerate(routes):
        m = len(route)
        if m <= 3:
            continue

        # Relocate within route
        if use_intra_relocate:
            for i in range(1, m - 1):
                if route[i] == 0:
                    continue
                for j in range(1, m):
                    if j == i or j == i + 1:
                        continue
                    if j <= 0 or j >= m:
                        continue

                    delta = delta_cost_relocate_intra(route, i, j, D)
                    candidate_cost = current_cost + delta

                    move = Move(kind="relocate_intra", delta=delta, r1=r_idx, i1=i, j=j)
                    moved_nodes = _moved_nodes_for_move(move, routes)
                    tabu = _is_tabu(moved_nodes, tabu_until, iteration)

                    # Aspiration: allow tabu move only if it beats global best
                    if tabu and candidate_cost >= best_cost_global - EPS:
                        continue

                    new_route = _apply_relocate_intra(route, i, j)
                    if not is_route_feasible(new_route, inst, D):
                        continue

                    if (best_move is None) or (candidate_cost < best_cost_after - EPS):
                        best_move = move
                        best_cost_after = candidate_cost

        # Swap within route
        if use_intra_swap:
            for i in range(1, m - 1):
                if route[i] == 0:
                    continue
                for j in range(i + 1, m - 1):
                    if route[j] == 0:
                        continue

                    delta = delta_cost_swap_intra(route, i, j, D)
                    candidate_cost = current_cost + delta

                    move = Move(kind="swap_intra", delta=delta, r1=r_idx, i1=i, j=j)
                    moved_nodes = _moved_nodes_for_move(move, routes)
                    tabu = _is_tabu(moved_nodes, tabu_until, iteration)

                    if tabu and candidate_cost >= best_cost_global - EPS:
                        continue

                    new_route = _apply_swap_intra(route, i, j)
                    if not is_route_feasible(new_route, inst, D):
                        continue

                    if (best_move is None) or (candidate_cost < best_cost_after - EPS):
                        best_move = move
                        best_cost_after = candidate_cost

        # 2-opt within route
        if use_2opt:
            for i in range(1, m - 2):
                for j in range(i + 1, m - 1):
                    delta = delta_cost_2opt(route, i, j, D)
                    candidate_cost = current_cost + delta

                    move = Move(kind="2opt", delta=delta, r1=r_idx, i1=i, j=j)
                    moved_nodes = _moved_nodes_for_move(move, routes)
                    tabu = _is_tabu(moved_nodes, tabu_until, iteration)

                    if tabu and candidate_cost >= best_cost_global - EPS:
                        continue

                    new_route = _apply_2opt(route, i, j)
                    if not is_route_feasible(new_route, inst, D):
                        continue

                    if (best_move is None) or (candidate_cost < best_cost_after - EPS):
                        best_move = move
                        best_cost_after = candidate_cost

    # ---------------- Inter-route moves ----------------
    if n_routes > 1:
        # Relocate between routes
        if use_inter_relocate:
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
                    d_node = inst.demand[node]

                    for r_to in range(n_routes):
                        if r_to == r_from:
                            continue
                        route_to = routes[r_to]
                        m_to = len(route_to)
                        if m_to < 2:
                            continue
                        load_to = loads[r_to]

                        # Capacity check
                        if load_to + d_node > inst.capacity:
                            continue
                        if load_from - d_node < 0:
                            continue

                        for j in range(1, m_to):
                            try:
                                delta = delta_cost_relocate_inter(
                                    route_from, i, route_to, j, D
                                )
                            except ValueError:
                                continue

                            candidate_cost = current_cost + delta
                            move = Move(
                                kind="relocate_inter",
                                delta=delta,
                                r1=r_from,
                                i1=i,
                                r2=r_to,
                                j=j,
                            )

                            moved_nodes = _moved_nodes_for_move(move, routes)
                            tabu = _is_tabu(moved_nodes, tabu_until, iteration)
                            if tabu and candidate_cost >= best_cost_global - EPS:
                                continue

                            # Check only affected routes
                            new_from = list(route_from)
                            new_from.pop(i)
                            new_to = list(route_to)
                            jj = j if j <= len(new_to) else len(new_to)
                            new_to.insert(jj, node)

                            # incremental TW checks using cached schedules
                            sched_from = schedules[r_from]
                            sched_to = schedules[r_to]
                            pref_from = max(1, i)
                            jj = j if j <= len(new_to) else len(new_to)
                            pref_to = max(1, jj)
                            okf = is_route_time_feasible_with_prefix(route_from, sched_from, new_from, inst, D_local, pref_from)
                            if not okf:
                                continue
                            okt = is_route_time_feasible_with_prefix(route_to, sched_to, new_to, inst, D_local, pref_to)
                            if not okt:
                                continue

                            if (best_move is None) or (candidate_cost < best_cost_after - EPS):
                                best_move = move
                                best_cost_after = candidate_cost

        # Swap between routes
        if use_inter_swap:
            for r1 in range(n_routes):
                route1 = routes[r1]
                m1 = len(route1)
                if m1 <= 3:
                    continue
                load1 = loads[r1]

                for r2 in range(r1 + 1, n_routes):
                    route2 = routes[r2]
                    m2 = len(route2)
                    if m2 <= 3:
                        continue
                    load2 = loads[r2]

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

                            # Capacity quick check
                            if load1 - d1 + d2 > inst.capacity:
                                continue
                            if load2 - d2 + d1 > inst.capacity:
                                continue

                            try:
                                delta = delta_cost_swap_inter(route1, i, route2, j, D)
                            except ValueError:
                                continue

                            candidate_cost = current_cost + delta
                            move = Move(
                                kind="swap_inter",
                                delta=delta,
                                r1=r1,
                                i1=i,
                                r2=r2,
                                i2=j,
                            )

                            moved_nodes = _moved_nodes_for_move(move, routes)
                            tabu = _is_tabu(moved_nodes, tabu_until, iteration)
                            if tabu and candidate_cost >= best_cost_global - EPS:
                                continue

                            # Check only affected routes
                            new_r1 = list(route1)
                            new_r2 = list(route2)
                            new_r1[i], new_r2[j] = new_r2[j], new_r1[i]

                            sched1 = schedules[r1]
                            sched2 = schedules[r2]
                            pref1 = max(1, i)
                            pref2 = max(1, j)
                            ok1 = is_route_time_feasible_with_prefix(route1, sched1, new_r1, inst, D_local, pref1)
                            if not ok1:
                                continue
                            ok2 = is_route_time_feasible_with_prefix(route2, sched2, new_r2, inst, D_local, pref2)
                            if not ok2:
                                continue

                            if (best_move is None) or (candidate_cost < best_cost_after - EPS):
                                best_move = move
                                best_cost_after = candidate_cost

    return best_move, best_cost_after


def tabu_search_vrptw(
    inst: VRPTWInstance,
    routes: Sequence[Sequence[int]],
    D: Optional[np.ndarray] = None,
    max_iterations: int = 1000,
    tenure: int = 15,
    use_intra_relocate: bool = True,
    use_intra_swap: bool = True,
    use_2opt: bool = True,
    use_inter_relocate: bool = True,
    use_inter_swap: bool = True,
    time_limit_sec: Optional[float] = None,
    verbose: bool = False,
) -> List[List[int]]:
    """
    Tabu Search for VRPTW on top of an initial feasible solution.

    Parameters:
        inst: VRPTWInstance
        routes: initial feasible solution (e.g. Clarke–Wright or LS result)
        D: distance matrix (if None, computed via build_distance_matrix)
        max_iterations: max tabu iterations
        tenure: tabu tenure (iterations) for moved nodes
        use_*: toggles for different move types
        time_limit_sec: optional wall-clock time limit
        verbose: print progress

    Returns:
        Best solution (routes) found.
    """
    if D is None:
        D = build_distance_matrix(inst)

    # Normalize and clean routes
    current_routes = normalize_routes(routes, depot=0)
    current_routes = [r for r in current_routes if len(r) > 2]

    if not is_feasible(current_routes, inst, D, require_all_customers=True):
        raise ValueError("Initial solution is infeasible for VRPTW.")

    current_cost = solution_cost(current_routes, D)
    best_routes = [list(r) for r in current_routes]
    best_cost = current_cost
    best_time: Optional[float] = None

    if verbose:
        print(f"[TS] Initial cost: {current_cost:.4f}, routes: {len(current_routes)}")

    tabu_until: Dict[int, int] = {}  # node_id -> iteration where tabu expires

    start_time = time.time()

    for it in range(1, max_iterations + 1):
        if time_limit_sec is not None and (time.time() - start_time) > time_limit_sec:
            if verbose:
                print(f"[TS] Time limit {time_limit_sec}s reached at iter {it}. Stop.")
            break

        move, move_cost_after = _select_best_tabu_move(
            inst=inst,
            routes=current_routes,
            D=D,
            current_cost=current_cost,
            best_cost_global=best_cost,
            tabu_until=tabu_until,
            iteration=it,
            use_intra_relocate=use_intra_relocate,
            use_intra_swap=use_intra_swap,
            use_2opt=use_2opt,
            use_inter_relocate=use_inter_relocate,
            use_inter_swap=use_inter_swap,
        )

        if move is None or move_cost_after is None:
            if verbose:
                print(f"[TS] No admissible moves at iter {it}. Stop.")
            break

        # Determine which nodes become tabu BEFORE applying the move
        moved_nodes = _moved_nodes_for_move(move, current_routes)

        # Apply move
        new_routes = _apply_move(inst, current_routes, move)
        new_routes = normalize_routes(new_routes, depot=0)
        new_routes = [r for r in new_routes if len(r) > 2]

        # Safety: feasibility & cost consistency
        if not is_feasible(new_routes, inst, D, require_all_customers=True):
            if verbose:
                print("[TS][WARN] Selected move produced infeasible solution. Stop.")
            break

        real_cost = solution_cost(new_routes, D)
        if abs(real_cost - move_cost_after) > 1e-3 and verbose:
            print(
                f"[TS][WARN] Cost mismatch at iter {it}: "
                f"delta-based={move_cost_after:.4f}, recomputed={real_cost:.4f}"
            )

        current_routes = new_routes
        current_cost = real_cost

        # Update tabu list for moved nodes
        for nid in moved_nodes:
            if nid != 0:
                tabu_until[nid] = it + tenure

        # Update global best
        if current_cost + EPS < best_cost:
            best_cost = current_cost
            best_routes = [list(r) for r in current_routes]
            # record time to reach new global best
            best_time = time.time() - start_time
            if verbose:
                print(
                    f"[TS] Iter {it}: new best {best_cost:.4f}, "
                    f"routes={len(best_routes)}, move={move.kind}"
                )
        else:
            if verbose:
                print(
                    f"[TS] Iter {it}: cost {current_cost:.4f}, "
                    f"best {best_cost:.4f}, move={move.kind}"
                )

    # Print and store time-to-best metadata
    if best_time is None:
        if verbose:
            print("[TS] No improvement was found; time-to-best is undefined.")
    else:
        if verbose:
            print(f"[TS] Time to best solution: {best_time:.4f} s")

    # attach as attribute to the function for external access
    try:
        tabu_search_vrptw.time_to_best = best_time
    except Exception:
        # non-critical: if attaching fails, continue silently
        pass

    return best_routes
