# lns.py
from __future__ import annotations

import time
import math
import random
from typing import List, Sequence, Optional, Tuple

import numpy as np

from vrptw_parser import VRPTWInstance
from utils import (
    build_distance_matrix,
    normalize_routes,
    solution_cost,
    is_feasible,
    is_route_feasible,
    route_load,
)
from constructors import clarke_wright_vrptw
from local_search import local_search_vrptw

EPS = 1e-6


# =========================
# Helpers
# =========================

def _clone_routes(routes: Sequence[Sequence[int]]) -> List[List[int]]:
    return [list(r) for r in routes]


def _safe_normalize(routes: Sequence[Sequence[int]]) -> List[List[int]]:
    r = normalize_routes(routes, depot=0)
    return [route for route in r if len(route) > 2]


def _routes_equal(a: Sequence[Sequence[int]], b: Sequence[Sequence[int]]) -> bool:
    if len(a) != len(b):
        return False
    for r1, r2 in zip(a, b):
        if len(r1) != len(r2):
            return False
        for x, y in zip(r1, r2):
            if x != y:
                return False
    return True


def _all_customers(routes: Sequence[Sequence[int]]) -> List[int]:
    return [v for r in routes for v in r if v != 0]


# =========================
# Destroy operators (ruin)
# =========================

def _random_removal(
    routes: List[List[int]],
    num_remove: int,
    rng: random.Random,
) -> Tuple[List[List[int]], List[int]]:
    routes = _clone_routes(routes)
    positions: List[Tuple[int, int]] = []
    for ri, r in enumerate(routes):
        for i in range(1, len(r) - 1):
            if r[i] != 0:
                positions.append((ri, i))

    rng.shuffle(positions)
    removed: List[int] = []

    for ri, i in positions:
        if len(removed) >= num_remove:
            break
        # guard against route already shortened
        if ri >= len(routes):
            continue
        r = routes[ri]
        if i <= 0 or i >= len(r) - 1:
            continue
        node = r[i]
        if node == 0:
            continue
        removed.append(node)
        r.pop(i)

    routes = _safe_normalize(routes)
    return routes, removed


def _worst_removal(
    inst: VRPTWInstance,
    routes: List[List[int]],
    D: np.ndarray,
    num_remove: int,
    rng: random.Random,
) -> Tuple[List[List[int]], List[int]]:
    routes = _clone_routes(routes)
    contribs: List[Tuple[float, int, int]] = []  # (contrib, ri, pos)

    for ri, r in enumerate(routes):
        for i in range(1, len(r) - 1):
            node = r[i]
            if node == 0:
                continue
            prev_n = r[i - 1]
            next_n = r[i + 1]
            c = D[prev_n, node] + D[node, next_n] - D[prev_n, next_n]
            contribs.append((c, ri, i))

    rng.shuffle(contribs)
    contribs.sort(key=lambda x: x[0], reverse=True)

    removed: List[int] = []
    removed_set = set()

    for _, ri, i in contribs:
        if len(removed) >= num_remove:
            break
        if ri >= len(routes):
            continue
        r = routes[ri]
        if i <= 0 or i >= len(r) - 1:
            continue
        node = r[i]
        if node == 0 or node in removed_set:
            continue
        removed.append(node)
        removed_set.add(node)
        r.pop(i)

    routes = _safe_normalize(routes)
    return routes, removed


def _shaw_relatedness(
    inst: VRPTWInstance,
    D: np.ndarray,
    i: int,
    j: int,
    alpha: float = 1.0,
    beta: float = 0.1,
    gamma: float = 0.1,
) -> float:
    # distance
    dij = D[i, j]
    # time windows
    ri, di = inst.ready_time[i], inst.due_time[i]
    rj, dj = inst.ready_time[j], inst.due_time[j]
    tw = abs(ri - rj) + abs(di - dj)
    # demand difference
    dq = abs(inst.demand[i] - inst.demand[j])
    return alpha * dij + beta * tw + gamma * dq


def _shaw_removal(
    inst: VRPTWInstance,
    routes: List[List[int]],
    D: np.ndarray,
    num_remove: int,
    rng: random.Random,
) -> Tuple[List[List[int]], List[int]]:
    routes = _clone_routes(routes)
    customers = _all_customers(routes)
    if not customers:
        return routes, []

    seed = rng.choice(customers)
    removed = [seed]
    remaining = set(customers)
    remaining.discard(seed)

    while len(removed) < num_remove and remaining:
        best_c = None
        best_rel = float("inf")
        for c in remaining:
            rel = min(_shaw_relatedness(inst, D, c, r) for r in removed)
            if rel < best_rel:
                best_rel = rel
                best_c = c
        if best_c is None:
            break
        removed.append(best_c)
        remaining.discard(best_c)

    removed_set = set(removed)
    new_routes: List[List[int]] = []
    for r in routes:
        nr = [v for v in r if v == 0 or v not in removed_set]
        new_routes.append(nr)

    new_routes = _safe_normalize(new_routes)
    return new_routes, removed


# =========================
# Repair operators (recreate)
# =========================

def _best_insertion_position(
    inst: VRPTWInstance,
    route: List[int],
    customer: int,
    D: np.ndarray,
) -> Optional[Tuple[float, int]]:
    """
    Best feasible insertion (delta_cost, pos) for 'customer' in 'route'.
    Return None if no feasible position.
    """
    base_load = route_load(route, inst)
    dem = inst.demand[customer]
    if base_load + dem > inst.capacity:
        return None

    best_delta = None
    best_pos = None

    for pos in range(1, len(route)):  # insert before pos
        prev_n = route[pos - 1]
        next_n = route[pos]
        delta = D[prev_n, customer] + D[customer, next_n] - D[prev_n, next_n]
        cand = route[:pos] + [customer] + route[pos:]
        if not is_route_feasible(cand, inst, D):
            continue
        if best_delta is None or delta < best_delta:
            best_delta = delta
            best_pos = pos

    if best_pos is None:
        return None
    return float(best_delta), int(best_pos)


def _greedy_repair(
    inst: VRPTWInstance,
    routes: List[List[int]],
    removed: List[int],
    D: np.ndarray,
    rng: random.Random,
) -> Optional[List[List[int]]]:
    routes = _clone_routes(routes)
    pending = removed[:]

    while pending:
        best_c = None
        best_r = None
        best_pos = None
        best_delta = None

        for c in pending:
            for ri, r in enumerate(routes):
                res = _best_insertion_position(inst, r, c, D)
                if res is None:
                    continue
                delta, pos = res
                if best_delta is None or delta < best_delta:
                    best_delta = delta
                    best_c = c
                    best_r = ri
                    best_pos = pos

        if best_c is None:
            # open new route
            c = pending[0]
            if inst.demand[c] > inst.capacity:
                return None
            nr = [0, c, 0]
            if not is_route_feasible(nr, inst, D):
                return None
            routes.append(nr)
            pending.pop(0)
        else:
            routes[best_r] = (
                routes[best_r][:best_pos]
                + [best_c]
                + routes[best_r][best_pos:]
            )
            pending.remove(best_c)

    return _safe_normalize(routes)


def _regret2_repair(
    inst: VRPTWInstance,
    routes: List[List[int]],
    removed: List[int],
    D: np.ndarray,
    rng: random.Random,
) -> Optional[List[List[int]]]:
    routes = _clone_routes(routes)
    pending = removed[:]

    while pending:
        best_c = None
        best_r = None
        best_pos = None
        best_regret = None

        for c in pending:
            options: List[Tuple[float, int, int]] = []
            for ri, r in enumerate(routes):
                res = _best_insertion_position(inst, r, c, D)
                if res is not None:
                    delta, pos = res
                    options.append((delta, ri, pos))

            if not options:
                continue

            options.sort(key=lambda x: x[0])
            best = options[0][0]
            second = options[1][0] if len(options) > 1 else best
            regret = second - best

            if best_regret is None or regret > best_regret + EPS:
                best_regret = regret
                best_c = c
                best_r = options[0][1]
                best_pos = options[0][2]

        if best_c is None:
            # open new route for first pending
            c = pending[0]
            if inst.demand[c] > inst.capacity:
                return None
            nr = [0, c, 0]
            if not is_route_feasible(nr, inst, D):
                return None
            routes.append(nr)
            pending.pop(0)
        else:
            routes[best_r] = (
                routes[best_r][:best_pos]
                + [best_c]
                + routes[best_r][best_pos:]
            )
            pending.remove(best_c)

    return _safe_normalize(routes)


# =========================
# LNS main
# =========================

def lns_vrptw(
    inst: VRPTWInstance,
    initial_routes: Optional[Sequence[Sequence[int]]] = None,
    D: Optional[np.ndarray] = None,
    max_iterations: int = 500,
    remove_fraction_min: float = 0.05,
    remove_fraction_max: float = 0.20,
    use_shaw: bool = True,
    use_worst: bool = True,
    apply_ls_every: int = 1,
    ls_max_iterations: int = 3000,
    ls_use_inter_route: bool = True,
    exploratory_accept_prob: float = 0.1,
    accept_equal_cost: bool = True,
    time_limit_sec: Optional[float] = None,
    seed: Optional[int] = None,
    verbose: bool = False,
) -> List[List[int]]:
    """
    Large Neighborhood Search (ruin-and-recreate) for VRPTW.

    Simplified vs ALNS:
      - Fixed set of destroy/repair operators (no adaptive weights).
      - Ruin with random / worst / Shaw (configurable).
      - Recreate with greedy or regret-2 (random choice).
      - Optional local search after reconstruction.
      - SA-style + exploratory acceptance to escape local minima.
      - Single global time limit; records time-to-best.

    Parameters:
        inst: VRPTW instance
        initial_routes: starting solution (if None, Clarke-Wright + LS)
        D: distance matrix
        max_iterations: number of ruin-recreate iterations
        remove_fraction_min/max: fraction of customers to remove each iteration
        use_shaw: enable Shaw removal
        use_worst: enable worst removal
        apply_ls_every: run local search every k iterations (>=1)
        ls_max_iterations: local search iterations per call
        ls_use_inter_route: LS neighborhood scope
        exploratory_accept_prob: prob. to accept non-improving candidate
                                 (in addition to SA-style acceptance)
        accept_equal_cost: accept structurally different equal-cost solutions
        time_limit_sec: global wall-clock time budget
        seed: RNG seed
        verbose: logging

    Returns:
        Best routes found.
    """
    if D is None:
        D = build_distance_matrix(inst)

    rng = random.Random(seed)
    start_time = time.time()

    def remaining_time() -> Optional[float]:
        if time_limit_sec is None:
            return None
        rem = time_limit_sec - (time.time() - start_time)
        return max(0.0, rem)

    # ---- Initial solution ----
    if initial_routes is None:
        routes0 = clarke_wright_vrptw(inst, D=D)
    else:
        routes0 = [list(r) for r in initial_routes]

    routes0 = _safe_normalize(routes0)
    if not is_feasible(routes0, inst, D, require_all_customers=True):
        raise ValueError("Initial solution for LNS is infeasible.")

    # strengthen with LS if time allows
    ls_budget = remaining_time()
    if ls_budget is not None and ls_budget <= 0:
        current = routes0
        if verbose:
            print("[LNS] Time limit exhausted before initial LS; using initial solution.")
    else:
        current = local_search_vrptw(
            inst,
            routes0,
            D=D,
            max_iterations=ls_max_iterations,
            use_inter_route=ls_use_inter_route,
            time_limit_sec=ls_budget,
            verbose=False,
        )
        current = _safe_normalize(current)

    if not is_feasible(current, inst, D, require_all_customers=True):
        raise ValueError("Local search produced infeasible solution in LNS initialization.")

    current_cost = solution_cost(current, D)
    best = _clone_routes(current)
    best_cost = current_cost
    time_to_best: Optional[float] = None

    if verbose:
        print(f"[LNS] Start: cost={best_cost:.4f}, routes={len(best)}")

    # Pre-calc removal fractions
    n_customers = inst.n_customers
    remove_fraction_min = max(0.0, min(remove_fraction_min, 1.0))
    remove_fraction_max = max(remove_fraction_min, min(remove_fraction_max, 1.0))

    # Simple SA-like temperature schedule based on initial cost
    T0 = 0.01 * max(1.0, current_cost)
    T_end = 0.001 * max(1.0, current_cost)

    def temperature(iteration: int) -> float:
        if max_iterations <= 1:
            return T_end
        frac = min(1.0, iteration / float(max_iterations - 1))
        log_t0 = math.log(T0 + 1e-12)
        log_t1 = math.log(T_end + 1e-12)
        return math.exp(log_t0 + (log_t1 - log_t0) * frac)

    # ---- Main ruin & recreate loop ----
    for it in range(1, max_iterations + 1):
        rem = remaining_time()
        if rem is not None and rem <= 0:
            if verbose:
                print(f"[LNS] Time limit {time_limit_sec}s reached at iter {it}.")
            break

        # choose q
        q_min = max(1, int(remove_fraction_min * n_customers))
        q_max = max(q_min, int(remove_fraction_max * n_customers))
        q = rng.randint(q_min, max(q_min, q_max))

        # choose destroy operator
        destroy_ops = ["random"]
        if use_worst:
            destroy_ops.append("worst")
        if use_shaw:
            destroy_ops.append("shaw")
        destroy_choice = rng.choice(destroy_ops)

        # RUIN
        if destroy_choice == "random":
            partial, removed = _random_removal(current, q, rng)
        elif destroy_choice == "worst":
            partial, removed = _worst_removal(inst, current, D, q, rng)
        else:  # "shaw"
            partial, removed = _shaw_removal(inst, current, D, q, rng)

        if not removed:
            continue

        # choose repair operator
        repair_choice = rng.choice(["greedy", "regret2"])

        # RECREATE
        if repair_choice == "greedy":
            candidate = _greedy_repair(inst, partial, removed, D, rng)
        else:
            candidate = _regret2_repair(inst, partial, removed, D, rng)

        if candidate is None:
            continue

        candidate = _safe_normalize(candidate)
        if not is_feasible(candidate, inst, D, require_all_customers=True):
            continue

        # optional LS
        if apply_ls_every > 0 and (it % apply_ls_every == 0):
            ls_budget = remaining_time()
            if ls_budget is not None and ls_budget <= 0:
                # no time left for LS; keep raw candidate
                pass
            else:
                candidate = local_search_vrptw(
                    inst,
                    candidate,
                    D=D,
                    max_iterations=ls_max_iterations,
                    use_inter_route=ls_use_inter_route,
                    time_limit_sec=ls_budget,
                    verbose=False,
                )
                candidate = _safe_normalize(candidate)
                if not is_feasible(candidate, inst, D, require_all_customers=True):
                    continue

        cand_cost = solution_cost(candidate, D)
        cand_differs = not _routes_equal(candidate, current)

        # Acceptance rule:
        #  - Always accept strictly better.
        #  - Accept equal if allowed and structurally different.
        #  - Otherwise SA-style + exploratory random accept.
        accepted = False
        T = temperature(it)

        if cand_cost < current_cost - EPS:
            accepted = True
        elif accept_equal_cost and abs(cand_cost - current_cost) <= EPS and cand_differs:
            accepted = True
        else:
            # SA-style for worse candidates
            if cand_cost > current_cost + EPS and T > 0:
                prob_sa = math.exp(-(cand_cost - current_cost) / max(T, 1e-12))
            else:
                prob_sa = 0.0

            prob_expl = exploratory_accept_prob if cand_differs else 0.0
            if rng.random() < max(prob_sa, prob_expl):
                accepted = True

        if accepted:
            current = _clone_routes(candidate)
            current_cost = cand_cost

        # update global best
        if cand_cost < best_cost - EPS:
            best = _clone_routes(candidate)
            best_cost = cand_cost
            time_to_best = time.time() - start_time
            if verbose:
                print(
                    f"[LNS] Iter {it}: new best {best_cost:.4f}, "
                    f"routes={len(best)}, ruin={destroy_choice}, repair={repair_choice}"
                )
        elif verbose and it % max(1, max_iterations // 10) == 0:
            print(
                f"[LNS] Iter {it}: current={current_cost:.4f}, "
                f"best={best_cost:.4f}, T={T:.4g}"
            )

    if verbose:
        if time_to_best is not None:
            print(f"[LNS] Final best={best_cost:.4f}, routes={len(best)}")
            print(f"[LNS] Time to best: {time_to_best:.4f} s")
        else:
            print("[LNS] No improvement over initial solution.")

    try:
        lns_vrptw.time_to_best = time_to_best
    except Exception:
        pass

    return best
