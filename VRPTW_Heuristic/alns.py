# alns.py
from __future__ import annotations

import time
import math
import random
from dataclasses import dataclass
from typing import List, Sequence, Optional, Dict, Tuple

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


def _get_all_customers(routes: Sequence[Sequence[int]]) -> List[int]:
    return [v for r in routes for v in r if v != 0]


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


# =========================
# Destroy operators
# =========================

def _random_removal(
    inst: VRPTWInstance,
    routes: List[List[int]],
    D: np.ndarray,
    num_remove: int,
    rng: random.Random,
) -> Tuple[List[List[int]], List[int]]:
    """
    Randomly remove `num_remove` customers from the solution.
    """
    routes = _clone_routes(routes)
    all_positions: List[Tuple[int, int]] = []  # (route_idx, pos)
    for r_idx, r in enumerate(routes):
        for i in range(1, len(r) - 1):
            if r[i] != 0:
                all_positions.append((r_idx, i))
    rng.shuffle(all_positions)

    removed: List[int] = []
    for r_idx, i in all_positions:
        if len(removed) >= num_remove:
            break
        if i <= 0 or i >= len(routes[r_idx]) - 1:
            continue
        node = routes[r_idx][i]
        if node == 0:
            continue
        removed.append(node)
        routes[r_idx].pop(i)

    routes = _safe_normalize(routes)
    return routes, removed


def _worst_removal(
    inst: VRPTWInstance,
    routes: List[List[int]],
    D: np.ndarray,
    num_remove: int,
    rng: random.Random,
) -> Tuple[List[List[int]], List[int]]:
    """
    Remove customers with largest cost contribution.
    Contribution of node i in route: D[prev,i] + D[i,next] - D[prev,next]
    """
    routes = _clone_routes(routes)
    contribs: List[Tuple[float, int, int]] = []  # (contrib, r_idx, pos)

    for r_idx, r in enumerate(routes):
        for i in range(1, len(r) - 1):
            node = r[i]
            if node == 0:
                continue
            prev_n = r[i - 1]
            next_n = r[i + 1]
            c = D[prev_n, node] + D[node, next_n] - D[prev_n, next_n]
            contribs.append((c, r_idx, i))

    # remove by descending contribution, tiebreak random
    rng.shuffle(contribs)
    contribs.sort(key=lambda x: x[0], reverse=True)

    removed: List[int] = []
    removed_marks = set()

    for _, r_idx, i in contribs:
        if len(removed) >= num_remove:
            break
        # guard if route already modified
        if r_idx >= len(routes):
            continue
        r = routes[r_idx]
        if i <= 0 or i >= len(r) - 1:
            continue
        node = r[i]
        if node == 0 or node in removed_marks:
            continue
        removed.append(node)
        removed_marks.add(node)
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
    """
    Shaw relatedness measure between customers i and j.
    Combines distance and time-window proximity.
    """
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
    """
    Shaw removal: start from random seed, iteratively remove related customers.
    """
    routes = _clone_routes(routes)
    customers = _get_all_customers(routes)
    if not customers:
        return routes, []

    seed = rng.choice(customers)
    removed = [seed]
    remaining = set(customers)
    remaining.discard(seed)

    while len(removed) < num_remove and remaining:
        # pick next most related to any already removed
        best_c = None
        best_score = float("inf")
        for c in remaining:
            rel = min(_shaw_relatedness(inst, D, c, r) for r in removed)
            if rel < best_score:
                best_score = rel
                best_c = c
        if best_c is None:
            break
        removed.append(best_c)
        remaining.discard(best_c)

    # physically remove from routes
    removed_set = set(removed)
    new_routes: List[List[int]] = []
    for r in routes:
        nr = [v for v in r if v == 0 or v not in removed_set]
        new_routes.append(nr)

    new_routes = _safe_normalize(new_routes)
    return new_routes, removed


# =========================
# Repair operators
# =========================

def _best_insertion_position(
    inst: VRPTWInstance,
    route: List[int],
    customer: int,
    D: np.ndarray,
) -> Optional[Tuple[float, int]]:
    """
    Find best feasible insertion (delta_cost, position) for 'customer' in 'route'.
    Returns None if no feasible position.
    """
    best_delta = None
    best_pos = None

    demand_c = inst.demand[customer]

    # quick capacity check: we handle at solution level; here only TW.
    base_load = route_load(route, inst)

    if base_load + demand_c > inst.capacity:
        return None

    for pos in range(1, len(route)):  # insert before pos
        prev_n = route[pos - 1]
        next_n = route[pos]
        delta = (
            D[prev_n, customer] + D[customer, next_n] - D[prev_n, next_n]
        )

        # try candidate route
        cand = route[:pos] + [customer] + route[pos:]

        # Time windows check
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
    """
    Greedy insertion: repeatedly insert the customer with the globally best
    insertion cost.
    """
    routes = _clone_routes(routes)
    pending = removed[:]

    while pending:
        best_c = None
        best_r = None
        best_pos = None
        best_delta = None

        for c in pending:
            for r_idx, r in enumerate(routes):
                res = _best_insertion_position(inst, r, c, D)
                if res is None:
                    continue
                delta, pos = res
                if best_delta is None or delta < best_delta:
                    best_delta = delta
                    best_c = c
                    best_r = r_idx
                    best_pos = pos

        # if no feasible insertion in existing routes, try opening a new one
        if best_c is None:
            c = pending[0]
            if inst.demand[c] > inst.capacity:
                return None  # impossible
            new_r = [0, c, 0]
            if not is_route_feasible(new_r, inst, D):
                return None
            routes.append(new_r)
            pending.pop(0)
        else:
            routes[best_r] = (
                routes[best_r][:best_pos]
                + [best_c]
                + routes[best_r][best_pos:]
            )
            pending.remove(best_c)

    return _safe_normalize(routes)


def _regret_k_repair(
    inst: VRPTWInstance,
    routes: List[List[int]],
    removed: List[int],
    D: np.ndarray,
    rng: random.Random,
    k: int = 2,
) -> Optional[List[List[int]]]:
    """
    Regret-k repair: at each step, for each customer compute its best k insertion
    positions; select customer with largest regret (cost_2 - cost_1, etc.).
    """
    routes = _clone_routes(routes)
    pending = removed[:]

    while pending:
        best_c = None
        best_r = None
        best_pos = None
        best_regret = None

        for c in pending:
            insertion_options: List[Tuple[float, int, int]] = []  # (delta, r_idx, pos)
            for r_idx, r in enumerate(routes):
                res = _best_insertion_position(inst, r, c, D)
                if res is not None:
                    delta, pos = res
                    insertion_options.append((delta, r_idx, pos))

            if not insertion_options:
                continue

            insertion_options.sort(key=lambda x: x[0])
            best = insertion_options[0][0]
            # regret = (2nd + 3rd + ... up to k) - best
            regret = 0.0
            for t in range(1, min(k, len(insertion_options))):
                regret += insertion_options[t][0] - best

            if best_regret is None or regret > best_regret + EPS:
                best_regret = regret
                best_c = c
                best_r = insertion_options[0][1]
                best_pos = insertion_options[0][2]

        # if nobody has feasible insertion into existing routes -> open new
        if best_c is None:
            c = pending[0]
            if inst.demand[c] > inst.capacity:
                return None
            new_r = [0, c, 0]
            if not is_route_feasible(new_r, inst, D):
                return None
            routes.append(new_r)
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
# Operator statistics
# =========================

@dataclass
class OperatorStats:
    weight: float = 1.0
    score: float = 0.0
    usage: int = 0


def _select_operator(ops: Dict[str, OperatorStats], rng: random.Random) -> str:
    total = sum(max(0.0, op.weight) for op in ops.values())
    if total <= 0.0:
        # fallback: uniform
        return rng.choice(list(ops.keys()))
    r = rng.random() * total
    acc = 0.0
    for name, op in ops.items():
        acc += max(0.0, op.weight)
        if r <= acc:
            return name
    # fallback
    return next(iter(ops.keys()))


def _update_operator_weights(
    ops: Dict[str, OperatorStats],
    reaction: float,
    sigma1: float,
    sigma2: float,
    sigma3: float,
) -> None:
    """
    Apply ALNS-style adaptive weight update.
    """
    for st in ops.values():
        if st.usage <= 0:
            continue
        avg_score = st.score / float(st.usage)
        st.weight = (1.0 - reaction) * st.weight + reaction * avg_score
        # reset period stats
        st.score = 0.0
        st.usage = 0


# =========================
# ALNS main
# =========================

def alns_vrptw(
    inst: VRPTWInstance,
    initial_routes: Optional[Sequence[Sequence[int]]] = None,
    D: Optional[np.ndarray] = None,
    max_iterations: int = 1000,
    remove_fraction_min: float = 0.05,
    remove_fraction_max: float = 0.25,
    segment_size: int = 50,
    reaction: float = 0.2,
    # scores for operator rewards (sigma1: new global best, sigma2: better than current, sigma3: accepted)
    sigma1: float = 6.0,
    sigma2: float = 3.0,
    sigma3: float = 1.0,
    time_limit_sec: Optional[float] = None,
    seed: Optional[int] = None,
    ls_max_iterations: int = 3000,
    ls_use_inter_route: bool = True,
    apply_ls_every: int = 1,
    initial_temperature: Optional[float] = None,
    final_temperature: Optional[float] = None,
    verbose: bool = False,
) -> List[List[int]]:
    """
    Adaptive Large Neighborhood Search for VRPTW.

    Workflow:
      - Start from given initial solution or Clarke–Wright (+ local search).
      - Loop:
          1. Select destroy & repair operators adaptively.
          2. Remove q customers (destroy).
          3. Reinsert them (repair) to get new solution.
          4. Optionally run local search on new solution.
          5. Accept new solution via SA-style criterion.
          6. Update operator weights every `segment_size` iterations.
      - Track global best and time-to-best.

    Tunable parameters:
      - remove_fraction_min/max: how "large" the neighborhood is.
      - segment_size, reaction, sigma*: ALNS adaptation behavior.
      - apply_ls_every: call local search every k iterations (>=1).
      - initial_temperature/final_temperature: for SA acceptance
            (defaults: scaled from instance if None).
      - time_limit_sec: global time limit.

    Returns:
      Best solution (routes) found.
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

    # ---------- Initial solution ----------
    if initial_routes is None:
        routes0 = clarke_wright_vrptw(inst, D=D)
    else:
        routes0 = [list(r) for r in initial_routes]

    routes0 = _safe_normalize(routes0)

    if not is_feasible(routes0, inst, D, require_all_customers=True):
        raise ValueError("Initial solution for ALNS is infeasible.")

    # Initial LS to get a strong starting point
    ls_budget = remaining_time()
    if ls_budget is not None and ls_budget <= 0:
        current = routes0
        if verbose:
            print("[ALNS] Time limit exhausted before initial LS; using given solution.")
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
        raise ValueError("Local search produced infeasible solution in ALNS initialization.")

    current_cost = solution_cost(current, D)
    best = _clone_routes(current)
    best_cost = current_cost
    time_to_best: Optional[float] = None

    if verbose:
        print(f"[ALNS] Start: cost={best_cost:.4f}, routes={len(best)}")

    # ---------- Operator sets ----------
    destroy_ops: Dict[str, OperatorStats] = {
        "random": OperatorStats(),
        "worst": OperatorStats(),
        "shaw": OperatorStats(),
    }

    repair_ops: Dict[str, OperatorStats] = {
        "greedy": OperatorStats(),
        "regret2": OperatorStats(),
        "regret3": OperatorStats(),
    }

    # ---------- SA temperature schedule ----------
    if initial_temperature is None:
        # scale based on current_cost
        initial_temperature = 0.01 * max(1.0, current_cost)
    if final_temperature is None:
        final_temperature = 0.001 * max(1.0, current_cost)

    def temperature(iteration: int) -> float:
        if max_iterations <= 1:
            return final_temperature
        frac = min(1.0, iteration / float(max_iterations - 1))
        # geometric-like schedule in log-space
        log_t0 = math.log(initial_temperature + 1e-12)
        log_t1 = math.log(final_temperature + 1e-12)
        return math.exp(log_t0 + (log_t1 - log_t0) * frac)

    # ---------- Main loop ----------
    num_customers = inst.n_customers
    remove_fraction_min = max(0.0, min(remove_fraction_min, 1.0))
    remove_fraction_max = max(remove_fraction_min, min(remove_fraction_max, 1.0))

    for it in range(1, max_iterations + 1):
        # time check
        rem = remaining_time()
        if rem is not None and rem <= 0:
            if verbose:
                print(f"[ALNS] Time limit {time_limit_sec}s reached at iter {it}.")
            break

        # choose q (number of removed customers)
        q_min = max(1, int(remove_fraction_min * num_customers))
        q_max = max(q_min, int(remove_fraction_max * num_customers))
        q = rng.randint(q_min, max(q_min, q_max))

        # select operators
        destroy_name = _select_operator(destroy_ops, rng)
        repair_name = _select_operator(repair_ops, rng)
        destroy_stats = destroy_ops[destroy_name]
        repair_stats = repair_ops[repair_name]

        # ----- Destroy -----
        if destroy_name == "random":
            partial, removed = _random_removal(inst, current, D, q, rng)
        elif destroy_name == "worst":
            partial, removed = _worst_removal(inst, current, D, q, rng)
        else:  # "shaw"
            partial, removed = _shaw_removal(inst, current, D, q, rng)

        if not removed:
            # fallback: no change
            continue

        # ----- Repair -----
        if repair_name == "greedy":
            candidate = _greedy_repair(inst, partial, removed, D, rng)
        elif repair_name == "regret2":
            candidate = _regret_k_repair(inst, partial, removed, D, rng, k=2)
        else:  # "regret3"
            candidate = _regret_k_repair(inst, partial, removed, D, rng, k=3)

        if candidate is None:
            # failed to reconstruct a feasible solution; penalize operators weakly
            destroy_stats.usage += 1
            repair_stats.usage += 1
            continue

        candidate = _safe_normalize(candidate)
        if not is_feasible(candidate, inst, D, require_all_customers=True):
            # infeasible candidate; penalize and continue
            destroy_stats.usage += 1
            repair_stats.usage += 1
            continue

        # Optional local search on candidate
        if apply_ls_every > 0 and (it % apply_ls_every == 0):
            ls_budget = remaining_time()
            if ls_budget is not None and ls_budget <= 0:
                # no more time for LS; skip LS but keep candidate as-is
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
                    # LS somehow produced infeasible: skip this move
                    destroy_stats.usage += 1
                    repair_stats.usage += 1
                    continue

        cand_cost = solution_cost(candidate, D)
        destroy_stats.usage += 1
        repair_stats.usage += 1

        # ----- Evaluate & accept -----
        accepted = False
        improved_current = cand_cost < current_cost - EPS
        improved_best = cand_cost < best_cost - EPS
        equal_current = abs(cand_cost - current_cost) <= EPS

        T = temperature(it)

        if improved_current or equal_current:
            accepted = True
        else:
            # SA-style acceptance for worse solutions
            diff = cand_cost - current_cost
            prob = math.exp(-diff / max(T, 1e-12))
            if rng.random() < prob:
                accepted = True

        if accepted:
            current = _clone_routes(candidate)
            current_cost = cand_cost

            # scoring rules
            if improved_best:
                destroy_stats.score += sigma1
                repair_stats.score += sigma1
            elif improved_current:
                destroy_stats.score += sigma2
                repair_stats.score += sigma2
            else:
                destroy_stats.score += sigma3
                repair_stats.score += sigma3

            if improved_best:
                best = _clone_routes(candidate)
                best_cost = cand_cost
                time_to_best = time.time() - start_time
                if verbose:
                    print(
                        f"[ALNS] Iter {it}: new best {best_cost:.4f}, "
                        f"routes={len(best)}, op=({destroy_name}, {repair_name})"
                    )
        else:
            # not accepted; still very small reward to avoid zero-division stagnation
            destroy_stats.score += 0.0
            repair_stats.score += 0.0

        if verbose and it % max(1, segment_size // 2) == 0:
            print(
                f"[ALNS] Iter {it}: current={current_cost:.4f}, "
                f"best={best_cost:.4f}, T={T:.4g}"
            )

        # ----- Adapt operator weights -----
        if segment_size > 0 and it % segment_size == 0:
            _update_operator_weights(destroy_ops, reaction, sigma1, sigma2, sigma3)
            _update_operator_weights(repair_ops, reaction, sigma1, sigma2, sigma3)

    # ----- Final reporting -----
    if verbose:
        if time_to_best is not None:
            print(f"[ALNS] Final best={best_cost:.4f}, routes={len(best)}")
            print(f"[ALNS] Time to best: {time_to_best:.4f} s")
        else:
            print(
                f"[ALNS] No improvement over initial solution. "
                f"Cost={best_cost:.4f}, routes={len(best)}"
            )

    # expose time-to-best metadata
    try:
        alns_vrptw.time_to_best = time_to_best
    except Exception:
        pass

    return best
