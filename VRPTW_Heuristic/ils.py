# ils.py
from __future__ import annotations

import time
import random
from typing import List, Sequence, Optional

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


def _clone_routes(routes: Sequence[Sequence[int]]) -> List[List[int]]:
    return [list(r) for r in routes]


def _safe_normalize(routes: Sequence[Sequence[int]]) -> List[List[int]]:
    r = normalize_routes(routes, depot=0)
    return [route for route in r if len(route) > 2]


def _routes_equal(a: Sequence[Sequence[int]], b: Sequence[Sequence[int]]) -> bool:
    """Return True if two solutions have identical route structure."""
    if len(a) != len(b):
        return False
    for ra, rb in zip(a, b):
        if len(ra) != len(rb):
            return False
        for va, vb in zip(ra, rb):
            if va != vb:
                return False
    return True


def _random_customer_position(
    rng: random.Random,
    route: Sequence[int],
) -> Optional[int]:
    """
    Return a random internal position (index) of a customer (non-depot),
    or None if no such position exists.
    """
    candidates = [i for i in range(1, len(route) - 1) if route[i] != 0]
    if not candidates:
        return None
    return rng.choice(candidates)


def _perturb_solution(
    inst: VRPTWInstance,
    routes: Sequence[Sequence[int]],
    D: np.ndarray,
    strength: int,
    rng: random.Random,
) -> List[List[int]]:
    """
    Apply a small random but *feasible* perturbation to the solution.

    Moves:
      - random inter-route relocate
      - random (intra/inter) swap

    We:
      - only accept a move if the affected routes remain feasible
        (capacity + time windows).
      - keep it simple and robust; this is a "shake", not a full search.

    Returns a new solution (list of routes).
    """
    if strength <= 0:
        return _clone_routes(routes)

    new_routes = _safe_normalize(routes)
    if not new_routes:
        return new_routes

    loads = [route_load(r, inst) for r in new_routes]
    applied = 0
    attempts = 0
    max_attempts = max(10, strength * 6)

    while applied < max(1, strength) and attempts < max_attempts:
        attempts += 1

        if not new_routes:
            break

        move_applied = False
        move_type = rng.random()

        # -------- Relocate (prefer cross-route) --------
        if move_type < 0.5 and len(new_routes) > 1:
            # pick source route with at least one customer
            from_candidates = [ri for ri, r in enumerate(new_routes) if len(r) > 3]
            if not from_candidates:
                continue
            r_from = rng.choice(from_candidates)
            route_from = new_routes[r_from]

            i = _random_customer_position(rng, route_from)
            if i is None:
                continue
            node = route_from[i]
            dem = inst.demand[node]

            # pick target route different from source
            to_candidates = [ri for ri in range(len(new_routes)) if ri != r_from]
            if not to_candidates:
                continue
            r_to = rng.choice(to_candidates)
            route_to = new_routes[r_to]

            j = rng.randint(1, len(route_to) - 1)

            # capacity checks
            load_from = loads[r_from]
            load_to = loads[r_to]
            if load_to + dem > inst.capacity:
                continue
            if load_from - dem < 0:
                continue

            # build candidate routes
            cand_from = route_from[:i] + route_from[i + 1 :]
            cand_to = route_to[:j] + [node] + route_to[j:]

            if not cand_from or not cand_to:
                continue

            if is_route_feasible(cand_from, inst, D) and is_route_feasible(cand_to, inst, D):
                new_routes[r_from] = cand_from
                new_routes[r_to] = cand_to
                loads[r_from] = route_load(cand_from, inst)
                loads[r_to] = route_load(cand_to, inst)
                applied += 1
                move_applied = True

        # -------- Swap (within or across routes) --------
        else:
            if len(new_routes) == 1:
                # intra-route swap
                r_idx = 0
                route = new_routes[r_idx]
                if len(route) <= 3:
                    continue

                i = _random_customer_position(rng, route)
                j = _random_customer_position(rng, route)
                if i is None or j is None or i == j:
                    continue

                cand = list(route)
                cand[i], cand[j] = cand[j], cand[i]

                if is_route_feasible(cand, inst, D):
                    new_routes[r_idx] = cand
                    loads[r_idx] = route_load(cand, inst)
                    applied += 1
                    move_applied = True
            else:
                # inter-route swap
                r1, r2 = rng.sample(range(len(new_routes)), 2)
                route1 = new_routes[r1]
                route2 = new_routes[r2]
                if len(route1) <= 3 or len(route2) <= 3:
                    continue

                i = _random_customer_position(rng, route1)
                j = _random_customer_position(rng, route2)
                if i is None or j is None:
                    continue

                n1 = route1[i]
                n2 = route2[j]
                d1 = inst.demand[n1]
                d2 = inst.demand[n2]

                load1 = loads[r1]
                load2 = loads[r2]

                # capacity after swap
                if load1 - d1 + d2 > inst.capacity:
                    continue
                if load2 - d2 + d1 > inst.capacity:
                    continue

                cand1 = list(route1)
                cand2 = list(route2)
                cand1[i], cand2[j] = cand2[j], cand1[i]

                if is_route_feasible(cand1, inst, D) and is_route_feasible(cand2, inst, D):
                    new_routes[r1] = cand1
                    new_routes[r2] = cand2
                    loads[r1] = route_load(cand1, inst)
                    loads[r2] = route_load(cand2, inst)
                    applied += 1
                    move_applied = True

        # clean after each successful move to drop empty routes and ensure depot bounds
        if move_applied:
            new_routes = _safe_normalize(new_routes)
            loads = [route_load(r, inst) for r in new_routes]

    if applied == 0:
        # fallback: random route shuffle to escape identical solution
        rng.shuffle(new_routes)

    return _safe_normalize(new_routes)


def iterated_local_search_vrptw(
    inst: VRPTWInstance,
    initial_routes: Optional[Sequence[Sequence[int]]] = None,
    D: Optional[np.ndarray] = None,
    max_iterations: int = 50,
    perturb_strength: int = 3,
    time_limit_sec: Optional[float] = None,
    seed: Optional[int] = None,
    # local search tuning (forwarded)
    ls_max_iterations: int = 10_000,
    ls_use_inter_route: bool = True,
    max_perturb_strength: Optional[int] = None,
    adapt_perturb_every: int = 6,
    exploratory_accept_prob: float = 0.1,
    accept_equal_cost: bool = True,
    verbose: bool = False,
) -> List[List[int]]:
    """
    Iterated Local Search (ILS) for VRPTW.

    Outline:
      1) Build or take an initial feasible solution.
      2) Apply local search -> x*
      3) Repeat:
           - Perturb x* -> x'
           - Local search on x' -> x*'
           - Accept x*' according to simple policy (better than current).
           - Track global best.
         until iteration or time limit.

    Parameters:
        inst: VRPTWInstance
        initial_routes: if None, uses Clarke-Wright constructor.
        D: distance matrix (if None, computed)
        max_iterations: number of ILS outer iterations
        perturb_strength: base number of random perturbation moves per iteration
        max_perturb_strength: cap for adaptive perturbation strength (default: base + 5)
        adapt_perturb_every: increase perturbation strength after this many consecutive
            non-improving iterations
        exploratory_accept_prob: probability of accepting a non-improving candidate to diversify
        accept_equal_cost: accept candidates with equal cost if they differ structurally
        time_limit_sec: overall wall-clock time limit; each local search call receives the
            remaining budget
        seed: RNG seed for reproducibility
        ls_max_iterations: max iterations for each local search call
        ls_use_inter_route: whether LS explores inter-route moves
        verbose: log progress

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
        remaining = time_limit_sec - (time.time() - start_time)
        if remaining <= 0:
            return 0.0
        return remaining

    # ---- Initial solution ----
    if initial_routes is None:
        init_routes = clarke_wright_vrptw(inst, D=D)
    else:
        init_routes = [list(r) for r in initial_routes]

    init_routes = _safe_normalize(init_routes)

    if not is_feasible(init_routes, inst, D, require_all_customers=True):
        raise ValueError("Initial solution for ILS is infeasible.")

    # First local search to reach a local optimum
    ls_budget = remaining_time()
    if ls_budget is not None and ls_budget <= 0:
        current = _safe_normalize(init_routes)
        if verbose:
            print("[ILS] Time limit exhausted before initial local search; using constructor output.")
    else:
        current = local_search_vrptw(
            inst,
            init_routes,
            D=D,
            max_iterations=ls_max_iterations,
            use_inter_route=ls_use_inter_route,
            time_limit_sec=ls_budget,
            verbose=verbose,
        )
        current = _safe_normalize(current)
    current_cost = solution_cost(current, D)

    if not is_feasible(current, inst, D, require_all_customers=True):
        raise ValueError("Local search produced infeasible solution in ILS initialization.")

    best = _clone_routes(current)
    best_cost = current_cost
    time_to_best: Optional[float] = None

    if verbose:
        print(f"[ILS] Start: cost={best_cost:.4f}, routes={len(best)}")

    base_strength = max(1, perturb_strength)
    strength_cap = max_perturb_strength if max_perturb_strength is not None else base_strength + 5
    strength_cap = max(strength_cap, base_strength)
    perturb_strength_cur = base_strength
    stagnation_counter = 0
    exploratory_accept_prob = min(max(exploratory_accept_prob, 0.0), 1.0)

    # ---- ILS main loop ----
    for it in range(1, max_iterations + 1):
        if time_limit_sec is not None and (time.time() - start_time) >= time_limit_sec:
            if verbose:
                print(f"[ILS] Global time limit {time_limit_sec}s reached at iter {it}.")
            break

        # adapt perturbation strength if stagnating
        if adapt_perturb_every > 0 and stagnation_counter >= adapt_perturb_every:
            if perturb_strength_cur < strength_cap:
                perturb_strength_cur += 1
                if verbose:
                    print(f"[ILS] Increasing perturbation strength to {perturb_strength_cur}")
            stagnation_counter = 0

        # 1) Perturb current solution
        pert = _perturb_solution(inst, current, D, perturb_strength_cur, rng)
        if not pert:
            # if perturbation collapsed everything, restart from global best
            pert = _clone_routes(best)

        # ensure feasibility (if broken, skip this iteration)
        if not is_feasible(pert, inst, D, require_all_customers=True):
            # light restart strategy: start from best
            pert = _clone_routes(best)
            if not is_feasible(pert, inst, D, require_all_customers=True):
                # if even best somehow infeasible, bail
                break

        # 2) Local search on perturbed solution
        #    Respect remaining global time if both limits are set.
        ls_budget = remaining_time()
        if ls_budget is not None and ls_budget <= 0:
            if verbose:
                print(f"[ILS] No time remaining for local search at iter {it}; stopping.")
            break
        cand = local_search_vrptw(
            inst,
            pert,
            D=D,
            max_iterations=ls_max_iterations,
            use_inter_route=ls_use_inter_route,
            time_limit_sec=ls_budget,
            verbose=False,
        )
        cand = _safe_normalize(cand)
        if not is_feasible(cand, inst, D, require_all_customers=True):
            # skip invalid candidate
            continue

        cand_cost = solution_cost(cand, D)
        cand_differs = not _routes_equal(cand, current)

        # 3) Acceptance criterion:
        #    - Accept if strictly better than current.
        #    - Otherwise, accept with small probability (to escape plateaus).
        if cand_cost < current_cost - EPS or (
            accept_equal_cost and abs(cand_cost - current_cost) <= EPS and cand_differs
        ):
            current = _clone_routes(cand)
            current_cost = cand_cost
            stagnation_counter = 0
            perturb_strength_cur = base_strength
        else:
            # Simple diversification: occasional non-improving acceptance
            if rng.random() < exploratory_accept_prob and cand_differs:
                current = _clone_routes(cand)
                current_cost = cand_cost
                stagnation_counter = 0
                perturb_strength_cur = base_strength
            else:
                stagnation_counter += 1

        # 4) Update global best
        if cand_cost < best_cost - EPS:
            best = _clone_routes(cand)
            best_cost = cand_cost
            time_to_best = time.time() - start_time
            perturb_strength_cur = base_strength
            stagnation_counter = 0
            if verbose:
                print(f"[ILS] Iter {it}: new best {best_cost:.4f}, routes={len(best)}")
        elif verbose:
            print(f"[ILS] Iter {it}: cand={cand_cost:.4f}, best={best_cost:.4f}")

    # Report / store time-to-best
    if verbose:
        if time_to_best is None:
            print("[ILS] No improvement over initial solution.")
        else:
            print(f"[ILS] Time to best: {time_to_best:.4f} s")

    try:
        iterated_local_search_vrptw.time_to_best = time_to_best
    except Exception:
        pass

    return best
