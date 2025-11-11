# constructors.py
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from vrptw_parser import VRPTWInstance
from utils import (
    build_distance_matrix,
    is_route_feasible,
)


def clarke_wright_vrptw(
    inst: VRPTWInstance,
    D: np.ndarray | None = None,
    lambda_: float = 1.0,
    mu: float = 0.0,
    nu: float = 0.0,
) -> List[List[int]]:
    """
    Parallel Clarke–Wright Savings heuristic adapted for VRPTW.

    Conventions:
      - Uses VRPTWInstance from vrptw_parser.
      - Uses distance matrix D[i,j] (precomputed if not provided).
      - Route format: [0, ..., 0], depot = 0 at start & end.
      - One route per customer as initialization: [0, i, 0].
      - Merging condition:
          * Capacity respected.
          * Time windows respected (via is_route_feasible on merged route).
          * Standard CW endpoint rule:
              - Merge tail of one route with head of another.
      - Savings formula (generalized):
          s(i, j) = d(0, i) + d(0, j)
                    - λ * d(i, j)
                    + μ * |d(0, i) - d(0, j)|
                    + ν * (q_i + q_j)
        Default λ=1, μ=ν=0 => classical Clarke–Wright.

    Returns:
      List of routes (each a list of node indices) forming a feasible solution
      w.r.t. capacity and time windows (if a merge would break feasibility,
      it is rejected).

    Notes:
      - This is a solid constructive baseline:
          * Proper VRPTW feasibility checks on each merge.
          * Parallel implementation.
      - It does NOT guarantee using ≤ inst.n_vehicles; if the heuristic
        gets stuck early, you may end up with more routes. Post-processing
        or a repair/merge phase can be added on top.
    """
    # --- Precompute distances ---
    if D is None:
        D = build_distance_matrix(inst)

    n = inst.n_customers
    if n <= 0:
        # Trivial instance: only depot.
        return [[0, 0]]

    # --- Initialization: one route per customer: [0, i, 0] ---
    routes: List[List[int]] = []
    loads: List[int] = []
    # route_of[node] = index in `routes` of the route currently containing node.
    route_of = [-1] * (n + 1)  # index 0 unused for customers; depot not tracked here

    for idx, cust in enumerate(range(1, n + 1)):
        r = [0, cust, 0]
        routes.append(r)
        loads.append(int(inst.demand[cust]))
        route_of[cust] = idx

    # --- Precompute savings ---
    # Only consider customer pairs (i, j) with i < j.
    savings: List[Tuple[float, int, int]] = []
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            sij = (
                D[0, i]
                + D[0, j]
                - lambda_ * D[i, j]
                + mu * abs(D[0, i] - D[0, j])
                + nu * (inst.demand[i] + inst.demand[j])
            )
            savings.append((sij, i, j))

    # Sort in descending order of savings
    savings.sort(key=lambda x: x[0], reverse=True)

    # --- Parallel Clarke–Wright merging with VRPTW constraints ---
    for s, i, j in savings:
        # Try both orientations: (i tail, j head) and (j tail, i head)
        # This allows connecting route_i -> route_j or route_j -> route_i.
        for tail, head in ((i, j), (j, i)):
            ri = route_of[tail]
            rj = route_of[head]

            # Both must currently be assigned and belong to different routes
            if ri == -1 or rj == -1 or ri == rj:
                continue

            route_i = routes[ri]
            route_j = routes[rj]
            if not route_i or not route_j:
                continue

            # tail must be the last customer before depot in route_i
            if len(route_i) < 3 or route_i[-2] != tail:
                continue

            # head must be the first customer after depot in route_j
            if len(route_j) < 3 or route_j[1] != head:
                continue

            # Capacity check (constant-time using stored loads)
            new_load = loads[ri] + loads[rj]
            if new_load > inst.capacity:
                continue

            # Build merged route:
            # [0, ..., tail, 0] + [0, head, ... , 0]
            #    -> [0, ..., tail, head, ..., 0]
            new_route = route_i[:-1] + route_j[1:]

            # Full VRPTW feasibility check for candidate route
            if not is_route_feasible(new_route, inst, D):
                continue

            # Merge is accepted
            routes[ri] = new_route
            loads[ri] = new_load

            # Reassign customers from route_j to route_i
            for node in route_j[1:-1]:
                route_of[node] = ri

            # Invalidate old route_j
            routes[rj] = []
            loads[rj] = 0

            # Important: break orientation loop after a successful merge.
            # We don't want to try the opposite orientation once merged.
            break

    # --- Collect non-empty routes ---
    final_routes = [r for r in routes if r]

    # Note:
    # - final_routes respects capacity & time windows by construction.
    # - A downstream metaheuristic can:
    #       * check global feasibility (all customers, no duplicates),
    #       * repair if #routes > inst.n_vehicles,
    #       * improve using local search with delta_cost_* utilities.
    return final_routes
