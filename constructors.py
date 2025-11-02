# constructors.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict
import math
import numpy as np

from utils import (
    build_distance_matrix,
    route_cost,
    total_cost,
    route_load,
    is_feasible,
)

@dataclass
class Solution:
    routes: List[List[int]]
    cost: float


# =============== Clarke–Wright Savings (Parallel) ===============

def _cw_merge_routes(
    routes: List[List[int]],
    route_of: Dict[int, int],
    load_of_route: List[int],
    i: int,
    j: int,
    Q: int,
    D: np.ndarray,
) -> bool:
    """
    Try to merge the two routes that contain customers i and j.
    Parallel CW merge with all four endpoint orientations (reversals where needed).
    Returns True if merged; False otherwise.
    """
    ri = route_of[i]
    rj = route_of[j]
    if ri == rj:
        return False

    A = routes[ri]
    B = routes[rj]
    loadA = load_of_route[ri]
    loadB = load_of_route[rj]
    if loadA + loadB > Q:
        return False

    # Identify endpoints
    Ai, Aj = A[0], A[-1]
    Bi, Bj = B[0], B[-1]

    # We need to connect an endpoint of A to an endpoint of B.
    # Consider four possibilities; choose the best (max saving).
    candidates: List[Tuple[float, str]] = []

    # A tail to B head: A ... Ai ... Aj  +  B Bi ... Bj  =>  A ... Aj - Bi ... Bj
    if Aj == i and Bi == j:
        delta = - D[0, Aj] - D[0, Bi] + D[Aj, Bi]
        candidates.append((delta, "A_tail__B_head"))

    # B tail to A head
    if Bj == j and Ai == i:
        delta = - D[0, Bj] - D[0, Ai] + D[Bj, Ai]
        candidates.append((delta, "B_tail__A_head"))

    # A tail to B tail (reverse B)
    if Aj == i and Bj == j:
        delta = - D[0, Aj] - D[0, Bj] + D[Aj, Bj]
        candidates.append((delta, "A_tail__B_tail"))

    # A head to B head (reverse A)
    if Ai == i and Bi == j:
        delta = - D[0, Ai] - D[0, Bi] + D[Ai, Bi]
        candidates.append((delta, "A_head__B_head"))

    if not candidates:
        return False

    # Pick the best (most negative delta = biggest improvement)
    best_delta, mode = min(candidates, key=lambda x: x[0])

    # Perform the merge (apply orientation if needed)
    if mode == "A_tail__B_head":
        new_route = A + B
    elif mode == "B_tail__A_head":
        new_route = B + A
    elif mode == "A_tail__B_tail":
        new_route = A + list(reversed(B))
    elif mode == "A_head__B_head":
        new_route = list(reversed(A)) + B
    else:
        return False  # shouldn't get here

    # Commit merge: replace route ri with new_route, delete rj
    routes[ri] = new_route
    routes[rj] = []  # mark empty; we'll skip later
    load_of_route[ri] = loadA + loadB
    load_of_route[rj] = 0

    # Update route_of for customers moved
    for v in new_route:
        route_of[v] = ri
    return True


def clarke_wright_parallel(coords: np.ndarray, demand: np.ndarray, Q: int, edge_weight_type: str = "EUC_2D",
                           round_euclidean: bool = False) -> Solution:
    """
    Classic parallel Clarke–Wright. Starts with n singleton routes and greedily merges.
    Returns a feasible solution with closed routes.
    """
    n_plus_1 = coords.shape[0]
    n = n_plus_1 - 1
    assert n >= 1, "No customers."

    D = build_distance_matrix(coords, edge_weight_type, round_euclidean=round_euclidean)

    # Start with one route per customer
    routes = [[i] for i in range(1, n_plus_1)]
    load_of_route = [int(demand[i]) for i in range(1, n_plus_1)]
    route_of = {i: idx for idx, i in enumerate(range(1, n_plus_1))}

    # Precompute savings list (i < j)
    savings = []
    for i in range(1, n_plus_1):
        for j in range(i + 1, n_plus_1):
            sij = D[0, i] + D[0, j] - D[i, j]
            savings.append((sij, i, j))
    # Sort descending by savings (largest first)
    savings.sort(key=lambda x: x[0], reverse=True)

    # Merge loop
    for _, i, j in savings:
        # Skip if either route was deleted
        if i not in route_of or j not in route_of:
            continue
        ri = route_of[i]; rj = route_of[j]
        if ri == rj or not routes[ri] or not routes[rj]:
            continue

        merged = _cw_merge_routes(routes, route_of, load_of_route, i, j, Q, D)
        # No need to track delta cost here; final cost computed afterward

    # Compact routes (remove empties)
    routes = [r for r in routes if r]
    sol_cost = total_cost(D, routes)
    return Solution(routes=routes, cost=sol_cost)


# ===================== Sweep + Best Insertion =====================

def _angles(coords: np.ndarray) -> np.ndarray:
    """
    Polar angles of customers around the depot (row 0).
    Returns angles for indices 1..n in an array of length n+1 with angle[0]=nan.
    """
    depot = coords[0]
    vecs = coords - depot
    ang = np.full((coords.shape[0],), np.nan, dtype=float)
    for i in range(1, coords.shape[0]):
        ang[i] = math.atan2(vecs[i, 1], vecs[i, 0])
    return ang


def _best_insertion_position(D: np.ndarray, route: List[int], cust: int) -> Tuple[int, float]:
    """
    For a given route and candidate customer, return (pos, delta_cost)
    where 'pos' is the index to insert cust BEFORE, minimizing the closed-tour delta.
    """
    if not route:
        # Depot -> cust -> Depot
        delta = - D[0, 0] + D[0, cust] + D[cust, 0]  # (0->cust->0) - 0
        return 0, float(delta)

    best_pos, best_delta = 0, float("inf")
    # Try inserting before each position j (including end j=len(route))
    for j in range(0, len(route) + 1):
        u = 0 if j == 0 else route[j - 1]
        v = 0 if j == len(route) else route[j]
        delta = - D[u, v] + D[u, cust] + D[cust, v]
        if delta < best_delta:
            best_delta = float(delta)
            best_pos = j
    return best_pos, best_delta


def sweep_best_insertion(coords: np.ndarray, demand: np.ndarray, Q: int, edge_weight_type: str = "EUC_2D",
                         round_euclidean: bool = False) -> Solution:
    """
    Sweep order by polar angle around the depot; insert each customer
    into the best feasible route position or start a new route.
    """
    D = build_distance_matrix(coords, edge_weight_type, round_euclidean=round_euclidean)
    n = coords.shape[0] - 1
    ang = _angles(coords)

    order = list(range(1, n + 1))
    order.sort(key=lambda i: ang[i])

    routes: List[List[int]] = []
    loads: List[int] = []

    for cust in order:
        dem = int(demand[cust])

        # Try to place into an existing route
        best_r_idx, best_pos, best_delta = None, None, float("inf")
        for r_idx, r in enumerate(routes):
            if loads[r_idx] + dem > Q:
                continue
            pos, delta = _best_insertion_position(D, r, cust)
            if delta < best_delta:
                best_delta = delta
                best_pos = pos
                best_r_idx = r_idx

        if best_r_idx is None:
            # Start a new route
            routes.append([cust])
            loads.append(dem)
        else:
            routes[best_r_idx].insert(best_pos, cust)
            loads[best_r_idx] += dem

    cost = total_cost(D, routes)
    return Solution(routes=routes, cost=cost)
