# utils.py
from __future__ import annotations
from typing import List, Tuple, Optional
import numpy as np


# -----------------------------
# Distance / cost infrastructure
# -----------------------------

def build_distance_matrix(
    coords: np.ndarray,
    edge_weight_type: str = "EUC_2D",
    round_euclidean: bool = False,
    dtype=np.float64,
) -> np.ndarray:
    """
    Precompute pairwise distances D[i, j].

    Args:
        coords: (n+1, 2) coordinates array; row 0 is the depot.
        edge_weight_type: "EUC_2D" or "CEIL_2D".
        round_euclidean: If True and edge_weight_type == "EUC_2D", round to nearest int.
                         Uchoa X-instances are often evaluated with raw Euclidean as float;
                         keep this False unless you specifically want rounded EUC_2D.
        dtype: output dtype.

    Returns:
        D: (n+1, n+1) array with symmetric distances and zeros on the diagonal.
    """
    coords = np.asarray(coords, dtype=float)
    n = coords.shape[0]
    # (x_i - x_j)^2 + (y_i - y_j)^2 via broadcasting
    dx = coords[:, 0][:, None] - coords[:, 0][None, :]
    dy = coords[:, 1][:, None] - coords[:, 1][None, :]
    D = np.hypot(dx, dy)

    edge_weight_type = (edge_weight_type or "EUC_2D").upper()
    if edge_weight_type == "CEIL_2D":
        D = np.ceil(D, dtype=dtype)
    elif edge_weight_type == "EUC_2D" and round_euclidean:
        D = np.rint(D).astype(dtype, copy=False)
    else:
        D = D.astype(dtype, copy=False)

    np.fill_diagonal(D, 0.0)
    return D


def route_cost(D: np.ndarray, route: List[int]) -> float:
    """Closed-tour cost of one route (depot->...->depot)."""
    if not route:
        return 0.0
    c = D[0, route[0]]
    for i in range(len(route) - 1):
        c += D[route[i], route[i + 1]]
    c += D[route[-1], 0]
    return float(c)


def total_cost(D: np.ndarray, routes: List[List[int]]) -> float:
    """Sum of closed-tour costs for all routes."""
    return float(sum(route_cost(D, r) for r in routes))


# -----------------------------
# Feasibility helpers
# -----------------------------

def route_load(demand: np.ndarray, route: List[int]) -> int:
    """Sum of demands on a route. demand[0] (depot) is assumed 0."""
    if not route:
        return 0
    return int(np.sum(demand[np.array(route, dtype=int)]))


def is_feasible(
    routes: List[List[int]],
    demand: np.ndarray,
    capacity: int,
    must_cover_all: bool = False,
    n_customers: Optional[int] = None,
) -> bool:
    """
    Checks capacity feasibility; optionally checks coverage (every customer visited once).

    Args:
        routes: list of routes (each a list of customer indices >= 1)
        demand: array with demand[0] == 0
        capacity: vehicle capacity Q
        must_cover_all: if True, verify every customer 1..n_customers is visited exactly once
        n_customers: required when must_cover_all=True

    Returns:
        True if feasible (and covers all if requested), False otherwise.
    """
    # Capacity
    for r in routes:
        if route_load(demand, r) > capacity:
            return False

    if must_cover_all:
        if n_customers is None:
            raise ValueError("n_customers must be provided when must_cover_all=True.")
        seen = np.zeros(n_customers + 1, dtype=int)  # 0..n
        for r in routes:
            for v in r:
                if v <= 0 or v > n_customers:
                    return False
                seen[v] += 1
        # Every customer 1..n must be visited exactly once
        if not np.all(seen[1:] == 1):
            return False

    return True


# -----------------------------
# Nearest neighbors (pruning)
# -----------------------------

def nearest_neighbors(D: np.ndarray, k: int) -> List[List[int]]:
    """
    For each node i, return the list of its k nearest neighbors (excluding itself and depot optional?).
    Here we return neighbors among all nodes except i itself.
    """
    n = D.shape[0]
    # argsort each row, drop self
    order = np.argsort(D, axis=1)
    nn_lists: List[List[int]] = []
    for i in range(n):
        row = order[i].tolist()
        row = [j for j in row if j != i]
        nn_lists.append(row[:k])
    return nn_lists


# --------------------------------------
# Δ-evaluators (constant-time move costs)
# --------------------------------------
# Assumptions:
# - Routes are lists of customer indices, depot=0 is implied at both ends.
# - D is a full matrix of distances.
# - Indices i, j refer to positions in the route lists (0-based over customers).


def _at(route: List[int], idx: int) -> int:
    """Safe access with depot sentinels: route[-1] -> last customer, route[len] -> depot."""
    if idx == -1:
        return 0  # depot before first
    if idx == len(route):
        return 0  # depot after last
    return route[idx]


def delta_relocate_intra(D: np.ndarray, route: List[int], i: int, j: int) -> float:
    """
    Move customer at position i to be inserted BEFORE position j in the SAME route.
    j references the ORIGINAL route indexing (before removal).
    Valid when j != i and j != i+1.
    """
    if i < 0 or i >= len(route) or j < 0 or j > len(route) or j == i or j == i + 1:
        return 0.0

    a = _at(route, i - 1)
    b = route[i]
    c = _at(route, i + 1)

    # remove b from between (a, b, c)
    delta = - D[a, b] - D[b, c] + D[a, c]

    # insertion context is always (route[j-1], route[j]) in the ORIGINAL route
    u = _at(route, j - 1)        # depot if j==0
    v = _at(route, j)            # depot if j==len(route)

    # insert b between (u, v)
    delta += - D[u, v] + D[u, b] + D[b, v]

    return float(delta)

def delta_relocate_inter(
    D: np.ndarray,
    route_a: List[int], i: int,
    route_b: List[int], j: int,
) -> float:
    """
    Move customer route_a[i] into route_b before position j.
    Returns Δ cost (new - old). Capacity feasibility must be checked by caller.
    """
    if i < 0 or i >= len(route_a) or j < 0 or j > len(route_b):
        return 0.0

    # Removal from A
    a = _at(route_a, i - 1)
    b = route_a[i]
    c = _at(route_a, i + 1)
    delta = - D[a, b] - D[b, c] + D[a, c]

    # Insertion into B before j
    u = _at(route_b, j - 1)
    v = _at(route_b, j)
    delta += - D[u, v] + D[u, b] + D[b, v]

    return float(delta)


def delta_swap_inter(
    D: np.ndarray,
    route_a: List[int], ia: int,
    route_b: List[int], ib: int,
) -> float:
    """
    Swap single customers route_a[ia] <-> route_b[ib].
    Returns Δ cost (new - old). Capacity feasibility must be checked by caller.
    """
    if ia < 0 or ia >= len(route_a) or ib < 0 or ib >= len(route_b):
        return 0.0

    a_prev = _at(route_a, ia - 1)
    x = route_a[ia]
    a_next = _at(route_a, ia + 1)

    b_prev = _at(route_b, ib - 1)
    y = route_b[ib]
    b_next = _at(route_b, ib + 1)

    delta = 0.0

    # Remove x from A: (a_prev - x - a_next) -> (a_prev - a_next)
    delta += - D[a_prev, x] - D[x, a_next] + D[a_prev, a_next]
    # Insert y in A at ia: (a_prev - y - a_next)
    delta += - D[a_prev, a_next] + D[a_prev, y] + D[y, a_next]

    # Remove y from B
    delta += - D[b_prev, y] - D[y, b_next] + D[b_prev, b_next]
    # Insert x in B at ib
    delta += - D[b_prev, b_next] + D[b_prev, x] + D[x, b_next]

    return float(delta)


def delta_2opt_intra(D: np.ndarray, route: List[int], i: int, k: int) -> float:
    """
    2-opt on a single route: reverse segment route[i..k] (inclusive).
    Cutting edges (i-1,i) and (k,k+1); adding (i-1,k) and (i,k+1).
    Returns Δ cost (new - old).
    """
    if i < 0 or k < 0 or i >= len(route) or k >= len(route) or i >= k:
        return 0.0

    a = _at(route, i - 1)
    b = route[i]
    c = route[k]
    d = _at(route, k + 1)

    # Remove (a-b) and (c-d), add (a-c) and (b-d)
    delta = - D[a, b] - D[c, d] + D[a, c] + D[b, d]
    return float(delta)


def delta_2opt_star(
    D: np.ndarray,
    route_a: List[int], ia: int,
    route_b: List[int], ib: int,
) -> float:
    """
    2-opt* between two routes: cut between (ia, ia+1) in A and (ib, ib+1) in B,
    then re-connect the prefix of A to the suffix of B and vice versa.
    This does NOT reverse segments (the classic 2-opt* variant with no reversal).

    Returns:
        Δ cost (new - old). You must check capacity of the two resulting routes.
    """
    if ia < -1 or ia >= len(route_a) or ib < -1 or ib >= len(route_b):
        return 0.0

    a_left = _at(route_a, ia)          # last in left part of A (or depot if ia == -1)
    a_right = _at(route_a, ia + 1)     # first in right part of A (or depot if ia == len-1 -> depot)

    b_left = _at(route_b, ib)
    b_right = _at(route_b, ib + 1)

    # Old edges: (a_left, a_right) and (b_left, b_right)
    # New edges: (a_left, b_right) and (b_left, a_right)
    delta = - D[a_left, a_right] - D[b_left, b_right] + D[a_left, b_right] + D[b_left, a_right]
    return float(delta)


def delta_oropt_intra(
    D: np.ndarray,
    route: List[int],
    i: int,
    length: int,
    j: int,
) -> float:
    """
    Or-opt (intra-route): remove a chain of 'length' customers starting at i and
    insert it BEFORE position j (both i and j refer to the ORIGINAL route).
    We forbid "no-op" insertions where j ∈ [i, i+length].
    """
    if length <= 0 or length > 3:
        return 0.0
    if i < 0 or i + length > len(route) or j < 0 or j > len(route):
        return 0.0
    # inserting inside the removed block is a no-op
    if j >= i and j <= i + length:
        return 0.0

    left  = _at(route, i - 1)
    first = route[i]
    last  = route[i + length - 1]
    right = _at(route, i + length)

    # removal of the block
    delta = - D[left, first] - D[last, right] + D[left, right]

    # insertion context in the ORIGINAL route (no index shifting):
    u = _at(route, j - 1)   # depot if j==0
    v = _at(route, j)       # depot if j==len(route)

    # insert the block between (u, v)
    delta += - D[u, v] + D[u, first] + D[last, v]

    return float(delta)

# -----------------------------
# Apply helpers (optional)
# -----------------------------

def apply_relocate_intra(route: List[int], i: int, j: int) -> None:
    """In-place relocate within a route: move element at i to be before position j."""
    if i == j or i < 0 or i >= len(route) or j < 0 or j > len(route):
        return
    node = route.pop(i)
    if j > i:
        j -= 1
    route.insert(j, node)


def apply_relocate_inter(route_a: List[int], i: int, route_b: List[int], j: int) -> None:
    """In-place relocate from route_a[i] into route_b before j."""
    if i < 0 or i >= len(route_a) or j < 0 or j > len(route_b):
        return
    node = route_a.pop(i)
    route_b.insert(j, node)


def apply_swap_inter(route_a: List[int], ia: int, route_b: List[int], ib: int) -> None:
    """In-place swap single customers."""
    if ia < 0 or ia >= len(route_a) or ib < 0 or ib >= len(route_b):
        return
    route_a[ia], route_b[ib] = route_b[ib], route_a[ia]


def apply_2opt_intra(route: List[int], i: int, k: int) -> None:
    """In-place reverse segment route[i..k]."""
    if i < 0 or k < 0 or i >= len(route) or k >= len(route) or i >= k:
        return
    route[i : k + 1] = reversed(route[i : k + 1])


def apply_2opt_star(route_a: List[int], ia: int, route_b: List[int], ib: int) -> Tuple[List[int], List[int]]:
    """
    Perform 2-opt* split&join (no reversal):
    A = A_left | A_right, split after ia
    B = B_left | B_right, split after ib
    NewA = A_left | B_right
    NewB = B_left | A_right
    Returns new lists (does not mutate originals).
    """
    A_left = route_a[: ia + 1]         # up to ia inclusive
    A_right = route_a[ia + 1 :]        # after ia
    B_left = route_b[: ib + 1]
    B_right = route_b[ib + 1 :]
    new_a = A_left + B_right
    new_b = B_left + A_right
    return new_a, new_b


def apply_oropt_intra(route: List[int], i: int, length: int, j: int) -> None:
    """In-place Or-opt intra (remove block at [i:i+length) and insert before j)."""
    if length <= 0 or i < 0 or i + length > len(route) or j < 0 or j > len(route):
        return
    block = route[i : i + length]
    del route[i : i + length]
    if j > i:
        j -= length
    route[j:j] = block
