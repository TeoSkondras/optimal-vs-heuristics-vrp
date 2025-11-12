from __future__ import annotations

from dataclasses import dataclass
from typing import List, Iterable, Sequence, Tuple

import numpy as np

# Adjust this import to match your actual parser module name.
from vrptw_parser import VRPTWInstance


# ------------- Distance / Cost Precomputation -------------


def build_distance_matrix(inst: VRPTWInstance) -> np.ndarray:
    """
    Precompute full distance matrix D[i,j] using instance.dist().
    Shape: (n+1, n+1).
    """
    n_nodes = inst.coords.shape[0]
    D = np.zeros((n_nodes, n_nodes), dtype=float)
    # exploit symmetry: compute only for j > i and mirror
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            d = inst.dist(i, j)
            D[i, j] = d
            D[j, i] = d
    return D


def route_cost(route: Sequence[int], D: np.ndarray) -> float:
    """Sum of arc costs along a single route [0, ..., 0]."""
    if len(route) < 2:
        return 0.0
    return float(sum(D[route[i], route[i + 1]] for i in range(len(route) - 1)))


def solution_cost(solution: Sequence[Sequence[int]], D: np.ndarray) -> float:
    """Total cost over all routes."""
    return float(sum(route_cost(r, D) for r in solution))


# ------------- Loads & Time Windows -------------


def route_load(route: Sequence[int], inst: VRPTWInstance) -> int:
    """
    Total demand on a route (excluding the depot).
    Assumes route includes depot at start/end.
    """
    return int(sum(inst.demand[node] for node in route if node != 0))


@dataclass
class RouteSchedule:
    """
    Time-related information along a route.
    All arrays have length == len(route).
    """
    arrival: np.ndarray
    start_service: np.ndarray
    departure: np.ndarray


def compute_route_schedule(
    route: Sequence[int],
    inst: VRPTWInstance,
    D: np.ndarray,
) -> RouteSchedule:
    """
    Forward simulation of arrival/start/departure times.

    Assumes:
      - route[0] == 0 and route[-1] == 0
      - No depot in the middle
    """
    m = len(route)
    arrival = np.zeros(m, dtype=float)
    start = np.zeros(m, dtype=float)
    depart = np.zeros(m, dtype=float)

    # Depot start
    first = route[0]
    if first != 0:
        raise ValueError("Route must start at depot (0).")
    start[0] = max(0.0, float(inst.ready_time[first]))
    depart[0] = start[0] + float(inst.service_time[first])

    for k in range(1, m):
        i = route[k - 1]
        j = route[k]
        travel = D[i, j]
        arrival[k] = depart[k - 1] + travel
        start[k] = max(arrival[k], float(inst.ready_time[j]))
        depart[k] = start[k] + float(inst.service_time[j])

    return RouteSchedule(arrival=arrival, start_service=start, departure=depart)


def is_route_feasible(
    route: Sequence[int],
    inst: VRPTWInstance,
    D: np.ndarray,
    check_capacity: bool = True,
    check_time_windows: bool = True,
) -> bool:
    """
    Check feasibility of a single route:
      - starts & ends at depot 0
      - no depot in the middle
      - capacity respected
      - time windows & service respected
    """
    if not route:
        return True

    if route[0] != 0 or route[-1] != 0:
        return False

    # No depot in the middle
    for v in route[1:-1]:
        if v == 0:
            return False
        if v < 0 or v >= inst.coords.shape[0]:
            return False

    if check_capacity:
        if route_load(route, inst) > inst.capacity:
            return False

    if check_time_windows:
        sched = compute_route_schedule(route, inst, D)
        # All must start service within their due time
        for idx, node in enumerate(route):
            if sched.start_service[idx] > float(inst.due_time[node]) + 1e-9:
                return False

    return True


def is_feasible(
    solution: Sequence[Sequence[int]],
    inst: VRPTWInstance,
    D: np.ndarray,
    require_all_customers: bool = True,
) -> bool:
    """
    Global feasibility:
      - Each route feasible.
      - Each customer 1..n appears at most once.
      - If require_all_customers: each 1..n appears exactly once.
    """
    n = inst.n_customers
    seen = np.zeros(n + 1, dtype=int)  # index by node id, 0..n

    for route in solution:
        if not is_route_feasible(route, inst, D):
            return False
        for node in route:
            if node == 0:
                continue
            if node < 0 or node > n:
                return False
            seen[node] += 1
            if seen[node] > 1:
                return False

    if require_all_customers:
        # All customers 1..n visited exactly once
        for node in range(1, n + 1):
            if seen[node] != 1:
                return False

    return True


# ------------- Constant-time Delta Cost Utilities -------------
# These operate on arc cost only (using D).
# Use together with feasibility checks (or incremental TW checks).


def delta_cost_relocate_intra(
    route: Sequence[int],
    i: int,
    j: int,
    D: np.ndarray,
) -> float:
    """
    Constant-time cost delta for relocating a customer inside the same route.

    Semantics:
      - route: [0, ..., 0]
      - i: index of node to move (1 <= i <= len(route)-2)
      - j: target position index BEFORE which the node will be inserted,
           interpreted on the ORIGINAL route.
      - Depots (position 0 and last) are not moved.

    Returns:
      new_cost - old_cost  (can be negative).

    Notes:
      - If j == i or j == i+1 => no change (delta = 0).
      - j is clamped to [1, len(route)-1] so insertion is never before start
        depot nor beyond final depot (it effectively becomes "before last 0").
    """
    n = len(route)
    if i <= 0 or i >= n - 1:
        raise ValueError("i must be an internal (non-depot) position.")
    if route[i] == 0:
        raise ValueError("Cannot relocate depot.")

    # clamp j
    if j < 1:
        j = 1
    if j > n - 1:
        j = n - 1

    # no-op cases
    if j == i or j == i + 1:
        return 0.0

    node = route[i]
    prev_i = route[i - 1]
    next_i = route[i + 1]

    # cost for removing node from (prev_i -> node -> next_i)
    delta = - (D[prev_i, node] + D[node, next_i] - D[prev_i, next_i])

    # insertion: insert node before position j (on original route)
    left = route[j - 1]
    right = route[j]
    delta += D[left, node] + D[node, right] - D[left, right]

    return float(delta)


def delta_cost_swap_intra(
    route: Sequence[int],
    i: int,
    j: int,
    D: np.ndarray,
) -> float:
    """
    Constant-time cost delta for swapping two customers in the same route.

    Assumes:
      - route: [0, ..., 0]
      - 1 <= i, j <= len(route)-2 (no depots)
    """
    if i == j:
        return 0.0
    if i > j:
        i, j = j, i

    n = len(route)
    if i <= 0 or j >= n - 1:
        raise ValueError("Swap indices must be internal (non-depot).")

    a = route[i]
    b = route[j]
    if a == 0 or b == 0:
        raise ValueError("Do not swap depot nodes.")

    # Adjacent swap
    if j == i + 1:
        left = route[i - 1]
        right = route[j + 1]
        before = D[left, a] + D[a, b] + D[b, right]
        after = D[left, b] + D[b, a] + D[a, right]
        return float(after - before)

    # Non-adjacent swap
    left_i = route[i - 1]
    right_i = route[i + 1]
    left_j = route[j - 1]
    right_j = route[j + 1]

    before = (
        D[left_i, a] + D[a, right_i] +
        D[left_j, b] + D[b, right_j]
    )
    after = (
        D[left_i, b] + D[b, right_i] +
        D[left_j, a] + D[a, right_j]
    )
    return float(after - before)


def delta_cost_2opt(
    route: Sequence[int],
    i: int,
    j: int,
    D: np.ndarray,
) -> float:
    """
    Constant-time 2-opt move delta within a single route.

    2-opt on a single route:
      - Choose edges (route[i-1], route[i]) and (route[j], route[j+1])
      - Reverse segment [i..j]
      - Assumes:
          1 <= i < j <= len(route)-2
          (no depot indices involved)

    Works as expected for symmetric distances (EUC_2D etc.).
    """
    if i >= j:
        return 0.0

    n = len(route)
    if i <= 0 or j >= n - 1:
        raise ValueError("2-opt cannot involve depot positions.")

    a = route[i - 1]
    b = route[i]
    c = route[j]
    d = route[j + 1]

    before = D[a, b] + D[c, d]
    after = D[a, c] + D[b, d]
    return float(after - before)


def delta_cost_relocate_inter(
    route_from: Sequence[int],
    i_from: int,
    route_to: Sequence[int],
    j_to: int,
    D: np.ndarray,
) -> float:
    """
    Constant-time delta for relocating a customer from one route to another.

    Semantics:
      - route_from, route_to are full routes including depot at ends.
      - i_from index in route_from (1..len(route_from)-2)
      - j_to insertion index BEFORE which node will be inserted in route_to
        interpreted on the original route_to (1..len(route_to)-1 inclusive).

    Returns new_cost - old_cost.
    """
    n_from = len(route_from)
    n_to = len(route_to)
    if i_from <= 0 or i_from >= n_from - 1:
        raise ValueError("i_from must be internal (non-depot) position.")

    # clamp j_to
    if j_to < 1:
        j_to = 1
    if j_to > n_to - 1:
        j_to = n_to - 1

    if route_from is route_to:
        # relocating inside same route should use intra version
        raise ValueError("Use intra-relocate for same-route moves.")

    x = route_from[i_from]
    a = route_from[i_from - 1]
    b = route_from[i_from + 1]

    left = route_to[j_to - 1]
    right = route_to[j_to]

    # removal from route_from: remove arcs (a->x) + (x->b), add (a->b)
    delta_from = - (D[a, x] + D[x, b] - D[a, b])

    # insertion into route_to: remove (left->right), add (left->x) + (x->right)
    delta_to = - (D[left, right]) + (D[left, x] + D[x, right])

    return float(delta_from + delta_to)


def delta_cost_swap_inter(
    route1: Sequence[int],
    i1: int,
    route2: Sequence[int],
    i2: int,
    D: np.ndarray,
) -> float:
    """
    Constant-time delta for swapping nodes at (route1,i1) and (route2,i2).

    Assumes routes are distinct. Returns new_cost - old_cost.
    """
    if route1 is route2:
        raise ValueError("Use intra-swap for same-route moves.")

    n1 = len(route1)
    n2 = len(route2)
    if i1 <= 0 or i1 >= n1 - 1:
        raise ValueError("i1 must be internal (non-depot)")
    if i2 <= 0 or i2 >= n2 - 1:
        raise ValueError("i2 must be internal (non-depot)")

    a = route1[i1]
    b = route2[i2]

    left1 = route1[i1 - 1]
    right1 = route1[i1 + 1]
    left2 = route2[i2 - 1]
    right2 = route2[i2 + 1]

    before1 = D[left1, a] + D[a, right1]
    after1 = D[left1, b] + D[b, right1]

    before2 = D[left2, b] + D[b, right2]
    after2 = D[left2, a] + D[a, right2]

    return float((after1 - before1) + (after2 - before2))


# ------------- Small Helper Utilities -------------


def normalize_routes(
    routes: Iterable[Iterable[int]],
    depot: int = 0,
) -> List[List[int]]:
    """
    Normalize a list of routes:
      - Ensure each route starts and ends with depot.
      - Remove duplicated leading/trailing depots.
    """
    norm: List[List[int]] = []
    for r in routes:
        rr = [v for v in r]
        if not rr:
            continue
        if rr[0] != depot:
            rr.insert(0, depot)
        if rr[-1] != depot:
            rr.append(depot)
        # collapse any accidental internal consecutive depots at ends
        while len(rr) > 1 and rr[0] == depot and rr[1] == depot:
            rr.pop(0)
        while len(rr) > 1 and rr[-1] == depot and rr[-2] == depot:
            rr.pop()
        norm.append(rr)
    return norm


def is_route_time_feasible_with_prefix(
    route_old: Sequence[int],
    sched_old: RouteSchedule,
    new_route: Sequence[int],
    inst: VRPTWInstance,
    D: np.ndarray,
    prefix_keep: int,
) -> bool:
    """
    Check time-window feasibility of `new_route` by reusing the prefix schedule
    from `route_old` up to `prefix_keep-1`. Only simulates times from prefix_keep
    onward.

    Parameters:
      - prefix_keep: number of nodes at the start to keep from the old route
        (0 <= prefix_keep <= len(new_route)). If prefix_keep == 0, simulation
        starts from depot with fresh timings.

    Returns True if time windows are respected.
    """
    m_new = len(new_route)
    if m_new == 0:
        return True

    # Basic checks
    if new_route[0] != 0 or new_route[-1] != 0:
        return False

    # Validate prefix_keep bounds
    if prefix_keep < 0:
        prefix_keep = 0
    if prefix_keep > m_new:
        prefix_keep = m_new

    # If prefix_keep == 0, start from depot fresh
    arrival = 0.0
    start_service = 0.0
    depart_prev = 0.0
    if prefix_keep == 0:
        # depot at position 0
        depart_prev = max(0.0, float(inst.ready_time[0])) + float(inst.service_time[0])
        k_start = 1
    else:
        # Use old schedule departure at prefix_keep-1 if compatible length
        if len(sched_old.departure) >= prefix_keep:
            depart_prev = float(sched_old.departure[prefix_keep - 1])
        else:
            # Fall back to recomputing from scratch
            depart_prev = max(0.0, float(inst.ready_time[0])) + float(inst.service_time[0])
            k_start = 1
        k_start = prefix_keep

    # Simulate forward from k_start..m_new-1
    for k in range(k_start, m_new):
        i = new_route[k - 1]
        j = new_route[k]
        travel = D[i, j]
        arrival = depart_prev + travel
        start_service = max(arrival, float(inst.ready_time[j]))
        # Check due time
        if start_service > float(inst.due_time[j]) + 1e-9:
            return False
        depart_prev = start_service + float(inst.service_time[j])

    return True
