# autotune_and_run.py
from __future__ import annotations

import os
import csv
import time
from typing import Callable, Dict, List, Tuple, Sequence, Optional

import numpy as np

from vrptw_parser import VRPTWInstance, parse_homberger_1999_vrptw_file
from utils import (
    build_distance_matrix,
    normalize_routes,
    solution_cost,
    is_feasible,
)
from constructors import clarke_wright_vrptw
from local_search import local_search_vrptw
from tabu_search import tabu_search_vrptw
from ils import iterated_local_search_vrptw
from alns import alns_vrptw
from lns import lns_vrptw
from plot_utils import _plot_routes  # internal plotting helper

EPS = 1e-6


# ============================================================
# Time budget rules (base, before adaptive extensions)
# ============================================================

def get_base_time_budgets(n_customers: int) -> Dict[str, float]:
    """
    Rule-of-thumb base time budgets (seconds) per algorithm, by instance size.
    Keys: 'ls', 'ils', 'tabu', 'alns', 'lns'
    """
    if n_customers <= 130:
        return {'ls': 6.0, 'ils': 40.0, 'tabu': 40.0, 'alns': 90.0, 'lns': 90.0}
    elif n_customers <= 300:
        return {'ls': 8.0, 'ils': 50.0, 'tabu': 50.0, 'alns': 120.0, 'lns': 120.0}
    elif n_customers <= 700:
        return {'ls': 10.0, 'ils': 90.0, 'tabu': 90.0, 'alns': 120.0, 'lns': 120.0}
    elif n_customers <= 1500:
        return {'ls': 12.0, 'ils': 100.0, 'tabu': 100.0, 'alns': 120.0, 'lns': 120.0}
    else:
        return {'ls': 15.0, 'ils': 120.0, 'tabu': 120.0, 'alns': 120.0, 'lns': 120.0}


# ============================================================
# Simple parameter tuning (rules-of-thumb)
# ============================================================

def tune_tabu_params(inst: VRPTWInstance, base_time: float, verbose: bool) -> Dict:
    n = inst.n_customers
    tenure = max(5, min(40, int(np.sqrt(n))))     # ~sqrt(n) classic
    max_iter = int(5_000 + 10 * n)                # high; time_limit is binding
    return dict(
        max_iterations=max_iter,
        tenure=tenure,
        use_intra_relocate=True,
        use_intra_swap=True,
        use_2opt=True,
        use_inter_relocate=True,
        use_inter_swap=True,
        verbose=verbose,
    )


def tune_ils_params(inst: VRPTWInstance, base_time: float, verbose: bool) -> Dict:
    n = inst.n_customers
    max_it = 40 if n <= 300 else 60 if n <= 700 else 80
    base_strength = 2 if n <= 300 else 3 if n <= 700 else 4
    return dict(
        max_iterations=max_it,
        perturb_strength=base_strength,
        max_perturb_strength=base_strength + 6,
        adapt_perturb_every=5,
        exploratory_accept_prob=0.15,
        accept_equal_cost=True,
        ls_max_iterations=5_000,
        ls_use_inter_route=True,
        verbose=verbose,
    )


def tune_alns_params(inst: VRPTWInstance, base_time: float, verbose: bool) -> Dict:
    n = inst.n_customers
    max_it = 800 if n <= 300 else 1000 if n <= 700 else 1200
    return dict(
        max_iterations=max_it,
        remove_fraction_min=0.05,
        remove_fraction_max=0.25,
        segment_size=50,
        reaction=0.2,
        sigma1=6.0,
        sigma2=3.0,
        sigma3=1.0,
        ls_max_iterations=3_000,
        ls_use_inter_route=True,
        apply_ls_every=3,
        initial_temperature=None,
        final_temperature=None,
        verbose=verbose,
    )


def tune_lns_params(inst: VRPTWInstance, base_time: float, verbose: bool) -> Dict:
    n = inst.n_customers
    max_it = 400 if n <= 300 else 600 if n <= 700 else 800
    return dict(
        max_iterations=max_it,
        remove_fraction_min=0.05,
        remove_fraction_max=0.20,
        use_shaw=True,
        use_worst=True,
        apply_ls_every=3,
        ls_max_iterations=3_000,
        ls_use_inter_route=True,
        exploratory_accept_prob=0.10,
        accept_equal_cost=True,
        verbose=verbose,
    )


# ============================================================
# Saving utilities (solutions & plots)
# ============================================================

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_solution_with_suffix(
    inst: VRPTWInstance,
    routes: Sequence[Sequence[int]],
    D: np.ndarray,
    algo_name: str,
    base_dir: str = "solutions",
) -> Tuple[str, str]:
    """
    Save solution & plot as:
      solutions/{inst.name}/{inst.name}_{algo}.sol
      solutions/{inst.name}/{inst.name}_{algo}.png
    Returns (sol_path, png_path).
    """
    inst_dir = os.path.join(base_dir, inst.name)
    _ensure_dir(inst_dir)

    sol_path = os.path.join(inst_dir, f"{inst.name}_{algo_name}.sol")
    png_path = os.path.join(inst_dir, f"{inst.name}_{algo_name}.png")

    # Normalize & clean
    norm_routes = normalize_routes(routes, depot=0)
    norm_routes = [r for r in norm_routes if len(r) > 2]

    # Write .sol
    total_cost = solution_cost(norm_routes, D)
    lines: List[str] = []
    route_idx = 1
    for r in norm_routes:
        customers = [str(v) for v in r if v != 0]
        if not customers:
            continue
        lines.append(f"Route #{route_idx}: " + " ".join(customers))
        route_idx += 1
    lines.append(f"Cost {total_cost:.4f}")
    with open(sol_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    # Plot
    if norm_routes:
        _plot_routes(inst, norm_routes, png_path)

    return sol_path, png_path


# ============================================================
# Adaptive time wrapper (15s extension logic)
# ============================================================

def run_with_adaptive_time(
    name: str,
    base_limit: float,
    inst: VRPTWInstance,
    D: np.ndarray,
    initial_routes: Sequence[Sequence[int]],
    run_segment_fn: Callable[[Sequence[Sequence[int]], float], Tuple[Sequence[Sequence[int]], Optional[float]]],
    verbose: bool = False,
    improvement_time_window: float = 15.0,
) -> Tuple[List[List[int]], float, float]:
    """Run an algorithm with adaptive +15s extensions near the time frontier.

    Parameters
    ----------
    name : str
        Short identifier used for logs.
    base_limit : float
        Initial time budget (seconds) before any extension.
    inst : VRPTWInstance
        Current VRPTW instance.
    D : np.ndarray
        Distance matrix.
    initial_routes : Sequence[Sequence[int]]
        Starting solution (assumed feasible).
    run_segment_fn : Callable
        Closure that accepts (routes, remaining_time) and returns
        (best_routes_segment, time_to_best_segment_relative).
    verbose : bool
        Enable logging.
    improvement_time_window : float
        Trailing window (seconds) considered "close" to the frontier for extensions.
    """
    current_best = normalize_routes(initial_routes, depot=0)
    current_best = [r for r in current_best if len(r) > 2]
    if not is_feasible(current_best, inst, D, require_all_customers=True):
        raise ValueError(f"Initial routes for {name} are infeasible.")

    best_routes = current_best
    best_cost = solution_cost(best_routes, D)
    time_to_best_global = 0.0

    if base_limit <= 0:
        if verbose:
            print(f"[{name.upper()}] Base time limit <= 0, skipping.")
        return best_routes, best_cost, time_to_best_global

    start_global = time.time()
    total_allowed = float(base_limit)
    adaptive_window = max(0.0, float(improvement_time_window))
    extension_len = 15.0

    if verbose:
        print(f"[{name.upper()}] Start adaptive run. Base limit={base_limit:.1f}s")

    while True:
        now = time.time()
        elapsed = now - start_global
        remaining = total_allowed - elapsed
        if remaining <= 0:
            if verbose:
                print(f"[{name.upper()}] Time limit reached (total_allowed={total_allowed:.1f}s).")
            break

        if verbose:
            print(
                f"[{name.upper()}] New segment. Remaining budget={remaining:.2f}s, "
                f"current best={best_cost:.4f}"
            )

        segment_start = time.time()
        seg_routes, seg_ttb = run_segment_fn(best_routes, remaining)
        segment_end = time.time()
        seg_duration = segment_end - segment_start

        seg_routes = normalize_routes(seg_routes, depot=0)
        seg_routes = [r for r in seg_routes if len(r) > 2]

        if not is_feasible(seg_routes, inst, D, require_all_customers=True):
            if verbose:
                print(f"[{name.upper()}][WARN] Segment returned infeasible solution. Stopping.")
            break

        seg_cost = solution_cost(seg_routes, D)

        if not isinstance(seg_ttb, (int, float)) or not np.isfinite(seg_ttb) or seg_ttb < 0:
            seg_ttb = seg_duration
        else:
            seg_ttb = min(seg_ttb, seg_duration)

        if seg_cost + EPS < best_cost:
            improvement_time = (segment_start - start_global) + float(seg_ttb)
            improvement_time = max(0.0, min(improvement_time, total_allowed))

            best_routes = seg_routes
            best_cost = seg_cost
            time_to_best_global = improvement_time

            if verbose:
                print(
                    f"[{name.upper()}] Improvement: cost={best_cost:.4f} "
                    f"at t={improvement_time:.2f}s (allowed={total_allowed:.1f}s)"
                )

            if adaptive_window > 0.0:
                window = min(adaptive_window, total_allowed)
                if improvement_time >= (total_allowed - window - 1e-9):
                    total_allowed += extension_len
                    if verbose:
                        print(
                            f"[{name.upper()}] Extending time by +{extension_len:.0f}s. "
                            f"New limit={total_allowed:.1f}s"
                        )

    if verbose:
        print(
            f"[{name.upper()}] Finished. Best cost={best_cost:.4f}, "
            f"time_to_best={time_to_best_global:.2f}s, total_allowed={total_allowed:.1f}s"
        )

    return best_routes, best_cost, time_to_best_global


# ============================================================
# Per-algorithm segment runners
# ============================================================

def make_ls_runner(inst: VRPTWInstance, D: np.ndarray, ls_params: Dict, verbose: bool):
    def run_segment(routes: Sequence[Sequence[int]], remaining: float):
        if remaining <= 0:
            return routes, 0.0
        t0 = time.time()
        out = local_search_vrptw(
            inst,
            routes,
            D=D,
            max_iterations=ls_params.get("max_iterations", 10_000),
            use_inter_route=ls_params.get("use_inter_route", True),
            time_limit_sec=remaining,
            verbose=verbose,
        )
        t1 = time.time()
        seg_time = max(0.0, min(t1 - t0, remaining))
        # LS is strictly improving; assume best at end of segment.
        return out, seg_time
    return run_segment


def make_tabu_runner(inst: VRPTWInstance, D: np.ndarray, tabu_params: Dict):
    def run_segment(routes: Sequence[Sequence[int]], remaining: float):
        if remaining <= 0:
            return routes, 0.0
        setattr(tabu_search_vrptw, "time_to_best", None)
        out = tabu_search_vrptw(
            inst,
            routes,
            D=D,
            time_limit_sec=remaining,
            **{k: v for k, v in tabu_params.items() if k != "time_limit_sec"},
        )
        seg_ttb = getattr(tabu_search_vrptw, "time_to_best", None)
        return out, seg_ttb
    return run_segment


def make_ils_runner(inst: VRPTWInstance, D: np.ndarray, ils_params: Dict):
    def run_segment(routes: Sequence[Sequence[int]], remaining: float):
        if remaining <= 0:
            return routes, 0.0
        setattr(iterated_local_search_vrptw, "time_to_best", None)
        out = iterated_local_search_vrptw(
            inst,
            initial_routes=routes,
            D=D,
            time_limit_sec=remaining,
            **{k: v for k, v in ils_params.items() if k != "time_limit_sec"},
        )
        seg_ttb = getattr(iterated_local_search_vrptw, "time_to_best", None)
        return out, seg_ttb
    return run_segment


def make_alns_runner(inst: VRPTWInstance, D: np.ndarray, alns_params: Dict):
    def run_segment(routes: Sequence[Sequence[int]], remaining: float):
        if remaining <= 0:
            return routes, 0.0
        setattr(alns_vrptw, "time_to_best", None)
        out = alns_vrptw(
            inst,
            initial_routes=routes,
            D=D,
            time_limit_sec=remaining,
            **{k: v for k, v in alns_params.items() if k != "time_limit_sec"},
        )
        seg_ttb = getattr(alns_vrptw, "time_to_best", None)
        return out, seg_ttb
    return run_segment


def make_lns_runner(inst: VRPTWInstance, D: np.ndarray, lns_params: Dict):
    def run_segment(routes: Sequence[Sequence[int]], remaining: float):
        if remaining <= 0:
            return routes, 0.0
        setattr(lns_vrptw, "time_to_best", None)
        out = lns_vrptw(
            inst,
            initial_routes=routes,
            D=D,
            time_limit_sec=remaining,
            **{k: v for k, v in lns_params.items() if k != "time_limit_sec"},
        )
        seg_ttb = getattr(lns_vrptw, "time_to_best", None)
        return out, seg_ttb
    return run_segment


# ============================================================
# Main driver for a single instance
# ============================================================

def solve_instance_with_all(
    inst_path: str,
    results_rows: List[List[object]],
    base_dir: str = "solutions",
    verbose: bool = False,
    start_with_ls_solution: bool = True,
    improvement_time_window: float = 15.0,
) -> None:
    inst = parse_homberger_1999_vrptw_file(inst_path)
    D = build_distance_matrix(inst)
    n = inst.n_customers

    if verbose:
        print(f"\n=== Instance {inst.name} | n={n}, vehicles={inst.n_vehicles}, cap={inst.capacity} ===")

    base_times = get_base_time_budgets(n)

    # ---------- 1) Clarke-Wright ----------
    if verbose:
        print("[CW] Constructing initial solution...")
    cw_routes = clarke_wright_vrptw(inst, D=D)
    cw_routes = normalize_routes(cw_routes, depot=0)
    cw_routes = [r for r in cw_routes if len(r) > 2]
    cw_cost = solution_cost(cw_routes, D)
    cw_ttb = 0.0
    cw_sol, cw_png = save_solution_with_suffix(inst, cw_routes, D, "cw", base_dir)
    if verbose:
        print(f"[CW] Cost={cw_cost:.4f}, routes={len(cw_routes)}")
    results_rows.append([inst.name, "cw", os.path.basename(cw_sol), cw_cost, cw_ttb])

    # ---------- 2) Local Search ----------
    if verbose:
        print(f"[LS] Starting from CW. Base time={base_times['ls']:.1f}s")
    ls_params = dict(
        max_iterations=20_000,
        use_inter_route=True,
    )
    ls_runner = make_ls_runner(inst, D, ls_params, verbose=verbose)
    ls_routes, ls_cost, ls_ttb = run_with_adaptive_time(
        "ls",
        base_times["ls"],
        inst,
        D,
        cw_routes,
        ls_runner,
        verbose=verbose,
        improvement_time_window=improvement_time_window,
    )
    ls_sol, ls_png = save_solution_with_suffix(inst, ls_routes, D, "ls", base_dir)
    results_rows.append([inst.name, "ls", os.path.basename(ls_sol), ls_cost, ls_ttb])

    if not is_feasible(ls_routes, inst, D, require_all_customers=True):
        if verbose:
            print("[LS][WARN] LS solution infeasible; falling back to CW.")
        ls_routes, ls_cost, ls_ttb = cw_routes, cw_cost, cw_ttb

    # Decide whether subsequent metaheuristics start from LS or CW
    if start_with_ls_solution:
        start_routes = ls_routes
    else:
        start_routes = cw_routes

    # ---------- 3) Tabu Search ----------
    if verbose:
        print(f"[TABU] Base time={base_times['tabu']:.1f}s (auto-tuned params)")
    tabu_params = tune_tabu_params(inst, base_times["tabu"], verbose=verbose)
    tabu_runner = make_tabu_runner(inst, D, tabu_params)
    tabu_routes, tabu_cost, tabu_ttb = run_with_adaptive_time(
        "tabu",
        base_times["tabu"],
        inst,
        D,
        start_routes,
        tabu_runner,
        verbose=verbose,
        improvement_time_window=improvement_time_window,
    )
    tabu_sol, tabu_png = save_solution_with_suffix(inst, tabu_routes, D, "tabu", base_dir)
    results_rows.append([inst.name, "tabu", os.path.basename(tabu_sol), tabu_cost, tabu_ttb])

    # ---------- 4) ILS ----------
    if verbose:
        print(f"[ILS] Base time={base_times['ils']:.1f}s (auto-tuned params)")
    ils_params = tune_ils_params(inst, base_times["ils"], verbose=verbose)
    ils_runner = make_ils_runner(inst, D, ils_params)
    ils_routes, ils_cost, ils_ttb = run_with_adaptive_time(
        "ils",
        base_times["ils"],
        inst,
        D,
        start_routes,
        ils_runner,
        verbose=verbose,
        improvement_time_window=improvement_time_window,
    )
    ils_sol, ils_png = save_solution_with_suffix(inst, ils_routes, D, "ils", base_dir)
    results_rows.append([inst.name, "ils", os.path.basename(ils_sol), ils_cost, ils_ttb])

    # ---------- 5) ALNS ----------
    if verbose:
        print(f"[ALNS] Base time={base_times['alns']:.1f}s (auto-tuned params)")
    alns_params = tune_alns_params(inst, base_times["alns"], verbose=verbose)
    alns_runner = make_alns_runner(inst, D, alns_params)
    alns_routes, alns_cost, alns_ttb = run_with_adaptive_time(
        "alns",
        base_times["alns"],
        inst,
        D,
        start_routes,
        alns_runner,
        verbose=verbose,
        improvement_time_window=improvement_time_window,
    )
    alns_sol, alns_png = save_solution_with_suffix(inst, alns_routes, D, "alns", base_dir)
    results_rows.append([inst.name, "alns", os.path.basename(alns_sol), alns_cost, alns_ttb])

    # ---------- 6) LNS ----------
    if verbose:
        print(f"[LNS] Base time={base_times['lns']:.1f}s (auto-tuned params)")
    lns_params = tune_lns_params(inst, base_times["lns"], verbose=verbose)
    lns_runner = make_lns_runner(inst, D, lns_params)
    lns_routes, lns_cost, lns_ttb = run_with_adaptive_time(
        "lns",
        base_times["lns"],
        inst,
        D,
        start_routes,
        lns_runner,
        verbose=verbose,
        improvement_time_window=improvement_time_window,
    )
    lns_sol, lns_png = save_solution_with_suffix(inst, lns_routes, D, "lns", base_dir)
    results_rows.append([inst.name, "lns", os.path.basename(lns_sol), lns_cost, lns_ttb])

    # ---------- 7) Best-of-all ----------
    algo_solutions = [
        ("cw", cw_routes, cw_cost, cw_ttb),
        ("ls", ls_routes, ls_cost, ls_ttb),
        ("tabu", tabu_routes, tabu_cost, tabu_ttb),
        ("ils", ils_routes, ils_cost, ils_ttb),
        ("alns", alns_routes, alns_cost, alns_ttb),
        ("lns", lns_routes, lns_cost, lns_ttb),
    ]
    best_algo, best_routes, best_cost, best_ttb = min(algo_solutions, key=lambda x: x[2])
    best_sol, best_png = save_solution_with_suffix(inst, best_routes, D, "best", base_dir)
    results_rows.append([inst.name, "best", os.path.basename(best_sol), best_cost, best_ttb])

    if verbose:
        print(f"[SUMMARY] {inst.name}: "
              f"best_algo={best_algo}, best_cost={best_cost:.4f}, "
              f"best_time_to_best={best_ttb:.2f}s")


# ============================================================
# Batch runner over directory
# ============================================================

def run_batch(
    directory: str,
    base_dir: str = "solutions",
    results_csv: str = "solutions/results.csv",
    verbose: bool = False,
    start_with_ls_solution: bool = True,
    improvement_time_window: float = 15.0,
) -> None:
    """Run `solve_instance_with_all` for each .txt instance in `directory`.

    The `improvement_time_window` can be controlled by passing it through
    `solve_instance_with_all` (default 15s). To change it for a batch run,
    set the parameter when calling `run_batch` if you wrap this function.
    """
    txt_files = [
        f for f in os.listdir(directory)
        if f.lower().endswith(".txt")
    ]
    txt_files.sort()

    all_rows: List[List[object]] = []
    header = ["instance", "algo", "solution", "cost", "time_to_best"]

    if verbose:
        print(f"Scanning directory: {directory}")
        print(f"Found {len(txt_files)} .txt instances")

    for fname in txt_files:
        inst_path = os.path.join(directory, fname)
        solve_instance_with_all(
            inst_path,
            all_rows,
            base_dir=base_dir,
            verbose=verbose,
            start_with_ls_solution=start_with_ls_solution,
            improvement_time_window=improvement_time_window,
        )

    _ensure_dir(os.path.dirname(results_csv) or ".")
    if verbose:
        print(f"\nWriting results CSV to: {results_csv}")
    with open(results_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(all_rows)

    if verbose:
        print("Done.")
