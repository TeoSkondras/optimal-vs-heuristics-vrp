# autotune.py
from __future__ import annotations
import contextlib
from dataclasses import dataclass
import sys
from typing import Dict, Any, List, Tuple, Optional
import math
import random
import time
import logging
import os
import shutil
import numpy as np
import re
import io
import csv
import json

from utils import (
    build_distance_matrix,
    route_cost,
    route_load,
)
from constructors import clarke_wright_parallel
from plot_utils import save_solution, plot_solution_from_file

# Solvers
from local_search import local_search
from ils import ILSConfig, run_ils
from tabu_search import tabu_search
from alns import alns as alns_run
from lns import lns as lns_run


# -------------------------------------------------------------------
# Feature extraction (unchanged) + SA scale sampling
# -------------------------------------------------------------------

@dataclass
class InstanceFeatures:
    n: int
    Q: int
    dem_mean: float
    dem_std: float
    dem_max: int
    total_demand: int
    tightness: float
    bbox_area: float
    density: float
    route_count: int
    avg_route_len: float
    med_route_len: float
    nn_k_default: int


def extract_features(coords: np.ndarray,
                     demand: np.ndarray,
                     Q: int,
                     seed_routes: List[List[int]],
                     edge_weight_type: str = "EUC_2D",
                     round_euclidean: bool = False) -> Tuple[InstanceFeatures, np.ndarray]:
    n = coords.shape[0] - 1
    xs, ys = coords[:, 0], coords[:, 1]
    minx, maxx = float(xs.min()), float(xs.max())
    miny, maxy = float(ys.min()), float(ys.max())
    width, height = max(1e-6, maxx - minx), max(1e-6, maxy - miny)
    area = width * height
    density = n / math.sqrt(area) if area > 0 else n

    dem = demand[1:].astype(float)
    dem_mean = float(dem.mean() if len(dem) else 0.0)
    dem_std = float(dem.std() if len(dem) else 0.0)
    dem_max = int(dem.max() if len(dem) else 0)
    total_dem = int(dem.sum())

    route_count = len(seed_routes)
    rlens = [len(r) for r in seed_routes] or [0]
    avg_rl = float(np.mean(rlens))
    med_rl = float(np.median(rlens))

    min_routes_by_dem = max(1, math.ceil(total_dem / max(1, Q)))
    routes_guess = max(min_routes_by_dem, route_count)
    tightness = total_dem / max(1, Q * routes_guess)

    # For very small instances, use a smaller k-nearest default (but keep a sensible lower bound).
    # Original default (0.05*n clipped to [8,40]) is too large for n<=70, so reduce bounds there.
    if n <= 70:
        nn_k = int(np.clip(round(0.08 * n), 4, 20))
    else:
        nn_k = int(np.clip(round(0.05 * n), 8, 40))

    D = build_distance_matrix(coords, edge_weight_type, round_euclidean=round_euclidean)
    feats = InstanceFeatures(
        n=n, Q=Q, dem_mean=dem_mean, dem_std=dem_std, dem_max=dem_max,
        total_demand=total_dem, tightness=tightness, bbox_area=area,
        density=density, route_count=route_count, avg_route_len=avg_rl,
        med_route_len=med_rl, nn_k_default=nn_k
    )
    return feats, D


def sample_insertion_scale(D: np.ndarray, routes: List[List[int]], samples: int = 400, rng: Optional[random.Random] = None) -> float:
    rng = rng or random.Random(0)
    all_customers = [v for r in routes for v in r]
    routes_nonempty = [r for r in routes if r]
    if not all_customers or not routes_nonempty:
        return 1.0

    def best_insert_delta(route: List[int], cust: int) -> float:
        if not route:
            return float(D[0, cust] + D[cust, 0])
        best = float("inf")
        for j in range(len(route) + 1):
            u = 0 if j == 0 else route[j - 1]
            v = 0 if j == len(route) else route[j]
            d = -D[u, v] + D[u, cust] + D[cust, v]
            if d < best:
                best = d
        return float(best)

    pos = []
    for _ in range(samples):
        v = rng.choice(all_customers)
        r = rng.choice(routes_nonempty)
        d = best_insert_delta(r, v)
        if d > 0:
            pos.append(d)

    if pos:
        return float(np.median(pos))
    # fallback: typical arc scale
    depot_to_any = [float(D[0, i]) for i in range(1, D.shape[0])]
    return float((np.mean(depot_to_any) * 2.0) if depot_to_any else 1.0)


# -------------------------------------------------------------------
# Best-time catcher: attaches to algorithm loggers, no code changes
# -------------------------------------------------------------------

class AdvancedBestCatcher(logging.Handler):
    """
    Tracks 'time-to-best' by watching log lines.
    Triggers on:
      - 'cur=<number>' (take the minimum seen)  <-- NEW
      - 'NEW BEST' / 'new best'
      - 'best=<number>'
    """
    RE_NEW_BEST = re.compile(r"\bnew\s+best\b", re.IGNORECASE)
    RE_BEST_NUM = re.compile(r"\bbest\s*=\s*([0-9]*\.?[0-9]+)")
    RE_CUR_NUM  = re.compile(r"\bcur\s*=\s*([0-9]*\.?[0-9]+)", re.IGNORECASE)

    def __init__(self, start_time: float):
        super().__init__(level=logging.INFO)
        self.start_time = start_time
        self.best_time: float | None = None
        self.best_val: float | None = None  # track min cost seen

    def emit(self, record: logging.LogRecord) -> None:
        msg = record.getMessage()
        now = time.perf_counter()

        # 1) Prefer explicit current cost lines: [ACC] ... cur=XXXXX
        mcur = self.RE_CUR_NUM.search(msg)
        if mcur:
            try:
                v = float(mcur.group(1))
            except Exception:
                v = None
            if v is not None and ((self.best_val is None) or (v < self.best_val - 1e-12)):
                self.best_val = v
                self.best_time = now - self.start_time
            return  # already processed

        # 2) Explicit NEW BEST tokens (fallback)
        if self.RE_NEW_BEST.search(msg):
            self.best_time = now - self.start_time
            return

        # 3) Lines with best=...
        mbest = self.RE_BEST_NUM.search(msg)
        if mbest:
            try:
                v = float(mbest.group(1))
            except Exception:
                return
            if (self.best_val is None) or (v < self.best_val - 1e-12):
                self.best_val = v
                self.best_time = now - self.start_time


def attach_best_catcher(logger: logging.Logger):
    start = time.perf_counter()
    catcher = AdvancedBestCatcher(start)
    logger.setLevel(logging.INFO)
    logger.propagate = True
    logger.addHandler(catcher)
    return catcher, start

class RealtimeStdoutBestCatcher(io.TextIOBase):
    RE_NEW_BEST = re.compile(r"\bnew\s+best\b", re.IGNORECASE)
    RE_BEST_NUM = re.compile(r"\bbest\s*=\s*([0-9]*\.?[0-9]+)")
    RE_CUR_NUM  = re.compile(r"\bcur\s*=\s*([0-9]*\.?[0-9]+)", re.IGNORECASE)

    def __init__(self, start_time: float, fallback_stream=None):
        super().__init__()
        self.start_time = start_time
        self.best_time: float | None = None
        self.best_val: float | None = None
        self._buf = ""
        self._fallback = fallback_stream or sys.__stdout__

    def write(self, s: str) -> int:
        self._fallback.write(s)
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self._process_line(line)
        return len(s)

    def flush(self) -> None:
        self._fallback.flush()

    def _process_line(self, line: str) -> None:
        now = time.perf_counter()
        if self.RE_NEW_BEST.search(line):
            self.best_time = now - self.start_time
            return
        mcur = self.RE_CUR_NUM.search(line)
        if mcur:
            try:
                v = float(mcur.group(1))
            except Exception:
                v = None
            if v is not None and ((self.best_val is None) or (v < self.best_val - 1e-12)):
                self.best_val = v
                self.best_time = now - self.start_time
            return
        mbest = self.RE_BEST_NUM.search(line)
        if mbest:
            try:
                v = float(mbest.group(1))
            except Exception:
                return
            if (self.best_val is None) or (v < self.best_val - 1e-12):
                self.best_val = v
                self.best_time = now - self.start_time

# -------------------------------------------------------------------
# Saving utilities (names & files)
# -------------------------------------------------------------------

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def save_named_solution_and_plot(inst, routes: List[List[int]], cost: float, algo_tag: str, inst_name: Optional[str] = None, runtime_sec: Optional[float] = None) -> Tuple[str, str]:
    """
    Uses plot_utils.save_solution / plot_solution_from_file,
    then renames to '<inst_name>_<algo>.sol|.png' inside 'solutions/<inst_name>/'.
    """
    # Save with the existing helper (creates directory solutions/inst.name/)
    sol_path = save_solution(inst, routes, cost, runtime_sec=runtime_sec)
    png_path = plot_solution_from_file(inst, sol_path)

    # Compute new names
    # Try to read the directory from sol_path and derive inst_name from caller if provided
    sol_dir = os.path.dirname(sol_path)
    if inst_name is None:
        # derive inst name from directory name
        inst_name = os.path.basename(sol_dir)

    _ensure_dir(sol_dir)
    sol_new = os.path.join(sol_dir, f"{inst_name}_{algo_tag}.sol")
    png_new = os.path.join(sol_dir, f"{inst_name}_{algo_tag}.png")

    try:
        shutil.copyfile(sol_path, sol_new)
    except Exception:
        pass
    try:
        shutil.copyfile(png_path, png_new)
    except Exception:
        pass

    return sol_new, png_new


# -------------------------------------------------------------------
# Auto-tuners (same logic as before) + CW and plain LS runners
# -------------------------------------------------------------------

def tune_and_run_cw(inst, logger: Optional[logging.Logger] = None, verbose: bool = True) -> Tuple[Dict[str, Any], Any, float]:
    """
    Clarke–Wright baseline. Returns (cfg, solution_obj, time_to_best_sec).
    For CW, best is reached at end; we report wall time.
    """
    start = time.perf_counter()
    cw = clarke_wright_parallel(inst.coords, inst.demand, inst.capacity, inst.edge_weight_type)
    elapsed = time.perf_counter() - start
    if logger and verbose:
        logger.info(f"[CW] routes={len(cw.routes)} cost={cw.cost:.2f} time={elapsed:.2f}s")
    cfg = dict()
    return cfg, cw, elapsed


def tune_and_run_local_search(
    inst,
    routes_seed: List[List[int]],
    time_budget_sec: float,
    logger: Optional[logging.Logger] = None,
    verbose: bool = True,
    seed: int = 0,
) -> Tuple[Dict[str, Any], Any, float]:
    """
    Plain Local Search applied to seed routes (no perturbations).
    We reuse feature-driven k and a small time budget slice.
    We capture 'time-to-best' via logger hook (if LS logs best updates; else use wall time).
    """
    feats, D = extract_features(inst.coords, inst.demand, inst.capacity, routes_seed, inst.edge_weight_type, False)
    k_nearest = feats.nn_k_default
    # For very small instances, prefer a smaller local-search slice to avoid overspending
    # on brief problems; otherwise keep previous heuristics.
    if feats.n <= 70:
        ls_slice = min(time_budget_sec, 0.35)
    else:
        ls_slice = min(time_budget_sec, 1.0 if feats.n >= 300 else 0.6)

    # Prepare a child logger
    lgr = logging.getLogger("LocalSearch")
    lgr.setLevel(logging.INFO)
    catcher, start = attach_best_catcher(lgr)

    start_call = time.perf_counter()
    ls = local_search(
        coords=inst.coords, demand=inst.demand, Q=inst.capacity,
        routes=[r[:] for r in routes_seed],
        edge_weight_type=inst.edge_weight_type, round_euclidean=False,
        k_nearest=int(k_nearest),
        max_passes_without_gain=2 if feats.n >= 300 else 1,
        time_limit_sec=float(ls_slice),
        verbose=verbose,
        logger=lgr,
    )
    total_elapsed = time.perf_counter() - start_call
    time_to_best = catcher.best_time if catcher.best_time is not None else total_elapsed

    if logger and verbose:
        logger.info(f"[LS] routes={len(ls.routes)} cost={ls.cost:.2f} time={total_elapsed:.2f}s best_at={time_to_best:.2f}s")

    cfg = dict(k_nearest=int(k_nearest), time_limit_sec=float(ls_slice), max_passes_without_gain=(2 if feats.n >= 300 else 1))
    return cfg, ls, time_to_best


def tune_and_run_ils(
    coords: np.ndarray, demand: np.ndarray, Q: int, routes_initial: List[List[int]],
    edge_weight_type: str, round_euclidean: bool, time_budget_sec: float,
    seed: int = 0, logger: Optional[logging.Logger] = None, verbose: bool = True,
) -> Tuple[Dict[str, Any], Any, float]:
    rng = random.Random(seed)
    feats, D = extract_features(coords, demand, Q, routes_initial, edge_weight_type, round_euclidean)

    # Small-instance adjustment: use a tighter LS slice for n<=70 to keep iterations quick
    if feats.n <= 70:
        ls_slice = min(1.0, max(0.4, 0.01 * feats.n))
    elif feats.n <= 200:
        ls_slice = min(2.5, max(1.0, 0.015 * feats.n))
    elif feats.n <= 1000:
        ls_slice = min(2.0, max(0.8, 0.01 * feats.n))
    else:
        ls_slice = 1.0

    max_passes = 3 if feats.tightness > 0.9 else 2
    k_nearest = feats.nn_k_default

    scale = sample_insertion_scale(D, routes_initial, rng=rng)
    T_init = max(1e-3, 0.08 * scale)
    T_cooling = 0.985 if feats.n <= 300 else (0.99 if feats.n <= 1000 else 0.993)

    cfg = ILSConfig(
        time_limit_sec=float(time_budget_sec),
        ls_time_slice_sec=float(ls_slice),
        max_passes_without_gain=int(max_passes),
        k_nearest=int(k_nearest),
        T_init_factor=float(T_init),
        T_cooling=float(T_cooling),
        seed=int(seed),
        verbose=bool(verbose),
    )

    # Attach catcher to ILS logger (run_ils uses its own logging if passed)
    lgr = logging.getLogger("ILS")
    lgr.setLevel(logging.INFO)
    catcher, start = attach_best_catcher(lgr)  # logger-based catcher (kept)

    stdout_catcher = RealtimeStdoutBestCatcher(start, fallback_stream=sys.stdout)

    start_call = time.perf_counter()
    with contextlib.redirect_stdout(stdout_catcher):
        out = run_ils(
            coords=coords, demand=demand, Q=Q,
            routes_initial=[r[:] for r in routes_initial],
            edge_weight_type=edge_weight_type, round_euclidean=round_euclidean,
            cfg=cfg,
            # if your run_ils accepts logger=, pass logger=lgr here as well
            # logger=lgr,
        )
    total_elapsed = time.perf_counter() - start_call
    time_to_best  = (catcher.best_time
                    if catcher.best_time is not None
                    else (stdout_catcher.best_time
                        if stdout_catcher.best_time is not None
                        else total_elapsed))


    if logger and verbose:
        logger.info(f"[ILS] routes={len(out.routes)} cost={out.cost:.2f} time={total_elapsed:.2f}s best_at={time_to_best:.2f}s")

    return cfg.__dict__, out, time_to_best


def tune_and_run_tabu(
    coords: np.ndarray, demand: np.ndarray, Q: int, routes_init: List[List[int]],
    edge_weight_type: str, round_euclidean: bool, time_budget_sec: float,
    seed: int = 42, logger: Optional[logging.Logger] = None, verbose: bool = True,
) -> Tuple[Dict[str, Any], Any, float]:
    rng = random.Random(seed)
    feats, D = extract_features(coords, demand, Q, routes_init, edge_weight_type, round_euclidean)

    base_tenure = int(max(7, round(0.6 * math.sqrt(max(1, feats.n)) + 0.8 * math.sqrt(max(1, feats.med_route_len)))))
    if feats.tightness > 0.95 or feats.density > (0.02 * feats.n):
        base_tenure = int(base_tenure * 1.2)
    tenure_range = (max(5, int(0.8 * base_tenure)), int(1.2 * base_tenure))

    intra_cap = int(np.clip(1.5 * feats.med_route_len, 32, 128))
    inter_cap = int(np.clip(2.5 * feats.med_route_len, 64, 256))
    # For small instances we don't need extremely long no-improve windows.
    if feats.n <= 70:
        max_no_improve = 200
    else:
        max_no_improve = 800 if feats.n >= 800 else (600 if feats.n >= 300 else 400)
    k_nearest = feats.nn_k_default

    lgr = logging.getLogger("TabuSearch")
    lgr.setLevel(logging.INFO)
    catcher, start = attach_best_catcher(lgr)

    start_call = time.perf_counter()
    out = tabu_search(
        coords=coords, demand=demand, Q=Q,
        routes_init=[r[:] for r in routes_init],
        edge_weight_type=edge_weight_type, round_euclidean=round_euclidean,
        tabu_tenure_range=tenure_range, tabu_tenure=tenure_range[0],
        aspiration=True,
        time_limit_sec=float(time_budget_sec),
        max_iters=200000,
        max_no_improve=int(max_no_improve),
        improvement_eps=1e-6,
        k_nearest=int(k_nearest),
        max_intra_candidates_per_route=int(intra_cap),
        max_inter_candidates_per_route=int(inter_cap),
        verbose=bool(verbose),
        seed=int(seed),
        logger=lgr,
    )
    total_elapsed = time.perf_counter() - start_call
    time_to_best = catcher.best_time if catcher.best_time is not None else total_elapsed

    if logger and verbose:
        logger.info(f"[TABU] routes={len(out.routes)} cost={out.cost:.2f} time={total_elapsed:.2f}s best_at={time_to_best:.2f}s")

    cfg = dict(
        time_limit_sec=float(time_budget_sec),
        tabu_tenure_range=tenure_range,
        aspiration=True,
        k_nearest=int(k_nearest),
        max_intra_candidates_per_route=int(intra_cap),
        max_inter_candidates_per_route=int(inter_cap),
        max_no_improve=int(max_no_improve),
        seed=int(seed),
    )
    return cfg, out, time_to_best


def _choose_remove_fraction(n: int, density: float, tightness: float) -> Tuple[float, float]:
    # Slightly smaller ruin sizes for very small instances to avoid over-destroying
    # (less disruptive moves when few customers exist).
    if n <= 70:
        a, b = 0.06, 0.14
    elif n <= 200:
        a, b = 0.08, 0.18
    elif n <= 700:
        a, b = 0.10, 0.22
    else:
        a, b = 0.12, 0.26
    if density > 0.02 * n:
        a, b = a + 0.02, b + 0.04
    if tightness > 0.95:
        a, b = a + 0.02, b + 0.03
    return (min(0.25, a), min(0.4, b))


def tune_and_run_alns(
    coords: np.ndarray, demand: np.ndarray, Q: int, routes_init: List[List[int]],
    edge_weight_type: str, round_euclidean: bool, time_budget_sec: float,
    seed: int = 44, logger: Optional[logging.Logger] = None, verbose: bool = True,
) -> Tuple[Dict[str, Any], Any, float]:
    rng = random.Random(seed)
    feats, D = extract_features(coords, demand, Q, routes_init, edge_weight_type, round_euclidean)
    scale = sample_insertion_scale(D, routes_init, rng=rng)
    T0 = max(1e-3, 0.6 * scale if feats.n <= 300 else 0.8 * scale)
    Tend = max(1.0, 0.02 * T0)
    cool = 0.996 if feats.n <= 700 else 0.997
    remove_fraction = _choose_remove_fraction(feats.n, feats.density, feats.tightness)
    use_destroy = ("route", "clustered_routes", "shaw", "worst") if feats.n >= 300 else ("shaw", "worst", "random")
    use_repair = ("regret3", "regret2", "greedy") if feats.n >= 200 else ("regret2", "greedy")
    ls_slice = 0.8 if feats.n >= 300 else 0.5
    if time_budget_sec <= 40: ls_slice = max(0.3, ls_slice * 0.75)
    reaction = 0.35 if feats.n >= 300 else 0.3
    segment = 30 if feats.n >= 300 else 40

    lgr = logging.getLogger("ALNS")
    lgr.setLevel(logging.INFO)
    catcher, start = attach_best_catcher(lgr)

    start_call = time.perf_counter()
    out = alns_run(
        coords=coords, demand=demand, Q=Q,
        routes_init=[r[:] for r in routes_init],
        edge_weight_type=edge_weight_type, round_euclidean=round_euclidean,
        start_temperature=float(T0), end_temperature=float(Tend), cooling_rate=float(cool),
        remove_fraction=tuple(remove_fraction),
        sigma1=6.0, sigma2=3.0, sigma3=1.0,
        reaction=float(reaction), segment_length=int(segment),
        time_limit_sec=float(time_budget_sec), max_iters=300000,
        seed=int(seed), verbose=bool(verbose), log_every=10,
        use_destroy=use_destroy, use_repair=use_repair,
        use_local_search=True, ls_time_limit_sec=float(ls_slice), ls_max_passes_without_gain=1,
        logger=lgr,
    )
    total_elapsed = time.perf_counter() - start_call
    time_to_best = catcher.best_time if catcher.best_time is not None else total_elapsed

    if logger and verbose:
        logger.info(f"[ALNS] routes={len(out.routes)} cost={out.cost:.2f} time={total_elapsed:.2f}s best_at={time_to_best:.2f}s")

    cfg = dict(
        start_temperature=float(T0), end_temperature=float(Tend), cooling_rate=float(cool),
        remove_fraction=tuple(remove_fraction),
        use_destroy=use_destroy, use_repair=use_repair,
        reaction=float(reaction), segment_length=int(segment),
        ls_time_limit_sec=float(ls_slice),
        time_limit_sec=float(time_budget_sec), seed=int(seed),
    )
    return cfg, out, time_to_best


def tune_and_run_lns(
    coords: np.ndarray, demand: np.ndarray, Q: int, routes_init: List[List[int]],
    edge_weight_type: str, round_euclidean: bool, time_budget_sec: float,
    seed: int = 7, logger: Optional[logging.Logger] = None, verbose: bool = True,
) -> Tuple[Dict[str, Any], Any, float]:
    rng = random.Random(seed)
    feats, D = extract_features(coords, demand, Q, routes_init, edge_weight_type, round_euclidean)
    scale = sample_insertion_scale(D, routes_init, rng=rng)
    T0 = max(1e-3, 0.7 * scale if feats.n > 300 else 0.5 * scale)
    Tend = max(1.0, 0.02 * T0)
    cool = 0.996 if feats.n <= 700 else 0.997
    remove_fraction = _choose_remove_fraction(feats.n, feats.density, feats.tightness)
    use_ruin = ("route", "clustered_routes", "shaw", "worst", "random") if feats.n >= 300 else ("shaw", "worst", "random")
    use_repair = ("regret3", "regret2", "greedy") if feats.n >= 200 else ("regret2", "greedy")
    ls_slice = 0.8 if feats.n >= 300 else 0.5
    if time_budget_sec <= 40: ls_slice = max(0.3, ls_slice * 0.75)

    lgr = logging.getLogger("LNS")
    lgr.setLevel(logging.INFO)
    catcher, start = attach_best_catcher(lgr)

    start_call = time.perf_counter()
    out = lns_run(
        coords=coords, demand=demand, Q=Q,
        routes_init=[r[:] for r in routes_init],
        edge_weight_type=edge_weight_type, round_euclidean=round_euclidean,
        remove_fraction=tuple(remove_fraction),
        use_sa=True, start_temperature=float(T0), end_temperature=float(Tend), cooling_rate=float(cool),
        p_accept_worse=0.0,
        use_ruin=use_ruin, use_repair=use_repair,
        use_local_search=True, ls_time_limit_sec=float(ls_slice), ls_max_passes_without_gain=1,
        time_limit_sec=float(time_budget_sec), max_iters=200000,
        seed=int(seed), verbose=bool(verbose), log_every=10, log_detail=True,
        logger=lgr,
    )
    total_elapsed = time.perf_counter() - start_call
    time_to_best = catcher.best_time if catcher.best_time is not None else total_elapsed


    if logger and verbose:
        logger.info(f"[LNS] routes={len(out.routes)} cost={out.cost:.2f} time={total_elapsed:.2f}s best_at={time_to_best:.2f}s")

    cfg = dict(
        start_temperature=float(T0), end_temperature=float(Tend), cooling_rate=float(cool),
        remove_fraction=tuple(remove_fraction),
        use_ruin=use_ruin, use_repair=use_repair,
        ls_time_limit_sec=float(ls_slice),
        time_limit_sec=float(time_budget_sec), seed=int(seed),
    )
    return cfg, out, time_to_best


# -------------------------------------------------------------------
# Orchestrator: runs all 6 algos, saves all 7 outputs, prints summary
# -------------------------------------------------------------------

def autotune_and_run_all(
    inst,
    routes_seed: Optional[List[List[int]]] = None,
    time_budgets_sec: Dict[str, float] = None,
    seeds: Dict[str, int] = None,
    logger: Optional[logging.Logger] = None,
    verbose: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """
    Runs: CW, LS, ILS, TABU, ALNS, LNS (6 algos), saves each solution & plot as:
      solutions/<inst_name>/<inst_name>_<algo>.sol|.png
    Also writes the best-of-6 as:
      solutions/<inst_name>/<inst_name>_best.sol|.png

    Returns a dict with per-algo config, cost, routes, and time_to_best.
    """
    time_budgets_sec = time_budgets_sec or {'ls': 12.0, 'ils': 70.0, 'tabu': 70.0, 'alns': 120.0, 'lns': 90.0}
    seeds = seeds or {'ls': 0, 'ils': 0, 'tabu': 42, 'alns': 44, 'lns': 7}

    inst_name = getattr(inst, "name", "instance")
    if routes_seed is None:
        # default seed: CW
        cw_cfg, cw_sol, _ = tune_and_run_cw(inst, logger=logger, verbose=verbose)
        routes_seed = [r[:] for r in cw_sol.routes]
    else:
        # still compute CW later for baseline
        pass

    report: Dict[str, Dict[str, Any]] = {}

    # 0) CW baseline
    cw_cfg, cw_sol, cw_time_best = tune_and_run_cw(inst, logger=logger, verbose=verbose)
    report['cw'] = {'cfg': cw_cfg, 'cost': float(cw_sol.cost), 'routes': [r[:] for r in cw_sol.routes], 'time_to_best': float(cw_time_best)}
    save_named_solution_and_plot(inst, cw_sol.routes, cw_sol.cost, "cw", inst_name, runtime_sec=cw_time_best)

    # 1) LS (plain local search on CW)
    ls_cfg, ls_sol, ls_time_best = tune_and_run_local_search(inst, routes_seed, time_budgets_sec['ls'], logger=logger, verbose=verbose, seed=seeds['ls'])
    report['ls'] = {'cfg': ls_cfg, 'cost': float(ls_sol.cost), 'routes': [r[:] for r in ls_sol.routes], 'time_to_best': float(ls_time_best)}
    save_named_solution_and_plot(inst, ls_sol.routes, ls_sol.cost, "ls", inst_name, runtime_sec=ls_time_best)

    # 2) ILS
    ils_cfg, ils_out, ils_time_best = tune_and_run_ils(
        inst.coords, inst.demand, inst.capacity, routes_seed,
        inst.edge_weight_type, False, time_budgets_sec['ils'],
        seed=seeds['ils'], logger=logger, verbose=verbose
    )
    report['ils'] = {'cfg': ils_cfg, 'cost': float(ils_out.cost), 'routes': [r[:] for r in ils_out.routes], 'time_to_best': float(ils_time_best)}
    save_named_solution_and_plot(inst, ils_out.routes, ils_out.cost, "ils", inst_name, runtime_sec=ils_time_best)

    # 3) TABU
    ts_cfg, ts_out, ts_time_best = tune_and_run_tabu(
        inst.coords, inst.demand, inst.capacity, routes_seed,
        inst.edge_weight_type, False, time_budgets_sec['tabu'],
        seed=seeds['tabu'], logger=logger, verbose=verbose
    )
    report['tabu'] = {'cfg': ts_cfg, 'cost': float(ts_out.cost), 'routes': [r[:] for r in ts_out.routes], 'time_to_best': float(ts_time_best)}
    save_named_solution_and_plot(inst, ts_out.routes, ts_out.cost, "tabu", inst_name, runtime_sec=ts_time_best)

    # 4) ALNS
    alns_cfg, alns_out, alns_time_best = tune_and_run_alns(
        inst.coords, inst.demand, inst.capacity, routes_seed,
        inst.edge_weight_type, False, time_budgets_sec['alns'],
        seed=seeds['alns'], logger=logger, verbose=verbose
    )
    report['alns'] = {'cfg': alns_cfg, 'cost': float(alns_out.cost), 'routes': [r[:] for r in alns_out.routes], 'time_to_best': float(alns_time_best)}
    save_named_solution_and_plot(inst, alns_out.routes, alns_out.cost, "alns", inst_name, runtime_sec=alns_time_best)

    # 5) LNS
    lns_cfg, lns_out, lns_time_best = tune_and_run_lns(
        inst.coords, inst.demand, inst.capacity, routes_seed,
        inst.edge_weight_type, False, time_budgets_sec['lns'],
        seed=seeds['lns'], logger=logger, verbose=verbose
    )
    report['lns'] = {'cfg': lns_cfg, 'cost': float(lns_out.cost), 'routes': [r[:] for r in lns_out.routes], 'time_to_best': float(lns_time_best)}
    save_named_solution_and_plot(inst, lns_out.routes, lns_out.cost, "lns", inst_name, runtime_sec=lns_time_best)

    # ---- 7th: Best overall
    best_name = None
    best_cost = float('inf')
    best_routes: List[List[int]] = []
    for name, res in report.items():
        if res['cost'] < best_cost:
            best_cost = res['cost']
            best_routes = [r[:] for r in res['routes']]
            best_name = name
            best_time_to_best = res['time_to_best']
    save_named_solution_and_plot(inst, best_routes, best_cost, "best", inst_name, runtime_sec=best_time_to_best)

    # ---- Summary log
    if logger and verbose:
        logger.info(f"[SUMMARY] Instance={inst_name}")
        for name in ["cw", "ls", "ils", "tabu", "alns", "lns"]:
            res = report[name]
            logger.info(f"  - {name.upper():5s} | cost={res['cost']:.2f} | routes={len(res['routes'])} | best@{res['time_to_best']:.2f}s")
        logger.info(f"  => BEST: {best_name.upper()} | cost={best_cost:.2f}")

    # Also save a CSV summary into the instance solutions folder.
    try:
        sol_dir = os.path.join("solutions", inst_name)
        _ensure_dir(sol_dir)
        csv_path = os.path.join(sol_dir, f"{inst_name}_summary.csv")
        with open(csv_path, 'w', newline='', encoding='utf-8') as fh:
            writer = csv.writer(fh)
            writer.writerow(["algo", "cost", "route_count", "time_to_best", "cfg", "is_best"])
            for name in ["cw", "ls", "ils", "tabu", "alns", "lns"]:
                res = report[name]
                cfg = res.get('cfg', {})
                try:
                    cfg_s = json.dumps(cfg, ensure_ascii=False)
                except Exception:
                    cfg_s = str(cfg)
                writer.writerow([
                    name,
                    float(res.get('cost', float('nan'))),
                    len(res.get('routes', [])),
                    float(res.get('time_to_best', float('nan'))),
                    cfg_s,
                    (name == best_name),
                ])
    except Exception:
        # Do not raise on CSV write errors; summary logging already done.
        pass

    return report
