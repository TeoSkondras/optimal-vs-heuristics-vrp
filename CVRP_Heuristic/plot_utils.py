# plot_utils.py
from __future__ import annotations
import os
from typing import List, Tuple, Optional
import re
import matplotlib.pyplot as plt
import numpy as np


# --------------------------
# Filesystem helpers
# --------------------------

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _default_paths(inst_name: str, base_dir: str = "solutions") -> Tuple[str, str]:
    """
    Returns (dir_path, stem) where:
      dir_path = solutions/<inst_name>
      stem     = solutions/<inst_name>/<inst_name>
    """
    dir_path = os.path.join(base_dir, inst_name)
    stem = os.path.join(dir_path, inst_name)
    return dir_path, stem


# --------------------------
# I/O: Save & Read .sol
# --------------------------

def save_solution(
    inst,
    routes: List[List[int]],
    cost: float,
    base_dir: str = "solutions",
    filename: Optional[str] = None,
) -> str:
    """
    Save solution in the format:
      Route #1: 31 46 35
      Route #2: 15 22 41 20
      ...
      Cost 27591

    Args:
        inst: Instance with .name
        routes: list of routes (each a list of customers, depot omitted)
        cost: total closed-tour cost
        base_dir: root folder (default 'solutions')
        filename: override full path (including .sol). If None -> solutions/<name>/<name>.sol

    Returns:
        The full path to the written .sol file.
    """
    dir_path, stem = _default_paths(inst.name, base_dir)
    _ensure_dir(dir_path)
    sol_path = filename if filename else f"{stem}.sol"

    with open(sol_path, "w", encoding="utf-8") as f:
        for k, r in enumerate(routes, start=1):
            if r:
                f.write(f"Route #{k}: {' '.join(str(v) for v in r)}\n")
            else:
                f.write(f"Route #{k}: \n")
        # cost as integer if it's very close to an int; else keep as float with 6 decimals
        if abs(cost - round(cost)) < 1e-6:
            f.write(f"Cost {int(round(cost))}\n")
        else:
            f.write(f"Cost {cost:.6f}\n")

    return sol_path


def read_solution_file(sol_path: str) -> Tuple[List[List[int]], float]:
    """
    Read a .sol file with lines like:
        Route #1: 31 46 35
        ...
        Cost 27591
    Returns (routes, cost).
    """
    routes: List[List[int]] = []
    cost: Optional[float] = None

    route_re = re.compile(r"^\s*Route\s*#\s*(\d+)\s*:\s*(.*)$", re.IGNORECASE)
    cost_re = re.compile(r"^\s*Cost\s+([+-]?\d+(?:\.\d+)?)\s*$", re.IGNORECASE)

    with open(sol_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            m_route = route_re.match(line)
            if m_route:
                # Grab remainder and split into ints if present
                tail = m_route.group(2).strip()
                if tail:
                    customers = [int(x) for x in tail.split()]
                else:
                    customers = []
                routes.append(customers)
                continue

            m_cost = cost_re.match(line)
            if m_cost:
                cost = float(m_cost.group(1))
                continue

    if cost is None:
        # If absent, set to NaN; we can still plot routes
        cost = float("nan")

    return routes, cost


# --------------------------
# Plotting
# --------------------------

def _compute_bounds(coords: np.ndarray, pad_ratio: float = 0.05) -> Tuple[float, float, float, float]:
    xmin, ymin = coords.min(axis=0)
    xmax, ymax = coords.max(axis=0)
    dx, dy = xmax - xmin, ymax - ymin
    px, py = dx * pad_ratio, dy * pad_ratio
    return xmin - px, xmax + px, ymin - py, ymax + py


def plot_solution(
    inst,
    routes: List[List[int]],
    out_path: str,
    title: Optional[str] = None,
    show_node_ids: bool = False,
    dpi: int = 140,
) -> str:
    """
    Plot routes to a PNG:
      - Depot (index 0) highlighted
      - Each route drawn from depot -> customers -> depot
      - Optional customer labels

    Args:
        inst: Instance with .coords (n+1, 2) and .name
        routes: list of routes (customers only)
        out_path: output PNG path
        title: optional title
        show_node_ids: put node IDs near points (may clutter for large instances)
        dpi: output resolution

    Returns:
        out_path
    """
    coords = np.asarray(inst.coords, dtype=float)
    depot = coords[0]
    customers = coords[1:]

    _ensure_dir(os.path.dirname(out_path) or ".")

    fig = plt.figure(figsize=(8, 8), dpi=dpi)
    ax = fig.add_subplot(111)

    # depot
    ax.scatter([depot[0]], [depot[1]], marker="*", s=180, zorder=5, edgecolors="k", linewidths=1.0, label="Depot")

    # customers
    ax.scatter(customers[:, 0], customers[:, 1], s=18, zorder=3, alpha=0.9, label="Customers")

    # draw each route
    for ridx, route in enumerate(routes, start=1):
        if not route:
            continue
        path = [0] + route + [0]
        xs = coords[[p for p in path], 0]
        ys = coords[[p for p in path], 1]
        # Default line style/colors cycle from matplotlib
        ax.plot(xs, ys, linewidth=1.4, alpha=0.9, label=f"Route {ridx}")

    if show_node_ids:
        # Label customers with their IDs (1..n)
        for vid in range(1, coords.shape[0]):
            x, y = coords[vid]
            ax.text(x, y, str(vid), fontsize=7, ha="left", va="bottom")

        # Label depot as 0
        ax.text(depot[0], depot[1], "0", fontsize=8, fontweight="bold", ha="right", va="top")

    xmin, xmax, ymin, ymax = _compute_bounds(coords)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")

    if title is None:
        title = f"{inst.name} – {len(routes)} route(s)"
    ax.set_title(title)
    ax.grid(True, linewidth=0.4, alpha=0.4)
    # Too many legend entries can clutter; show only for small instances
    if len(routes) <= 12:
        ax.legend(loc="best", fontsize=8, framealpha=0.8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def plot_solution_from_file(
    inst,
    sol_path: str,
    base_dir: str = "solutions",
    out_filename: Optional[str] = None,
    show_node_ids: bool = False,
    dpi: int = 140,
) -> str:
    """
    Read a .sol file and write a PNG under solutions/<inst.name>/<inst.name>.png (by default).

    Args:
        inst: Instance (for coordinates and name)
        sol_path: path to .sol file (must match 'Route #k: ...' format + 'Cost <val>')
        base_dir: root directory for output
        out_filename: override full PNG path; if None => solutions/<name>/<name>.png
        show_node_ids: annotate nodes with their IDs
        dpi: resolution

    Returns:
        The path to the written PNG.
    """
    routes, cost = read_solution_file(sol_path)
    dir_path, stem = _default_paths(inst.name, base_dir)
    _ensure_dir(dir_path)
    out_path = out_filename if out_filename else f"{stem}.png"

    title = f"{inst.name} – {len(routes)} route(s)"
    if cost == cost:  # not NaN
        title += f" – Cost {int(round(cost)) if abs(cost-round(cost))<1e-6 else f'{cost:.2f}'}"

    return plot_solution(inst, routes, out_path, title=title, show_node_ids=show_node_ids, dpi=dpi)
