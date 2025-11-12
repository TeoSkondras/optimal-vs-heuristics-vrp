# plot_utils.py
from __future__ import annotations

import os
import re
from typing import List, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt

from vrptw_parser import VRPTWInstance
from utils import (
    normalize_routes,
    solution_cost,
)


# ---------------- IO Helpers ----------------


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _default_solution_paths(inst: VRPTWInstance, base_dir: str) -> Tuple[str, str]:
    """
    Returns (solution_path, png_path) for a given instance and base dir.
    """
    inst_dir = os.path.join(base_dir, inst.name)
    _ensure_dir(inst_dir)
    sol_path = os.path.join(inst_dir, f"{inst.name}.sol")
    png_path = os.path.join(inst_dir, f"{inst.name}.png")
    return sol_path, png_path


# ---------------- Save Solution (.sol) ----------------


def save_solution(
    inst: VRPTWInstance,
    routes: Sequence[Sequence[int]],
    D: np.ndarray,
    base_dir: str = "solutions",
) -> str:
    """
    Save solution in CVRPLIB-style route format:

        Route #1: 20 41 85 80 31 ...
        Route #2: 21 23 ...
        ...
        Cost 2698.6

    Conventions:
      - `routes` may or may not include depot; we normalize.
      - Only customer IDs (non-zero) are printed on route lines.
      - File path: solutions/{inst.name}/{inst.name}.sol

    Returns:
      Path to the written .sol file.
    """
    sol_path, _ = _default_solution_paths(inst, base_dir)

    norm_routes = normalize_routes(routes, depot=0)

    # Filter out empty routes (after normalization)
    norm_routes = [r for r in norm_routes if len(r) > 2]

    total_cost = solution_cost(norm_routes, D)

    lines: List[str] = []
    route_idx = 1
    for r in norm_routes:
        customers = [str(v) for v in r if v != 0]
        if not customers:
            continue
        line = f"Route #{route_idx}: " + " ".join(customers)
        lines.append(line)
        route_idx += 1

    lines.append(f"Cost {total_cost:.4f}")

    with open(sol_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    return sol_path


# ---------------- Load Solution (.sol) ----------------


def load_solution_from_file(
    file_path: str,
    depot: int = 0,
) -> List[List[int]]:
    """
    Parse a .sol file in the format:

        Route #k: i j k ...
        ...
        Cost X

    Returns:
      List of routes as [0, ..., 0] (with depot added at both ends).
    """
    routes: List[List[int]] = []

    with open(file_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.lower().startswith("cost"):
                # Ignore cost line; we recompute if needed.
                continue
            if line.lower().startswith("route"):
                # Extract part after ':'
                # e.g. "Route #1: 20 41 85"
                parts = line.split(":", 1)
                if len(parts) != 2:
                    continue
                rhs = parts[1].strip()
                if not rhs:
                    continue
                # Split into customer ids
                custs = []
                for tok in rhs.split():
                    if tok.isdigit():
                        custs.append(int(tok))
                    else:
                        # Be tolerant of stray chars; ignore non-ints
                        try:
                            custs.append(int(tok))
                        except ValueError:
                            pass
                if custs:
                    route = [depot] + custs + [depot]
                    routes.append(route)

    return routes


# ---------------- Plotting ----------------


def _plot_routes(
    inst: VRPTWInstance,
    routes: Sequence[Sequence[int]],
    out_path: str,
    depot_size: int = 60,
    customer_size: int = 20,
    linewidth: float = 1.0,
) -> None:
    """
    Internal: plot given routes and save to out_path as PNG.

    Each route is drawn as a polyline through node coordinates.
    """
    coords = inst.coords
    depot = 0

    # Basic figure
    plt.figure(figsize=(8, 8))

    # Plot each route as a line
    for r in routes:
        if len(r) < 2:
            continue
        xs = [coords[i, 0] for i in r]
        ys = [coords[i, 1] for i in r]
        plt.plot(xs, ys, marker="o", linewidth=linewidth, markersize=customer_size / 6.0)

    # Highlight depot
    plt.scatter(
        [coords[depot, 0]],
        [coords[depot, 1]],
        s=depot_size,
        marker="s",
        edgecolors="k",
        linewidths=1.5,
        zorder=5,
    )

    plt.title(inst.name)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis("equal")
    plt.tight_layout()

    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_solution_from_file(
    inst: VRPTWInstance,
    base_dir: str = "solutions",
) -> str:
    """
    Read {base_dir}/{inst.name}/{inst.name}.sol
    and create {base_dir}/{inst.name}/{inst.name}.png
    plotting the routes.

    Returns:
      Path to the written .png file.
    """
    sol_path, png_path = _default_solution_paths(inst, base_dir)

    if not os.path.exists(sol_path):
        raise FileNotFoundError(f"Solution file not found: {sol_path}")

    routes = load_solution_from_file(sol_path, depot=0)

    if not routes:
        raise ValueError(f"No routes parsed from solution file: {sol_path}")

    _plot_routes(inst, routes, png_path)
    return png_path
