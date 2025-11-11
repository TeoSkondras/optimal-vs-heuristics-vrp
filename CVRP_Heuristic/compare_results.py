#!/usr/bin/env python3
"""
Compare costs between Uchoa_et_al_2014/*.sol and solutions/<name>/<name>_best.sol.

For each file Uchoa_et_al_2014/<name>.sol that has a corresponding
solutions/<name>/<name>_best.sol, print:
<name> <ratio_of_first_to_second>

Usage (defaults work if you keep the described layout):
    python compare_costs.py
or:
    python compare_costs.py --instances Uchoa_et_al_2014 --solutions solutions
"""

from __future__ import annotations
import argparse
import re
from pathlib import Path
from typing import Optional, Tuple

COST_RE = re.compile(r"^Cost\s+([0-9]+(?:\.[0-9]+)?)\s*$")

def find_cost_in_file(file_path: Path) -> Optional[float]:
    """
    Return the numeric cost from the first line that matches "Cost <number>".
    Returns None if not found or file unreadable.
    """
    try:
        with file_path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                m = COST_RE.match(line.strip())
                if m:
                    try:
                        return float(m.group(1))
                    except ValueError:
                        return None
    except OSError:
        return None
    return None

def gather_instance_costs(instances_dir: Path) -> list[Tuple[str, float, Path]]:
    """
    Collect (name, cost, path) for each *.sol in instances_dir with a valid cost.
    name is the stem (filename without suffix).
    """
    results: list[Tuple[str, float, Path]] = []
    for sol_path in sorted(instances_dir.glob("*.sol")):
        cost = find_cost_in_file(sol_path)
        if cost is not None:
            results.append((sol_path.stem, cost, sol_path))
    return results

def best_solution_path(solutions_dir: Path, name: str) -> Path:
    """
    Expected path for the best solution file for a given instance name.
    """
    return solutions_dir / name / f"{name}_best.sol"

def main(instances_dir: Path, solutions_dir: Path) -> int:
    """
    For each instance with a corresponding best solution, print name and ratio.
    Ratio = (instance_cost / best_cost).
    """
    any_output = False
    for name, instance_cost, _ in gather_instance_costs(instances_dir):
        best_path = best_solution_path(solutions_dir, name)
        if not best_path.is_file():
            # Skip if folder/file doesn't exist (as requested)
            continue
        best_cost = find_cost_in_file(best_path)
        if best_cost is None:
            # Skip if best cost not found or unparsable
            continue
        if best_cost == 0:
            # Avoid division by zero; skip this case silently
            continue
        ratio = instance_cost / best_cost
        # Print name and ratio; keep it simple and stable
        print(f"{name} {ratio:.6f}")
        any_output = True

    # Return 0 even if no output; nothing to print just means nothing matched
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare costs for .sol files.")
    parser.add_argument(
        "--instances",
        type=Path,
        default=Path("Uchoa_et_al_2014"),
        help="Directory containing the original .sol files (default: Uchoa_et_al_2014)",
    )
    parser.add_argument(
        "--solutions",
        type=Path,
        default=Path("solutions"),
        help="Root directory containing per-instance subfolders with *_best.sol (default: solutions)",
    )
    args = parser.parse_args()

    raise SystemExit(main(args.instances, args.solutions))
