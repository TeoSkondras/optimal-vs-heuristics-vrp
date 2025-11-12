from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional
import re
import numpy as np


@dataclass
class VRPTWInstance:
    """
    Homberger & Gehring (1999) / Solomon-style VRPTW instance.

    Conventions:
      - Node 0 is the depot.
      - Nodes 1..n are customers.
      - All arrays are length n+1.
      - Coordinates in 'coords' are Euclidean; dist() uses straight Euclidean.
    """
    name: str
    coords: np.ndarray        # shape: (n+1, 2), index 0 is depot
    demand: np.ndarray        # shape: (n+1,), demand[0] usually 0
    ready_time: np.ndarray    # shape: (n+1,)
    due_time: np.ndarray      # shape: (n+1,)
    service_time: np.ndarray  # shape: (n+1,)
    capacity: int
    n_vehicles: int
    edge_weight_type: str = "EUC_2D"
    comment: Optional[str] = None

    @property
    def n_customers(self) -> int:
        return self.coords.shape[0] - 1

    def dist(self, i: int, j: int) -> float:
        dx, dy = self.coords[i] - self.coords[j]
        return float(np.hypot(dx, dy))


class VRPTWParseError(ValueError):
    pass


def _normalize_text(text: str) -> str:
    # Normalize newlines and strip BOM if any
    return text.replace("\r\n", "\n").replace("\r", "\n").lstrip("\ufeff")


def _strip_comment_and_trim(line: str) -> str:
    # Homberger files usually don't have '#', but be defensive.
    if "#" in line and '"' not in line:
        line = line.split("#", 1)[0]
    return line.strip()


def parse_homberger_1999_vrptw(text: str) -> VRPTWInstance:
    """
    Parse a Homberger & Gehring (1999) VRPTW instance from a string.

    Expected structure (canonical):

        C1_2_1

        VEHICLE
        NUMBER     CAPACITY
         50          200

        CUSTOMER
        CUST NO.  XCOORD.  YCOORD.  DEMAND  READY TIME  DUE DATE  SERVICE TIME

            0      70       70        0        0        1351      0
            1      33       78       20      750         809      90
            ...
    """
    text = _normalize_text(text)
    raw_lines = text.split("\n")
    lines = [_strip_comment_and_trim(ln) for ln in raw_lines]

    # --- Find NAME (first non-empty line) ---
    idx = 0
    n = len(lines)
    while idx < n and not lines[idx]:
        idx += 1
    if idx >= n:
        raise VRPTWParseError("Empty file or missing instance name.")
    name = lines[idx].strip()
    if not name:
        raise VRPTWParseError("Missing instance name.")
    idx += 1

    # --- Find VEHICLE section ---
    # Look for a line equal to "VEHICLE" (case-insensitive)
    while idx < n and lines[idx].upper() != "VEHICLE":
        idx += 1
    if idx >= n:
        raise VRPTWParseError("Missing 'VEHICLE' section header.")
    idx += 1  # move past "VEHICLE"

    # Skip blank lines
    while idx < n and not lines[idx]:
        idx += 1
    if idx >= n:
        raise VRPTWParseError("Unexpected end of file after 'VEHICLE' header.")

    # Optional header row: "NUMBER     CAPACITY"
    if "NUMBER" in lines[idx].upper() and "CAPACITY" in lines[idx].upper():
        idx += 1
        # Skip blank lines after header
        while idx < n and not lines[idx]:
            idx += 1

    if idx >= n:
        raise VRPTWParseError("Missing vehicle data line with 'NUMBER CAPACITY'.")

    # Parse vehicle line: "<n_vehicles> <capacity>"
    parts = re.split(r"\s+", lines[idx])
    if len(parts) < 2:
        raise VRPTWParseError(f"Malformed vehicle line: {lines[idx]!r}")
    try:
        n_vehicles = int(parts[0])
        capacity = int(parts[1])
    except Exception as e:
        raise VRPTWParseError(f"Failed to parse vehicle line '{lines[idx]}': {e}")
    idx += 1

    # --- Find CUSTOMER section ---
    while idx < n and lines[idx].upper() != "CUSTOMER":
        idx += 1
    if idx >= n:
        raise VRPTWParseError("Missing 'CUSTOMER' section header.")
    idx += 1  # move past "CUSTOMER"

    # Skip blank lines
    while idx < n and not lines[idx]:
        idx += 1
    if idx >= n:
        raise VRPTWParseError("Unexpected end of file after 'CUSTOMER' header.")

    # Optional customer header:
    # "CUST NO.  XCOORD.  YCOORD.  DEMAND  READY TIME  DUE DATE  SERVICE TIME"
    header_line = lines[idx].upper()
    if "CUST" in header_line and "XCOORD" in header_line and "YCOORD" in header_line:
        idx += 1
        while idx < n and not lines[idx]:
            idx += 1

    # --- Parse customer/depot rows ---
    # Each non-empty data line: id, x, y, demand, ready, due, service
    node_data: Dict[int, tuple] = {}
    started = False

    for j in range(idx, n):
        line = lines[j]
        if not line:
            continue

        parts = re.split(r"\s+", line)
        # Minimal sanity: expect at least 7 tokens for a data row
        if len(parts) < 7:
            # If we've already started collecting nodes and now see something shorter,
            # assume we're past the data section.
            if started:
                break
            else:
                continue

        try:
            nid = int(parts[0])
            x = float(parts[1])
            y = float(parts[2])
            dem = int(float(parts[3]))
            ready = int(float(parts[4]))
            due = int(float(parts[5]))
            service = int(float(parts[6]))
        except ValueError:
            # Same logic: if already started, treat as end-of-section
            if started:
                break
            else:
                continue

        started = True
        if nid in node_data:
            raise VRPTWParseError(f"Duplicate node id {nid} in customer section.")
        node_data[nid] = (x, y, dem, ready, due, service)

    if not node_data:
        raise VRPTWParseError("No customer/depot records found in CUSTOMER section.")

    # --- Validation & normalization ---
    if 0 not in node_data:
        raise VRPTWParseError("Depot (id 0) not found in CUSTOMER section.")

    max_id = max(node_data.keys())
    expected_ids = set(range(0, max_id + 1))
    missing = expected_ids.difference(node_data.keys())
    if missing:
        raise VRPTWParseError(f"Missing node ids in CUSTOMER section: {sorted(missing)}")

    n_nodes = max_id + 1  # includes depot at 0

    coords = np.zeros((n_nodes, 2), dtype=float)
    demand = np.zeros((n_nodes,), dtype=int)
    ready_time = np.zeros((n_nodes,), dtype=int)
    due_time = np.zeros((n_nodes,), dtype=int)
    service_time = np.zeros((n_nodes,), dtype=int)

    for nid, (x, y, dem, rt, dd, st) in node_data.items():
        if nid < 0 or nid >= n_nodes:
            raise VRPTWParseError(f"Node id {nid} out of expected range 0..{n_nodes-1}.")
        coords[nid, 0] = x
        coords[nid, 1] = y
        demand[nid] = dem
        ready_time[nid] = rt
        due_time[nid] = dd
        service_time[nid] = st

    # Standardize depot demand = 0 (just in case)
    demand[0] = 0

    return VRPTWInstance(
        name=name,
        coords=coords,
        demand=demand,
        ready_time=ready_time,
        due_time=due_time,
        service_time=service_time,
        capacity=capacity,
        n_vehicles=n_vehicles,
        edge_weight_type="EUC_2D",
        comment=None,
    )


def parse_homberger_1999_vrptw_file(path: str, encoding: str = "utf-8") -> VRPTWInstance:
    with open(path, "r", encoding=encoding) as f:
        text = f.read()
    return parse_homberger_1999_vrptw(text)
