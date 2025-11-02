from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Iterable
import re
import numpy as np


@dataclass
class Instance:
    name: str
    coords: np.ndarray       # shape: (n+1, 2). Index 0 is depot.
    demand: np.ndarray       # shape: (n+1,). demand[0] == 0
    capacity: int
    edge_weight_type: str    # e.g., "EUC_2D" (CVRPLIB), "CEIL_2D"
    comment: Optional[str] = None

    @property
    def n_customers(self) -> int:
        return self.coords.shape[0] - 1

    def dist(self, i: int, j: int) -> float:
        """Distance consistent with CVRPLIB's EUC_2D / CEIL_2D conventions."""
        dx, dy = self.coords[i] - self.coords[j]
        d = float(np.hypot(dx, dy))
        if self.edge_weight_type.upper() == "CEIL_2D":
            # Traditional CVRPLIB CEIL_2D uses ceil of Euclidean distance.
            return float(np.ceil(d))
        # EUC_2D usually implies rounding to nearest int in some CVRPLIB sets,
        # but Uchoa's X-instances generally use straight Euclidean length as float.
        # Keep float here; you can round when building a full distance matrix if desired.
        return d


class CVRPLIBParseError(ValueError):
    pass


_SECTION_NONE = 0
_SECTION_COORDS = 1
_SECTION_DEMANDS = 2
_SECTION_DEPOT = 3

_HEADER_KV = {
    "NAME",
    "TYPE",
    "COMMENT",
    "DIMENSION",
    "EDGE_WEIGHT_TYPE",
    "CAPACITY",
}


def _strip_inline_comment(line: str) -> str:
    # CVRPLIB rarely uses inline comments, but be defensive: remove trailing comments after a '#'.
    # Keep quoted strings intact.
    if "#" in line and '"' not in line:
        return line.split("#", 1)[0]
    return line


def _parse_header_value(raw: str) -> str:
    # Lines may be like: "NAME : X-n101-k25" or "COMMENT : "some text""
    # Normalize whitespace and strip surrounding quotes.
    val = raw.strip()
    if val.startswith(":"):
        val = val[1:].strip()
    # Remove surrounding quotes if present
    if (len(val) >= 2) and ((val[0] == '"' and val[-1] == '"') or (val[0] == "'" and val[-1] == "'")):
        val = val[1:-1]
    return val.strip()


def parse_cvrplib_uchoa_2014(text: str) -> Instance:
    """
    Parse a CVRPLIB (Uchoa et al.) CVRP instance from a string and return a normalized Instance:
      - depot at index 0
      - customers 1..n
    """
    # Normalize newlines and strip BOM if any
    text = text.replace("\r\n", "\n").replace("\r", "\n").lstrip("\ufeff")
    lines = [ln.rstrip() for ln in text.split("\n")]

    # Header accumulators
    header: Dict[str, str] = {}
    section = _SECTION_NONE

    # Temporary storage keyed by original 1-based IDs
    coord_by_id: Dict[int, Tuple[float, float]] = {}
    demand_by_id: Dict[int, int] = {}
    depot_ids: List[int] = []

    for raw_line in lines:
        line = _strip_inline_comment(raw_line).strip()
        if not line:
            continue

        # End of file marker
        if line.upper() == "EOF":
            break

        # Section switches
        if line.upper().startswith("NODE_COORD_SECTION"):
            section = _SECTION_COORDS
            continue
        if line.upper().startswith("DEMAND_SECTION"):
            section = _SECTION_DEMANDS
            continue
        if line.upper().startswith("DEPOT_SECTION"):
            section = _SECTION_DEPOT
            continue

        # Header key: value lines
        if section == _SECTION_NONE:
            # Sometimes headers have tabs; split on first ':' if present, else on whitespace
            if any(line.upper().startswith(k + " ") or line.upper().startswith(k + " :") for k in _HEADER_KV) \
               or any(line.upper().startswith(k + "\t") for k in _HEADER_KV) \
               or any(line.upper().startswith(k + ":") for k in _HEADER_KV):
                # Try to split at the first ':'
                if ":" in line:
                    k, v = line.split(":", 1)
                else:
                    # Fallback "KEY  value"
                    parts = line.split(None, 1)
                    if len(parts) == 1:
                        k, v = parts[0], ""
                    else:
                        k, v = parts
                key = k.strip().upper()
                if key in _HEADER_KV:
                    header[key] = _parse_header_value(v)
                continue
            else:
                # Not in a declared section; ignore empty/unknown lines
                continue

        # Inside sections
        if section == _SECTION_COORDS:
            # Format: id x y (numbers; can be separated by spaces or tabs)
            parts = line.split()
            if len(parts) < 3:
                raise CVRPLIBParseError(f"Malformed NODE_COORD_SECTION line: {line}")
            try:
                nid = int(parts[0])
                x = float(parts[1]); y = float(parts[2])
            except Exception as e:
                raise CVRPLIBParseError(f"Failed to parse coord line '{line}': {e}")
            coord_by_id[nid] = (x, y)
            continue

        if section == _SECTION_DEMANDS:
            # Format: id demand
            parts = line.split()
            if len(parts) < 2:
                raise CVRPLIBParseError(f"Malformed DEMAND_SECTION line: {line}")
            try:
                nid = int(parts[0]); dem = int(float(parts[1]))
            except Exception as e:
                raise CVRPLIBParseError(f"Failed to parse demand line '{line}': {e}")
            demand_by_id[nid] = dem
            continue

        if section == _SECTION_DEPOT:
            # One or more depot IDs, terminated by -1
            try:
                val = int(line.split()[0])
            except Exception as e:
                raise CVRPLIBParseError(f"Failed to parse DEPOT_SECTION line '{line}': {e}")
            if val == -1:
                # End of depot list
                section = _SECTION_NONE
            else:
                depot_ids.append(val)
            continue

    # --- Validation & normalization ---
    # Required header fields
    required = ["NAME", "TYPE", "DIMENSION", "EDGE_WEIGHT_TYPE", "CAPACITY"]
    missing = [k for k in required if k not in header]
    if missing:
        raise CVRPLIBParseError(f"Missing header keys: {missing}")

    if header["TYPE"].upper() not in {"CVRP", "VRP"}:
        raise CVRPLIBParseError(f"Unsupported TYPE: {header['TYPE']}")

    try:
        dim = int(header["DIMENSION"])
    except Exception:
        raise CVRPLIBParseError(f"Invalid DIMENSION: {header['DIMENSION']}")

    if not depot_ids:
        raise CVRPLIBParseError("No depot specified in DEPOT_SECTION.")
    depot_id = depot_ids[0]  # CVRPLIB may list multiple; take the first.

    # Sanity checks on counts
    if len(coord_by_id) != dim:
        raise CVRPLIBParseError(f"Coordinate count ({len(coord_by_id)}) != DIMENSION ({dim}).")
    if len(demand_by_id) != dim:
        raise CVRPLIBParseError(f"Demand count ({len(demand_by_id)}) != DIMENSION ({dim}).")
    if depot_id not in coord_by_id:
        raise CVRPLIBParseError(f"DEPOT id {depot_id} not found in NODE_COORD_SECTION.")
    if depot_id not in demand_by_id:
        raise CVRPLIBParseError(f"DEPOT id {depot_id} not found in DEMAND_SECTION.")

    # Map original 1..dim IDs -> new indices with depot first (0), then remaining customers in ascending id
    customers = [i for i in sorted(coord_by_id.keys()) if i != depot_id]
    id_to_idx = {depot_id: 0}
    for k, nid in enumerate(customers, start=1):
        id_to_idx[nid] = k

    # Build arrays
    coords = np.zeros((dim, 2), dtype=float)  # But we want (n+1,2) => here dim already includes depot
    demand = np.zeros((dim,), dtype=int)

    for nid, (x, y) in coord_by_id.items():
        idx = id_to_idx[nid]
        coords[idx, 0] = x
        coords[idx, 1] = y
    for nid, dem in demand_by_id.items():
        idx = id_to_idx[nid]
        demand[idx] = int(dem)

    # Force depot demand = 0 (some files include 0 already; we standardize)
    demand[0] = 0

    try:
        capacity = int(header["CAPACITY"])
    except Exception:
        raise CVRPLIBParseError(f"Invalid CAPACITY: {header['CAPACITY']}")

    edge_type = header["EDGE_WEIGHT_TYPE"].upper()
    if edge_type not in {"EUC_2D", "CEIL_2D"}:
        # You can extend this if you need other types later.
        raise CVRPLIBParseError(f"Unsupported EDGE_WEIGHT_TYPE: {edge_type}")

    return Instance(
        name=header["NAME"],
        coords=coords,
        demand=demand,
        capacity=capacity,
        edge_weight_type=edge_type,
        comment=header.get("COMMENT"),
    )


def parse_cvrplib_file(path: str, encoding: str = "utf-8") -> Instance:
    with open(path, "r", encoding=encoding) as f:
        text = f.read()
    return parse_cvrplib_uchoa_2014(text)
