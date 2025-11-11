#!/usr/bin/env python3
"""
Extract summary blocks from a long log.

A "summary block" is defined as:
- Starts at a line containing "[INFO] [SUMMARY]"
- Continues line-by-line
- Ends on (and includes) the first line that contains "=> BEST:"

All such blocks are written to the output file, separated by a single blank line.

Usage:
    python extract_summaries.py input.log output.txt
"""

from __future__ import annotations
import argparse
from pathlib import Path

def extract_summaries(in_path: Path, out_path: Path) -> int:
    """
    Read in_path, write only [INFO] [SUMMARY] ... => BEST: blocks to out_path.
    Returns the number of blocks written.
    """
    blocks_written = 0
    collecting = False
    buf: list[str] = []

    with in_path.open("r", encoding="utf-8", errors="replace") as fin, \
         out_path.open("w", encoding="utf-8", errors="strict") as fout:

        for raw_line in fin:
            line = raw_line.rstrip("\n")

            if not collecting:
                # Look for the start of a summary block
                if "[INFO] [SUMMARY]" in line:
                    collecting = True
                    buf = [line]
                # else: ignore
            else:
                # We are inside a block; keep collecting
                buf.append(line)

                # End condition: first line that contains "=> BEST:"
                if "=> BEST:" in line:
                    # Flush the block
                    fout.write("\n".join(buf) + "\n")
                    fout.write("\n")  # separate blocks by one blank line
                    blocks_written += 1
                    collecting = False
                    buf = []

        # If file ended while collecting but never hit "=> BEST:", discard partial
        # (spec wants to end at => BEST:, so incomplete blocks are skipped)

    return blocks_written


def main() -> None:
    p = argparse.ArgumentParser(description="Extract [INFO] [SUMMARY] ... => BEST: blocks from logs.")
    p.add_argument("input", type=Path, help="Path to the input log file")
    p.add_argument("output", type=Path, help="Path to write the extracted summaries")
    args = p.parse_args()

    count = extract_summaries(args.input, args.output)
    print(f"Wrote {count} summary block(s) to {args.output}")

if __name__ == "__main__":
    main()
