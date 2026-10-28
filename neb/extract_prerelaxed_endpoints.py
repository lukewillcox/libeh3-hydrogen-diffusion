"""
Extract *raw* MD endpoint geometries (no relaxation) for NEB testing.

Writes:
- initial_md.xyz  (initial frame)
- final_md.xyz    (final frame)
- window_md.xyz   (initial..final frames)

Notes
-----
- Works for ASE-readable trajectories (.traj, .extxyz, etc).
- For .traj, ASE reads via ULM backend.
- For .extxyz, ASE reads via extxyz reader.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
from ase.io import read, write
from ase.geometry import find_mic


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--traj", required=True, help="Trajectory file (e.g. md_340K.traj)")
    p.add_argument("--initial-frame", type=int, required=True)
    p.add_argument("--final-frame", type=int, required=True)
    p.add_argument("--h-index", type=int, default=None, help="Optional: print MIC displacement for this atom index")
    p.add_argument("--out-prefix", default="", help="Optional prefix for output files")
    return p.parse_args()


def mic_displacement(a0, a1, i: int) -> float:
    v = a1.positions[i] - a0.positions[i]
    _, d = find_mic(v, cell=a0.cell, pbc=a0.pbc)
    return float(d)


def main() -> None:
    args = parse_args()

    traj_path = Path(args.traj).expanduser().resolve()
    if not traj_path.exists():
        raise FileNotFoundError(f"Trajectory not found: {traj_path}")

    i0 = args.initial_frame
    i1 = args.final_frame
    if i1 < i0:
        raise ValueError("--final-frame must be >= --initial-frame")

    # Read endpoints (single frames)
    initial = read(str(traj_path), index=i0)
    final = read(str(traj_path), index=i1)

    # Read window (inclusive) for debugging/movie
    window = read(str(traj_path), index=f"{i0}:{i1 + 1}")

    prefix = args.out_prefix

    initial_fn = f"{prefix}initial_md.xyz"
    final_fn = f"{prefix}final_md.xyz"
    window_fn = f"{prefix}window_md.xyz"

    write(initial_fn, initial)
    write(final_fn, final)
    write(window_fn, window)

    print(f"Saved {initial_fn} (frame {i0})")
    print(f"Saved {final_fn}   (frame {i1})")
    print(f"Saved {window_fn}  (frames {i0} → {i1}, count={len(window)})")

    if args.h_index is not None:
        h = args.h_index
        if h < 0 or h >= len(initial):
            raise IndexError(f"h-index {h} out of range (0..{len(initial)-1})")
        d = mic_displacement(initial, final, h)
        r0 = initial.positions[h]
        r1 = final.positions[h]
        print(f"\nAtom index {h} raw MIC Δr (Å): {d:.6f}")
        print(f"r0: {r0}")
        print(f"r1: {r1}")


if __name__ == "__main__":
    main()