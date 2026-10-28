#!/usr/bin/env python3
"""
Seed-sweep jump histogram for ONE hydrogen index (e.g. H28) across many .traj files.

- Reads many ASE .traj files via glob
- Unwraps the chosen atom under PBC using MIC steps
- Computes |Δr| over a lag time
- Prints per-seed stats + top events overall
- Saves a combined histogram PNG + per-seed CSV summary

Example:
  conda activate xtb-md
  cd ~/Library/CloudStorage/OneDrive-CardiffUniversity/XTB-340K/seed_sweep_1947_2045_H28
  python ../../jump_hist_seed_sweep_honly.py \
    --glob "md_restart_seed*.traj" \
    --h-index 28 \
    --timestep-fs 0.25 \
    --save-interval 10 \
    --lag-ps 1.0 \
    --max-frames 6000 \
    --top 30 \
    --bins 80 \
    --out-prefix H28_lag1ps
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt

from ase.io.trajectory import Trajectory
from ase.geometry import find_mic


@dataclass(frozen=True)
class Event:
    dist: float
    seedfile: str
    frame0: int
    frame1: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--glob", dest="pattern", required=True, help='Glob for traj files, e.g. "md_restart_seed*.traj"')
    p.add_argument("--h-index", type=int, required=True, help="Atom index of the hydrogen to track, e.g. 28")
    p.add_argument("--timestep-fs", type=float, required=True, help="Integrator timestep in fs (e.g. 0.25)")
    p.add_argument("--save-interval", type=int, required=True, help="Saved frame every N MD steps (e.g. 10)")
    p.add_argument("--lag-ps", type=float, default=1.0, help="Lag time in ps for displacement (default 1.0)")
    p.add_argument("--max-frames", type=int, default=6000, help="Max frames read per traj (default 6000)")
    p.add_argument("--top", type=int, default=30, help="Top N events overall to print (default 30)")
    p.add_argument("--bins", type=int, default=80, help="Histogram bins (default 80)")
    p.add_argument("--out-prefix", default="jump_hist", help="Prefix for outputs (png/csv)")
    return p.parse_args()


def load_frames(traj_path: str, max_frames: int):
    traj = Trajectory(traj_path, "r")
    n_total = len(traj)
    n_use = min(max_frames, n_total)
    frames = [traj[i] for i in range(n_use)]
    return frames, n_total, n_use


def unwrap_single_atom(frames, atom_index: int) -> np.ndarray:
    """
    Unwrap ONE atom position over time using MIC between consecutive frames.
    Returns array (nframes, 3) of unwrapped positions in Å.
    """
    n = len(frames)
    if n == 0:
        raise ValueError("No frames loaded.")

    cell = frames[0].get_cell()
    pbc = frames[0].get_pbc()

    unwrapped = np.zeros((n, 3), dtype=float)
    unwrapped[0] = frames[0].positions[atom_index]

    for t in range(1, n):
        pa = frames[t - 1].positions[atom_index]
        pb = frames[t].positions[atom_index]
        dr = pb - pa
        dr_mic, _ = find_mic(dr, cell=cell, pbc=pbc)
        unwrapped[t] = unwrapped[t - 1] + dr_mic

    return unwrapped


def main() -> None:
    args = parse_args()

    files = sorted(glob.glob(args.pattern))
    if not files:
        raise FileNotFoundError(f"No files match pattern: {args.pattern}")

    dt_fs_per_frame = args.timestep_fs * args.save_interval
    dt_ps_per_frame = dt_fs_per_frame / 1000.0
    lag_frames = int(round(args.lag_ps / dt_ps_per_frame))

    if lag_frames < 1:
        raise ValueError(
            f"--lag-ps too small. Each saved frame is {dt_ps_per_frame:.6f} ps. "
            f"Need lag >= {dt_ps_per_frame:.6f} ps."
        )

    print(f"Found {len(files)} traj files.")
    print(f"Tracking atom index: H{args.h_index}")
    print(f"Frame dt: {dt_ps_per_frame:.6f} ps  (timestep_fs={args.timestep_fs}, save_interval={args.save_interval})")
    print(f"Lag: {args.lag_ps:.3f} ps => {lag_frames} frames\n")

    all_dist: List[float] = []
    all_events: List[Event] = []
    per_seed_rows = []

    for fpath in files:
        frames, n_total, n_use = load_frames(fpath, args.max_frames)
        if lag_frames >= n_use:
            print(f"[skip] {os.path.basename(fpath)}: only {n_use} frames loaded; lag needs {lag_frames+1}+")
            continue

        atoms0 = frames[0]
        if args.h_index < 0 or args.h_index >= len(atoms0):
            raise IndexError(f"H index {args.h_index} out of range for {fpath} (natoms={len(atoms0)})")

        # optional sanity check: ensure it's actually H
        sym = atoms0[args.h_index].symbol
        if sym != "H":
            print(f"[warn] {os.path.basename(fpath)}: index {args.h_index} is '{sym}', not 'H'")

        uw = unwrap_single_atom(frames, args.h_index)  # (n_use, 3)
        disp = uw[lag_frames:] - uw[:-lag_frames]      # (n_use-lag, 3)
        dist = np.linalg.norm(disp, axis=1)            # (n_use-lag,)

        all_dist.extend(dist.tolist())

        # per-seed stats
        max_i = int(np.argmax(dist))
        max_d = float(dist[max_i])
        frame0 = max_i
        frame1 = max_i + lag_frames

        p50 = float(np.percentile(dist, 50))
        p90 = float(np.percentile(dist, 90))
        p99 = float(np.percentile(dist, 99))

        per_seed_rows.append({
            "file": os.path.basename(fpath),
            "frames_total": n_total,
            "frames_used": n_use,
            "lag_ps": args.lag_ps,
            "lag_frames": lag_frames,
            "max_jump_A": max_d,
            "max_frame0": frame0,
            "max_frame1": frame1,
            "p50_A": p50,
            "p90_A": p90,
            "p99_A": p99,
        })

        all_events.append(Event(dist=max_d, seedfile=os.path.basename(fpath), frame0=frame0, frame1=frame1))

        print(f"{os.path.basename(fpath)}  max={max_d:7.3f} Å  at frame {frame0} -> {frame1}  (used {n_use}/{n_total})")

    if not all_dist:
        raise RuntimeError("No distances computed (all files skipped?). Check lag/max-frames.")

    # top events overall (by max per seed)
    all_events.sort(key=lambda e: e.dist, reverse=True)
    print("\nTop max-jump seeds (one max per seed file):")
    for e in all_events[: args.top]:
        print(f"  {e.dist:7.3f} Å   {e.seedfile:>22s}   frame {e.frame0} -> {e.frame1}")

    # histogram
    out_png = f"{args.out_prefix}_H{args.h_index}_lag{args.lag_ps:g}ps_hist.png"
    plt.figure()
    plt.hist(np.array(all_dist), bins=args.bins)
    plt.xlabel(f"|Δr| over lag = {args.lag_ps:.3f} ps (Å)  [H{args.h_index}]")
    plt.ylabel("Count")
    plt.title("Hydrogen displacement distribution (unwrapped, one H index)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.show()
    print(f"\nSaved histogram: {out_png}")

    # CSV summary
    out_csv = f"{args.out_prefix}_H{args.h_index}_lag{args.lag_ps:g}ps_summary.csv"
    with open(out_csv, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=list(per_seed_rows[0].keys()))
        w.writeheader()
        w.writerows(per_seed_rows)
    print(f"Saved per-seed summary CSV: {out_csv}")


if __name__ == "__main__":
    main()