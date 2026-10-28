# conda activate xtb-md
"""
Measure hydrogen displacement over a chosen lag time using an extxyz trajectory.

Example:
  conda activate xtb-md
  python jump_hist_extxyz.py \
    --traj simulation.pos_0.extxyz \
    --lag-ps 1.0 \
    --timestep-fs 0.5 \
    --save-interval 1 \
    --max-frames 6000 \
    --top 30
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List

import numpy as np
import matplotlib.pyplot as plt

from ase.io import read
from ase.geometry import find_mic


@dataclass(frozen=True)
class JumpEvent:
    dist: float
    frame0: int
    frame1: int
    h_index: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--traj", required=True, help="Trajectory file (extxyz)")
    p.add_argument("--timestep-fs", type=float, required=True, help="MD timestep in fs")
    p.add_argument("--save-interval", type=int, default=1, help="MD steps per saved frame")
    p.add_argument("--lag-ps", type=float, default=1.0, help="Lag time in ps")
    p.add_argument("--max-frames", type=int, default=2000, help="Read only first N frames")
    p.add_argument("--top", type=int, default=20, help="Print top N largest displacements")
    p.add_argument("--bins", type=int, default=60, help="Histogram bins")
    p.add_argument("--min-jump", type=float, default=0.0, help="Minimum jump distance in Å")
    p.add_argument("--max-jump", type=float, default=None, help="Maximum jump distance in Å")
    return p.parse_args()


def unwrap_positions(frames, atom_indices: List[int]) -> np.ndarray:
    nframes = len(frames)
    nat = len(atom_indices)
    unwrapped = np.zeros((nframes, nat, 3), dtype=float)

    p0 = frames[0].get_positions()
    for k, idx in enumerate(atom_indices):
        unwrapped[0, k] = p0[idx]

    cell = frames[0].get_cell()
    pbc = frames[0].get_pbc()

    for t in range(1, nframes):
        pa = frames[t - 1].get_positions()
        pb = frames[t].get_positions()
        for k, idx in enumerate(atom_indices):
            dr = pb[idx] - pa[idx]
            dr_mic, _ = find_mic(dr, cell=cell, pbc=pbc)
            unwrapped[t, k] = unwrapped[t - 1, k] + dr_mic

    return unwrapped


def main() -> None:
    args = parse_args()

    all_frames = read(args.traj, index=":")
    n_total = len(all_frames)
    n_use = min(args.max_frames, n_total)
    frames = all_frames[:n_use]

    atoms0 = frames[0]
    h_indices = [i for i, a in enumerate(atoms0) if a.symbol == "H"]
    if not h_indices:
        raise RuntimeError("No H atoms found.")

    dt_fs_per_frame = args.timestep_fs * args.save_interval
    dt_ps_per_frame = dt_fs_per_frame / 1000.0
    lag_frames = int(round(args.lag_ps / dt_ps_per_frame))

    if lag_frames < 1:
        raise ValueError(f"lag_ps too small. Each saved frame = {dt_ps_per_frame:.4f} ps.")
    if lag_frames >= n_use:
        raise ValueError(f"lag_frames={lag_frames} >= nframes={n_use}. Increase --max-frames or reduce --lag-ps.")

    print(f"Frames loaded: {n_use} (of {n_total})")
    print(f"Frame dt: {dt_ps_per_frame:.4f} ps")
    print(f"Lag: {args.lag_ps:.3f} ps  => {lag_frames} frames")
    print(f"Jump filter: {args.min_jump:.3f} Å to "
          f"{args.max_jump:.3f} Å" if args.max_jump is not None else f"Jump filter: >= {args.min_jump:.3f} Å")

    unwrapped = unwrap_positions(frames, h_indices)
    disp = unwrapped[lag_frames:] - unwrapped[:-lag_frames]
    dist = np.linalg.norm(disp, axis=2)

    events: List[JumpEvent] = []
    n_time = dist.shape[0]
    for t0 in range(n_time):
        t1 = t0 + lag_frames
        for k, h in enumerate(h_indices):
            d = float(dist[t0, k])
            if d < args.min_jump:
                continue
            if args.max_jump is not None and d > args.max_jump:
                continue
            events.append(JumpEvent(d, t0, t1, h))

    events.sort(key=lambda e: e.dist, reverse=True)

    filtered_dist = np.array([e.dist for e in events], dtype=float)

    print("\nTop jump events (over lag):")
    if not events:
        print("  No events found in the requested distance window.")
    else:
        for e in events[: args.top]:
            print(f"  dist={e.dist:7.3f} Å   H{e.h_index:2d}   frame {e.frame0} -> {e.frame1}")

    plt.figure()
    if filtered_dist.size > 0:
        plt.hist(filtered_dist, bins=args.bins)
    else:
        plt.hist([], bins=args.bins)
    plt.xlabel(f"|Δr| over lag = {args.lag_ps:.3f} ps (Å)")
    plt.ylabel("Count")
    plt.title("Hydrogen displacement distribution (unwrapped)")
    plt.tight_layout()
    plt.savefig("jump_hist.png", dpi=300)
    plt.show()
    print("\nSaved histogram: jump_hist.png")


if __name__ == "__main__":
    main()