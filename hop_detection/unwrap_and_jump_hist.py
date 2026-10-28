"""
Unwrap PBC trajectory and measure H displacement over a chosen time lag.

Outputs
-------
- Prints the top N largest H displacements over the lag
- Saves histogram: jump_hist.png

Usage example
-------------
conda activate xtb-md
python unwrap_and_jump_hist.py --traj md_340K.traj --timestep-fs 0.5 --save-interval 10 --lag-ps 1.0 --max-frames 2000
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt

from ase.io.trajectory import Trajectory
from ase.geometry import find_mic


@dataclass(frozen=True)
class JumpEvent:
    dist: float
    frame0: int
    frame1: int
    h_index: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--traj", required=True, help="ASE .traj file (e.g. md_340K.traj)")
    p.add_argument("--timestep-fs", type=float, required=True, help="MD integrator timestep in fs (e.g. 0.5)")
    p.add_argument("--save-interval", type=int, required=True, help="How many MD steps per saved frame (e.g. 10)")
    p.add_argument("--lag-ps", type=float, default=1.0, help="Lag time in ps for displacement (default: 1.0)")
    p.add_argument("--max-frames", type=int, default=2000, help="Read only first N frames (default: 2000)")
    p.add_argument("--top", type=int, default=20, help="Print top N largest jumps (default: 20)")
    p.add_argument("--bins", type=int, default=60, help="Histogram bins (default: 60)")
    return p.parse_args()


def unwrap_positions(frames, atom_indices: List[int]) -> np.ndarray:
    """
    Build unwrapped positions for selected atoms.

    Returns
    -------
    unwrapped : (nframes, natoms_sel, 3)
    """
    nframes = len(frames)
    nat = len(atom_indices)
    unwrapped = np.zeros((nframes, nat, 3), dtype=float)

    # frame 0
    p0 = frames[0].get_positions()
    for k, idx in enumerate(atom_indices):
        unwrapped[0, k] = p0[idx]

    cell = frames[0].get_cell()
    pbc = frames[0].get_pbc()

    # accumulate MIC displacement frame-to-frame
    for t in range(1, nframes):
        pa = frames[t - 1].get_positions()
        pb = frames[t].get_positions()

        for k, idx in enumerate(atom_indices):
            dr = pb[idx] - pa[idx]                # wrapped step
            dr_mic, _ = find_mic(dr, cell=cell, pbc=pbc)  # smallest equivalent step
            unwrapped[t, k] = unwrapped[t - 1, k] + dr_mic

    return unwrapped


def main() -> None:
    args = parse_args()

    traj = Trajectory(args.traj, "r")
    n_total = len(traj)
    n_use = min(args.max_frames, n_total)
    frames = [traj[i] for i in range(n_use)]
    atoms0 = frames[0]

    h_indices = [i for i, a in enumerate(atoms0) if a.symbol == "H"]
    if not h_indices:
        raise RuntimeError("No H atoms found in trajectory.")

    dt_fs_per_frame = args.timestep_fs * args.save_interval
    dt_ps_per_frame = dt_fs_per_frame / 1000.0

    lag_frames = int(round(args.lag_ps / dt_ps_per_frame))
    if lag_frames < 1:
        raise ValueError(
            f"lag_ps too small for your saved-frame spacing. "
            f"Each frame is {dt_ps_per_frame:.4f} ps."
        )
    if lag_frames >= n_use:
        raise ValueError(f"lag_frames={lag_frames} >= nframes={n_use}. Increase --max-frames or reduce --lag-ps.")

    print(f"Frames loaded: {n_use} (of {n_total})")
    print(f"Frame dt: {dt_ps_per_frame:.4f} ps  (timestep_fs={args.timestep_fs}, save_interval={args.save_interval})")
    print(f"Lag: {args.lag_ps:.3f} ps  => {lag_frames} frames")

    # 1) unwrap
    unwrapped = unwrap_positions(frames, h_indices)  # (nframes, nH, 3)

    # 2) compute lag displacement magnitudes
    disp = unwrapped[lag_frames:] - unwrapped[:-lag_frames]          # (nframes-lag, nH, 3)
    dist = np.linalg.norm(disp, axis=2)                              # (nframes-lag, nH)

    # flatten to all H, all time origins
    dist_flat = dist.reshape(-1)

    # 3) report biggest events
    events: List[JumpEvent] = []
    n_time = dist.shape[0]
    for t0 in range(n_time):
        t1 = t0 + lag_frames
        for k, h in enumerate(h_indices):
            events.append(JumpEvent(dist=float(dist[t0, k]), frame0=t0, frame1=t1, h_index=h))

    events.sort(key=lambda e: e.dist, reverse=True)
    print("\nTop jump events (over lag):")
    for e in events[: args.top]:
        print(f"  dist={e.dist:7.3f} Å   H{e.h_index:2d}   frame {e.frame0} -> {e.frame1}")

    # 4) histogram plot
    plt.figure()
    plt.hist(dist_flat, bins=args.bins)
    plt.xlabel(f"|Δr| over lag = {args.lag_ps:.3f} ps (Å)")
    plt.ylabel("Count")
    plt.title("Hydrogen displacement distribution (unwrapped)")
    plt.tight_layout()
    plt.savefig("jump_hist.png", dpi=300)
    plt.show()

    print("\nSaved histogram: jump_hist.png")


if __name__ == "__main__":
    main()