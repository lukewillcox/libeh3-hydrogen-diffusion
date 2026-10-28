# jump_freq_anytraj.py
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import List

import numpy as np
import matplotlib.pyplot as plt

from ase.geometry import find_mic
from ase.io import read
from ase.io.trajectory import Trajectory


@dataclass(frozen=True)
class JumpEvent:
    dist: float
    frame0: int
    frame1: int
    h_index: int


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--traj", required=True, help="Trajectory file (.traj or .extxyz)")
    p.add_argument("--timestep-fs", type=float, required=True, help="MD timestep (fs)")
    p.add_argument("--save-interval", type=int, required=True, help="MD steps per saved frame")
    p.add_argument("--lag-ps", type=float, default=1.0, help="Lag time (ps)")
    p.add_argument("--max-frames", type=int, default=6000, help="Max frames to analyze")
    p.add_argument("--threshold", type=float, default=2.0, help="Count events with |Δr| >= threshold (Å)")
    p.add_argument("--top", type=int, default=20, help="Print top N events")
    p.add_argument("--tag", default="run", help="Label for output files")
    return p.parse_args()


def load_frames(path: str, nmax: int):
    # Use Trajectory for .traj, read() for extxyz/xyz
    if path.endswith(".traj"):
        tr = Trajectory(path, "r")
        n = min(len(tr), nmax)
        frames = [tr[i] for i in range(n)]
        return frames, len(tr)
    else:
        all_frames = read(path, index=":")
        n = min(len(all_frames), nmax)
        return all_frames[:n], len(all_frames)


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


def main():
    args = parse_args()

    frames, n_total = load_frames(args.traj, args.max_frames)
    n_use = len(frames)
    if n_use < 2:
        raise RuntimeError("Not enough frames loaded to analyze.")

    atoms0 = frames[0]
    h_indices = [i for i, a in enumerate(atoms0) if a.symbol == "H"]
    if not h_indices:
        raise RuntimeError("No H atoms found.")

    dt_fs_per_frame = args.timestep_fs * args.save_interval
    dt_ps_per_frame = dt_fs_per_frame / 1000.0
    lag_frames = int(round(args.lag_ps / dt_ps_per_frame))

    if lag_frames < 1:
        raise ValueError(f"lag_ps too small. Each saved frame = {dt_ps_per_frame:.6f} ps.")
    if lag_frames >= n_use:
        raise ValueError(f"lag_frames={lag_frames} >= nframes={n_use}. Increase --max-frames or reduce --lag-ps.")

    print(f"Trajectory: {args.traj}")
    print(f"Frames loaded: {n_use} (of {n_total})")
    print(f"Frame dt: {dt_ps_per_frame:.6f} ps")
    print(f"Lag: {args.lag_ps:.3f} ps  => {lag_frames} frames")
    print(f"Threshold: {args.threshold:.3f} Å")

    unwrapped = unwrap_positions(frames, h_indices)
    disp = unwrapped[lag_frames:] - unwrapped[:-lag_frames]
    dist = np.linalg.norm(disp, axis=2)  # (nframes-lag, nH)

    # events
    events: List[JumpEvent] = []
    n_time = dist.shape[0]
    for t0 in range(n_time):
        t1 = t0 + lag_frames
        for k, h in enumerate(h_indices):
            d = float(dist[t0, k])
            events.append(JumpEvent(d, t0, t1, h))

    events.sort(key=lambda e: e.dist, reverse=True)

    print("\nTop events:")
    for e in events[: args.top]:
        print(f"  dist={e.dist:7.3f} Å   H{e.h_index:2d}   frame {e.frame0} -> {e.frame1}")

    # hop frequency (counts per ps)
    hot = dist.reshape(-1) >= args.threshold
    count = int(hot.sum())
    total_time_ps = (n_use - lag_frames) * dt_ps_per_frame
    rate = count / total_time_ps if total_time_ps > 0 else float("nan")

    print(f"\nCount(|Δr|>={args.threshold}Å) = {count}")
    print(f"Analyzed time window ≈ {total_time_ps:.3f} ps")
    print(f"Event rate ≈ {rate:.3f} events/ps (all H, all time origins)")

    # histogram
    plt.figure()
    plt.hist(dist.reshape(-1), bins=60)
    plt.axvline(args.threshold, linestyle="--")
    plt.xlabel(f"|Δr| over lag={args.lag_ps:.3f} ps (Å)")
    plt.ylabel("Count")
    plt.title(f"Jump distribution ({args.tag})")
    plt.tight_layout()
    out = f"jump_hist_{args.tag}.png"
    plt.savefig(out, dpi=300)
    print(f"Saved: {out}")

    # save summary line for later temperature plotting
    with open("jump_rate_summary.csv", "a") as f:
        if os.path.getsize("jump_rate_summary.csv") == 0:
            f.write("tag,threshold_A,lag_ps,frames_used,dt_ps_per_frame,count,window_ps,rate_events_per_ps\n")
        f.write(f"{args.tag},{args.threshold},{args.lag_ps},{n_use},{dt_ps_per_frame},{count},{total_time_ps},{rate}\n")
    print("Appended: jump_rate_summary.csv")


if __name__ == "__main__":
    main()