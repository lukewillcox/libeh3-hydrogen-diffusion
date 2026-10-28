# jump_frequency_extxyz.py
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from ase.geometry import find_mic
from ase.io import read


@dataclass(frozen=True)
class JumpEvent:
    dist: float
    frame0: int
    frame1: int
    h_index: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--traj", required=True, help="Trajectory file (extxyz)")
    p.add_argument("--temperature", type=float, required=True, help="Temperature in K for reporting")
    p.add_argument("--timestep-fs", type=float, required=True, help="MD timestep in fs")
    p.add_argument("--save-interval", type=int, default=1, help="MD steps per saved frame")
    p.add_argument("--lag-ps", type=float, default=1.0, help="Lag time in ps")
    p.add_argument("--max-frames", type=int, default=2000, help="Read only first N frames")
    p.add_argument("--top", type=int, default=20, help="Print top N largest displacements")
    p.add_argument("--bins", type=int, default=60, help="Histogram bins")
    p.add_argument("--jump-threshold", type=float, default=1.0, help="Å threshold for a jump")
    p.add_argument(
        "--summary-csv",
        default="jump_frequency_summary.csv",
        help="Append one summary row to this CSV file",
    )
    p.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip histogram plotting",
    )
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


def count_jump_events(
    dist: np.ndarray,
    h_indices: List[int],
    threshold: float,
    lag_frames: int,
) -> List[JumpEvent]:
    events: List[JumpEvent] = []
    n_time, n_h = dist.shape

    for k in range(n_h):
        above = dist[:, k] >= threshold
        prev = False
        for t0, is_above in enumerate(above):
            if is_above and not prev:
                t1 = t0 + lag_frames
                events.append(
                    JumpEvent(
                        dist=float(dist[t0, k]),
                        frame0=t0,
                        frame1=t1,
                        h_index=h_indices[k],
                    )
                )
            prev = is_above

    events.sort(key=lambda e: (e.frame0, e.h_index))
    return events


def append_summary_csv(path: Path, row: dict) -> None:
    write_header = not path.exists()
    fieldnames = [
        "temperature_K",
        "traj",
        "n_frames",
        "n_H",
        "lag_ps",
        "jump_threshold_A",
        "total_time_ps",
        "n_jump_events",
        "jump_rate_per_ps",
        "jump_rate_per_H_per_ps",
        "mean_displacement_A",
        "max_displacement_A",
    ]

    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = parse_args()

    all_frames = read(args.traj, index=":")
    n_total = len(all_frames)
    n_use = min(args.max_frames, n_total)
    frames = all_frames[:n_use]

    if n_use < 2:
        raise RuntimeError("Need at least 2 frames.")

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
        raise ValueError(
            f"lag_frames={lag_frames} >= nframes={n_use}. Increase --max-frames or reduce --lag-ps."
        )

    print(f"Frames loaded: {n_use} (of {n_total})")
    print(f"H atoms: {len(h_indices)}")
    print(f"Frame dt: {dt_ps_per_frame:.6f} ps")
    print(f"Lag: {args.lag_ps:.3f} ps => {lag_frames} frames")
    print(f"Jump threshold: {args.jump_threshold:.3f} Å")

    unwrapped = unwrap_positions(frames, h_indices)
    disp = unwrapped[lag_frames:] - unwrapped[:-lag_frames]
    dist = np.linalg.norm(disp, axis=2)

    top_events: List[JumpEvent] = []
    n_time = dist.shape[0]
    for t0 in range(n_time):
        t1 = t0 + lag_frames
        for k, h in enumerate(h_indices):
            top_events.append(JumpEvent(float(dist[t0, k]), t0, t1, h))
    top_events.sort(key=lambda e: e.dist, reverse=True)

    print("\nTop jump-like displacements:")
    for e in top_events[: args.top]:
        print(f"  dist={e.dist:7.3f} Å   H{e.h_index:2d}   frame {e.frame0} -> {e.frame1}")

    jump_events = count_jump_events(
        dist=dist,
        h_indices=h_indices,
        threshold=args.jump_threshold,
        lag_frames=lag_frames,
    )

    total_time_ps = (n_use - 1) * dt_ps_per_frame
    n_jump_events = len(jump_events)
    jump_rate_per_ps = n_jump_events / total_time_ps if total_time_ps > 0 else np.nan
    jump_rate_per_h_per_ps = (
        n_jump_events / (len(h_indices) * total_time_ps) if total_time_ps > 0 else np.nan
    )

    print("\nJump frequency summary:")
    print(f"  Temperature:           {args.temperature:.1f} K")
    print(f"  Total time:            {total_time_ps:.4f} ps")
    print(f"  Number of H atoms:     {len(h_indices)}")
    print(f"  Jump events:           {n_jump_events}")
    print(f"  Jump rate:             {jump_rate_per_ps:.6f} jumps/ps")
    print(f"  Jump rate per H:       {jump_rate_per_h_per_ps:.6f} jumps/(H·ps)")
    print(f"  Mean lag displacement: {float(np.mean(dist)):.4f} Å")
    print(f"  Max lag displacement:  {float(np.max(dist)):.4f} Å")

    csv_row = {
        "temperature_K": args.temperature,
        "traj": args.traj,
        "n_frames": n_use,
        "n_H": len(h_indices),
        "lag_ps": args.lag_ps,
        "jump_threshold_A": args.jump_threshold,
        "total_time_ps": total_time_ps,
        "n_jump_events": n_jump_events,
        "jump_rate_per_ps": jump_rate_per_ps,
        "jump_rate_per_H_per_ps": jump_rate_per_h_per_ps,
        "mean_displacement_A": float(np.mean(dist)),
        "max_displacement_A": float(np.max(dist)),
    }
    append_summary_csv(Path(args.summary_csv), csv_row)
    print(f"\nAppended summary to: {args.summary_csv}")

    if not args.no_plot:
        plt.figure()
        plt.hist(dist.reshape(-1), bins=args.bins)
        plt.axvline(args.jump_threshold, linestyle="--")
        plt.xlabel(f"|Δr| over lag = {args.lag_ps:.3f} ps (Å)")
        plt.ylabel("Count")
        plt.title(f"Hydrogen displacement distribution ({args.temperature:.0f} K)")
        plt.tight_layout()
        out_png = f"jump_hist_{int(round(args.temperature))}K.png"
        plt.savefig(out_png, dpi=300)
        plt.show()
        print(f"Saved histogram: {out_png}")


if __name__ == "__main__":
    main()