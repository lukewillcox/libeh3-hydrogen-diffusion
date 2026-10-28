# temp_jump_summary.py
from __future__ import annotations

import argparse
import re
import numpy as np
import matplotlib.pyplot as plt

from ase.io import read
from ase.io.trajectory import Trajectory
from ase.geometry import find_mic


def load_frames(path: str, max_frames: int):
    """Loads up to max_frames frames from .traj or extxyz/xyz."""
    if path.endswith(".traj"):
        tr = Trajectory(path, "r")
        n = min(len(tr), max_frames)
        return [tr[i] for i in range(n)]
    else:
        frames = read(path, index=":")
        return frames[: min(len(frames), max_frames)]


def unwrap_positions(frames, atom_indices):
    nframes = len(frames)
    nat = len(atom_indices)
    unwrapped = np.zeros((nframes, nat, 3), dtype=float)

    pos0 = frames[0].get_positions()
    for k, idx in enumerate(atom_indices):
        unwrapped[0, k] = pos0[idx]

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


def parse_temp_from_path(path: str) -> float | None:
    m = re.search(r"(\d+)\s*K", path)
    if m:
        return float(m.group(1))
    m = re.search(r"_(\d+)K", path)
    if m:
        return float(m.group(1))
    m = re.search(r"md_(\d+)K", path)
    if m:
        return float(m.group(1))
    return None


def summarize(path: str, timestep_fs: float, save_interval: int, lag_ps: float,
              max_frames: int, threshold_A: float):
    frames = load_frames(path, max_frames=max_frames)
    if len(frames) < 2:
        raise RuntimeError(f"Not enough frames in {path}")

    atoms0 = frames[0]
    h_indices = [i for i, a in enumerate(atoms0) if a.symbol == "H"]
    if not h_indices:
        raise RuntimeError(f"No H atoms in {path}")

    dt_fs_per_frame = timestep_fs * save_interval
    dt_ps_per_frame = dt_fs_per_frame / 1000.0
    lag_frames = int(round(lag_ps / dt_ps_per_frame))
    if lag_frames < 1:
        raise ValueError(f"lag_ps too small: frame dt = {dt_ps_per_frame:.5f} ps")
    if lag_frames >= len(frames):
        raise ValueError(f"lag_frames={lag_frames} >= nframes={len(frames)} for {path}")

    # unwrapped positions for H only
    unwrapped = unwrap_positions(frames, h_indices)
    disp = unwrapped[lag_frames:] - unwrapped[:-lag_frames]
    dist = np.linalg.norm(disp, axis=2)  # (ntime, nH)
    dist_flat = dist.reshape(-1)

    # metrics
    tail_prob = float(np.mean(dist_flat > threshold_A))
    p95 = float(np.percentile(dist_flat, 95))
    p99 = float(np.percentile(dist_flat, 99))

    # hop rate: count events above threshold per time
    # Here: each (t0, H) is an event candidate; total time origins = ntime
    n_events = int(np.sum(dist > threshold_A))
    total_time_ps = (dist.shape[0]) * dt_ps_per_frame  # time origins span
    hop_rate_per_ps = float(n_events / total_time_ps) if total_time_ps > 0 else 0.0

    return {
        "nframes": len(frames),
        "dt_ps": dt_ps_per_frame,
        "lag_frames": lag_frames,
        "tail_prob": tail_prob,
        "p95": p95,
        "p99": p99,
        "hop_rate_ps": hop_rate_per_ps,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", nargs="+", required=True, help="List of trajectories (.traj or extxyz)")
    ap.add_argument("--timestep-fs", type=float, required=True)
    ap.add_argument("--save-interval", type=int, required=True)
    ap.add_argument("--lag-ps", type=float, default=1.0)
    ap.add_argument("--max-frames", type=int, default=6000)
    ap.add_argument("--threshold", type=float, default=2.0)
    ap.add_argument("--out", default="temp_vs_hops.png")
    args = ap.parse_args()

    rows = []
    for path in args.traj:
        T = parse_temp_from_path(path)
        if T is None:
            raise ValueError(f"Could not infer temperature from filename: {path} (include e.g. md_340K.traj)")
        s = summarize(path, args.timestep_fs, args.save_interval, args.lag_ps, args.max_frames, args.threshold)
        rows.append((T, path, s))

    rows.sort(key=lambda x: x[0])

    temps = np.array([r[0] for r in rows], float)
    hop_rate = np.array([r[2]["hop_rate_ps"] for r in rows], float)
    tail = np.array([r[2]["tail_prob"] for r in rows], float)
    p95 = np.array([r[2]["p95"] for r in rows], float)

    # print a clean table
    print("T(K)   hop_rate(/ps)   tail_prob(>|thr|)   p95(Å)   file")
    for T, path, s in rows:
        print(f"{T:4.0f}   {s['hop_rate_ps']:12.3f}   {s['tail_prob']:16.6f}   {s['p95']:6.3f}   {path}")

    # plot (one figure with two y-axes)
    fig, ax1 = plt.subplots()
    ax1.plot(temps, hop_rate, marker="o")
    ax1.set_xlabel("Temperature (K)")
    ax1.set_ylabel(f"Hop rate (/ps), threshold={args.threshold} Å over {args.lag_ps} ps")

    ax2 = ax1.twinx()
    ax2.plot(temps, p95, marker="o")
    ax2.set_ylabel(f"95th percentile |Δr| (Å) over {args.lag_ps} ps")

    plt.title("Hydrogen mobility vs temperature (same lag + threshold)")
    plt.tight_layout()
    plt.savefig(args.out, dpi=300)
    print(f"\nSaved plot: {args.out}")


if __name__ == "__main__":
    main()