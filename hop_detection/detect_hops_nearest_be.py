# detect_hops_nearest_be.py

from __future__ import annotations

import argparse
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

from ase.io.trajectory import Trajectory
from ase.geometry import find_mic
from ase.io import write
from ase.visualize import view


# --- tunable parameters ---
SITE_CUTOFF = 2.5          # Å: H must be within this to count as assigned to a Be
MIN_DISP = 0.8             # Å: total MIC displacement across hop window
MIN_STABLE = 10            # frames: old and new site must each persist this long
PAD = 20                   # frames saved before/after selected hop
DIST_MARGIN = 0.25         # Å: nearest Be must be clearly closer than the alternative

# Match these to your MD output:
TIMESTEP_FS = 0.5
SAVE_INTERVAL = 10
FRAME_DT_FS = TIMESTEP_FS * SAVE_INTERVAL


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--traj", default="md.traj", help="Input trajectory")
    p.add_argument("--show-hop", action="store_true", help="Open selected hop window")
    p.add_argument("--show-full", action="store_true", help="Open full MD movie")
    p.add_argument("--plot", action="store_true", help="Plot old/new Be distances vs time")
    return p.parse_args()


def mic_distance(atoms, i: int, j: int) -> float:
    pos = atoms.get_positions()
    cell = atoms.get_cell()
    pbc = atoms.get_pbc()
    v = pos[i] - pos[j]
    _, d = find_mic(v, cell=cell, pbc=pbc)
    return float(d)


def mic_displacement(atoms_a, atoms_b, i: int) -> float:
    pos_a = atoms_a.get_positions()
    pos_b = atoms_b.get_positions()
    cell = atoms_a.get_cell()
    pbc = atoms_a.get_pbc()
    v = pos_b[i] - pos_a[i]
    _, d = find_mic(v, cell=cell, pbc=pbc)
    return float(d)


def nearest_be_index(atoms, h_index: int, be_indices: list[int]) -> tuple[int, float, dict[int, float]]:
    dmap: dict[int, float] = {}
    best_be = -1
    best_d = 1e9

    for j in be_indices:
        d = mic_distance(atoms, h_index, j)
        dmap[j] = d
        if d < best_d and d < SITE_CUTOFF:
            best_d = d
            best_be = j

    return best_be, best_d, dmap


def stable_site(labels: list[int], start: int, n: int, site: int) -> bool:
    return all(labels[k] == site for k in range(start, start + n))


def main() -> None:
    args = parse_args()

    traj = Trajectory(args.traj, "r")
    atoms0 = traj[0]

    h_indices = [i for i, a in enumerate(atoms0) if a.symbol == "H"]
    be_indices = [i for i, a in enumerate(atoms0) if a.symbol == "Be"]

    labels = {h: [] for h in h_indices}
    nearest_dist = {h: [] for h in h_indices}
    dmaps = {h: [] for h in h_indices}

    # Collect per-frame data for every H
    for atoms in traj:
        for h in h_indices:
            be, d, dmap = nearest_be_index(atoms, h, be_indices)
            labels[h].append(be)
            nearest_dist[h].append(d)
            dmaps[h].append(dmap)

    nframes = len(traj)
    events: list[tuple[int, int, int, int, int, float]] = []

    for h in h_indices:
        lab = labels[h]
        t = MIN_STABLE

        while t < nframes - MIN_STABLE:
            old_site = lab[t - 1]
            new_site = lab[t]

            if old_site == -1 or new_site == -1 or old_site == new_site:
                t += 1
                continue

            # old site stable before, new site stable after
            old_ok = stable_site(lab, t - MIN_STABLE, MIN_STABLE, old_site)
            new_ok = stable_site(lab, t, MIN_STABLE, new_site)

            if not (old_ok and new_ok):
                t += 1
                continue

            # Compare distances at the start/end of the event window
            before_frame = t - MIN_STABLE
            after_frame = t + MIN_STABLE - 1

            before_old = dmaps[h][before_frame][old_site]
            before_new = dmaps[h][before_frame][new_site]
            after_old = dmaps[h][after_frame][old_site]
            after_new = dmaps[h][after_frame][new_site]

            # Require a clear switch in local environment
            before_clear = (
                before_old < SITE_CUTOFF
                and before_new - before_old > DIST_MARGIN
            )
            after_clear = (
                after_new < SITE_CUTOFF
                and after_old - after_new > DIST_MARGIN
            )

            if not (before_clear and after_clear):
                t += 1
                continue

            disp = mic_displacement(traj[before_frame], traj[after_frame], h)

            if disp < MIN_DISP:
                t += 1
                continue

            events.append((t - 1, t, h, old_site, new_site, disp))

            # Skip ahead so one physical hop is not counted repeatedly
            t += MIN_STABLE
        # end while

    if not events:
        print("No real H hops found after stability, margin, and displacement filters.")
        return

    # Summary by hydrogen
    counts = Counter(e[2] for e in events)
    print("Hop counts by H index:")
    for h in sorted(counts):
        print(f"  H{h}: {counts[h]}")

    # Sort by displacement so strongest/clearest hops come first
    events.sort(key=lambda x: x[5], reverse=True)

    print("\nTop hop events (frame_before, frame_after, H, oldBe, newBe, disp Å):")
    for e in events[:20]:
        print("  ", e)

    # Pick the strongest hop
    frame_before, frame_after, h, old_be, new_be, disp = events[0]

    initial_frame = max(0, frame_before - PAD)
    final_frame = min(nframes - 1, frame_after + PAD)

    initial = traj[initial_frame].copy()
    final = traj[final_frame].copy()

    write("initial.xyz", initial)
    write("final.xyz", final)

    hop_traj = [traj[i].copy() for i in range(initial_frame, final_frame + 1)]
    write("hop_window.xyz", hop_traj)
    write("hop_window.traj", hop_traj)

    print("\nSelected hop:")
    print(f"  frames {initial_frame} → {final_frame}")
    print(f"  H{h}: Be{old_be} → Be{new_be} | displacement = {disp:.2f} Å")
    print("  Wrote: initial.xyz, final.xyz, hop_window.xyz, hop_window.traj")

    if args.plot:
        time_fs = np.arange(nframes) * FRAME_DT_FS
        old_series = np.array([dmaps[h][i][old_be] for i in range(nframes)])
        new_series = np.array([dmaps[h][i][new_be] for i in range(nframes)])

        plt.figure()
        plt.plot(time_fs, old_series, label=f"H{h}–Be{old_be}")
        plt.plot(time_fs, new_series, label=f"H{h}–Be{new_be}")
        plt.axvline(frame_before * FRAME_DT_FS, linestyle="--", label="detected hop")
        plt.axvline(frame_after * FRAME_DT_FS, linestyle="--")
        plt.xlabel("Time (fs)")
        plt.ylabel("Distance (Å)")
        plt.title(f"H{h} distances to old/new Be vs time")
        plt.legend()
        plt.tight_layout()
        plt.savefig("h_distance_plot.png", dpi=300)
        print("Saved: h_distance_plot.png")

    if args.show_hop:
        view(hop_traj)

    if args.show_full:
        full_traj = [traj[i].copy() for i in range(nframes)]
        view(full_traj)


if __name__ == "__main__":
    main()