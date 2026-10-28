# hops_final_attempt.py

from __future__ import annotations

import argparse
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from ase.geometry import find_mic
from ase.io import write
from ase.io.trajectory import Trajectory
from ase.visualize import view


# --- stricter parameters ---
SITE_CUTOFF = 2.2          # H must be within this to count as attached to a Be
INNER_CUTOFF = 1.95        # H must be clearly inside old/new site before/after
DIST_MARGIN = 0.40         # old/new distance difference must be at least this large
MIN_STABLE = 50            # 50 frames before + 50 after = 100-frame hop window
PAD = 40                   # frames saved before/after selected hop

# Be-Be filter:
# reject "hops" where Be atoms just come too close and confuse nearest-neighbour logic
MIN_BE_BE = 2.6            # minimum allowed Be-Be separation during candidate event

# Match these to your MD output
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
    dmaps = {h: [] for h in h_indices}

    # collect per-frame data
    for atoms in traj:
        for h in h_indices:
            be, _, dmap = nearest_be_index(atoms, h, be_indices)
            labels[h].append(be)
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

            old_ok = stable_site(lab, t - MIN_STABLE, MIN_STABLE, old_site)
            new_ok = stable_site(lab, t, MIN_STABLE, new_site)

            if not (old_ok and new_ok):
                t += 1
                continue

            before_frame = t - MIN_STABLE
            after_frame = t + MIN_STABLE - 1

            before_old = dmaps[h][before_frame][old_site]
            before_new = dmaps[h][before_frame][new_site]
            after_old = dmaps[h][after_frame][old_site]
            after_new = dmaps[h][after_frame][new_site]

            # H must clearly belong to old site before and new site after
            before_clear = (
                before_old < INNER_CUTOFF
                and before_new - before_old > DIST_MARGIN
            )
            after_clear = (
                after_new < INNER_CUTOFF
                and after_old - after_new > DIST_MARGIN
            )

            if not (before_clear and after_clear):
                t += 1
                continue

            # Be-Be distance must stay reasonable:
            # reject cases where Be atoms just approach too closely and fake a hop
            bebe_before = mic_distance(traj[before_frame], old_site, new_site)
            bebe_after = mic_distance(traj[after_frame], old_site, new_site)
            bebe_mid = mic_distance(traj[t], old_site, new_site)

            min_bebe = min(bebe_before, bebe_mid, bebe_after)

            if min_bebe < MIN_BE_BE:
                t += 1
                continue

            # reaction-coordinate-like check:
            # before: old Be clearly preferred; after: new Be clearly preferred
            rc_before = before_new - before_old
            rc_after = after_new - after_old

            if not (rc_before > DIST_MARGIN and rc_after < -DIST_MARGIN):
                t += 1
                continue

            # use full old/new site switch strength as score
            score = abs(rc_before) + abs(rc_after)

            events.append((t - 1, t, h, old_site, new_site, score))

            # skip ahead so one hop is not counted repeatedly
            t += MIN_STABLE

    if not events:
        print("No real H hops found after strict stability, margin, and Be-Be filters.")
        return

    counts = Counter(e[2] for e in events)
    print("Hop counts by H index:")
    for h in sorted(counts):
        print(f"  H{h}: {counts[h]}")

    # strongest site-switch first
    events.sort(key=lambda x: x[5], reverse=True)

    print("\nTop hop events (frame_before, frame_after, H, oldBe, newBe, score):")
    for e in events[:20]:
        print("  ", e)

    frame_before, frame_after, h, old_be, new_be, score = events[0]

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
    print(f"  H{h}: Be{old_be} → Be{new_be} | score = {score:.2f}")
    print("  Wrote: initial.xyz, final.xyz, hop_window.xyz, hop_window.traj")

    if args.plot:
        time_fs = np.arange(nframes) * FRAME_DT_FS
        old_series = np.array([dmaps[h][i][old_be] for i in range(nframes)])
        new_series = np.array([dmaps[h][i][new_be] for i in range(nframes)])
        rc_series = new_series - old_series

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

        plt.figure()
        plt.plot(time_fs, rc_series, label="d(H,newBe) - d(H,oldBe)")
        plt.axhline(0.0, linestyle="--")
        plt.axvline(frame_before * FRAME_DT_FS, linestyle="--", label="detected hop")
        plt.axvline(frame_after * FRAME_DT_FS, linestyle="--")
        plt.xlabel("Time (fs)")
        plt.ylabel("Distance difference (Å)")
        plt.title(f"H{h} reaction-coordinate-like signal")
        plt.legend()
        plt.tight_layout()
        plt.savefig("reaction_coordinate_plot.png", dpi=300)

        print("Saved: h_distance_plot.png, reaction_coordinate_plot.png")

    if args.show_hop:
        view(hop_traj)

    if args.show_full:
        full_traj = [traj[i].copy() for i in range(nframes)]
        view(full_traj)


if __name__ == "__main__":
    main()