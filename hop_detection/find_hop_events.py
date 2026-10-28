from __future__ import annotations

import argparse
import numpy as np
from ase.io import read, write
from ase.geometry import find_mic


def mic_displacement(atoms_a, atoms_b, i: int) -> float:
    """Minimum-image displacement magnitude of atom i between two frames."""
    v = atoms_b.positions[i] - atoms_a.positions[i]
    _, d = find_mic(v, cell=atoms_a.cell, pbc=atoms_a.pbc)
    return float(d)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--traj", required=True, help="Trajectory file (e.g. md_340K.traj)")
    p.add_argument("--lag-frames", type=int, required=True, help="Lag in saved frames")
    p.add_argument("--min-disp", type=float, default=2.0, help="Min displacement Å for candidate")
    p.add_argument("--top", type=int, default=20, help="Also report top-N events by displacement")
    p.add_argument("--pad", type=int, default=50, help="Frames on either side to write event window")
    p.add_argument("--only-H", action="store_true", help="Only consider hydrogen atoms (recommended)")
    args = p.parse_args()

    frames = read(args.traj, index=":")
    nframes = len(frames)
    if nframes <= args.lag_frames:
        raise ValueError(f"Trajectory has {nframes} frames, lag_frames={args.lag_frames} is too big.")

    symbols = frames[0].get_chemical_symbols()
    idxs = [i for i, s in enumerate(symbols) if (s == "H" or not args.only_H)]

    events = []  # (disp, t, i)
    for t in range(0, nframes - args.lag_frames):
        a = frames[t]
        b = frames[t + args.lag_frames]
        for i in idxs:
            d = mic_displacement(a, b, i)
            if d >= args.min_disp:
                events.append((d, t, i))

    # Sort by largest displacement
    events.sort(reverse=True, key=lambda x: x[0])

    if not events:
        print("No candidates found.")
        print("Try: smaller min-disp, larger lag-frames, higher T, or longer MD.")
        return

    print(f"Found {len(events)} candidates with disp >= {args.min_disp} Å")
    print("Top events: (disp Å, frame_t, atom_index, frame_t+lag)")
    for d, t, i in events[:args.top]:
        print(f"  {d:7.3f}  {t:7d}  {i:5d}  {t+args.lag_frames:7d}")

    # Pick the single best event and write files for NEB
    d, t, i = events[0]
    t2 = t + args.lag_frames

    initial = frames[t].copy()
    final = frames[t2].copy()
    write("initial.xyz", initial)
    write("final.xyz", final)

    # Window for VMD inspection
    lo = max(0, t - args.pad)
    hi = min(nframes - 1, t2 + args.pad)
    window = [frames[k].copy() for k in range(lo, hi + 1)]
    write("event_window.xyz", window)

    print("\nSelected best candidate:")
    print(f"  atom index: {i} (symbol {symbols[i]})")
    print(f"  frames: {t} -> {t2} (lag={args.lag_frames})")
    print(f"  displacement: {d:.3f} Å")
    print(f"Wrote: initial.xyz, final.xyz, event_window.xyz (frames {lo}..{hi})")


if __name__ == "__main__":
    main()