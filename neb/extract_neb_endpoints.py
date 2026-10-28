from __future__ import annotations

import argparse
from ase.io.trajectory import Trajectory
from ase.io import write


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--traj", required=True, help="Trajectory file (e.g. md.traj)")
    p.add_argument("--initial-frame", type=int, required=True)
    p.add_argument("--final-frame", type=int, required=True)
    p.add_argument("--pad", type=int, default=5, help="Extra frames for window")
    return p.parse_args()


def main():
    args = parse_args()

    traj = Trajectory(args.traj, "r")
    nframes = len(traj)

    i0 = max(0, args.initial_frame)
    i1 = min(nframes - 1, args.final_frame)

    initial = traj[i0].copy()
    final = traj[i1].copy()

    write("initial.xyz", initial)
    write("final.xyz", final)

    # also save a window for visual inspection
    w0 = max(0, i0 - args.pad)
    w1 = min(nframes - 1, i1 + args.pad)

    window = [traj[i].copy() for i in range(w0, w1 + 1)]
    write("event_window.xyz", window)
    write("event_window.traj", window)

    print(f"Saved initial.xyz (frame {i0})")
    print(f"Saved final.xyz   (frame {i1})")
    print(f"Saved window      ({w0} → {w1})")


if __name__ == "__main__":
    main()