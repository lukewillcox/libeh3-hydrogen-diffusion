# md_analysis_tools.py
#
# Usage examples
# --------------
# MSD only:
#   python md_analysis_tools.py msd --traj md.traj --timestep-fs 0.5 --save-interval 10 --show
#
# Density map only (x-z projection):
#   python md_analysis_tools.py density --traj md.traj --axes x z --bins 200 --show
#
# Full movie:
#   python md_analysis_tools.py movie --traj md.traj
#
# Candidate frame window movie:
#   python md_analysis_tools.py movie --traj md.traj --start 22000 --stop 22300
#
# Notes
# -----
# - MSD and density maps do NOT directly identify exact hop frames.
# - They tell you whether diffusion is happening and where H prefers to sit / move.
# - For exact frames, use the movie or a dedicated hop-extraction script.


from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ase.geometry import find_mic
from ase.io.trajectory import Trajectory
from ase.visualize import view


AXIS_TO_INDEX = {"x": 0, "y": 1, "z": 2}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MSD, density-map, and movie tools for H diffusion analysis from ASE trajectories."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Shared traj arg helper
    def add_traj_arg(p: argparse.ArgumentParser) -> None:
        p.add_argument("--traj", default="md.traj", help="Input ASE trajectory file.")

    # MSD subcommand
    msd = subparsers.add_parser("msd", help="Compute hydrogen MSD under PBC.")
    add_traj_arg(msd)
    msd.add_argument("--timestep-fs", type=float, required=True, help="MD timestep in fs.")
    msd.add_argument(
        "--save-interval",
        type=int,
        default=10,
        help="Number of MD steps between saved frames in the trajectory.",
    )
    msd.add_argument(
        "--max-lag",
        type=int,
        default=None,
        help="Maximum lag in frames for MSD. Default: half the trajectory length.",
    )
    msd.add_argument(
        "--outfile-prefix",
        default="msd_h",
        help="Prefix for output files, e.g. msd_h.csv and msd_h.png.",
    )
    msd.add_argument("--show", action="store_true", help="Show MSD plot interactively.")

    # Density subcommand
    density = subparsers.add_parser("density", help="Build 2D projected H density map.")
    add_traj_arg(density)
    density.add_argument(
        "--axes",
        nargs=2,
        default=["x", "z"],
        choices=["x", "y", "z"],
        help="Two Cartesian axes to project onto, e.g. x z.",
    )
    density.add_argument("--bins", type=int, default=200, help="Number of bins per axis.")
    density.add_argument(
        "--outfile-prefix",
        default="h_density",
        help="Prefix for density outputs, e.g. h_density.png and h_density.npz.",
    )
    density.add_argument("--show", action="store_true", help="Show density plot interactively.")

    # Movie subcommand
    movie = subparsers.add_parser("movie", help="Open full trajectory or a selected frame window.")
    add_traj_arg(movie)
    movie.add_argument("--start", type=int, default=None, help="Start frame (inclusive).")
    movie.add_argument("--stop", type=int, default=None, help="Stop frame (exclusive).")

    return parser.parse_args()


def hydrogen_indices(atoms0) -> list[int]:
    return [i for i, a in enumerate(atoms0) if a.symbol == "H"]


def load_traj(traj_path: str) -> Trajectory:
    path = Path(traj_path)
    if not path.exists():
        raise FileNotFoundError(f"Trajectory not found: {traj_path}")
    return Trajectory(traj_path, "r")


def mic_displacement_vector(atoms_a, atoms_b, atom_index: int) -> np.ndarray:
    pos_a = atoms_a.get_positions()[atom_index]
    pos_b = atoms_b.get_positions()[atom_index]
    cell = atoms_a.get_cell()
    pbc = atoms_a.get_pbc()
    vec = pos_b - pos_a
    mic_vec, _ = find_mic(vec, cell=cell, pbc=pbc)
    return np.asarray(mic_vec, dtype=float)


def run_msd(
    traj_path: str,
    timestep_fs: float,
    save_interval: int,
    max_lag: int | None,
    outfile_prefix: str,
    show: bool,
) -> None:
    traj = load_traj(traj_path)
    nframes = len(traj)
    if nframes < 2:
        raise ValueError("Need at least 2 frames for MSD.")

    atoms0 = traj[0]
    h_idx = hydrogen_indices(atoms0)
    if not h_idx:
        raise ValueError("No H atoms found in trajectory.")

    if max_lag is None:
        max_lag = nframes // 2
    max_lag = max(1, min(max_lag, nframes - 1))

    frame_dt_fs = timestep_fs * save_interval
    times_fs = np.arange(1, max_lag + 1) * frame_dt_fs
    msd_vals = np.zeros(max_lag, dtype=float)

    # Average over all time origins and all H atoms
    for lag in range(1, max_lag + 1):
        sq_disps: list[float] = []
        for t0 in range(nframes - lag):
            a0 = traj[t0]
            a1 = traj[t0 + lag]
            for h in h_idx:
                dv = mic_displacement_vector(a0, a1, h)
                sq_disps.append(float(np.dot(dv, dv)))
        msd_vals[lag - 1] = float(np.mean(sq_disps))

    # Save CSV
    csv_path = f"{outfile_prefix}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["lag_frame", "time_fs", "msd_angstrom2"])
        for i, (tfs, msd) in enumerate(zip(times_fs, msd_vals), start=1):
            writer.writerow([i, tfs, msd])

    # Plot
    plt.figure()
    plt.plot(times_fs, msd_vals)
    plt.xlabel("Time lag (fs)")
    plt.ylabel("MSD (Å$^2$)")
    plt.title("Hydrogen mean square displacement")
    plt.tight_layout()
    png_path = f"{outfile_prefix}.png"
    plt.savefig(png_path, dpi=300)

    print(f"Wrote {csv_path}")
    print(f"Wrote {png_path}")
    print("Interpretation:")
    print("  - Flat MSD: mostly local vibration.")
    print("  - Rising MSD: hydrogen diffusion / hopping.")
    print("  - Compare MSD across temperatures to judge mobility.")

    if show:
        plt.show()
    else:
        plt.close()


def wrap_fractional_positions(atoms, atom_indices: list[int]) -> np.ndarray:
    scaled = atoms.get_scaled_positions(wrap=True)
    return np.asarray([scaled[i] for i in atom_indices], dtype=float)


def run_density(
    traj_path: str,
    axes: list[str],
    bins: int,
    outfile_prefix: str,
    show: bool,
) -> None:
    traj = load_traj(traj_path)
    atoms0 = traj[0]
    h_idx = hydrogen_indices(atoms0)
    if not h_idx:
        raise ValueError("No H atoms found in trajectory.")

    ax0 = AXIS_TO_INDEX[axes[0]]
    ax1 = AXIS_TO_INDEX[axes[1]]

    # Use wrapped fractional coords for stable periodic-density representation,
    # then convert to Cartesian lengths along chosen axes.
    frac_points: list[np.ndarray] = []
    for atoms in traj:
        frac_h = wrap_fractional_positions(atoms, h_idx)
        frac_points.append(frac_h[:, [ax0, ax1]])
    frac_all = np.vstack(frac_points)

    lengths = atoms0.cell.lengths()
    cart_points = np.empty_like(frac_all)
    cart_points[:, 0] = frac_all[:, 0] * lengths[ax0]
    cart_points[:, 1] = frac_all[:, 1] * lengths[ax1]

    hist, xedges, yedges = np.histogram2d(
        cart_points[:, 0],
        cart_points[:, 1],
        bins=bins,
        range=[[0.0, lengths[ax0]], [0.0, lengths[ax1]]],
        density=True,
    )

    npz_path = f"{outfile_prefix}.npz"
    np.savez(npz_path, hist=hist, xedges=xedges, yedges=yedges, axes=np.array(axes))

    plt.figure()
    plt.imshow(
        hist.T,
        origin="lower",
        aspect="auto",
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
    )
    plt.xlabel(f"{axes[0]} (Å)")
    plt.ylabel(f"{axes[1]} (Å)")
    plt.title("Hydrogen spatial density map")
    plt.colorbar(label="Probability density")
    plt.tight_layout()

    png_path = f"{outfile_prefix}.png"
    plt.savefig(png_path, dpi=300)

    print(f"Wrote {npz_path}")
    print(f"Wrote {png_path}")
    print("Interpretation:")
    print("  - Bright regions: preferred hydrogen sites.")
    print("  - Connections between bright regions: likely migration pathways.")
    print("  - Density maps show where hopping occurs statistically, not exact hop frames.")

    if show:
        plt.show()
    else:
        plt.close()


def run_movie(traj_path: str, start: int | None, stop: int | None) -> None:
    traj = load_traj(traj_path)
    nframes = len(traj)

    if start is None and stop is None:
        frames = [traj[i].copy() for i in range(nframes)]
        print(f"Opening full trajectory movie: 0 to {nframes - 1}")
        view(frames)
        return

    s = 0 if start is None else max(0, start)
    e = nframes if stop is None else min(nframes, stop)
    if s >= e:
        raise ValueError(f"Invalid frame window: start={s}, stop={e}, nframes={nframes}")

    frames = [traj[i].copy() for i in range(s, e)]
    print(f"Opening trajectory window: {s} to {e - 1}")
    view(frames)


def main() -> None:
    args = parse_args()

    if args.command == "msd":
        run_msd(
            traj_path=args.traj,
            timestep_fs=args.timestep_fs,
            save_interval=args.save_interval,
            max_lag=args.max_lag,
            outfile_prefix=args.outfile_prefix,
            show=args.show,
        )
    elif args.command == "density":
        run_density(
            traj_path=args.traj,
            axes=args.axes,
            bins=args.bins,
            outfile_prefix=args.outfile_prefix,
            show=args.show,
        )
    elif args.command == "movie":
        run_movie(
            traj_path=args.traj,
            start=args.start,
            stop=args.stop,
        )
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()