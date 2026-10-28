#!/usr/bin/env python3
# neb_average_std.py
#
# Compute an average NEB profile (mean ± std) across multiple NEB runs (folders),
# and plot it with ERROR BARS (not a shaded band).
#
# Usage:
#   conda activate xtb-md
#   python neb_average_std.py \
#     /path/to/neb_run1 \
#     /path/to/neb_run2 \
#     /path/to/neb_run3 \
#     --out neb_avg_std_3runs_errorbars.png
#
# Notes:
# - Each NEB folder should contain either:
#     - neb_image_00.xyz ... neb_image_08.xyz  (preferred), OR
#     - neb_stage2.traj (fallback)
# - Energies are taken from potential_energy stored in the xyz/traj (if present).
#   If energies are missing, you must rerun NEB saving images with energies.

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from ase.io import read


def load_images_from_folder(folder: Path) -> List:
    """Load NEB images from a folder."""
    # Prefer explicit images:
    files = sorted(folder.glob("neb_image_*.xyz"))
    if files:
        return [read(str(f)) for f in files]

    # Fallback: read last band from stage2 trajectory (if it exists)
    stage2 = folder / "neb_stage2.traj"
    if stage2.exists():
        traj = read(str(stage2), index=":")
        # Heuristic: band size = count of unique images in your NEB (often 9)
        # If you used a different number, set it here:
        n_images_guess = 9
        if len(traj) >= n_images_guess:
            return traj[-n_images_guess:]
        raise RuntimeError(
            f"{stage2} exists but has fewer frames ({len(traj)}) than expected band ({n_images_guess})."
        )

    raise FileNotFoundError(
        f"Could not find neb_image_*.xyz or neb_stage2.traj in: {folder}"
    )


def band_coordinate(images: List) -> np.ndarray:
    """
    Compute a band coordinate s (Å) from cumulative RMS displacement between successive images.
    """
    s = [0.0]
    for i in range(1, len(images)):
        dr = images[i].get_positions() - images[i - 1].get_positions()
        step = float(np.sqrt((dr * dr).sum() / len(dr)))  # RMS displacement
        s.append(s[-1] + step)
    return np.array(s, dtype=float)


def energies_relative(images: List) -> np.ndarray:
    """
    Extract energies and shift so image 0 is 0 (relative energy in eV).
    """
    e = np.array([float(img.get_potential_energy()) for img in images], dtype=float)
    return e - e[0]


def resample_to_common_grid(
    s: np.ndarray, e_rel: np.ndarray, x_common: np.ndarray
) -> np.ndarray:
    """
    Interpolate E(s) onto a common normalized x grid [0..1].
    """
    if s[-1] <= 0:
        # Degenerate band (shouldn't happen); return constant
        return np.full_like(x_common, e_rel[0], dtype=float)

    x = s / s[-1]  # normalize to [0..1]
    # np.interp needs increasing x:
    order = np.argsort(x)
    x_sorted = x[order]
    e_sorted = e_rel[order]
    return np.interp(x_common, x_sorted, e_sorted)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("folders", nargs="+", help="NEB run folders to average")
    p.add_argument("--out", default="neb_avg_std_errorbars.png", help="Output PNG filename")
    p.add_argument("--npoints", type=int, default=25, help="Number of points on normalized band coordinate")
    p.add_argument("--capsize", type=float, default=3.0, help="Errorbar cap size")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    folders = [Path(f).expanduser().resolve() for f in args.folders]
    for f in folders:
        if not f.exists() or not f.is_dir():
            raise FileNotFoundError(f"Not a folder: {f}")

    # Common normalized coordinate grid
    x_common = np.linspace(0.0, 1.0, args.npoints)

    all_profiles: List[np.ndarray] = []
    used: List[str] = []

    for folder in folders:
        imgs = load_images_from_folder(folder)
        s = band_coordinate(imgs)
        e_rel = energies_relative(imgs)
        prof = resample_to_common_grid(s, e_rel, x_common)

        all_profiles.append(prof)
        used.append(folder.name)

    Y = np.vstack(all_profiles)  # shape: (nruns, npoints)
    mean = Y.mean(axis=0)
    std = Y.std(axis=0, ddof=1) if Y.shape[0] > 1 else np.zeros_like(mean)

    # ---- Plot with error bars ----
    plt.figure()
    plt.errorbar(
        x_common,
        mean,
        yerr=std,
        fmt="o-",
        capsize=args.capsize,
        elinewidth=1,
    )
    plt.xlabel("Normalized band coordinate (0 → 1)")
    plt.ylabel("Energy (eV, relative to initial)")
    plt.title("Average NEB profile (mean ± std)")
    plt.tight_layout()
    plt.savefig(args.out, dpi=300)
    plt.show()

    print(f"Used folders: {', '.join(used)}")
    print(f"Saved: {args.out}")
    print("\nnode   x_norm    E_mean(eV)    E_std(eV)")
    for i, (xv, m, sdev) in enumerate(zip(x_common, mean, std)):
        print(f"{i:3d}   {xv:6.3f}    {m:10.6f}   {sdev:10.6f}")


if __name__ == "__main__":
    main()