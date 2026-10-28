# ~/Library/CloudStorage/OneDrive-CardiffUniversity/neb_plot_nodes.py
"""
Plot NEB energy profile: energies vs band nodes (reaction coordinate s).

Usage (recommended):
    cd /path/to/your/neb_run_folder
    python ~/Library/CloudStorage/OneDrive-CardiffUniversity/neb_plot_nodes.py

Inputs supported (auto-detected in current directory):
    1) neb_image_*.xyz   (preferred)
    2) neb_stage2.traj   (fallback)
    3) neb_stage1.traj   (fallback)
    4) neb_opt.traj      (fallback)

Outputs:
    - neb_energy_vs_nodes.png
    - prints node, s(Å), E_rel(eV)

Notes:
    - "s" is computed as cumulative RMS displacement between consecutive images.
    - Energies are taken from Atoms.get_potential_energy().
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")  # always save plot, no GUI required
import matplotlib.pyplot as plt

from ase.io import read
from ase.atoms import Atoms


DEFAULT_TRAJ_CANDIDATES = ("neb_stage2.traj", "neb_stage1.traj", "neb_opt.traj")


def _read_images_from_xyz(pattern: str = "neb_image_*.xyz") -> List[Atoms]:
    files = sorted(glob.glob(pattern))
    if not files:
        return []
    return [read(f) for f in files]


def _read_last_band_from_traj(traj_file: str, n_images: int) -> List[Atoms]:
    frames = read(traj_file, index=":")
    if not frames or len(frames) < n_images:
        return []
    return frames[-n_images:]


def _guess_n_images_from_xyz() -> int | None:
    files = sorted(glob.glob("neb_image_*.xyz"))
    if not files:
        return None
    return len(files)


def load_neb_images(n_images: int | None, traj_candidates: Tuple[str, ...]) -> List[Atoms]:
    # 1) preferred: neb_image_*.xyz
    images = _read_images_from_xyz("neb_image_*.xyz")
    if images:
        return images

    # if no xyz images, we need a band size guess
    if n_images is None:
        # common default if you used 9 images
        n_images = 9

    # 2) fallback: trajectories
    for tfn in traj_candidates:
        if Path(tfn).exists():
            images = _read_last_band_from_traj(tfn, n_images=n_images)
            if images:
                return images

    raise FileNotFoundError(
        "Could not find NEB images in this folder.\n"
        "Expected either:\n"
        "  - neb_image_*.xyz\n"
        "  - neb_stage2.traj / neb_stage1.traj / neb_opt.traj\n"
        "Run this script inside the NEB run directory."
    )


def band_coordinate_rms(images: List[Atoms]) -> np.ndarray:
    """Cumulative band coordinate s (Å) using RMS displacement between consecutive images."""
    s = [0.0]
    for i in range(1, len(images)):
        dr = images[i].get_positions() - images[i - 1].get_positions()
        step = float(np.sqrt((dr * dr).sum() / len(dr)))  # RMS Å
        s.append(s[-1] + step)
    return np.array(s, dtype=float)


def energies_relative(images: List[Atoms]) -> np.ndarray:
    e = np.array([float(img.get_potential_energy()) for img in images], dtype=float)
    return e - e[0]


def save_plot(s: np.ndarray, e_rel: np.ndarray, out: str, title: str) -> None:
    plt.figure(figsize=(6, 4))
    plt.plot(s, e_rel, marker="o")
    plt.xlabel("Band coordinate s (Å)")
    plt.ylabel("Energy (eV, relative to node 0)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out, dpi=300)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot NEB energy vs band nodes (s).")
    p.add_argument(
        "--n-images",
        type=int,
        default=None,
        help="Band size used in NEB (only needed if no neb_image_*.xyz exists). Default: auto or 9.",
    )
    p.add_argument(
        "--traj-candidates",
        nargs="*",
        default=list(DEFAULT_TRAJ_CANDIDATES),
        help="Trajectory filenames to try if neb_image_*.xyz is missing.",
    )
    p.add_argument(
        "--out",
        default="neb_energy_vs_nodes.png",
        help="Output png filename (saved in current directory).",
    )
    p.add_argument(
        "--title",
        default=None,
        help="Plot title (default: folder name).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # If neb_image_*.xyz exist, use that to determine n_images automatically
    n_guess = _guess_n_images_from_xyz()
    n_images = n_guess if n_guess is not None else args.n_images

    images = load_neb_images(n_images=n_images, traj_candidates=tuple(args.traj_candidates))
    s = band_coordinate_rms(images)
    e_rel = energies_relative(images)

    folder = os.path.basename(os.getcwd())
    title = args.title or f"NEB energy vs nodes ({folder})"

    save_plot(s, e_rel, out=args.out, title=title)

    print(f"Saved: {os.path.abspath(args.out)}")
    print("node   s(Å)      E_rel(eV)")
    for i, (si, ei) in enumerate(zip(s, e_rel)):
        print(f"{i:3d}  {si:8.4f}  {ei:12.6f}")


if __name__ == "__main__":
    main()