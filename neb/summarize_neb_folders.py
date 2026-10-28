from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import numpy as np
from ase.io import read
from ase.geometry import find_mic


def parse_h_index(folder: str) -> int | None:
    m = re.search(r"_H(\d+)", folder)
    if not m:
        return None
    return int(m.group(1))


def mic_displacement(a, b, idx: int) -> float:
    dr = b.positions[idx] - a.positions[idx]
    dr_mic, d = find_mic(dr, cell=a.cell, pbc=a.pbc)
    return float(d)


def load_band_energies(folder: Path) -> tuple[np.ndarray, int] | None:
    """
    Returns (E_rel, n_images) or None if cannot get energies.
    Priority:
      1) neb_with_energy.extxyz
      2) neb_stage2_climb.traj / neb_stage2.traj (take last n_images frames)
      3) neb_image_*.xyz if energies embedded (rare)
    """

    # 1) explicit extxyz with energies
    extxyz = folder / "neb_with_energy.extxyz"
    if extxyz.exists():
        imgs = read(str(extxyz), index=":")
        E = np.array([img.get_potential_energy() for img in imgs], dtype=float)
        Erel = E - E[0]
        return Erel, len(imgs)

    # infer n_images from files if possible
    neb_imgs = sorted(folder.glob("neb_image_*.xyz"))
    n_images = len(neb_imgs) if neb_imgs else 9

    # 2) stage2 trajectory
    for trajname in ["neb_stage2_climb.traj", "neb_stage2.traj", "neb_stage1_climb.traj", "neb_stage1.traj"]:
        trj = folder / trajname
        if trj.exists():
            frames = read(str(trj), index=":")
            if len(frames) >= n_images:
                imgs = frames[-n_images:]
                E = np.array([img.get_potential_energy() for img in imgs], dtype=float)
                Erel = E - E[0]
                return Erel, n_images

    # 3) try neb_image_*.xyz energies (only if embedded)
    if neb_imgs:
        imgs = [read(str(f)) for f in neb_imgs]
        try:
            E = np.array([img.get_potential_energy() for img in imgs], dtype=float)
            Erel = E - E[0]
            return Erel, len(imgs)
        except Exception:
            return None

    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("folders", nargs="+", help="NEB folders to summarize")
    p.add_argument("--out", default="neb_summary.csv", help="CSV output filename")
    args = p.parse_args()

    rows = []
    print("\nfolder, H, jump_MiC_A, barrier_eV, endpoint_dE_eV, n_images, energy_source")
    print("-" * 95)

    for f in args.folders:
        folder = Path(f).expanduser().resolve()
        name = folder.name
        h = parse_h_index(name)

        # endpoints
        ini_fn = folder / "initial_relaxed.xyz"
        fin_fn = folder / "final_relaxed.xyz"
        if not ini_fn.exists() or not fin_fn.exists():
            print(f"{name}: MISSING initial_relaxed.xyz/final_relaxed.xyz")
            continue
        ini = read(str(ini_fn))
        fin = read(str(fin_fn))

        # jump length
        jump = None
        if h is not None:
            try:
                jump = mic_displacement(ini, fin, h)
            except Exception:
                jump = None

        # energies
        Edata = load_band_energies(folder)
        if Edata is None:
            barrier = None
            dE = None
            nimg = None
            src = "none"
        else:
            Erel, nimg = Edata
            barrier = float(np.max(Erel))
            dE = float(Erel[-1])
            # source label
            if (folder / "neb_with_energy.extxyz").exists():
                src = "neb_with_energy.extxyz"
            elif (folder / "neb_stage2_climb.traj").exists():
                src = "neb_stage2_climb.traj"
            elif (folder / "neb_stage2.traj").exists():
                src = "neb_stage2.traj"
            else:
                src = "neb_image_*.xyz"

        print(f"{name:30s}  H={str(h):>3s}  jump={str(jump):>8s}  "
              f"barrier={str(barrier):>10s}  dE={str(dE):>10s}  n={str(nimg):>3s}  {src}")

        rows.append((name, h, jump, barrier, dE, nimg, src))

    # write CSV
    out = Path(args.out).expanduser().resolve()
    with out.open("w") as fp:
        fp.write("folder,H_index,jump_MIC_A,barrier_eV,endpoint_dE_eV,n_images,energy_source\n")
        for r in rows:
            fp.write(",".join("" if v is None else str(v) for v in r) + "\n")

    print(f"\nWrote: {out}")


if __name__ == "__main__":
    main()