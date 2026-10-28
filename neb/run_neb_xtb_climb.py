from __future__ import annotations

import numpy as np
from ase.io import read, write
from ase.mep import NEB
from ase.optimize import BFGS
from xtb.ase.calculator import XTB

suffix = "_climb"

def attach_xtb(atoms) -> None:
    atoms.set_pbc(True)
    atoms.calc = XTB(
        method="GFN0-xTB",
        accuracy=1.0,
        electronic_temperature=300.0,
    )


def preflight(images) -> None:
    for i, img in enumerate(images):
        try:
            e = float(img.get_potential_energy())
            f = img.get_forces()
            fmax = float(np.linalg.norm(f, axis=1).max())
            print(f"image {i}: OK  E={e:.6f} eV  fmax={fmax:.3f} eV/Å")
        except Exception as exc:
            print(f"image {i}: FAIL -> {exc}")
            write(f"FAILED_image_{i:02d}.xyz", img)
            raise


def main() -> None:
    # ---- ONLY CHANGE HERE: output suffix ----
    suffix = "_climb"
    # ----------------------------------------

    initial = read("initial_relaxed.xyz")
    final = read("final_relaxed.xyz")

    cell = initial.get_cell()

    initial.set_pbc(True)
    final.set_pbc(True)
    initial.set_cell(cell)
    final.set_cell(cell)

    n_images = 9
    images = [initial] + [initial.copy() for _ in range(n_images - 2)] + [final]

    for img in images:
        img.set_pbc(True)
        img.set_cell(cell)
        attach_xtb(img)

    neb = NEB(images, climb=True)

    try:
        neb.interpolate(method="idpp", mic=True)
    except TypeError:
        try:
            neb.interpolate(mic=True)
        except TypeError:
            neb.interpolate()

    preflight(images)

    # ---- Stage 1 outputs renamed ----
    opt1 = BFGS(neb, logfile=f"neb_stage1{suffix}.log", trajectory=f"neb_stage1{suffix}.traj")
    opt1.run(fmax=0.10, steps=500)

    # ---- Stage 2 outputs renamed ----
    neb = NEB(images, climb=True)

    for img in images:
        img.set_pbc(True)
        img.set_cell(cell)
        attach_xtb(img)

    opt2 = BFGS(neb, logfile=f"neb_stage2{suffix}.log", trajectory=f"neb_stage2{suffix}.traj")
    opt2.run(fmax=0.03, steps=500)

    # ---- Images renamed (so old neb_image_*.xyz remains) ----
    for i, img in enumerate(images):
        write(f"neb_image_{i:02d}{suffix}.xyz", img)

    energies = np.array([img.get_potential_energy() for img in images])
    energies_rel = energies - energies[0]

    print("Barrier (eV):", float(energies_rel.max()))
    print("Endpoint ΔE (eV):", float(energies_rel[-1]))
    for i, e in enumerate(energies_rel):
        print(i, float(e))

def print_progress(images, opt, label: str) -> None:
    energies = np.array([img.get_potential_energy() for img in images], dtype=float)
    energies_rel = energies - energies[0]

    fmax_list = []
    for img in images[1:-1]:
        forces = img.get_forces()
        fmax_img = float(np.linalg.norm(forces, axis=1).max())
        fmax_list.append(fmax_img)

    band_fmax = max(fmax_list) if fmax_list else 0.0
    e_string = "  ".join(f"{e:.3f}" for e in energies_rel)

    print(f"[{label}] step={opt.nsteps:4d}  band_fmax={band_fmax:.3f} eV/Å", flush=True)
    print(f"[{label}] Erel(eV): {e_string}", flush=True)

if __name__ == "__main__":
    main()