# file: run_neb_xtb.py
from __future__ import annotations

import argparse
import numpy as np
from ase.constraints import FixAtoms
from ase.io import read, write
from ase.mep import NEB
from ase.optimize import BFGS
from xtb.ase.calculator import XTB


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--initial", default="initial_relaxed.xyz")
    p.add_argument("--final", default="final_relaxed.xyz")
    p.add_argument("--suffix", default="_climb")
    p.add_argument("--n-images", type=int, default=9)
    p.add_argument("--climb", action="store_true", default=True)
    p.add_argument("--fmax-stage1", type=float, default=0.10)
    p.add_argument("--steps-stage1", type=int, default=500)
    p.add_argument("--fmax-stage2", type=float, default=0.03)
    p.add_argument("--steps-stage2", type=int, default=500)
    p.add_argument("--freeze-indices", type=int, nargs="*", default=[])
    p.add_argument("--method", default="GFN0-xTB")
    p.add_argument("--accuracy", type=float, default=1.0)
    p.add_argument("--electronic-temperature", type=float, default=300.0)
    return p.parse_args()


def attach_xtb(atoms, args: argparse.Namespace) -> None:
    atoms.set_pbc(True)
    atoms.calc = XTB(
        method=args.method,
        accuracy=args.accuracy,
        electronic_temperature=args.electronic_temperature,
    )


def apply_constraints(atoms, freeze_indices: list[int]) -> None:
    if freeze_indices:
        atoms.set_constraint(FixAtoms(indices=freeze_indices))


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


def prepare_images(initial, final, n_images: int, args: argparse.Namespace):
    cell = initial.get_cell()

    initial.set_pbc(True)
    final.set_pbc(True)
    initial.set_cell(cell)
    final.set_cell(cell)

    images = [initial] + [initial.copy() for _ in range(n_images - 2)] + [final]

    for img in images:
        img.set_pbc(True)
        img.set_cell(cell)
        apply_constraints(img, args.freeze_indices)
        attach_xtb(img, args)

    return images


def main() -> None:
    args = parse_args()

    initial = read(args.initial)
    final = read(args.final)

    images = prepare_images(initial, final, args.n_images, args)

    neb = NEB(images, climb=args.climb)

    try:
        neb.interpolate(method="idpp", mic=True)
    except TypeError:
        try:
            neb.interpolate(mic=True)
        except TypeError:
            neb.interpolate()

    preflight(images)

    opt1 = BFGS(
        neb,
        logfile=f"neb_stage1{args.suffix}.log",
        trajectory=f"neb_stage1{args.suffix}.traj",
    )
    opt1.attach(lambda: print_progress(images, opt1, "stage1"), interval=5)
    opt1.run(fmax=args.fmax_stage1, steps=args.steps_stage1)

    neb = NEB(images, climb=args.climb)

    for img in images:
        img.set_pbc(True)
        apply_constraints(img, args.freeze_indices)
        attach_xtb(img, args)

    opt2 = BFGS(
        neb,
        logfile=f"neb_stage2{args.suffix}.log",
        trajectory=f"neb_stage2{args.suffix}.traj",
    )
    opt2.attach(lambda: print_progress(images, opt2, "stage2"), interval=5)
    opt2.run(fmax=args.fmax_stage2, steps=args.steps_stage2)

    for i, img in enumerate(images):
        write(f"neb_image_{i:02d}{args.suffix}.xyz", img)

    energies = np.array([img.get_potential_energy() for img in images], dtype=float)
    energies_rel = energies - energies[0]

    print("Barrier (eV):", float(energies_rel.max()))
    print("Endpoint ΔE (eV):", float(energies_rel[-1]))
    for i, e in enumerate(energies_rel):
        print(i, float(e))


if __name__ == "__main__":
    main()