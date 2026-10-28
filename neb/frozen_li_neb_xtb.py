from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.constraints import FixAtoms
from ase.geometry import cellpar_to_cell
from ase.io import read, write
from ase.mep import NEB
from ase.optimize import BFGS
from xtb.ase.calculator import XTB


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="mode", required=True)

    def add_common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--a", type=float, required=True)
        sp.add_argument("--b", type=float, required=True)
        sp.add_argument("--c", type=float, required=True)
        sp.add_argument("--alpha", type=float, required=True)
        sp.add_argument("--beta", type=float, required=True)
        sp.add_argument("--gamma", type=float, required=True)
        sp.add_argument("--method", default="GFN0-xTB")
        sp.add_argument("--accuracy", type=float, default=1.0)
        sp.add_argument("--electronic-temperature", type=float, default=300.0)
        sp.add_argument("--frozen-indices", type=int, nargs="+", required=True)
        sp.add_argument("--fmax", type=float, default=0.05)
        sp.add_argument("--steps", type=int, default=300)

    relax = sub.add_parser("relax")
    add_common(relax)
    relax.add_argument("--input", required=True)
    relax.add_argument("--output", required=True)
    relax.add_argument("--traj", default="relax.traj")
    relax.add_argument("--log", default="relax.log")

    neb = sub.add_parser("neb")
    add_common(neb)
    neb.add_argument("--initial", required=True)
    neb.add_argument("--final", required=True)
    neb.add_argument("--nimages", type=int, default=7)
    neb.add_argument("--climb", action="store_true")
    neb.add_argument("--k", type=float, default=0.1)
    neb.add_argument("--traj", default="neb.traj")
    neb.add_argument("--log", default="neb.log")
    neb.add_argument("--images-dir", default="neb_images")

    return p.parse_args()


def make_cell(args: argparse.Namespace) -> np.ndarray:
    return cellpar_to_cell([args.a, args.b, args.c, args.alpha, args.beta, args.gamma])


def make_calc(args: argparse.Namespace) -> XTB:
    return XTB(
        method=args.method,
        accuracy=args.accuracy,
        electronic_temperature=args.electronic_temperature,
    )


def prepare_atoms(atoms: Atoms, args: argparse.Namespace) -> Atoms:
    atoms = atoms.copy()
    atoms.set_cell(make_cell(args))
    atoms.set_pbc(True)
    atoms.calc = make_calc(args)
    atoms.set_constraint(FixAtoms(indices=args.frozen_indices))
    return atoms


def run_relax(args: argparse.Namespace) -> None:
    atoms = read(args.input)
    atoms = prepare_atoms(atoms, args)
    opt = BFGS(atoms, trajectory=args.traj, logfile=args.log)
    opt.run(fmax=args.fmax, steps=args.steps)
    write(args.output, atoms)
    print(f"Wrote {args.output}")
    print(f"Energy = {atoms.get_potential_energy():.10f} eV")


def run_neb(args: argparse.Namespace) -> None:
    initial = read(args.initial)
    final = read(args.final)

    images = [prepare_atoms(initial, args)]
    for _ in range(args.nimages - 2):
        images.append(prepare_atoms(initial.copy(), args))
    images.append(prepare_atoms(final, args))

    neb = NEB(images, climb=args.climb, k=args.k)
    neb.interpolate()

    outdir = Path(args.images_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    for i, atoms in enumerate(images):
        write(outdir / f"image_{i:02d}_start.xyz", atoms)

    opt = BFGS(neb, trajectory=args.traj, logfile=args.log)
    opt.run(fmax=args.fmax, steps=args.steps)

    energies = []
    for i, atoms in enumerate(images):
        e = float(atoms.get_potential_energy())
        energies.append(e)
        write(outdir / f"image_{i:02d}_final.xyz", atoms)

    e0 = energies[0]
    rel = [e - e0 for e in energies]
    print("Relative image energies (eV):")
    for i, de in enumerate(rel):
        print(f"image_{i:02d} {de:.10f}")
    print(f"Approximate barrier = {max(rel):.10f} eV")


def main() -> None:
    args = parse_args()
    if args.mode == "relax":
        run_relax(args)
    else:
        run_neb(args)


if __name__ == "__main__":
    main()
