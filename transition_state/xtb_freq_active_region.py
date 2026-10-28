from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from ase.constraints import FixAtoms
from ase.geometry import cellpar_to_cell
from ase.io import read, write
from ase.optimize import BFGS
from ase.vibrations import Vibrations
from xtb.ase.calculator import XTB


DEFAULT_MOBILE = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="xTB finite-difference vibrational analysis for a periodic TS/minimum with an active region."
    )
    p.add_argument("xyz", help="Input structure file")
    p.add_argument("--a", type=float, required=True)
    p.add_argument("--b", type=float, required=True)
    p.add_argument("--c", type=float, required=True)
    p.add_argument("--alpha", type=float, required=True)
    p.add_argument("--beta", type=float, required=True)
    p.add_argument("--gamma", type=float, required=True)

    p.add_argument("--method", default="GFN0-xTB")
    p.add_argument("--accuracy", type=float, default=1.0)
    p.add_argument("--electronic-temperature", type=float, default=300.0)

    p.add_argument(
        "--mobile-indices",
        type=int,
        nargs="+",
        required=True,
        help="0-based ASE indices to keep mobile in the vibrational calculation",
    )
    p.add_argument(
        "--pre-relax-steps",
        type=int,
        default=0,
        help="Optional constrained pre-relaxation before frequencies",
    )
    p.add_argument(
        "--pre-relax-fmax",
        type=float,
        default=0.05,
        help="Force threshold for optional constrained pre-relaxation",
    )
    p.add_argument(
        "--delta",
        type=float,
        default=0.01,
        help="Finite-difference displacement in Angstrom",
    )
    p.add_argument(
        "--name",
        default="vib_active_region",
        help="Prefix for vibration files/folder",
    )
    p.add_argument(
        "--save-relaxed",
        default="freq_relaxed.xyz",
        help="Output file for the constrained-relaxed structure",
    )
    p.add_argument(
        "--save-modes",
        action="store_true",
        help="Write normal mode trajectories for all modes",
    )
    return p.parse_args()


def make_calc(args: argparse.Namespace) -> XTB:
    return XTB(
        method=args.method,
        accuracy=args.accuracy,
        electronic_temperature=args.electronic_temperature,
    )


def prepare_atoms(args: argparse.Namespace):
    atoms = read(args.xyz)
    cell = cellpar_to_cell([args.a, args.b, args.c, args.alpha, args.beta, args.gamma])
    atoms.set_cell(cell)
    atoms.set_pbc(True)

    natoms = len(atoms)
    mobile = sorted(set(args.mobile_indices))

    bad = [i for i in mobile if i < 0 or i >= natoms]
    if bad:
        raise ValueError(f"Mobile indices out of range for {natoms} atoms: {bad}")

    frozen = [i for i in range(natoms) if i not in mobile]
    atoms.set_constraint(FixAtoms(indices=frozen))
    atoms.calc = make_calc(args)
    return atoms, mobile, frozen


def run_prerelax(atoms, args: argparse.Namespace) -> None:
    if args.pre_relax_steps <= 0:
        return
    opt = BFGS(atoms, logfile=f"{args.name}_prerelax.log")
    opt.run(fmax=args.pre_relax_fmax, steps=args.pre_relax_steps)


def summarize_frequencies(vib: Vibrations, out_prefix: str) -> None:
    energies = vib.get_energies()
    freqs_cm = vib.get_frequencies()

    imag = []
    for i, f in enumerate(freqs_cm):
        f_real = float(np.real(f))
        f_imag = float(np.imag(f))
        if abs(f_imag) > 1e-8 or f_real < 0.0:
            imag.append((i, f_real, f_imag))

    zpe_ev = 0.5 * float(np.sum(np.real(energies)[np.real(energies) > 0.0]))

    summary_path = Path(f"{out_prefix}_summary.txt")
    with summary_path.open("w") as fh:
        fh.write("Vibrational frequency summary\n")
        fh.write(f"Number of modes returned: {len(freqs_cm)}\n")
        fh.write(f"Number of imaginary modes: {len(imag)}\n")
        fh.write(f"Zero-point energy (eV): {zpe_ev:.10f}\n\n")

        fh.write("Frequencies (cm^-1)\n")
        for i, f in enumerate(freqs_cm):
            fh.write(f"{i:3d}  {f}\n")

        fh.write("\nImaginary modes\n")
        if imag:
            for i, f_real, f_imag in imag:
                if abs(f_imag) > 1e-8:
                    fh.write(f"{i:3d}  {f_real:.6f}{f_imag:+.6f}j cm^-1\n")
                else:
                    fh.write(f"{i:3d}  {f_real:.6f} cm^-1\n")
        else:
            fh.write("None\n")

    print(f"Number of modes returned: {len(freqs_cm)}")
    print(f"Number of imaginary modes: {len(imag)}")
    print(f"Zero-point energy (eV): {zpe_ev:.10f}")
    print(f"Saved summary: {summary_path}")

    if len(imag) == 1:
        print("TS check: exactly one imaginary mode found.")
    elif len(imag) == 0:
        print("TS check: no imaginary modes found.")
    else:
        print("TS check: more than one imaginary mode found.")


def main() -> None:
    args = parse_args()
    atoms, mobile, frozen = prepare_atoms(args)

    print(f"Loaded structure: {args.xyz}")
    print(f"Total atoms: {len(atoms)}")
    print(f"Mobile atoms ({len(mobile)}): {mobile}")
    print(f"Frozen atoms ({len(frozen)}): {frozen}")

    run_prerelax(atoms, args)
    write(args.save_relaxed, atoms)
    print(f"Saved constrained-relaxed geometry: {args.save_relaxed}")

    vib = Vibrations(
        atoms,
        indices=mobile,
        name=args.name,
        delta=args.delta,
    )

    print("Running vibrational analysis...")
    vib.run()
    vib.summary(log=f"{args.name}_ase_summary.txt")
    summarize_frequencies(vib, args.name)

    if args.save_modes:
        freqs_cm = vib.get_frequencies()
        for i in range(len(freqs_cm)):
            try:
                vib.write_mode(i)
            except Exception as exc:
                print(f"Could not write mode {i}: {exc}")

    print(f"ASE summary: {args.name}_ase_summary.txt")
    print("Done.")


if __name__ == "__main__":
    main()
