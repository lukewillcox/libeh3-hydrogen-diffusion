# refine_ts_dimer_xtb.py
from __future__ import annotations

import argparse

from ase.constraints import FixAtoms
from ase.geometry import cellpar_to_cell
from ase.io import read, write
from ase.mep import DimerControl, MinModeAtoms, MinModeTranslate
from xtb.ase.calculator import XTB


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("xyz", help="Starting TS guess, e.g. a top NEB image")
    p.add_argument("--a", type=float, required=True)
    p.add_argument("--b", type=float, required=True)
    p.add_argument("--c", type=float, required=True)
    p.add_argument("--alpha", type=float, required=True)
    p.add_argument("--beta", type=float, required=True)
    p.add_argument("--gamma", type=float, required=True)

    p.add_argument("--mobile-indices", type=int, nargs="+", required=True)
    p.add_argument("--method", default="GFN0-xTB")
    p.add_argument("--accuracy", type=float, default=1.0)
    p.add_argument("--electronic-temperature", type=float, default=300.0)

    p.add_argument("--fmax", type=float, default=0.05)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--trial-angle", type=float, default=0.78539816339)
    p.add_argument("--trial-trans-step", type=float, default=0.01)
    p.add_argument("--maximum-translation", type=float, default=0.1)
    p.add_argument("--cg-translation", action="store_true")
    p.add_argument("--name", default="ts_dimer")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    atoms = read(args.xyz)
    cell = cellpar_to_cell([args.a, args.b, args.c, args.alpha, args.beta, args.gamma])
    atoms.set_cell(cell)
    atoms.set_pbc(True)

    natoms = len(atoms)
    mobile = sorted(set(args.mobile_indices))
    frozen = [i for i in range(natoms) if i not in mobile]
    atoms.set_constraint(FixAtoms(indices=frozen))

    atoms.calc = XTB(
        method=args.method,
        accuracy=args.accuracy,
        electronic_temperature=args.electronic_temperature,
    )

    with DimerControl(
        initial_eigenmode_method="gauss",
        displacement_method="vector",
        trial_angle=args.trial_angle,
        trial_trans_step=args.trial_trans_step,
        maximum_translation=args.maximum_translation,
        cg_translation=args.cg_translation,
        logfile=f"{args.name}_dimer.log",
    ) as d_control:
        d_atoms = MinModeAtoms(atoms, d_control)
        d_atoms.displace(displacement_vector=[[0.0, 0.0, 0.0] for _ in range(natoms)])

        opt = MinModeTranslate(
            d_atoms,
            trajectory=f"{args.name}_dimer.traj",
            logfile=f"{args.name}_translate.log",
        )
        opt.run(fmax=args.fmax, steps=args.steps)

    write(f"{args.name}_refined.xyz", atoms)
    print(f"Wrote {args.name}_refined.xyz")
    print(f"Final energy: {atoms.get_potential_energy():.10f} eV")


if __name__ == "__main__":
    main()