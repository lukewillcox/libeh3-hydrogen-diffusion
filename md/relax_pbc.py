# relax_pbc.py
from __future__ import annotations

from ase.geometry import cellpar_to_cell
from ase.io import read, write
from ase.optimize import BFGS
from xtb.ase.calculator import XTB


CELL = cellpar_to_cell([6.0138, 6.1251, 13.1242, 90.0, 82.3646, 90.0])


def attach_xtb(atoms) -> None:
    atoms.set_cell(CELL)
    atoms.set_pbc(True)
    atoms.calc = XTB(
        method="GFN0-xTB",
        accuracy=1.0,
        electronic_temperature=300.0,
    )


def relax_one(name: str, fmax: float = 0.01, steps: int = 300) -> None:
    atoms = read(f"{name}.xyz")
    attach_xtb(atoms)

    e0 = float(atoms.get_potential_energy())

    opt = BFGS(
        atoms,
        logfile=f"{name}_relax.log",
        trajectory=f"{name}_relax.traj",
    )
    opt.run(fmax=fmax, steps=steps)

    e1 = float(atoms.get_potential_energy())

    write(f"{name}_relaxed.xyz", atoms)

    print(f"{name}: E0={e0:.6f} eV  E1={e1:.6f} eV  dE={e1 - e0:.6f} eV")
    print(f"Wrote {name}_relaxed.xyz and {name}_relax.traj")


def main() -> None:
    relax_one("initial", fmax=0.01, steps=300)
    relax_one("final", fmax=0.01, steps=300)


if __name__ == "__main__":
    main()