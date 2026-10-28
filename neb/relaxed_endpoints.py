# relax_pbc.py

from __future__ import annotations

from ase.io import read, write
from ase.optimize import BFGS
from xtb.ase.calculator import XTB


def attach_xtb(atoms) -> None:
    atoms.set_pbc(True)
    atoms.calc = XTB(
        method="GFN0-xTB",
        accuracy=1.0,
        electronic_temperature=300.0,
    )


def relax_one(name: str, fmax: float = 0.05, steps: int = 300) -> None:
    atoms = read(f"{name}.xyz")

    # Keep the periodic crystal setup exactly as provided
    atoms.set_pbc(True)

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
    relax_one("initial", fmax=0.05, steps=300)
    relax_one("final", fmax=0.05, steps=300)


if __name__ == "__main__":
    main()