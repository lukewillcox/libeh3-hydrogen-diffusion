from __future__ import annotations

import glob
import numpy as np
from ase.io import read, write
from ase.calculators.singlepoint import SinglePointCalculator
from xtb.ase.calculator import XTB


def main():
    files = sorted(glob.glob("neb_image_*.xyz"))
    if not files:
        raise FileNotFoundError("No neb_image_*.xyz found in current folder.")

    images = [read(f) for f in files]

    # ensure consistent PBC/cell
    cell = images[0].cell
    for a in images:
        a.set_pbc(True)
        a.set_cell(cell)

    # xTB calculator (singlepoints for energy + forces)
    calc = XTB(method="GFN0-xTB", accuracy=1.0, electronic_temperature=300.0)

    energies = []
    for i, a in enumerate(images):
        a.calc = calc
        e = float(a.get_potential_energy())
        f = a.get_forces()
        energies.append(e)

        # freeze results into file (so ASE can read energy/forces later without re-running xTB)
        a.calc = SinglePointCalculator(a, energy=e, forces=f)

        print(f"image {i:02d}: E = {e:.8f} eV, fmax = {np.linalg.norm(f, axis=1).max():.3f} eV/Å")

    energies = np.array(energies)
    e_rel = energies - energies[0]

    # write outputs
    write("neb_with_energy.extxyz", images)     # contains energy/forces in extxyz metadata
    write("neb_with_energy.traj", images)       # ASE trajectory with stored results

    np.savetxt("neb_with_energy_Erel.txt", np.c_[np.arange(len(images)), e_rel],
               header="image_index  E_rel_eV")

    print("Wrote: neb_with_energy.extxyz")
    print("Wrote: neb_with_energy.traj")
    print("Wrote: neb_with_energy_Erel.txt")


if __name__ == "__main__":
    main()
