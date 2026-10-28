import matplotlib.pyplot as plt
import numpy as np
import os
from ase.io import read
from ase.geometry import cellpar_to_cell
from xtb.ase.calculator import XTB

cell = cellpar_to_cell([6.0138, 6.1251, 13.1242, 90.0, 82.3646, 90.0])

energies = []
i = 0
while True:
    fn = f"neb_image_{i:02d}_climb.xyz"
    if not os.path.exists(fn):
        break

    atoms = read(fn)
    atoms.set_cell(cell)
    atoms.set_pbc(True)
    atoms.calc = XTB(
        method="GFN0-xTB",
        accuracy=1.0,
        electronic_temperature=300.0,
    )
    energies.append(float(atoms.get_potential_energy()))
    i += 1

if not energies:
    raise RuntimeError("No neb_image_*_climb.xyz files found")

energies = np.array(energies)
energies -= energies[0]

images = list(range(len(energies)))

plt.figure()
plt.plot(images, energies, marker='o')
plt.xlabel("NEB Image")
plt.ylabel("Energy (eV, relative)")
plt.title("NEB Energy Profile")

folder_name = os.path.basename(os.getcwd())
filename = f"{folder_name}_neb_plot.png"

plt.tight_layout()
if not os.path.exists(filename):
    plt.savefig(filename, dpi=300)
    print(f"Saved plot as: {filename}")
else:
    print(f"Plot already exists: {filename}")

print(f"Barrier (eV): {energies.max():.10f}")
print(f"Endpoint ΔE (eV): {energies[-1]:.10f}")
plt.show()
