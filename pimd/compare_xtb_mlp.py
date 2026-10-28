from __future__ import annotations

import glob
import numpy as np
import matplotlib.pyplot as plt
from ase.io import read
from xtb.ase.calculator import XTB

from upet.calculator import UPETCalculator  # pip install upet


def load_images(pattern="neb_image_*.xyz"):
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No {pattern} found in {__import__('os').getcwd()}")
    imgs = [read(f) for f in files]
    return files, imgs


def main():
    files, images = load_images("neb_image_*.xyz")
    print(f"Loaded {len(images)} NEB images")

    # --- UPET / PET-MAD calculator (pretrained; downloads on first use) ---
    upet = UPETCalculator(model="pet-mad-s", version="1.5.0", device="cpu")

    # --- xTB singlepoints on the SAME geometries (so you can compare directly) ---
    # (This avoids relying on energies being stored inside the xyz files.)
    xtb = XTB(method="GFN0-xTB", accuracy=1.0, electronic_temperature=300.0)

    E_upet = []
    E_xtb = []

    for i, img in enumerate(images):
        img.set_pbc(True)

        img.calc = upet
        e1 = float(img.get_potential_energy())
        E_upet.append(e1)

        img.calc = xtb
        e2 = float(img.get_potential_energy())
        E_xtb.append(e2)

        print(f"img {i:02d}  UPET={e1: .6f} eV   xTB={e2: .6f} eV")

    E_upet = np.array(E_upet)
    E_xtb = np.array(E_xtb)
    rel_upet = E_upet - E_upet[0]
    rel_xtb = E_xtb - E_xtb[0]

    # --- save table ---
    np.savetxt(
        "xtb_vs_upet.csv",
        np.column_stack([np.arange(len(images)), rel_xtb, rel_upet, rel_upet - rel_xtb]),
        delimiter=",",
        header="image,xtb_rel_eV,upet_rel_eV,delta_rel_eV(upet-xtb)",
        comments="",
    )
    print("Wrote xtb_vs_upet.csv")

    # --- plot ---
    x = np.arange(len(images))
    plt.figure()
    plt.plot(x, rel_xtb, marker="o", label="xTB rel")
    plt.plot(x, rel_upet, marker="o", label="UPET (PET-MAD) rel")
    plt.xlabel("NEB image index")
    plt.ylabel("Energy (eV, relative to image 0)")
    plt.title("NEB energies on identical geometries: xTB vs UPET(PET-MAD)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("xtb_vs_upet.png", dpi=300)
    print("Wrote xtb_vs_upet.png")
    plt.show()


if __name__ == "__main__":
    main()