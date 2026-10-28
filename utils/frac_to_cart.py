"""
Convert fractional coordinates (paper 'internal positions xyz') to Cartesian coordinates (Å)
for α-LiBeH3 (CaSiO3_1-type, P21/c) using the lattice parameters.

Outputs:
- prints Cartesian coordinates table to terminal
- writes LiBeH3_cart.xyz
- writes LiBeH3_cart.txt
"""

from __future__ import annotations

import numpy as np
from ase import Atoms
from ase.io import write
from ase.geometry import find_mic


def monoclinic_cell(a: float, b: float, c: float, beta_deg: float) -> np.ndarray:
    beta = np.deg2rad(beta_deg)
    return np.array(
        [
            [a, 0.0, 0.0],
            [0.0, b, 0.0],
            [c * np.cos(beta), 0.0, c * np.sin(beta)],
        ],
        dtype=float,
    )


def frac_to_cart(frac: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """
    ASE convention: cell is (3,3) with ROWS as lattice vectors.
    Fractional coords (N,3) -> Cartesian (N,3): cart = frac @ cell
    """
    frac = np.asarray(frac, dtype=float)
    cell = np.asarray(cell, dtype=float)
    if frac.ndim != 2 or frac.shape[1] != 3:
        raise ValueError("frac must be shape (N,3)")
    if cell.shape != (3, 3):
        raise ValueError("cell must be shape (3,3)")
    return frac @ cell


def min_mic_distance(atoms: Atoms) -> tuple[float, tuple[int, int]]:
    pos = atoms.get_positions()
    cell = atoms.get_cell()
    pbc = atoms.get_pbc()

    best_d = 1e9
    best_pair = (-1, -1)
    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            v = pos[j] - pos[i]
            _, d = find_mic(v, cell=cell, pbc=pbc)
            d = float(d)
            if d < best_d:
                best_d = d
                best_pair = (i, j)
    return best_d, best_pair


def main() -> None:
    # Lattice parameters (Å, degrees)
    a, b, c, beta = 6.0138, 6.1251, 13.1242, 82.3646
    cell = monoclinic_cell(a, b, c, beta)

    # Fractional coordinates from the paper table (x, y, z)
    data = [
        ("Li", (0.01034, 0.11905, 0.24776)),
        ("Li", (0.23913, 0.88227, 0.40684)),
        ("Li", (0.25256, 0.37237, 0.39902)),
        ("Be", (0.22830, 0.59347, 0.58761)),
        ("Be", (0.22408, 0.16547, 0.59160)),
        ("Be", (0.56545, 0.11780, 0.31356)),
        ("H",  (0.33048, 0.11911, 0.30598)),
        ("H",  (0.29176, 0.61712, 0.28155)),
        ("H",  (0.02654, 0.64435, 0.65174)),
        ("H",  (0.02541, 0.12017, 0.65892)),
        ("H",  (0.22524, 0.62362, 0.48159)),
        ("H",  (0.19721, 0.13091, 0.48930)),
        ("H",  (0.42305, 0.69535, 0.61974)),
        ("H",  (0.42500, 0.06448, 0.61744)),
        ("H",  (0.29287, 0.38009, 0.61611)),
    ]

    symbols = [s for s, _ in data]
    frac = np.array([xyz for _, xyz in data], dtype=float)
    cart = frac_to_cart(frac, cell)

    atoms = Atoms(symbols=symbols, positions=cart, cell=cell, pbc=True)

    # Existing prints (cell + sanity)
    print("Built atoms:", len(atoms))
    print("Cell (Å):\n", atoms.get_cell().array)
    dmin, (i, j) = min_mic_distance(atoms)
    print("Min MIC distance (Å):", dmin, "pair:", (i, j), atoms[i].symbol, atoms[j].symbol)

    # >>> THIS IS WHAT YOU WANT: full Cartesian x,y,z <<<
    print("\nCartesian coordinates (Å):")
    for idx, (sym, xyz) in enumerate(zip(atoms.get_chemical_symbols(), atoms.get_positions())):
        x, y, z = xyz
        print(f"{idx:2d} {sym:2s}  {x:12.6f} {y:12.6f} {z:12.6f}")

    # Save for copy/paste / reuse
    np.savetxt("LiBeH3_cart.txt", atoms.get_positions(), fmt="%.8f")
    write("LiBeH3_cart.xyz", atoms)
    print("\nWrote LiBeH3_cart.txt and LiBeH3_cart.xyz in current directory.")


if __name__ == "__main__":
    main()