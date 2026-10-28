"""
LiBeH3 MD using xTB in CLUSTER mode (no PBC) with a Langevin thermostat.

Key ideas
---------
- Build structure with its crystal cell (for reference), then convert to cluster:
  pbc=False + big vacuum cell + center()
- Use Langevin thermostat to improve SCC/SCF stability for forces during MD.
- Start gently: smaller timestep + lower temperature first, then increase later.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from ase import Atoms, units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
from ase.md.langevin import Langevin
from ase.visualize.plot import plot_atoms

from xtb.ase.calculator import XTB
from ase.io.trajectory import Trajectory


def build_atoms() -> Atoms:
    a = 6.0138
    b = 6.1251
    c = 13.1242
    beta_deg = 82.3646
    beta = np.deg2rad(beta_deg)

    cell = np.array(
        [
            [a, 0.0, 0.0],
            [0.0, b, 0.0],
            [c * np.cos(beta), 0.0, c * np.sin(beta)],
        ],
        dtype=float,
    )

    cartesian_positions = np.array(
        [
            # Li
            [0.494226, 0.729193, 3.222821],
            [2.147526, 5.403992, 5.292108],
            [2.214655, 2.280803, 5.190387],
            # Be
            [2.397623, 3.635063, 7.643535],
            [2.379202, 1.013520, 7.695436],
            [3.947288, 0.721537, 4.078737],
            # H
            [2.521007, 0.729561, 3.980138],
            [2.245552, 3.779922, 3.662356],
            [1.296108, 3.946708, 8.477727],
            [1.301833, 0.736053, 8.571123],
            [2.194343, 3.819735, 6.264444],
            [2.039221, 0.801837, 6.364734],
            [3.624839, 4.259088, 8.061476],
            [3.632555, 0.394946, 8.031558],
            [2.835632, 2.328089, 8.014258],
        ],
        dtype=float,
    )

    symbols = (["Li"] * 3) + (["Be"] * 3) + (["H"] * 9)

    return Atoms(
        symbols=symbols,
        positions=cartesian_positions,
        cell=cell,
        pbc=True,  # start as crystal; we will convert to cluster in main()
    )


def make_cluster(atoms: Atoms, vacuum_ang: float = 40.0) -> None:
    atoms.pbc = False
    atoms.set_cell([vacuum_ang, vacuum_ang, vacuum_ang])
    atoms.center()


def plot_structure(atoms: Atoms) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    plot_atoms(atoms, ax, rotation=("45x,45y,0z"), show_unit_cell=2, radii=0.75)
    ax.set_axis_off()
    plt.tight_layout()
    plt.show(block=False)


def run_md_and_collect_energies(
    atoms: Atoms,
    timestep_fs: float = 0.5,
    temperature_K: float = 300.0,
    friction: float = 0.02,
    steps_per_block: int = 10,
    blocks: int = 2000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K)
    Stationary(atoms)
    ZeroRotation(atoms)

    dyn = Langevin(
        atoms,
        timestep_fs * units.fs,
        temperature_K=temperature_K,
        friction=friction,
    )
    traj = Trajectory("md.traj", "w", atoms)
    dyn.attach(traj.write, interval=10)

    time_ps: list[float] = []
    epot: list[float] = []
    ekin: list[float] = []

    step = 0
    for _ in range(blocks):
        dyn.run(steps_per_block)
        step += steps_per_block

        time_ps.append(step * timestep_fs / 1000.0)
        epot.append(float(atoms.get_potential_energy()))
        ekin.append(float(atoms.get_kinetic_energy()))

    t = np.array(time_ps)
    ep = np.array(epot)
    ek = np.array(ekin)
    et = ep + ek
    return t, ep, ek, et


def plot_energies(time_ps: np.ndarray, epot: np.ndarray, ekin: np.ndarray, etot: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(time_ps, epot, label="Potential energy")
    ax.plot(time_ps, ekin, label="Kinetic energy")
    ax.plot(time_ps, etot, label="Total energy")
    ax.set_xlabel("Time (ps)")
    ax.set_ylabel("Energy (eV)")
    ax.legend(loc="best")
    ax.grid(True, linewidth=0.5, alpha=0.5)
    plt.tight_layout()
    plt.show()


def main() -> None:
    atoms = build_atoms()
    make_cluster(atoms, vacuum_ang=40.0)
    plot_structure(atoms)

    atoms.calc = XTB(method="GFN2-xTB", charge=0, uhf=0)

    print("Single-point Epot (eV):", float(atoms.get_potential_energy()))

    t, ep, ek, et = run_md_and_collect_energies(
        atoms,
        timestep_fs=0.25,
        temperature_K=300.0,
        friction=0.02,
        steps_per_block=10,
        blocks=50,
    )

    plot_energies(t, ep, ek, et)


if __name__ == "__main__":
    main()
