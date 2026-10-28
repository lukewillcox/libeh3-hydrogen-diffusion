from __future__ import annotations

import argparse
import numpy as np

from ase.io import read, Trajectory
from ase import units
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import (
    MaxwellBoltzmannDistribution,
    Stationary,
    ZeroRotation,
)
from xtb.ase.calculator import XTB


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--traj", required=True, help="Input trajectory (e.g. md_340K.traj)")
    p.add_argument("--frame", type=int, required=True, help="Restart from this frame index")
    p.add_argument("--out", default="restart_md.traj", help="Output trajectory")
    p.add_argument("--log", default="restart_md.log", help="Output log")
    p.add_argument("--temperature", type=float, default=340.0, help="Target temperature (K)")
    p.add_argument("--timestep-fs", type=float, default=0.25, help="MD timestep (fs)")
    p.add_argument("--steps", type=int, default=50000, help="How many MD steps to run")
    p.add_argument("--save-interval", type=int, default=10, help="Write every N steps")
    p.add_argument("--friction", type=float, default=0.05, help="Langevin friction (1/fs)")
    p.add_argument("--seed", type=int, default=0, help="Random seed for velocities")
    return p.parse_args()


def main():
    args = parse_args()

    # Read a single frame as an Atoms object
    atoms = read(args.traj, index=args.frame)

    # Attach calculator (GFN0-xTB per your workflow)
    atoms.calc = XTB(method="GFN0-xTB", accuracy=1.0, electronic_temperature=300.0)

    # Set seed so the velocities are reproducible
    np.random.seed(args.seed)

    # Assign new velocities (new trajectory)
    MaxwellBoltzmannDistribution(atoms, temperature_K=args.temperature)
    Stationary(atoms)
    ZeroRotation(atoms)

    dt = args.timestep_fs * units.fs

    dyn = Langevin(
        atoms,
        timestep=dt,
        temperature_K=args.temperature,
        friction=args.friction / units.fs,
    )

    # Write trajectory manually so we control interval
    traj = Trajectory(args.out, "w", atoms)
    dyn.attach(traj.write, interval=args.save_interval)

    # Also print a simple status
    def status():
        epot = float(atoms.get_potential_energy())
        tinst = float(atoms.get_temperature())
        print(f"step={dyn.nsteps:7d}  T={tinst:7.2f} K  Epot={epot: .6f} eV")

    dyn.attach(status, interval=args.save_interval)

    print(f"Restarting from frame {args.frame}, seed={args.seed}")
    dyn.run(args.steps)
    traj.close()
    print("Done. Wrote:", args.out)


if __name__ == "__main__":
    main()