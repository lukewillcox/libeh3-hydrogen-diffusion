# xtb_md_340K.py
from __future__ import annotations

import argparse
import numpy as np
from ase import units
from ase.calculators.calculator import CalculationFailed
from ase.geometry import cellpar_to_cell
from ase.io import read, write
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import (
    MaxwellBoltzmannDistribution,
    Stationary,
    ZeroRotation,
)
from ase.neighborlist import neighbor_list
from ase.optimize import BFGS
from xtb.ase.calculator import XTB


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    # input / cell
    p.add_argument("xyz", help="Input xyz file with Cartesian coordinates")
    p.add_argument("--a", type=float, required=True)
    p.add_argument("--b", type=float, required=True)
    p.add_argument("--c", type=float, required=True)
    p.add_argument("--alpha", type=float, required=True)
    p.add_argument("--beta", type=float, required=True)
    p.add_argument("--gamma", type=float, required=True)

    # MD
    p.add_argument("--temperature", type=float, default=340.0, help="Target T (K)")
    p.add_argument("--timestep", type=float, default=0.25, help="fs (0.25 safer than 0.5)")
    p.add_argument("--steps", type=int, default=150000)
    p.add_argument("--loginterval", type=int, default=10, help="Write traj/log every N steps")
    p.add_argument("--traj", default="md_340K.traj")
    p.add_argument("--log", default="md_340K.log")

    # Langevin coupling (units: 1/fs)
    p.add_argument("--friction", type=float, default=0.05, help="1/fs (0.03–0.10 recommended)")

    # Temperature ramp
    p.add_argument("--t0", type=float, default=100.0, help="Start temp (K)")
    p.add_argument("--ramp-steps", type=int, default=20000, help="Steps to ramp t0 -> target")

    # xTB
    p.add_argument("--method", default="GFN0-xTB")
    p.add_argument("--accuracy", type=float, default=1.0)
    p.add_argument("--electronic-temperature", type=float, default=300.0)

    # pre-relax
    p.add_argument("--pre-relax-steps", type=int, default=100, help="0 disables")
    p.add_argument("--pre-relax-fmax", type=float, default=0.2, help="eV/Å")

    # geometry guard
    p.add_argument("--min-dist", type=float, default=0.7, help="Å: abort if any pair closer than this")

    # ✅ ONLY CHANGE: reproducible initial velocities
    p.add_argument("--seed", type=int, default=None, help="Random seed for initial velocities")

    return p.parse_args()


def min_pair_distance(atoms, cutoff: float = 3.0) -> float:
    i, j, d = neighbor_list("ijd", atoms, cutoff=cutoff)
    if len(d) == 0:
        return float("inf")
    return float(np.min(d))


def main() -> None:
    args = parse_args()

    atoms = read(args.xyz)
    cell = cellpar_to_cell([args.a, args.b, args.c, args.alpha, args.beta, args.gamma])
    atoms.set_cell(cell)
    atoms.set_pbc(True)

    atoms.calc = XTB(
        method=args.method,
        accuracy=args.accuracy,
        electronic_temperature=args.electronic_temperature,
    )

    if args.pre_relax_steps > 0:
        opt = BFGS(atoms, logfile="pre_relax.log")
        opt.run(fmax=args.pre_relax_fmax, steps=args.pre_relax_steps)

    # ✅ seed affects MaxwellBoltzmannDistribution
    rng = None
    if args.seed is not None:
        try:
            rng = np.random.default_rng(args.seed)  # newer ASE accepts rng=
        except Exception:
            rng = None
        np.random.seed(args.seed)  # fallback for older ASE

    if rng is not None:
        MaxwellBoltzmannDistribution(atoms, temperature_K=args.t0, rng=rng)
    else:
        MaxwellBoltzmannDistribution(atoms, temperature_K=args.t0)

    Stationary(atoms)
    ZeroRotation(atoms)

    dt = args.timestep * units.fs
    dyn = Langevin(
        atoms,
        timestep=dt,
        temperature_K=args.t0,
        friction=args.friction / units.fs,
        trajectory=args.traj,
        logfile=args.log,
        loginterval=args.loginterval,
    )

    ramp_steps = max(1, int(args.ramp_steps))

    def update_target_temperature() -> None:
        step = dyn.nsteps
        if step >= ramp_steps:
            tset = args.temperature
        else:
            frac = step / ramp_steps
            tset = args.t0 + frac * (args.temperature - args.t0)
        dyn.set_temperature(temperature_K=tset)

    def guard_geometry() -> None:
        dmin = min_pair_distance(atoms, cutoff=3.0)
        if dmin < args.min_dist:
            fn = f"FAILED_step_{dyn.nsteps}.xyz"
            write(fn, atoms)
            raise RuntimeError(f"Min distance {dmin:.3f} Å < {args.min_dist:.3f} Å. Wrote {fn}")

    def print_status() -> None:
        epot = float(atoms.get_potential_energy())
        ekin = float(atoms.get_kinetic_energy())
        tinst = float(atoms.get_temperature())
        print(
            f"step={dyn.nsteps:7d} "
            f"T_target={dyn.temp/units.kB:7.2f} K  T_inst={tinst:7.2f} K  "
            f"E_pot={epot: .6f} eV  E_kin={ekin: .6f} eV  E_tot={epot+ekin: .6f} eV"
        )

    dyn.attach(update_target_temperature, interval=1)
    dyn.attach(guard_geometry, interval=args.loginterval)
    dyn.attach(print_status, interval=args.loginterval)

    print("Starting MD...")
    try:
        dyn.run(args.steps)
    except (CalculationFailed, RuntimeError) as exc:
        fn = f"FAILED_last_{dyn.nsteps}.xyz"
        write(fn, atoms)
        print(f"\nMD failed at step {dyn.nsteps}. Saved {fn}")
        raise

    write("final.xyz", atoms)
    print("Done. Wrote final.xyz")


if __name__ == "__main__":
    main()