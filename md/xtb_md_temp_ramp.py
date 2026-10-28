from __future__ import annotations

import argparse
import csv
from pathlib import Path

from ase import units
from ase.io import read, write
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.io.trajectory import Trajectory
from ase.geometry import cellpar_to_cell
from xtb.ase.calculator import XTB


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run xTB MD with a linear temperature ramp and log actual temperature."
    )
    parser.add_argument("xyz", help="Input structure file, e.g. structure.xyz")
    parser.add_argument("--a", type=float, required=True, help="Cell length a in Å")
    parser.add_argument("--b", type=float, required=True, help="Cell length b in Å")
    parser.add_argument("--c", type=float, required=True, help="Cell length c in Å")
    parser.add_argument("--alpha", type=float, required=True, help="Cell angle alpha in degrees")
    parser.add_argument("--beta", type=float, required=True, help="Cell angle beta in degrees")
    parser.add_argument("--gamma", type=float, required=True, help="Cell angle gamma in degrees")
    parser.add_argument("--start-temperature", type=float, default=275.0, help="Start temperature in K")
    parser.add_argument("--end-temperature", type=float, default=310.0, help="End temperature in K")
    parser.add_argument("--timestep", type=float, default=0.5, help="MD timestep in fs")
    parser.add_argument("--steps", type=int, default=30000, help="Number of MD steps")
    parser.add_argument("--friction", type=float, default=0.02, help="Langevin friction in 1/fs")
    parser.add_argument("--method", type=str, default="GFN0-xTB", help="xTB method, e.g. GFN0-xTB")
    parser.add_argument("--traj", type=str, default="md_ramp_275_310K.traj", help="Output ASE trajectory")
    parser.add_argument("--xyz-out", type=str, default="md_ramp_275_310K_final.xyz", help="Final structure output")
    parser.add_argument("--log", type=str, default="md_ramp_275_310K.log", help="Text log file")
    parser.add_argument("--temp-log", type=str, default="md_ramp_275_310K_temperature.csv", help="CSV temperature log")
    parser.add_argument("--loginterval", type=int, default=10, help="Logging interval in MD steps")
    return parser


def linear_target_temperature(step: int, nsteps: int, t_start: float, t_end: float) -> float:
    if nsteps <= 1:
        return t_end
    fraction = step / (nsteps - 1)
    return t_start + fraction * (t_end - t_start)


def set_langevin_temperature(dyn: Langevin, temperature_k: float) -> None:
    # ASE Langevin stores the target temperature internally in eV.
    dyn.set_temperature(temperature_K=temperature_k)


def attach_logger(
    dyn: Langevin,
    atoms,
    nsteps: int,
    t_start: float,
    t_end: float,
    loginterval: int,
    text_log_path: Path,
    temp_log_path: Path,
) -> None:
    text_log = open(text_log_path, "w", encoding="utf-8")
    temp_csv = open(temp_log_path, "w", newline="", encoding="utf-8")
    writer = csv.writer(temp_csv)
    writer.writerow(
        [
            "step",
            "time_fs",
            "target_temperature_K",
            "instantaneous_temperature_K",
            "potential_energy_eV",
            "kinetic_energy_eV",
            "total_energy_eV",
        ]
    )

    def log_status() -> None:
        step = dyn.nsteps
        target_k = linear_target_temperature(step, nsteps, t_start, t_end)
        set_langevin_temperature(dyn, target_k)

        time_fs = step * dyn.dt / units.fs
        temp_k = atoms.get_temperature()
        epot = float(atoms.get_potential_energy())
        ekin = float(atoms.get_kinetic_energy())
        etot = epot + ekin

        line = (
            f"step={step:7d}  "
            f"time_fs={time_fs:10.2f}  "
            f"T_target={target_k:8.2f} K  "
            f"T_inst={temp_k:8.2f} K  "
            f"E_pot={epot:12.6f} eV  "
            f"E_kin={ekin:10.6f} eV  "
            f"E_tot={etot:12.6f} eV"
        )
        print(line)
        text_log.write(line + "\n")
        text_log.flush()

        writer.writerow([step, f"{time_fs:.4f}", f"{target_k:.4f}", f"{temp_k:.4f}", f"{epot:.8f}", f"{ekin:.8f}", f"{etot:.8f}"])
        temp_csv.flush()

    dyn.attach(log_status, interval=loginterval)

    def close_logs() -> None:
        text_log.close()
        temp_csv.close()

    dyn._close_logs_callback = close_logs


def main() -> None:
    args = build_parser().parse_args()

    atoms = read(args.xyz)

    cell = cellpar_to_cell([args.a, args.b, args.c, args.alpha, args.beta, args.gamma])
    atoms.set_cell(cell)
    atoms.set_pbc(True)

    atoms.calc = XTB(
        method=args.method,
        accuracy=1.0,
        electronic_temperature=300.0,
    )

    MaxwellBoltzmannDistribution(atoms, temperature_K=args.start_temperature)
    Stationary(atoms)

    dyn = Langevin(
        atoms,
        timestep=args.timestep * units.fs,
        temperature_K=args.start_temperature,
        friction=args.friction / units.fs,
    )

    traj = Trajectory(args.traj, "w", atoms)
    dyn.attach(traj.write, interval=args.loginterval)

    attach_logger(
        dyn=dyn,
        atoms=atoms,
        nsteps=args.steps,
        t_start=args.start_temperature,
        t_end=args.end_temperature,
        loginterval=args.loginterval,
        text_log_path=Path(args.log),
        temp_log_path=Path(args.temp_log),
    )

    print("Starting ramped MD...")
    print(
        f"Temperature ramp: {args.start_temperature:.1f} K -> {args.end_temperature:.1f} K "
        f"over {args.steps} steps"
    )

    try:
        dyn.run(args.steps)
    finally:
        traj.close()
        if hasattr(dyn, "_close_logs_callback"):
            dyn._close_logs_callback()

    write(args.xyz_out, atoms)
    print(f"Wrote final structure to {args.xyz_out}")
    print(f"Wrote trajectory to {args.traj}")
    print(f"Wrote text log to {args.log}")
    print(f"Wrote temperature CSV to {args.temp_log}")


if __name__ == "__main__":
    main()