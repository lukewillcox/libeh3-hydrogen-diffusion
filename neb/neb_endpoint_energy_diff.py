from __future__ import annotations

from pathlib import Path

from ase.geometry import cellpar_to_cell
from ase.io import read
from xtb.ase.calculator import XTB


CELL = cellpar_to_cell([6.0138, 6.1251, 13.1242, 90.0, 82.3646, 90.0])

TARGET_FOLDERS = [
    "neb_1947_2045_H28",
    "neb_1460_1324_H24",
    "neb_1717_2117_H32_CLIMB",
    "neb_1794_2149_RND",
]


def attach_xtb(atoms) -> None:
    atoms.set_cell(CELL)
    atoms.set_pbc(True)
    atoms.calc = XTB(
        method="GFN0-xTB",
        accuracy=1.0,
        electronic_temperature=300.0,
    )


def find_last_image(folder: Path) -> Path | None:
    images = sorted(folder.glob("neb_image_*_climb.xyz"))
    return images[-1] if images else None


def energy_and_kinetic(path: Path) -> tuple[float, float]:
    atoms = read(path)
    attach_xtb(atoms)
    epot = float(atoms.get_potential_energy())
    ekin = float(atoms.get_kinetic_energy())
    return epot, ekin


def main() -> None:
    base = Path(".")

    print(
        f"{'Folder':<35} {'First image':<22} {'Last image':<22} "
        f"{'ΔEpot (eV)':<16} {'ΔEkin (eV)':<16}"
    )
    print("-" * 115)

    for folder_name in TARGET_FOLDERS:
        folder = base / folder_name
        first = folder / "neb_image_00_climb.xyz"
        last = find_last_image(folder)

        if not folder.exists():
            print(f"{folder_name:<35} {'MISSING FOLDER':<22}")
            continue

        if not first.exists() or last is None:
            print(f"{folder_name:<35} {'MISSING IMAGES':<22}")
            continue

        try:
            epot_first, ekin_first = energy_and_kinetic(first)
            epot_last, ekin_last = energy_and_kinetic(last)

            print(
                f"{folder.name:<35} "
                f"{first.name:<22} "
                f"{last.name:<22} "
                f"{epot_last - epot_first:<16.10f} "
                f"{ekin_last - ekin_first:<16.10f}"
            )
        except Exception as exc:
            print(
                f"{folder.name:<35} "
                f"{'ERROR':<22} "
                f"{'ERROR':<22} "
                f"{str(exc):<16}"
            )

    print("\nNote: for standard NEB xyz images, kinetic energies are usually zero/not meaningful.")


if __name__ == "__main__":
    main()