# file: plot_neb_compare.py
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def parse_energy_file(path: Path) -> list[float]:
    rel_energies: list[float] = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[0].isdigit():
            rel_energies.append(float(parts[2]))
    if not rel_energies:
        raise ValueError(f"No relative energies found in {path}")
    return rel_energies


def main() -> None:
    free_file = Path("neb_image_energies_climb.txt")
    frozen_file = Path("neb_image_energies_frozenLi_climb.txt")

    free = parse_energy_file(free_file)
    frozen = parse_energy_file(frozen_file)

    x_free = list(range(len(free)))
    x_frozen = list(range(len(frozen)))

    plt.figure()
    plt.plot(x_free, free, marker="o", label="Free Li")
    plt.plot(x_frozen, frozen, marker="o", label="Frozen Li (13,16,19)")
    plt.xlabel("NEB image")
    plt.ylabel("Relative energy (eV)")
    plt.title("NEB comparison: free vs frozen Li")
    plt.xticks(range(max(len(free), len(frozen))))
    plt.legend()
    plt.tight_layout()
    plt.savefig("neb_free_vs_frozenLi.png", dpi=300)
    plt.show()

    print("Saved: neb_free_vs_frozenLi.png")


if __name__ == "__main__":
    main()