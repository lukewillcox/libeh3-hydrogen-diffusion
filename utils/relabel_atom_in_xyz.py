#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


def relabel_xyz(inp: Path, out: Path, atom_index: int, new_symbol: str) -> None:
    """
    Relabel the element symbol of atom `atom_index` (0-based) in EVERY frame of an XYZ file.
    Preserves all extra columns (forces/charges/etc).
    Works for standard multi-frame XYZ: [natoms][comment][natoms lines] repeated.
    """
    lines = inp.read_text().splitlines()

    i = 0
    out_lines: list[str] = []
    frame = 0

    while i < len(lines):
        # natoms line
        if i >= len(lines):
            break
        natoms_line = lines[i].strip()
        if not natoms_line:
            i += 1
            continue

        try:
            natoms = int(natoms_line.split()[0])
        except Exception:
            raise ValueError(f"Bad XYZ natoms line at line {i+1}: {lines[i]}")

        if atom_index < 0 or atom_index >= natoms:
            raise ValueError(f"atom_index {atom_index} out of range (0..{natoms-1}) in frame {frame}")

        # header
        out_lines.append(lines[i])       # natoms
        out_lines.append(lines[i + 1])   # comment

        # atom lines
        start = i + 2
        end = start + natoms
        if end > len(lines):
            raise ValueError(f"Truncated XYZ: frame {frame} expected {natoms} atoms")

        for k in range(natoms):
            line = lines[start + k]
            parts = line.split()
            if not parts:
                out_lines.append(line)
                continue

            if k == atom_index:
                parts[0] = new_symbol
                out_lines.append(" ".join(parts))
            else:
                out_lines.append(line)

        # advance
        i = end
        frame += 1

    out.write_text("\n".join(out_lines) + "\n")
    print(f"Wrote: {out}  (frames={frame})")


def main():
    if len(sys.argv) != 5:
        print("Usage: python relabel_atom_in_xyz.py INPUT.xyz OUTPUT.xyz ATOM_INDEX NEW_SYMBOL")
        print("Example: python relabel_atom_in_xyz.py neb_image_00_climb.xyz neb_image_00_climb_H32asN.xyz 32 N")
        sys.exit(2)

    inp = Path(sys.argv[1]).expanduser().resolve()
    out = Path(sys.argv[2]).expanduser().resolve()
    atom_index = int(sys.argv[3])
    new_symbol = sys.argv[4]

    if not inp.exists():
        raise FileNotFoundError(inp)

    relabel_xyz(inp, out, atom_index, new_symbol)


if __name__ == "__main__":
    main()