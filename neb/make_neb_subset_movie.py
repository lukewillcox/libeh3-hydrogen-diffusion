from __future__ import annotations

import argparse
import glob
from ase.io import read, write
from ase import Atoms


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pattern", default="neb_image_*.xyz", help="Input NEB images glob")
    p.add_argument("--indices", required=True,
                   help="Comma-separated atom indices to keep, e.g. '1,8,32'")
    p.add_argument("--out", default="neb_subset_movie.xyz", help="Output multi-frame xyz")
    args = p.parse_args()

    files = sorted(glob.glob(args.pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {args.pattern} in current folder.")

    keep = [int(x) for x in args.indices.split(",")]

    frames = []
    for fn in files:
        a = read(fn)
        sub: Atoms = a[keep]
        sub.set_cell(a.cell)
        sub.set_pbc(a.pbc)
        frames.append(sub)

    write(args.out, frames)
    print(f"Wrote {args.out} with {len(frames)} frames and {len(keep)} atoms/frame.")


if __name__ == "__main__":
    main()
