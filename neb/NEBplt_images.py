from __future__ import annotations

import glob
import numpy as np
import matplotlib.pyplot as plt
from ase.io import read
from ase.visualize.plot import plot_atoms


def load_neb_images():
    files = sorted(glob.glob("neb_image_*.xyz"))
    if files:
        return [read(f) for f in files]

    # fallback: last band from neb_stage2.traj
    try:
        traj = read("neb_stage2.traj", index=":")
    except Exception:
        traj = []
    if not traj:
        raise FileNotFoundError("No neb_image_*.xyz and cannot read neb_stage2.traj")

    # assume 9 images if you used n_images=9
    n_images = 9
    if len(traj) >= n_images:
        return traj[-n_images:]
    raise RuntimeError("neb_stage2.traj exists but too short.")


def pick_most_moving_H(images):
    """Pick the H with the largest displacement between endpoints."""
    a = images[0]
    b = images[-1]
    H = [i for i, at in enumerate(a) if at.symbol == "H"]
    disp = []
    for i in H:
        d = np.linalg.norm(b.positions[i] - a.positions[i])
        disp.append((d, i))
    disp.sort(reverse=True)
    return disp[0][1], disp[0][0]


def band_coordinate(images):
    """Cumulative RMS displacement between consecutive images (Å)."""
    s = [0.0]
    for i in range(1, len(images)):
        dr = images[i].get_positions() - images[i - 1].get_positions()
        step = float(np.sqrt((dr * dr).sum() / len(dr)))
        s.append(s[-1] + step)
    return np.array(s)


def main(h_index: int | None = None):
    images = load_neb_images()

    # energies
    E = np.array([img.get_potential_energy() for img in images], float)
    Erel = E - E[0]

    # x-axis: band coordinate OR just image index
    s = band_coordinate(images)

    # choose H index
    if h_index is None:
        h_index, d = pick_most_moving_H(images)
        print(f"Auto-picked H index {h_index} (endpoint displacement ~ {d:.3f} Å)")
    else:
        print(f"Using user H index {h_index}")

    # collect H positions along band
    hpos = np.array([img.positions[h_index] for img in images])

    # ----- FIGURE -----
    fig = plt.figure(figsize=(7.5, 4.8))

    # main NEB curve
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(s, Erel, marker="o")
    ax.set_xlabel("Band coordinate s (Å)")
    ax.set_ylabel("Energy (eV, relative to image 0)")
    ax.set_title("NEB energy vs band coordinate")

    # mark highest point
    i_ts = int(np.argmax(Erel))
    ax.annotate(
        f"TS ~ image {i_ts}",
        xy=(s[i_ts], Erel[i_ts]),
        xytext=(s[i_ts], Erel[i_ts] + 0.1),
        arrowprops=dict(arrowstyle="->"),
    )

    # inset axes for diffusion “beads”
    inset = ax.inset_axes([0.58, 0.20, 0.38, 0.62])  # [x0,y0,w,h] in axes fraction

    # draw host lattice from a representative image (TS image looks nice)
    host = images[i_ts].copy()

    # To make the “host” clearer, you can hide most H by turning them into dummy
    # but the simplest is: keep all atoms and overlay a bead path.
    plot_atoms(host, inset, rotation=("0x,0y,0z"), show_unit_cell=1)

    # overlay H bead path (projected onto inset axes)
    # We just scatter in 2D using the same rotation as plot_atoms uses by default.
    # A reliable simple approach: scatter in X-Y projection.
    inset.scatter(hpos[:, 0], hpos[:, 1], s=35)  # no custom colors

    inset.set_title(f"H{h_index} path (proj.)", fontsize=9)
    inset.set_xticks([])
    inset.set_yticks([])

    fig.tight_layout()
    fig.savefig("neb_with_diffusion_inset.png", dpi=300)
    print("Wrote: neb_with_diffusion_inset.png")


if __name__ == "__main__":
    # Option A: hard-code the H index you care about (recommended)
    # main(h_index=51)

    # Option B: auto-pick the most-moving H
    main(h_index=None)