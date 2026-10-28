from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from ase.io import read


@dataclass(frozen=True)
class SeedResult:
    seed: int
    status: str
    commit_frame: int | None
    commit_value: float | None
    n_frames: int
    q_min: float
    q_max: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=".")
    p.add_argument("--ts-xyz", default="../TS_image_05_climb.xyz")
    p.add_argument("--seed-min", type=int, default=1)
    p.add_argument("--seed-max", type=int, default=20)
    p.add_argument("--traj-template", default="md_seed{seed}.traj")
    p.add_argument("--plane", type=int, nargs=3, default=[24, 48, 40])
    p.add_argument("--flip", type=int, default=8, help="Be index")
    p.add_argument("--q-cut", type=float, default=0.20)
    p.add_argument("--consecutive", type=int, default=3)
    p.add_argument("--start-frame", type=int, default=1)
    p.add_argument("--dt-fs", type=float, default=0.25)
    p.add_argument("--save-interval", type=int, default=10)
    p.add_argument("--csv", default="committor_results.csv")
    p.add_argument("--summary", default="committor_summary.txt")
    p.add_argument("--plot", action="store_true")
    return p.parse_args()


def unit_normal(r0: np.ndarray, r1: np.ndarray, r2: np.ndarray) -> np.ndarray:
    n = np.cross(r1 - r0, r2 - r0)
    norm = np.linalg.norm(n)
    if norm < 1e-12:
        raise ValueError("Plane atoms are nearly collinear.")
    return n / norm


def signed_distance(atoms, plane: tuple[int, int, int], flip: int) -> float:
    pos = atoms.get_positions()
    r0, r1, r2 = pos[plane[0]], pos[plane[1]], pos[plane[2]]
    centroid = (r0 + r1 + r2) / 3.0
    n_hat = unit_normal(r0, r1, r2)
    return float(np.dot(n_hat, pos[flip] - centroid))


def signed_distance_series(frames, plane: tuple[int, int, int], flip: int) -> np.ndarray:
    return np.array([signed_distance(atoms, plane, flip) for atoms in frames], dtype=float)


def assign_commitment(
    q: np.ndarray, q_cut: float, consecutive: int, start_frame: int
) -> tuple[str, int | None, float | None]:
    pos_run = 0
    neg_run = 0

    for i in range(start_frame, len(q)):
        val = q[i]

        if val >= q_cut:
            pos_run += 1
        else:
            pos_run = 0

        if val <= -q_cut:
            neg_run += 1
        else:
            neg_run = 0

        if pos_run >= consecutive:
            j = i - consecutive + 1
            return "B", j, float(q[j])

        if neg_run >= consecutive:
            j = i - consecutive + 1
            return "A", j, float(q[j])

    return "unclassified", None, None


def analyze_seed(
    seed: int,
    root: Path,
    traj_template: str,
    plane: tuple[int, int, int],
    flip: int,
    s_ts: float,
    q_cut: float,
    consecutive: int,
    start_frame: int,
) -> SeedResult | None:
    traj_path = root / f"seed_{seed}" / traj_template.format(seed=seed)
    if not traj_path.exists():
        print(f"seed_{seed}: missing {traj_path.name}")
        return None

    try:
        frames = read(str(traj_path), index=":")
    except Exception as exc:
        print(f"seed_{seed}: failed to read trajectory: {exc}")
        return None

    if not frames:
        print(f"seed_{seed}: empty trajectory")
        return None

    s = signed_distance_series(frames, plane=plane, flip=flip)
    q = s - s_ts
    status, commit_frame, commit_value = assign_commitment(
        q=q, q_cut=q_cut, consecutive=consecutive, start_frame=start_frame
    )

    return SeedResult(
        seed=seed,
        status=status,
        commit_frame=commit_frame,
        commit_value=commit_value,
        n_frames=len(frames),
        q_min=float(np.min(q)),
        q_max=float(np.max(q)),
    )


def save_csv(path: Path, results: Iterable[SeedResult], dt_fs: float, save_interval: int) -> None:
    dt_ps_per_frame = dt_fs * save_interval / 1000.0
    fieldnames = [
        "seed",
        "status",
        "commit_frame",
        "commit_time_ps",
        "commit_value_A",
        "n_frames",
        "q_min_A",
        "q_max_A",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "seed": r.seed,
                    "status": r.status,
                    "commit_frame": r.commit_frame,
                    "commit_time_ps": None if r.commit_frame is None else r.commit_frame * dt_ps_per_frame,
                    "commit_value_A": r.commit_value,
                    "n_frames": r.n_frames,
                    "q_min_A": r.q_min,
                    "q_max_A": r.q_max,
                }
            )


def save_summary(
    path: Path,
    results: list[SeedResult],
    s_ts: float,
    q_cut: float,
    consecutive: int,
    start_frame: int,
) -> None:
    n_a = sum(r.status == "A" for r in results)
    n_b = sum(r.status == "B" for r in results)
    n_u = sum(r.status == "unclassified" for r in results)
    n_classified = n_a + n_b
    p_b = (n_b / n_classified) if n_classified else float("nan")

    lines = [
        f"s_TS = {s_ts:.6f} Å",
        f"q_cut = {q_cut:.3f} Å",
        f"consecutive = {consecutive}",
        f"start_frame = {start_frame}",
        f"A = {n_a}",
        f"B = {n_b}",
        f"unclassified = {n_u}",
        f"classified = {n_classified}",
        f"p_B = {p_b:.6f}" if n_classified else "p_B = NaN",
        "",
        "Per-seed results:",
    ]
    for r in results:
        lines.append(
            f"seed_{r.seed}: status={r.status}, commit_frame={r.commit_frame}, "
            f"commit_value={r.commit_value}, q_min={r.q_min:.4f}, q_max={r.q_max:.4f}"
        )

    path.write_text("\n".join(lines) + "\n")


def make_plot(
    seed: int,
    root: Path,
    traj_template: str,
    plane: tuple[int, int, int],
    flip: int,
    s_ts: float,
    q_cut: float,
    dt_fs: float,
    save_interval: int,
) -> None:
    traj_path = root / f"seed_{seed}" / traj_template.format(seed=seed)
    frames = read(str(traj_path), index=":")
    s = signed_distance_series(frames, plane=plane, flip=flip)
    q = s - s_ts
    dt_ps_per_frame = dt_fs * save_interval / 1000.0
    t = np.arange(len(q)) * dt_ps_per_frame

    plt.figure()
    plt.plot(t, q)
    plt.axhline(+q_cut, linestyle="--")
    plt.axhline(-q_cut, linestyle="--")
    plt.axhline(0.0, linestyle=":")
    plt.xlabel("Time (ps)")
    plt.ylabel("q = s - s_TS (Å)")
    plt.title(f"seed_{seed}")
    plt.tight_layout()
    out = root / f"seed_{seed}" / f"seed_{seed}_q_trace.png"
    plt.savefig(out, dpi=300)
    plt.close()


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    plane = tuple(args.plane)

    ts_atoms = read(args.ts_xyz)
    s_ts = signed_distance(ts_atoms, plane=plane, flip=args.flip)
    print(f"TS reference signed distance s_TS = {s_ts:.6f} Å")

    results: list[SeedResult] = []
    for seed in range(args.seed_min, args.seed_max + 1):
        r = analyze_seed(
            seed=seed,
            root=root,
            traj_template=args.traj_template,
            plane=plane,
            flip=args.flip,
            s_ts=s_ts,
            q_cut=args.q_cut,
            consecutive=args.consecutive,
            start_frame=args.start_frame,
        )
        if r is None:
            continue

        results.append(r)
        print(
            f"seed_{r.seed}: {r.status:12s} "
            f"commit_frame={r.commit_frame} commit_value={r.commit_value} "
            f"q_min={r.q_min:.4f} q_max={r.q_max:.4f}"
        )

        if args.plot:
            make_plot(
                seed=seed,
                root=root,
                traj_template=args.traj_template,
                plane=plane,
                flip=args.flip,
                s_ts=s_ts,
                q_cut=args.q_cut,
                dt_fs=args.dt_fs,
                save_interval=args.save_interval,
            )

    if not results:
        raise RuntimeError("No usable trajectories analyzed.")

    save_csv(Path(args.csv), results, dt_fs=args.dt_fs, save_interval=args.save_interval)
    save_summary(
        Path(args.summary),
        results,
        s_ts=s_ts,
        q_cut=args.q_cut,
        consecutive=args.consecutive,
        start_frame=args.start_frame,
    )

    n_a = sum(r.status == "A" for r in results)
    n_b = sum(r.status == "B" for r in results)
    n_u = sum(r.status == "unclassified" for r in results)
    n_classified = n_a + n_b
    p_b = (n_b / n_classified) if n_classified else float("nan")

    print("\nSummary")
    print(f"A = {n_a}")
    print(f"B = {n_b}")
    print(f"unclassified = {n_u}")
    print(f"classified = {n_classified}")
    print(f"p_B = {p_b:.6f}" if n_classified else "p_B = NaN")
    print(f"Saved CSV: {args.csv}")
    print(f"Saved summary: {args.summary}")


if __name__ == "__main__":
    main()