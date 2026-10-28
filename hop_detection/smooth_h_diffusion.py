# file: smooth_h_diffusion.py
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply moving-window averaging to H diffusion data."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input data file with two columns: x y or x,y",
    )
    parser.add_argument(
        "--output",
        default="h_diffusion_smoothed.csv",
        help="Output CSV for smoothed data",
    )
    parser.add_argument(
        "--plot",
        default="h_diffusion_smoothed.png",
        help="Output plot filename",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=5,
        help="Centered moving-average window size (odd integer recommended)",
    )
    parser.add_argument(
        "--delimiter",
        default=None,
        help="Optional delimiter override, e.g. ',' or '\\t'. Default: auto-detect",
    )
    parser.add_argument(
        "--skiprows",
        type=int,
        default=0,
        help="Number of header rows to skip",
    )
    parser.add_argument(
        "--xlabel",
        default="x",
        help="X-axis label for plot",
    )
    parser.add_argument(
        "--ylabel",
        default="H diffusion",
        help="Y-axis label for plot",
    )
    parser.add_argument(
        "--title",
        default="H diffusion data with moving-window smoothing",
        help="Plot title",
    )
    return parser.parse_args()


def detect_delimiter(path: Path) -> str | None:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if "," in stripped:
                return ","
            return None
    return None


def load_two_column_data(
    path: Path,
    delimiter: str | None,
    skiprows: int,
) -> tuple[np.ndarray, np.ndarray]:
    use_delimiter = delimiter if delimiter is not None else detect_delimiter(path)

    data = np.loadtxt(
        path,
        delimiter=use_delimiter,
        comments="#",
        skiprows=skiprows,
        dtype=float,
    )

    if data.ndim == 1:
        if data.size != 2:
            raise ValueError(
                "Input file must contain exactly two columns of numeric data."
            )
        data = data.reshape(1, 2)

    if data.shape[1] < 2:
        raise ValueError("Input file must contain at least two columns.")

    x = data[:, 0]
    y = data[:, 1]

    order = np.argsort(x)
    return x[order], y[order]


def moving_average(y: np.ndarray, window: int) -> np.ndarray:
    if window < 1:
        raise ValueError("Window size must be >= 1.")
    if window > len(y):
        raise ValueError("Window size cannot be larger than the number of data points.")

    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(y, kernel, mode="same")


def edge_corrected_moving_average(y: np.ndarray, window: int) -> np.ndarray:
    if window < 1:
        raise ValueError("Window size must be >= 1.")
    if window > len(y):
        raise ValueError("Window size cannot be larger than the number of data points.")

    weights = np.ones(window, dtype=float)
    numerator = np.convolve(y, weights, mode="same")
    denominator = np.convolve(np.ones_like(y, dtype=float), weights, mode="same")
    return numerator / denominator


def save_csv(path: Path, x: np.ndarray, y_raw: np.ndarray, y_smooth: np.ndarray) -> None:
    header = "x,raw_y,smoothed_y"
    out = np.column_stack([x, y_raw, y_smooth])
    np.savetxt(path, out, delimiter=",", header=header, comments="", fmt="%.10f")


def make_plot(
    path: Path,
    x: np.ndarray,
    y_raw: np.ndarray,
    y_smooth: np.ndarray,
    xlabel: str,
    ylabel: str,
    title: str,
) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(x, y_raw, marker="o", linestyle="-", label="Raw")
    plt.plot(x, y_smooth, marker="s", linestyle="-", label="Smoothed")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    plot_path = Path(args.plot)

    x, y = load_two_column_data(
        path=input_path,
        delimiter=args.delimiter,
        skiprows=args.skiprows,
    )

    y_smooth = edge_corrected_moving_average(y, args.window)

    save_csv(output_path, x, y, y_smooth)
    make_plot(
        path=plot_path,
        x=x,
        y_raw=y,
        y_smooth=y_smooth,
        xlabel=args.xlabel,
        ylabel=args.ylabel,
        title=args.title,
    )

    print(f"Loaded {len(x)} points from: {input_path}")
    print(f"Window size: {args.window}")
    print(f"Saved smoothed data: {output_path}")
    print(f"Saved plot: {plot_path}")


if __name__ == "__main__":
    main()