#!/usr/bin/env python3
"""
Animate plotting of (x, y) points in order at a fixed interval.

Usage:
  python animate_points.py points.csv --interval 0.5

Notes:
- CSV must have headers: x,y
- Interval is in seconds (default 0.5s).
- Aspect ratio is kept equal; limits auto-fit to data with small margins.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def read_points(path: Path) -> pd.DataFrame:
    """
    Reads a CSV with columns x,y. Handles comma or tab/space delimiters.
    """
    # Try comma first; if it fails, fall back to whitespace
    try:
        df = pd.read_csv(path)
        if not {"x", "y"}.issubset(df.columns):
            raise ValueError("CSV missing 'x' and 'y' headers.")
    except Exception:
        df = pd.read_csv(path, sep=r"\s+|\t|,", engine="python")
        if not {"x", "y"}.issubset(df.columns):
            raise ValueError("CSV missing 'x' and 'y' headers.")

    # Ensure numeric
    df = df[["x", "y"]].astype(float).reset_index(drop=True)
    return df

def compute_limits(df: pd.DataFrame, pad_ratio: float = 0.05):
    x_min, x_max = df["x"].min(), df["x"].max()
    y_min, y_max = df["y"].min(), df["y"].max()

    x_rng = x_max - x_min
    y_rng = y_max - y_min
    # Prevent zero range
    if x_rng == 0: x_rng = 1.0
    if y_rng == 0: y_rng = 1.0

    x_pad = x_rng * pad_ratio
    y_pad = y_rng * pad_ratio
    return (x_min - x_pad, x_max + x_pad, y_min - y_pad, y_max + y_pad)

def animate(df: pd.DataFrame, interval_s: float):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_title("Animating Path")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.4)
    ax.set_aspect("equal", adjustable="box")

    xlim_min, xlim_max, ylim_min, ylim_max = compute_limits(df)
    ax.set_xlim(xlim_min, xlim_max)
    ax.set_ylim(ylim_min, ylim_max)

    # Line for the path-so-far and a marker for the current point
    line, = ax.plot([], [], linewidth=1.8)
    current, = ax.plot([], [], marker="o", markersize=5)

    # Pre-extract arrays for speed
    xs = df["x"].to_numpy()
    ys = df["y"].to_numpy()

    def init():
        line.set_data([], [])
        current.set_data([], [])
        return line, current

    def update(i):
        # Draw up to index i (inclusive)
        line.set_data(xs[:i+1], ys[:i+1])
        current.set_data(xs[i], ys[i])
        return line, current

    # Convert seconds to ms for FuncAnimation
    ani = FuncAnimation(
        fig,
        update,
        frames=len(df),
        init_func=init,
        interval=max(1, int(interval_s * 1000)),
        blit=False,  # safer across backends
        repeat=False
    )

    plt.tight_layout()
    plt.show()

def main(argv=None):
    parser = argparse.ArgumentParser(description="Animate plotting of points.")
    parser.add_argument("csv", type=Path, help="pathpoints.csv")
    parser.add_argument("--interval", type=float, default=0.5,
                        help="Seconds between points (default: 0.5)")
    args = parser.parse_args(argv)

    df = read_points(args.csv)
    animate(df, args.interval)

if __name__ == "__main__":
    main()
