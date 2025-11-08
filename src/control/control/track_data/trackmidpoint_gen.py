# animate_xy_pair_midpoints.py
from pathlib import Path
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DELAY_SEC = 0.1
POINT_SIZE = 16
MIDPOINT_SIZE = 48  # larger + 'x' marker for visibility

VALID_TAGS = {"blue", "yellow"}
IGNORE_TAGS = {"orange", "big_orange"}  # explicit, though we already filter

def load_tagged_xy(csv_path: Path):
    """
    Reads CSV with at least: tag, x, y.
    Filters to only 'blue' and 'yellow'. Ignores others.
    Coerces x,y to numeric and drops NaNs.
    Returns two lists of (x, y) preserving file order: blues, yellows.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Basic guards
    for col in ("tag", "x", "y"):
        if col not in df.columns:
            raise ValueError(f"CSV must contain column '{col}' (found: {list(df.columns)})")

    # Filter to valid tags
    f = df["tag"].str.lower().isin(VALID_TAGS)
    df = df.loc[f, ["tag", "x", "y"]].copy()

    # Coerce x,y
    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df.dropna(subset=["x", "y"], inplace=True)

    # Split by tag, preserve order
    blues = df[df["tag"].str.lower() == "blue"][["x", "y"]].to_numpy()
    yellows = df[df["tag"].str.lower() == "yellow"][["x", "y"]].to_numpy()

    if len(blues) == 0 or len(yellows) == 0:
        print(f"Warning: insufficient points (blues={len(blues)}, yellows={len(yellows)}).")

    return blues, yellows

def pair_blues_to_nearest_yellows(blues: np.ndarray, yellows: np.ndarray):
    """
    Greedy matching: for each blue (in order), pick the nearest unused yellow.
    Returns:
      pairs: list of ((bx, by), (yx, yy)) for matched pairs (yellow not reused)
      unmatched_blues: indices of blues that could not find a yellow
    """
    if len(blues) == 0 or len(yellows) == 0:
        return [], list(range(len(blues)))

    used_y = np.zeros(len(yellows), dtype=bool)
    pairs = []
    unmatched = []

    Y = yellows  # shape (Ny, 2)
    for i, b in enumerate(blues):
        # distances to all unused yellows
        mask = ~used_y
        if not np.any(mask):
            unmatched.append(i)
            continue
        diffs = Y[mask] - b  # (k, 2)
        d2 = np.sum(diffs * diffs, axis=1)
        j_rel = int(np.argmin(d2))  # index among filtered
        # map back to absolute yellow index
        abs_candidates = np.flatnonzero(mask)
        j = abs_candidates[j_rel]
        used_y[j] = True
        pairs.append((tuple(b), tuple(Y[j])))

    return pairs, unmatched

def animate_pairs_and_midpoints(pairs, all_points_for_limits, delay_sec=DELAY_SEC):
    """
    Animate plotting:
      - points: blue as dots, yellow as dots (different scatter datasets)
      - midpoints: 'x' marker, larger size
      - one "frame" per pair: add blue, add yellow, add midpoint, pause
    Returns midpoints arrays (mx, my) for saving.
    """
    if len(pairs) == 0:
        return [], []

    # Axis limits from all relevant points for stable view
    pts = np.array(all_points_for_limits)
    x_min, x_max = float(np.min(pts[:,0])), float(np.max(pts[:,0]))
    y_min, y_max = float(np.min(pts[:,1])), float(np.max(pts[:,1]))
    pad_x = (x_max - x_min) * 0.05 or 1.0
    pad_y = (y_max - y_min) * 0.05 or 1.0

    plt.ion()
    fig, ax = plt.subplots()
    ax.set_title("Blue ↔ Nearest Yellow (live) + Midpoints")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect('equal')
    ax.set_xlim(x_min - pad_x, x_max + pad_x)
    ax.set_ylim(y_min - pad_y, y_max + pad_y)

    # separate scatters so you can see colors clearly; matplotlib default colors suffice
    scat_blue   = ax.scatter([], [], s=POINT_SIZE, label="blue")
    scat_yellow = ax.scatter([], [], s=POINT_SIZE, label="yellow")
    scat_mid    = ax.scatter([], [], marker='x', s=MIDPOINT_SIZE, label="midpoint")
    ax.legend(loc="best")

    # Accumulators for display
    bx_list, by_list = [], []
    yx_list, yy_list = [], []
    mx_list, my_list = [], []

    try:
        for (bx, by), (yx, yy) in pairs:
            # Add blue
            bx_list.append(bx); by_list.append(by)
            scat_blue.set_offsets(np.column_stack([bx_list, by_list]))

            # Add yellow
            yx_list.append(yx); yy_list.append(yy)
            scat_yellow.set_offsets(np.column_stack([yx_list, yy_list]))

            # Compute + add midpoint
            mx = (bx + yx) / 2.0
            my = (by + yy) / 2.0
            mx_list.append(mx); my_list.append(my)
            scat_mid.set_offsets(np.column_stack([mx_list, my_list]))

            fig.canvas.draw_idle()
            plt.pause(delay_sec)
    except KeyboardInterrupt:
        pass
    finally:
        plt.ioff()
        plt.show()

    return mx_list, my_list

def save_midpoints(output_dir: Path, base_name: str, mid_x, mid_y):
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{base_name}_midpoints.csv"
    pd.DataFrame({"mid_x": mid_x, "mid_y": mid_y}).to_csv(out_path, index=False)
    print(f"Midpoints written to: {out_path}")

def main():
    # Usage: python animate_xy_pair_midpoints.py filename.csv
    filename = sys.argv[1] if len(sys.argv) >= 2 else "hairpins_increasing_difficulty.csv"

    script_dir = Path(__file__).parent
    csv_path   = script_dir / "tracks" / filename
    out_dir    = script_dir / "track_midpoints"

    print(f"Reading: {csv_path}")
    blues, yellows = load_tagged_xy(csv_path)

    # Build all-points list (for limits and potential final context if needed)
    all_pts = []
    if len(blues):   all_pts.extend([(float(x), float(y)) for x, y in blues])
    if len(yellows): all_pts.extend([(float(x), float(y)) for x, y in yellows])
    if not all_pts:
        print("No blue/yellow points to display.")
        return

    # Pairing: each blue → nearest unused yellow
    pairs, unmatched = pair_blues_to_nearest_yellows(blues, yellows)
    if unmatched:
        print(f"Note: {len(unmatched)} blue point(s) had no available yellow to pair; skipped.")

    # Animate pairs and midpoints
    mid_x, mid_y = animate_pairs_and_midpoints(pairs, all_pts, delay_sec=DELAY_SEC)

    # Save midpoints
    save_midpoints(out_dir, Path(filename).stem, mid_x, mid_y)

if __name__ == "__main__":
    main()
