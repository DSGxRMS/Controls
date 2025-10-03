import time
import math
import numpy as np
import pandas as pd

def preprocess_path(xs, ys, loop=True):
    """
    Preprocess path coordinates to compute cumulative distances.

    Args:
        xs (array): X coordinates
        ys (array): Y coordinates
        loop (bool): Whether the path is a loop

    Returns:
        tuple: (xs, ys, s, total_len) where s is cumulative distances
    """
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    x_next = np.roll(xs, -1) if loop else np.concatenate((xs[1:], xs[-1:]))
    y_next = np.roll(ys, -1) if loop else np.concatenate((ys[1:], ys[-1:]))
    seglen = np.hypot(x_next - xs, y_next - ys)
    if not loop:
        seglen[-1] = 0.0
    s = np.concatenate(([0.0], np.cumsum(seglen[:-1])))
    return xs, ys, s, float(seglen.sum())

def resample_track(x_raw, y_raw, num_arc_points=800):
    """
    Resample track to uniform arc length using linear interpolation.

    Args:
        x_raw, y_raw (arrays): Raw coordinates
        num_arc_points (int): Number of points

    Returns:
        tuple: (resampled_x, resampled_y)
    """
    x_raw = np.asarray(x_raw, dtype=float)
    y_raw = np.asarray(y_raw, dtype=float)
    
    # Remove duplicate/near-duplicate points
    if len(x_raw) > 1:
        dx = np.diff(x_raw)
        dy = np.diff(y_raw)
        distances = np.hypot(dx, dy)
        # Keep first point and points with sufficient distance from previous
        keep = np.concatenate(([True], distances > 1e-6))
        x_filtered = x_raw[keep]
        y_filtered = y_raw[keep]
    else:
        x_filtered, y_filtered = x_raw, y_raw
    
    if len(x_filtered) < 2:
        raise ValueError(f"Insufficient points for interpolation: {len(x_filtered)}")
    
    # Calculate cumulative distances along the path
    dx = np.diff(x_filtered)
    dy = np.diff(y_filtered)
    segment_lengths = np.hypot(dx, dy)
    cumulative_distances = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    
    # Normalize to [0, 1]
    total_length = cumulative_distances[-1]
    if total_length > 0:
        normalized_distances = cumulative_distances / total_length
    else:
        normalized_distances = np.linspace(0, 1, len(x_filtered))
    
    # Create uniform spacing in normalized distance
    uniform_distances = np.linspace(0, 1, num_arc_points)
    
    # Interpolate x and y coordinates
    x_resampled = np.interp(uniform_distances, normalized_distances, x_filtered)
    y_resampled = np.interp(uniform_distances, normalized_distances, y_filtered)
    
    return x_resampled, y_resampled