import math
import numpy as np

def preprocess_path(xs, ys, loop=True):
    """
    Preprocess path coordinates to compute cumulative distances.
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
    """
    x_raw = np.asarray(x_raw, dtype=float)
    y_raw = np.asarray(y_raw, dtype=float)
    
    if len(x_raw) > 1:
        dx = np.diff(x_raw)
        dy = np.diff(y_raw)
        distances = np.hypot(dx, dy)
        keep = np.concatenate(([True], distances > 1e-6))
        x_filtered = x_raw[keep]
        y_filtered = y_raw[keep]
    else:
        x_filtered, y_filtered = x_raw, y_raw
    
    if len(x_filtered) < 2:
        raise ValueError(f"Insufficient points for interpolation: {len(x_filtered)}")
    
    dx = np.diff(x_filtered)
    dy = np.diff(y_filtered)
    segment_lengths = np.hypot(dx, dy)
    cumulative_distances = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    
    total_length = cumulative_distances[-1]
    if total_length > 0:
        normalized_distances = cumulative_distances / total_length
    else:
        normalized_distances = np.linspace(0, 1, len(x_filtered))
    
    uniform_distances = np.linspace(0, 1, num_arc_points)
    
    x_resampled = np.interp(uniform_distances, normalized_distances, x_filtered)
    y_resampled = np.interp(uniform_distances, normalized_distances, y_filtered)
    
    return x_resampled, y_resampled

def local_closest_index(cur_pos, xs, ys, cur_idx, search_window=100):
    """
    Find the closest path index in a local search window.
    """
    start_idx = max(0, cur_idx - search_window // 2)
    end_idx = min(len(xs), cur_idx + search_window // 2)
    
    search_indices = np.arange(start_idx, end_idx)
    
    if len(search_indices) == 0:
        return cur_idx

    dx = xs[search_indices] - cur_pos[0]
    dy = ys[search_indices] - cur_pos[1]
    
    distances_sq = dx**2 + dy**2
    
    closest_local_idx = np.argmin(distances_sq)
    return search_indices[closest_local_idx]

def calc_lookahead(v, la_dist_min, la_dist_max, la_vel_min, la_vel_max):
    """
    Calculate lookahead distance based on velocity.
    """
    lookahead_dist = (v - la_vel_min) * (la_dist_max - la_dist_min) / (la_vel_max - la_vel_min) + la_dist_min
    return np.clip(lookahead_dist, la_dist_min, la_dist_max)

def pure_pursuit_steer(x, y, yaw, xs, ys, s, total_len, cur_idx, la_dist):
    """
    Calculate steering angle using pure pursuit.
    """
    # Find lookahead point
    s_target = s[cur_idx] + la_dist
    if s_target >= total_len:
        s_target -= total_len

    s_diff = s - s_target
    idx_next = np.argmin(np.abs(s_diff))
    
    x_la = xs[idx_next]
    y_la = ys[idx_next]

    # Transform lookahead point to vehicle frame
    x_rel = x_la - x
    y_rel = y_la - y
    
    x_veh = x_rel * np.cos(-yaw) - y_rel * np.sin(-yaw)
    y_veh = x_rel * np.sin(-yaw) + y_rel * np.cos(-yaw)

    # Calculate curvature and steering angle
    if x_veh < 1e-3: # Avoid division by zero
        return 0.0

    curvature = 2 * y_veh / (x_veh**2 + y_veh**2)
    steer = np.arctan(curvature)
    
    return steer

def compute_signed_curvature(xs, ys):
    """
    Compute signed curvature of a path.
    """
    dx = np.gradient(xs)
    dy = np.gradient(ys)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    
    # Curvature formula
    k = (dx * ddy - dy * ddx) / ((dx**2 + dy**2)**1.5 + 1e-6)
    
    # Compute segment lengths
    seg_ds = np.hypot(np.gradient(xs), np.gradient(ys))
    
    return k, seg_ds