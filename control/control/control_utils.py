import math
import numpy as np

# Vehicle and control parameters
SEARCH_BACK = 10
SEARCH_FWD = 250
MAX_STEP = 60

WHEELBASE_M = 1.5
MAX_STEER_RAD = 0.2
LD_BASE = 3.5
LD_GAIN = 0.6
LD_MIN = 2.0
LD_MAX = 15.0

V_MAX = 8.0
AY_MAX = 4.0
AX_MAX = 5.0
AX_MIN = -4.0

PROFILE_WINDOW_M = 100.0
BRAKE_EXTEND_M = 60.0
NUM_ARC_POINTS = 800
PROFILE_HZ = 10.0
BRAKE_GAIN = 0.7

STOP_SPEED_THRESHOLD = 0.1   # m/s, vehicle considered stopped

# Jerk-limited velocity profile params (from Controls_final.m)
V_MIN = 0.5          # m/s (reduced to allow gentle startup)
A_MAX = 15.0         # m/s^2
D_MAX = 20.0         # m/s^2 (max decel)
J_MAX = 70.0         # m/s^3
CURVATURE_MAX = 0.9  # 1/m

# Startup parameters
STARTUP_THROTTLE = 0.3  # Initial throttle for getting vehicle moving
MIN_STARTUP_SPEED = 0.1  # m/s - minimum speed to consider vehicle started

# Utility Functions
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

def local_closest_index(xy, xs, ys, cur_idx, loop=True):
    """
    Find the closest path index to the given position.

    Args:
        xy (tuple): (x, y) position
        xs, ys (arrays): Path coordinates
        cur_idx (int): Current index hint
        loop (bool): Path loop flag

    Returns:
        int: Closest index
    """
    x0, y0 = xy
    N = len(xs)
    if N == 0:
        return 0
    if loop:
        start = (cur_idx - SEARCH_BACK) % N
        count = min(N, SEARCH_BACK + SEARCH_FWD + 1)
        idxs = (np.arange(start, start + count) % N)
        dx, dy = xs[idxs] - x0, ys[idxs] - y0
        j = int(np.argmin(dx * dx + dy * dy))
        return int(idxs[j])
    else:
        i0 = max(0, cur_idx - SEARCH_BACK)
        i1 = min(N, cur_idx + SEARCH_FWD + 1)
        dx, dy = xs[i0:i1] - x0, ys[i0:i1] - y0
        j = int(np.argmin(dx * dx + dy * dy))
        return i0 + j

def calc_lookahead(speed_mps):
    """
    Calculate lookahead distance based on speed.

    Args:
        speed_mps (float): Speed in m/s

    Returns:
        float: Lookahead distance
    """
    return max(LD_MIN, min(LD_MAX, LD_BASE + LD_GAIN * speed_mps))

def forward_index_by_distance(near_idx, Ld, s, total_len, loop=True):
    """
    Find index at a given distance ahead.

    Args:
        near_idx (int): Starting index
        Ld (float): Distance
        s (array): Cumulative distances
        total_len (float): Total path length
        loop (bool): Path loop flag

    Returns:
        int: Target index
    """
    if loop:
        target = (s[near_idx] + Ld) % total_len
        return int(np.searchsorted(s, target, side="left") % len(s))
    else:
        target = min(s[near_idx] + Ld, s[-1])
        return int(np.searchsorted(s, target, side="left"))

def pure_pursuit_steer(pos_xy, yaw, speed, xs, ys, near_idx, s, total_len, loop=True):
    """
    Compute pure pursuit steering command.

    Args:
        pos_xy (tuple): Current position
        yaw (float): Current yaw
        speed (float): Current speed
        xs, ys, s: Path data
        near_idx (int): Current index
        total_len (float): Total path length
        loop (bool): Path loop flag

    Returns:
        tuple: (normalized_steering, target_index)
    """
    Ld = calc_lookahead(speed)
    tgt_idx = forward_index_by_distance(near_idx, Ld, s, total_len, loop)
    tx, ty = xs[tgt_idx], ys[tgt_idx]
    dx, dy = tx - pos_xy[0], ty - pos_xy[1]
    cy, sy = math.cos(yaw), math.sin(yaw)
    x_rel, y_rel = cy * dx + sy * dy, -sy * dx + cy * dy
    kappa = 2.0 * y_rel / max(0.5, Ld) ** 2
    delta = math.atan(WHEELBASE_M * kappa)
    return max(-1, min(1, delta / MAX_STEER_RAD)), tgt_idx

def resample_track(x_raw, y_raw, num_arc_points=NUM_ARC_POINTS):
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

def compute_signed_curvature(x, y):
    """
    Compute signed curvature of the path.

    Args:
        x, y (arrays): Coordinates

    Returns:
        array: Signed curvature
    """
    dx = np.gradient(x)
    dy = np.gradient(y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    denom = np.power(dx * dx + dy * dy, 1.5) + 1e-12
    kappa = (dx * ddy - dy * ddx) / denom
    kappa = np.clip(kappa, -CURVATURE_MAX, CURVATURE_MAX)
    kappa[~np.isfinite(kappa)] = 0.0
    return kappa

def ackermann_curv_speed_limit(kappa):
    """
    Compute speed limit based on curvature using Ackermann model.

    Args:
        kappa (array): Curvature

    Returns:
        array: Speed limits
    """
    delta = np.arctan(kappa * WHEELBASE_M)
    denom = np.abs(np.tan(delta)) + 1e-6
    v = np.sqrt(np.maximum(0.0, (D_MAX * WHEELBASE_M) / denom))
    return np.minimum(v, V_MAX)

def segment_distances(xs, ys, loop=True):
    """
    Compute distances between consecutive points.

    Args:
        xs, ys (arrays): Coordinates
        loop (bool): Path loop flag

    Returns:
        array: Segment distances
    """
    x_next = np.roll(xs, -1)
    y_next = np.roll(ys, -1)
    ds = np.hypot(x_next - xs, y_next - ys)
    if not loop:
        ds[-1] = 0.0
    return ds

def jerk_limited_velocity_profile(v_limit, ds, v0, vf, v_min, v_max, a_max, d_max, j_max):
    """
    Compute jerk-limited velocity profile.

    Args:
        v_limit (array): Speed limits
        ds (array): Segment distances
        v0, vf (float): Initial and final speeds
        v_min, v_max (float): Min/max speeds
        a_max, d_max (float): Max accel/decel
        j_max (float): Max jerk

    Returns:
        array: Velocity profile
    """
    v_limit = np.asarray(v_limit, dtype=float)
    ds = np.asarray(ds, dtype=float)
    N = len(v_limit)
    if len(ds) != N:
        raise ValueError("ds length must equal v_limit length (ds[0] is 0 for the first point)")
    v_forward = np.zeros(N, dtype=float)
    v_forward[0] = min(max(v0, 0.0), v_limit[0], v_max)
    a_prev = 0.0
    for i in range(1, N):
        ds_i = max(ds[i], 1e-9)
        v_avg = max(v_min, v_forward[i - 1])
        dt = ds_i / v_avg
        a_curr = min(a_prev + j_max * dt, a_max)
        v_possible = math.sqrt(max(0.0, v_forward[i - 1] ** 2 + 2.0 * a_curr * ds_i))
        v_forward[i] = min(v_possible, v_limit[i], v_max)
        a_prev = (v_forward[i] ** 2 - v_forward[i - 1] ** 2) / (2.0 * ds_i)
    v_profile = v_forward.copy()
    v_profile[-1] = min(v_profile[-1], max(0.0, vf))
    a_prev = 0.0
    for i in range(N - 2, -1, -1):
        ds_i = max(ds[i + 1], 1e-9)
        v_avg = max(v_min, v_profile[i + 1])
        dt = ds_i / v_avg
        a_curr = min(a_prev + j_max * dt, d_max)
        v_possible = math.sqrt(max(0.0, v_profile[i + 1] ** 2 + 2.0 * a_curr * ds_i))
        v_profile[i] = min(v_profile[i], v_possible, v_max)
        a_prev = (v_profile[i + 1] ** 2 - v_profile[i] ** 2) / (2.0 * ds_i)
    v_profile = np.minimum(v_profile, v_max)
    if N > 1:
        v_profile[:-1] = np.maximum(v_profile[:-1], v_min)
    if vf <= 0.0:
        v_profile[-1] = 0.0
    return v_profile

class PID:
    """
    PID controller for throttle (output clamped to [0,1]).
    """
    def __init__(self, kp, ki, kd):
        self.kp, self.ki, self.kd = kp, ki, kd
        self._i = 0.0
        self._prev_err = None

    def reset(self):
        self._i, self._prev_err = 0.0, None

    def update(self, err, dt):
        dt = max(dt, 1e-3)
        self._i += err * dt
        d = 0 if self._prev_err is None else (err - self._prev_err) / dt
        self._prev_err = err
        return max(0, min(1, self.kp * err + self.ki * self._i + self.kd * d))

class PIDRange:
    """
    PID controller with symmetric output limits (for steering correction).
    """
    def __init__(self, kp, ki, kd, out_min=-1.0, out_max=1.0):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.out_min, self.out_max = out_min, out_max
        self._i = 0.0
        self._prev_err = None

    def reset(self):
        self._i, self._prev_err = 0.0, None

    def update(self, err, dt):
        dt = max(dt, 1e-3)
        self._i += err * dt
        d = 0.0 if self._prev_err is None else (err - self._prev_err) / dt
        self._prev_err = err
        u = self.kp * err + self.ki * self._i + self.kd * d
        return max(self.out_min, min(self.out_max, u))

def path_heading(xs, ys, idx, loop=True):
    """
    Compute path heading at index.

    Args:
        xs, ys (arrays): Coordinates
        idx (int): Index
        loop (bool): Path loop flag

    Returns:
        float: Heading angle
    """
    n = len(xs)
    i2 = (idx + 1) % n if loop else min(idx + 1, n - 1)
    dx = xs[i2] - xs[idx]
    dy = ys[i2] - ys[idx]
    return math.atan2(dy, dx)

def cross_track_error(cx, cy, xs, ys, idx, loop=True):
    """
    Compute cross-track error.

    Args:
        cx, cy (float): Current position
        xs, ys (arrays): Path coordinates
        idx (int): Current index
        loop (bool): Path loop flag

    Returns:
        tuple: (lateral_error, reference_heading)
    """
    theta_ref = path_heading(xs, ys, idx, loop)
    dx = cx - xs[idx]
    dy = cy - ys[idx]
    e_lat = -math.sin(theta_ref) * dx + math.cos(theta_ref) * dy
    return e_lat, theta_ref

def startup_control(speed, target_speed=2.0):
    """
    Generate startup control commands to get vehicle moving from standstill.
    
    Args:
        speed (float): Current speed
        target_speed (float): Target speed for startup
        
    Returns:
        tuple: (throttle, brake, is_started)
    """
    if speed < MIN_STARTUP_SPEED:
        # Vehicle is essentially stopped, apply startup throttle
        return STARTUP_THROTTLE, 0.0, False
    elif speed < target_speed:
        # Vehicle is moving but below target, continue accelerating
        throttle = min(0.5, STARTUP_THROTTLE * (target_speed - speed) / target_speed)
        return max(0.1, throttle), 0.0, False
    else:
        # Vehicle has reached startup speed
        return 0.0, 0.0, True