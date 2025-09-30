import time
import math
import numpy as np
import pandas as pd
from scipy.interpolate import splprep, splev

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
V_MIN = 5.0          # m/s
A_MAX = 15.0         # m/s^2
D_MAX = 20.0         # m/s^2 (max decel)
J_MAX = 70.0         # m/s^3
CURVATURE_MAX = 0.9  # 1/m

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
    Resample track to uniform arc length.

    Args:
        x_raw, y_raw (arrays): Raw coordinates
        num_arc_points (int): Number of points

    Returns:
        tuple: (resampled_x, resampled_y)
    """
    tck, _ = splprep([x_raw, y_raw], s=0, k=min(3, max(1, len(x_raw) - 1)))
    tt_dense = np.linspace(0, 1, 2000)
    xx, yy = splev(tt_dense, tck)
    s_dense = np.concatenate(([0.0], np.cumsum(np.hypot(np.diff(xx), np.diff(yy)))))
    s_dense /= s_dense[-1] if s_dense[-1] > 0 else 1.0
    s_uniform = np.linspace(0, 1, num_arc_points)
    return np.interp(s_uniform, s_dense, xx), np.interp(s_uniform, s_dense, yy)

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

class ControlAlgorithm:
    """
    Encapsulates path data, PID controllers, and control computation methods.
    """
    def __init__(self, xs, ys, loop=True):
        """
        Initialize with path coordinates.

        Args:
            xs, ys (arrays): Path coordinates
            loop (bool): Whether path is a loop
        """
        self.loop = loop
        self.xs, self.ys, self.s, self.total_len = preprocess_path(xs, ys, loop)
        self.seg_ds = segment_distances(self.xs, self.ys, loop)
        self.kappa_signed = compute_signed_curvature(self.xs, self.ys)
        self.v_limit_global = ackermann_curv_speed_limit(self.kappa_signed)
        self.route_v = self.v_limit_global.copy()

        # Controllers
        self.th_pid = PID(3.2, 0.5, 0.134)
        self.steer_pid = PIDRange(kp=0.15, ki=0.05, kd=0.20, out_min=-0.60, out_max=0.60)

        # State
        self.cur_idx = 0
        self.last_profile_t = 0.0

    def update_state(self, pos_xy, yaw, speed, dt, now):
        """
        Update internal state with current vehicle state.

        Args:
            pos_xy (tuple): (x, y) position
            yaw (float): Yaw angle
            speed (float): Speed
            dt (float): Time step
            now (float): Current time
        """
        self.cur_idx = local_closest_index(pos_xy, self.xs, self.ys, self.cur_idx, self.loop)

        # Refresh velocity profile if needed
        if now - self.last_profile_t >= (1.0 / PROFILE_HZ):
            self.last_profile_t = now
            end_idx = forward_index_by_distance(self.cur_idx, PROFILE_WINDOW_M, self.s, self.total_len, self.loop)

            if self.loop:
                if end_idx >= self.cur_idx:
                    idxs = np.arange(self.cur_idx, end_idx + 1)
                else:
                    idxs = np.concatenate((np.arange(self.cur_idx, len(self.xs)), np.arange(0, end_idx + 1)))
            else:
                idxs = np.arange(self.cur_idx, end_idx + 1)

            if self.loop:
                ds_win_list = [0.0]
                for ii in range(len(idxs) - 1):
                    ds_win_list.append(self.seg_ds[idxs[ii]])
                ds_win = np.asarray(ds_win_list, dtype=float)
            else:
                ds_win = np.concatenate(([0.0], self.seg_ds[idxs[:-1]]))

            v_lim_win = self.v_limit_global[idxs]
            v0 = speed
            vf = 0.0 if (not self.loop and idxs[-1] == len(self.xs) - 1) else float(v_lim_win[-1])

            v_prof_win = jerk_limited_velocity_profile(
                v_lim_win, ds_win, v0, vf, V_MIN, V_MAX, A_MAX, D_MAX, J_MAX
            )

            if self.loop and end_idx < self.cur_idx:
                n1 = len(self.xs) - self.cur_idx
                self.route_v[self.cur_idx:] = v_prof_win[:n1]
                self.route_v[:end_idx + 1] = v_prof_win[n1:]
            else:
                self.route_v[self.cur_idx:end_idx + 1] = v_prof_win

    def compute_controls(self, pos_xy, yaw, speed, dt):
        """
        Compute steering and throttle controls.

        Args:
            pos_xy (tuple): (x, y) position
            yaw (float): Yaw angle
            speed (float): Speed
            dt (float): Time step

        Returns:
            dict: Control outputs and logging data
        """
        # Pure Pursuit feedforward steering
        steering_ppc, tgt_idx = pure_pursuit_steer(pos_xy, yaw, speed, self.xs, self.ys, self.cur_idx, self.s, self.total_len, self.loop)

        # Lateral error feedback
        e_lat, theta_ref = cross_track_error(pos_xy[0], pos_xy[1], self.xs, self.ys, self.cur_idx, self.loop)
        steer_correction = self.steer_pid.update(e_lat, dt)

        
        steering_cmd = max(-1.0, min(1.0, steering_ppc + steer_correction))

        # Longitudinal control
        v_err = self.route_v[self.cur_idx] - speed
        if v_err >= 0:
            throttle = self.th_pid.update(v_err, dt)
            brake = 0.0
        else:
            self.th_pid.reset()
            throttle = 0.0
            brake = min(1.0, -v_err * BRAKE_GAIN)

        # Logging data
        tgt_x, tgt_y = self.xs[tgt_idx], self.ys[tgt_idx]
        Ld = calc_lookahead(speed)
        dx_t, dy_t = tgt_x - pos_xy[0], tgt_y - pos_xy[1]
        cyaw, syaw = math.cos(yaw), math.sin(yaw)
        x_rel, y_rel = cyaw * dx_t + syaw * dy_t, -syaw * dx_t + cyaw * dy_t
        kappa_ppc = 2.0 * y_rel / max(0.5, Ld) ** 2
        kappa_ref = self.kappa_signed[self.cur_idx]
        steering_ack = math.atan(WHEELBASE_M * kappa_ref) / MAX_STEER_RAD
        steering_ack = max(-1.0, min(1.0, steering_ack))

        return {
            'steering_cmd': steering_cmd,
            'throttle': throttle,
            'brake': brake,
            'steering_ppc': steering_ppc,
            'steer_correction': steer_correction,
            'v_err': v_err,
            'tgt_x': tgt_x,
            'tgt_y': tgt_y,
            'kappa_ppc': kappa_ppc,
            'e_lat': e_lat,
            'steering_ack': steering_ack
        }