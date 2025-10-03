import time
import math
import numpy as np
import pandas as pd

from scipy.interpolate import splprep, splev

# -------------------- Setup --------------------


PATHPOINTS_CSV = "control/control/pathpoints.csv"
ROUTE_IS_LOOP = False
scaling_factor = 1

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

# -------------------- Utility Functions --------------------
def preprocess_path(xs, ys, loop=True):
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    x_next = np.roll(xs, -1) if loop else np.concatenate((xs[1:], xs[-1:]))
    y_next = np.roll(ys, -1) if loop else np.concatenate((ys[1:], ys[-1:]))
    seglen = np.hypot(x_next - xs, y_next - ys)
    if not loop:
        seglen[-1] = 0.0
    s = np.concatenate(([0.0], np.cumsum(seglen[:-1])))
    return xs, ys, s, float(seglen.sum())

def get_xy_speed(state):
    pos = state.kinematics_estimated.position
    return (pos.x_val, pos.y_val), float(getattr(state, "speed", 0.0))

def local_closest_index(xy, xs, ys, cur_idx, loop=True):
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
    return max(LD_MIN, min(LD_MAX, LD_BASE + LD_GAIN * speed_mps))

def get_yaw(state):
    q = state.kinematics_estimated.orientation
    w, x, y, z = q.w_val, q.x_val, q.y_val, q.z_val
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

def forward_index_by_distance(near_idx, Ld, s, total_len, loop=True):
    if loop:
        target = (s[near_idx] + Ld) % total_len
        return int(np.searchsorted(s, target, side="left") % len(s))
    else:
        target = min(s[near_idx] + Ld, s[-1])
        return int(np.searchsorted(s, target, side="left"))

def pure_pursuit_steer(pos_xy, yaw, speed, xs, ys, near_idx, s, total_len, loop=True):
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
    tck, _ = splprep([x_raw, y_raw], s=0, k=min(3, max(1, len(x_raw) - 1)))
    tt_dense = np.linspace(0, 1, 2000)
    xx, yy = splev(tt_dense, tck)
    s_dense = np.concatenate(([0.0], np.cumsum(np.hypot(np.diff(xx), np.diff(yy)))))
    s_dense /= s_dense[-1] if s_dense[-1] > 0 else 1.0
    s_uniform = np.linspace(0, 1, num_arc_points)
    return np.interp(s_uniform, s_dense, xx), np.interp(s_uniform, s_dense, yy)

def compute_curvature(x, y):
    dx, dy = np.gradient(x), np.gradient(y)
    ddx, ddy = np.gradient(dx), np.gradient(dy)
    denom = np.power(dx*dx + dy*dy, 1.5)
    curv = np.abs(dx * ddy - dy * ddx) / (denom + 1e-12)
    curv[~np.isfinite(curv)] = 0.0
    return curv

def curvature_speed_limit(curvature):
    return np.minimum(np.sqrt(AY_MAX / (curvature + 1e-9)), V_MAX)

def profile_window(v_limit_win, ds_win, v0):
    Nw = len(v_limit_win)
    vp = np.zeros(Nw)
    vp[0] = min(v_limit_win[0], v0)
    for i in range(1, Nw):
        vp[i] = min(math.sqrt(vp[i-1]**2 + 2 * AX_MAX * ds_win[i-1]), v_limit_win[i])
    for i in range(Nw - 2, -1, -1):
        vp[i] = min(vp[i], math.sqrt(vp[i+1]**2 + 2 * abs(AX_MIN) * ds_win[i]), v_limit_win[i])
    return vp

def compute_signed_curvature(x, y):
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
    delta = np.arctan(kappa * WHEELBASE_M)
    denom = np.abs(np.tan(delta)) + 1e-6
    v = np.sqrt(np.maximum(0.0, (D_MAX * WHEELBASE_M) / denom))
    return np.minimum(v, V_MAX)

def segment_distances(xs, ys, loop=True):
    x_next = np.roll(xs, -1)
    y_next = np.roll(ys, -1)
    ds = np.hypot(x_next - xs, y_next - ys)
    if not loop:
        ds[-1] = 0.0
    return ds

def jerk_limited_velocity_profile(v_limit, ds, v0, vf, v_min, v_max, a_max, d_max, j_max):
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
    # Throttle PID (kept as-is; output clamped to [0,1])
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
    # Generic PID with symmetric output limits (for steering correction)
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
    n = len(xs)
    i2 = (idx + 1) % n if loop else min(idx + 1, n - 1)
    dx = xs[i2] - xs[idx]
    dy = ys[i2] - ys[idx]
    return math.atan2(dy, dx)

def cross_track_error(cx, cy, xs, ys, idx, loop=True):
    # Signed lateral error relative to path tangent at idx
    theta_ref = path_heading(xs, ys, idx, loop)
    dx = cx - xs[idx]
    dy = cy - ys[idx]
    # Rotate world error into path frame: y' is lateral error
    e_lat = -math.sin(theta_ref) * dx + math.cos(theta_ref) * dy
    return e_lat, theta_ref

