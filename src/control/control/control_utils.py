#!/usr/bin/env python3
import math
import numpy as np


# ============================
# Minimal Useful Path Utilities
# ============================

def preprocess_path(xs, ys, loop=False):
    """
    Convert to numpy arrays, remove duplicates,
    compute cumulative arc-length s[] and total length.
    """
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)

    if xs.size < 2:
        return xs, ys, np.zeros_like(xs), 0.0

    # Remove back-to-back duplicates
    dx = np.diff(xs)
    dy = np.diff(ys)
    keep = np.ones(xs.size, dtype=bool)
    keep[1:] = (dx != 0) | (dy != 0)
    xs = xs[keep]
    ys = ys[keep]

    if xs.size < 2:
        return xs, ys, np.zeros_like(xs), 0.0

    seg = np.hypot(np.diff(xs), np.diff(ys))
    s = np.concatenate(([0.0], np.cumsum(seg)))
    return xs, ys, s, float(s[-1])


def segment_distances(xs, ys, loop=False):
    """
    Distance between consecutive points.
    """
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)

    if xs.size < 2:
        return np.zeros_like(xs)

    ds = np.zeros_like(xs)
    ds[:-1] = np.hypot(np.diff(xs), np.diff(ys))
    return ds


def local_closest_index(position, xs, ys, start_idx=0):
    """
    Simple brute-force closest search over local window.
    Fast enough because planner provides short local paths.
    """
    x0, y0 = position
    xs = np.asarray(xs)
    ys = np.asarray(ys)

    dx = xs - x0
    dy = ys - y0
    dist2 = dx * dx + dy * dy
    return int(np.argmin(dist2))


# ============================
# Pure Pursuit Steering
# ============================

WHEELBASE_M = 1.5
MAX_STEER_RAD = 1.0

LD_BASE = 3.5
LD_GAIN = 0.6
LD_MIN = 2.0
LD_MAX = 8.0

def calc_lookahead(speed):
    return max(LD_MIN, min(LD_MAX, LD_BASE + LD_GAIN * speed))


def forward_index_by_distance(idx, Ld, s, total_len):
    if len(s) == 0:
        return idx
    target = min(s[idx] + Ld, s[-1])
    return int(np.searchsorted(s, target))


def pure_pursuit_steer(position, yaw, speed, xs, ys, near_idx, s, total_len):
    Ld = calc_lookahead(speed)
    tgt_idx = forward_index_by_distance(near_idx, Ld, s, total_len)

    tx, ty = xs[tgt_idx], ys[tgt_idx]
    dx = tx - position[0]
    dy = ty - position[1]

    cy = math.cos(yaw)
    sy = math.sin(yaw)

    x_rel = cy * dx + sy * dy
    y_rel = -sy * dx + cy * dy

    denom = max(Ld, 0.5)**2
    kappa = 2.0 * y_rel / denom

    delta = math.atan(WHEELBASE_M * kappa)
    delta = max(-MAX_STEER_RAD, min(MAX_STEER_RAD, delta))
    return delta, tgt_idx


# ============================
# Cross-Track Error
# ============================

def path_heading(xs, ys, idx):
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    if idx >= len(xs) - 1:
        idx = len(xs) - 2
    dx = xs[idx + 1] - xs[idx]
    dy = ys[idx + 1] - ys[idx]
    return math.atan2(dy, dx)


def cross_track_error(cx, cy, xs, ys, idx):
    psi = path_heading(xs, ys, idx)
    dx = cx - xs[idx]
    dy = cy - ys[idx]
    e_lat = -math.sin(psi) * dx + math.cos(psi) * dy
    return e_lat, psi


# ============================
# Simple PID controllers
# ============================

class PID:
    def __init__(self, kp, ki, kd):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.i = 0.0
        self.prev = None

    def reset(self):
        self.i = 0.0
        self.prev = None

    def update(self, err, dt):
        dt = max(dt, 1e-3)
        self.i += err * dt
        d = 0 if self.prev is None else (err - self.prev) / dt
        self.prev = err
        return self.kp * err + self.ki * self.i + self.kd * d


class PIDRange:
    def __init__(self, kp, ki, kd, out_min=-1, out_max=1):
        self.pid = PID(kp, ki, kd)
        self.min = out_min
        self.max = out_max

    def reset(self):
        self.pid.reset()

    def update(self, err, dt):
        return max(self.min, min(self.max, self.pid.update(err, dt)))

# for LQR
def compute_signed_curvature(xs, ys):
    """
    Signed geometric curvature for feedforward steering (for LQR).
    """
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    n = len(xs)

    if n < 3:
        return np.zeros_like(xs)

    kappa = np.zeros(n)

    for i in range(1, n - 1):
        x0, y0 = xs[i - 1], ys[i - 1]
        x1, y1 = xs[i], ys[i]
        x2, y2 = xs[i + 1], ys[i + 1]

        ax, ay = x1 - x0, y1 - y0
        bx, by = x2 - x1, y2 - y1

        cross = ax * by - ay * bx
        ab = math.hypot(ax, ay)
        bc = math.hypot(bx, by)
        ac = math.hypot(x2 - x0, y2 - y0)

        denom = ab * bc * ac + 1e-6
        kappa[i] = 2.0 * cross / denom

    kappa[0] = kappa[1]
    kappa[-1] = kappa[-2]
    return kappa
