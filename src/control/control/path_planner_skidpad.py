#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PathWindowNode: EUFS GT-odom driven path-window publisher (no plotting).

Circular reach threshold (no skipping out of order):
- On first odom, lock to the nearest CSV index: current_index = nearest_idx.
- Thereafter, advance ONLY when vehicle is within reach_radius of the current
  target point; may advance multiple indices in a tick, but only if each
  consecutive point is also within reach. Preserves strict CSV order.

I/O:
- Subscribes: nav_msgs/Odometry (default: /ground_truth/odom), BEST_EFFORT
- Publishes:  std_msgs/Float32MultiArray (default: /path_window), BEST_EFFORT
              data = [x1,y1, x2,y2, ...] for window_size points starting at current_index.

Params:
  odom_topic:     str  (default "/ground_truth/odom")
  publish_topic:  str  (default "/path_window")
  csv_file:       str  (default "pathpoints.csv")  # headers: x,y
  window_size:    int  (default 6)
  reach_radius:   float(default 2.0)               # meters
  hz:             float(default 20.0)
  lock_on_start:  bool (default True)
"""

from __future__ import annotations

import math
import threading
from pathlib import Path as FSPath
from typing import Optional

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy, QoSDurabilityPolicy
)
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32MultiArray


class PathWindowNode(Node):
    def __init__(self):
        super().__init__('path_window_node', automatically_declare_parameters_from_overrides=True)

        # ---------- Parameters ----------
        self.declare_parameter('odom_topic', '/ground_truth/odom')
        self.declare_parameter('publish_topic', '/path_window')
        self.declare_parameter('csv_file', 'pathpoints.csv')  # CSV with headers x,y
        self.declare_parameter('window_size', 6)              # publish target + next N-1 points
        self.declare_parameter('reach_radius', 2.0)           # meters: circular threshold to mark reached
        self.declare_parameter('hz', 20.0)                    # publish rate (Hz)
        self.declare_parameter('lock_on_start', True)         # lock to nearest once at startup

        # ---------- Load Parameters ----------
        self.odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
        self.publish_topic = self.get_parameter('publish_topic').get_parameter_value().string_value
        self.csv_file = self.get_parameter('csv_file').get_parameter_value().string_value
        self.window_size = max(1, int(self.get_parameter('window_size').value))
        self.reach_radius = float(self.get_parameter('reach_radius').value)
        self.hz = float(self.get_parameter('hz').value)
        self.lock_on_start = bool(self.get_parameter('lock_on_start').value)

        # ---------- QoS (Best Effort everywhere) ----------
        self.qos_best_effort = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # ---------- Load path points ----------
        self.path_xy = self._load_csv(self.csv_file)
        if self.path_xy.shape[0] < self.window_size:
            self.get_logger().warn(
                f"Path length {self.path_xy.shape[0]} < window_size {self.window_size}. Adjusting."
            )
            self.window_size = self.path_xy.shape[0]
        self.num_points = self.path_xy.shape[0]

        # ---------- State ----------
        self.current_pose: Optional[tuple[float, float]] = None
        self.current_index: Optional[int] = None     # Target index we are trying to reach
        self._start_locked = False                   # True once we've locked to initial nearest
        self._lock = threading.Lock()

        # ---------- ROS I/O ----------
        self.sub_odom = self.create_subscription(
            Odometry, self.odom_topic, self._odom_cb, self.qos_best_effort
        )
        self.pub_window = self.create_publisher(
            Float32MultiArray, self.publish_topic, self.qos_best_effort
        )
        self.get_logger().info(f"Subscribing (best-effort): {self.odom_topic}")
        self.get_logger().info(f"Publishing (best-effort): {self.publish_topic}")

        # ---------- Timer ----------
        self.timer = self.create_timer(1.0 / self.hz, self._tick)

    # ---------------- CSV loader ----------------
    def _load_csv(self, csv_path: str) -> np.ndarray:
        p = FSPath(csv_path)
        if not p.exists():
            alt = FSPath(__file__).parent / csv_path
            if alt.exists():
                p = alt
            else:
                raise FileNotFoundError(f"CSV path not found: {csv_path}")

        # Try comma; fallback to whitespace
        try:
            data = np.genfromtxt(p, delimiter=',', names=True, dtype=float)
            if 'x' not in data.dtype.names or 'y' not in data.dtype.names:
                raise ValueError
            xy = np.vstack([data['x'], data['y']]).T
        except Exception:
            data = np.genfromtxt(p, delimiter=None, names=True, dtype=float)
            if 'x' not in data.dtype.names or 'y' not in data.dtype.names:
                raise ValueError(f"CSV must have headers 'x,y' (any delimiter). File: {p}")
            xy = np.vstack([data['x'], data['y']]).T

        xy = xy[~np.isnan(xy).any(axis=1)]
        return xy

    # ---------------- Odom callback ----------------
    def _odom_cb(self, msg: Odometry):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        with self._lock:
            self.current_pose = (x, y)

            # Initial lock: choose the nearest CSV index ONCE.
            if not self._start_locked and self.lock_on_start:
                self.current_index = self._nearest_index_global(x, y)
                self._start_locked = True
                self.get_logger().info(f"Initial lock at index {self.current_index}")
                return

            # If not locked-on-start (param false) and current_index is None, fallback to index 0.
            if self.current_index is None:
                self.current_index = 0
                self._start_locked = True
                self.get_logger().info("No start lock requested; starting at index 0.")
                return

            # STRICT ORDER + CIRCULAR REACH:
            # Advance by +1 while EACH consecutive target is within reach_radius.
            self._advance_by_reach(x, y)

    # ---------------- Helpers ----------------
    def _nearest_index_global(self, x: float, y: float) -> int:
        diffs = self.path_xy - np.array([x, y])
        d2 = np.einsum('ij,ij->i', diffs, diffs)
        return int(np.argmin(d2))

    def _advance_by_reach(self, x: float, y: float):
        """Advance current_index while the car is within reach_radius of the
        current target point. This may advance multiple indices in one call
        but only if each *consecutive* point is inside the circular threshold."""
        if self.current_index is None:
            return

        # Clamp to valid range
        self.current_index = max(0, min(self.current_index, self.num_points - 1))

        # Try to advance as long as *current* target is within reach.
        while self.current_index < self.num_points:
            px, py = self.path_xy[self.current_index]
            if math.hypot(px - x, py - y) <= self.reach_radius:
                if self.current_index < self.num_points - 1:
                    self.current_index += 1
                else:
                    # We are at the final point; cannot advance further
                    break
            else:
                break

    # ---------------- Publisher tick ----------------
    def _tick(self):
        with self._lock:
            if self.current_pose is None or self.current_index is None:
                return

            start = self.current_index
            end = min(self.num_points, start + self.window_size)
            window_xy = self.path_xy[start:end]

            # Publish flattened [x1,y1, x2,y2, ...]
            msg = Float32MultiArray()
            msg.data = window_xy.reshape(-1).astype(float).tolist()
            self.pub_window.publish(msg)


def main():
    rclpy.init()
    node = PathWindowNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
