#!/usr/bin/env python3
import math
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header
from std_msgs.msg import Float64MultiArray


class LocalPathPlanner(Node):
    def __init__(self):
        super().__init__("local_path_planner")

        # ---- Parameters ----
        # CSV with columns: x,y
        self.declare_parameter("csv_path", str(Path(__file__).parent / "pathpoints.csv"))
        self.declare_parameter("window_length", 5.0)   # meters
        self.declare_parameter("n_points", 50)         # points in local path

        csv_path = Path(self.get_parameter("csv_path").get_parameter_value().string_value)
        self.window_length = float(self.get_parameter("window_length").value)
        self.n_points = int(self.get_parameter("n_points").value)

        self.get_logger().info(f"Loading global path from: {csv_path}")

        # ---- Load global path from CSV ----
        import pandas as pd
        df = pd.read_csv(csv_path)
        self.global_x = df["x"].to_numpy(dtype=float)
        self.global_y = df["y"].to_numpy(dtype=float)

        if len(self.global_x) < 2:
            raise RuntimeError("Path CSV must contain at least 2 points")

        # Precompute cumulative distance along path (s)
        self.s = self._compute_cumulative_s(self.global_x, self.global_y)

        # Subscribe to car odometry (adjust topic if needed for EUFSIM)
        self.odom_sub = self.create_subscription(
            Odometry,
            "/ground_truth/odom",
            self.odom_callback,
            10,
        )

        # Local smoothed path for visualization
        self.path_pub = self.create_publisher(Path, "/local_path", 10)

        # Local path points as raw arrays (for your controller)
        self.points_pub = self.create_publisher(Float64MultiArray, "/local_path_points", 10)

        self.get_logger().info("LocalPathPlanner node initialized.")

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------
    def _compute_cumulative_s(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Cumulative arc length along the polyline."""
        dx = np.diff(x)
        dy = np.diff(y)
        ds = np.sqrt(dx * dx + dy * dy)
        s = np.zeros_like(x)
        s[1:] = np.cumsum(ds)
        return s

    def _closest_index(self, x: float, y: float) -> int:
        """Index of the closest global path point to (x, y)."""
        dx = self.global_x - x
        dy = self.global_y - y
        dist2 = dx * dx + dy * dy
        return int(np.argmin(dist2))

    def _local_window_indices(self, i0: int) -> np.ndarray:
        """
        Get indices of the global path within window_length [m] ahead of i0.
        Non-looped version.
        """
        s0 = self.s[i0]
        s_max = s0 + self.window_length
        # since s is increasing, we can just search forward
        # If path is long, you can optimize; this is simple and robust.
        mask = (self.s >= s0) & (self.s <= s_max)
        idxs = np.nonzero(mask)[0]
        # Ensure at least a few points
        if len(idxs) < 4:
            # extend a bit if needed
            end_idx = min(i0 + 4, len(self.global_x) - 1)
            idxs = np.arange(i0, end_idx + 1)
        return idxs

    def _smooth_points_cubic(self, x_seg: np.ndarray, y_seg: np.ndarray, s_seg: np.ndarray):
        """
        Smooth segment using cubic polynomial fits x(s), y(s).
        Returns dense (xs, ys) along the window.
        """
        # Normalize s to start at 0 for better conditioning
        s0 = s_seg[0]
        s_norm = s_seg - s0
        length = s_norm[-1] if s_norm[-1] > 1e-6 else self.window_length

        # Choose evaluation grid
        s_grid = np.linspace(0.0, length, num=self.n_points)

        # If we have enough points, fit 3rd degree. Otherwise fall back to lower order or raw.
        if len(s_seg) >= 4:
            px = np.polyfit(s_norm, x_seg, 3)
            py = np.polyfit(s_norm, y_seg, 3)
            xs = np.polyval(px, s_grid)
            ys = np.polyval(py, s_grid)
        elif len(s_seg) == 3:
            px = np.polyfit(s_norm, x_seg, 2)
            py = np.polyfit(s_norm, y_seg, 2)
            xs = np.polyval(px, s_grid)
            ys = np.polyval(py, s_grid)
        elif len(s_seg) == 2:
            # simple linear interpolation
            xs = np.interp(s_grid, s_norm, x_seg)
            ys = np.interp(s_grid, s_norm, y_seg)
        else:
            # single point fallback
            xs = np.full(self.n_points, x_seg[0])
            ys = np.full(self.n_points, y_seg[0])

        return xs, ys

    # ------------------------------------------------------------------
    # Main callback
    # ------------------------------------------------------------------
    def odom_callback(self, msg: Odometry):
        # Extract car position
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        # Extract yaw from quaternion
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        self.yaw = yaw

        # Extract speed
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        self.speed = math.sqrt(vx*vx + vy*vy)

        # Find closest global path index
        i0 = self._closest_index(x, y)

        # Find local window of indices ahead of car
        idxs = self._local_window_indices(i0)

        idxs = self._local_window_indices(i0)
        if len(idxs) < 3:
         return

        x_seg = self.global_x[idxs]
        y_seg = self.global_y[idxs]
        s_seg = self.s[idxs]


        # Smooth using cubic polynomial
        xs, ys = self._smooth_points_cubic(x_seg, y_seg, s_seg)

        # Publish Path for visualization (RViz / EUFSIM)
        self._publish_path(xs, ys, msg.header.stamp, msg.header.frame_id or "map")

        # Publish raw x,y arrays for use by your controller node
        self._publish_points(xs, ys)

    def _publish_path(self, xs: np.ndarray, ys: np.ndarray, stamp, frame_id: str):
        path_msg = Path()
        path_msg.header = Header()
        path_msg.header.stamp = stamp
        path_msg.header.frame_id = frame_id

        for x, y in zip(xs, ys):
            ps = PoseStamped()
            ps.header = path_msg.header
            ps.pose.position.x = float(x)
            ps.pose.position.y = float(y)
            ps.pose.position.z = 0.0
            # Orientation can be left as identity for visualization
            path_msg.poses.append(ps)

        self.path_pub.publish(path_msg)

    def _publish_points(self, xs: np.ndarray, ys: np.ndarray):
        """
        Publish as Float64MultiArray: [x0, x1, ..., xN, y0, y1, ..., yN]
        Your control script can split it back into route_x, route_y.
        """
        data = np.concatenate((xs, ys))
        msg = Float64MultiArray()
        msg.data = data.tolist()
        self.points_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = LocalPathPlanner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
