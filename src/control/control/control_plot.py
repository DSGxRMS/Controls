#!/usr/bin/env python3
import time
import math
from collections import deque

import numpy as np
import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry, Path
from std_msgs.msg import Float32
from ackermann_msgs.msg import AckermannDriveStamped

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

# from your utilities
from control.control_utils import (
    preprocess_path,
    local_closest_index,
    cross_track_error,
    path_heading,
)

# ==============================
# CONFIG
# ==============================

TELEMETRY_BUFFER_SIZE = 1000
ERROR_PLOT_HISTORY    = 300

MAX_STEER_RAD = 1.0       # match your controllers
SCALING_FACTOR = 1.0      # in case path scaling changes
ROUTE_IS_LOOP  = False    # only used for completeness; here we treat path as open


# ==============================
# MATPLOTLIB VISUALIZER
# ==============================

class TelemetryVisualizer:
    def __init__(self):
        # Buffers
        self.telemetry = deque(maxlen=TELEMETRY_BUFFER_SIZE)

        # Path data (global)
        self.route_x = None
        self.route_y = None

        plt.ion()
        self.fig = plt.figure(figsize=(16, 9))
        self.fig.canvas.manager.set_window_title("Real-Time Autonomous Vehicle Telemetry")

        # Grid layout: left big path, right 4 stacked plots
        gs = self.fig.add_gridspec(4, 2, width_ratios=[2.5, 1], height_ratios=[1, 1, 1, 1],
                                   hspace=0.35, wspace=0.25)

        # Path & trajectory (all rows of col 0)
        self.ax_path = self.fig.add_subplot(gs[:, 0])

        # Speed plot (row 0, col 1)
        self.ax_speed = self.fig.add_subplot(gs[0, 1])

        # Accel plot (row 1, col 1)
        self.ax_accel = self.fig.add_subplot(gs[1, 1])

        # Cross-track error (row 2, col 1)
        self.ax_cte = self.fig.add_subplot(gs[2, 1])

        # Heading error (row 3, col 1)
        self.ax_heading = self.fig.add_subplot(gs[3, 1])

        self._setup_path_axes()
        self._setup_time_axes()

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    # ---------- Setup ----------

    def _setup_path_axes(self):
        self.ax_path.set_title("Global Path & Vehicle Pose", fontsize=14, fontweight='bold')
        self.ax_path.set_xlabel("X (m)")
        self.ax_path.set_ylabel("Y (m)")
        self.ax_path.grid(True, linestyle=':', alpha=0.4)
        self.ax_path.set_aspect('equal', adjustable='box')

        # static handles
        self.path_lc = None
        self.trace_line, = self.ax_path.plot([], [], 'k-', alpha=0.4, linewidth=1.5, label="Trajectory")
        self.car_dot, = self.ax_path.plot([], [], 'ro', markersize=8, label="Vehicle")
        # heading arrow via quiver
        self.heading_quiver = self.ax_path.quiver([], [], [], [], color='r', scale=25, width=0.006)

        self.ax_path.legend(loc="upper right", fontsize='small', framealpha=0.9)

    def _setup_time_axes(self):
        # speed
        self.ax_speed.set_title("Speed Tracking", fontsize=11, fontweight='bold')
        self.ax_speed.set_ylabel("Speed (m/s)")
        self.speed_actual_line, = self.ax_speed.plot([], [], 'b-', label="Actual")
        self.speed_ref_line,    = self.ax_speed.plot([], [], 'g--', label="Ref")
        self.ax_speed.grid(True, linestyle=':', alpha=0.6)
        self.ax_speed.legend(loc="upper left", fontsize='x-small')

        # acceleration
        self.ax_accel.set_title("Acceleration Command", fontsize=11, fontweight='bold')
        self.ax_accel.set_ylabel("Accel (m/s²)")
        self.accel_cmd_line, = self.ax_accel.plot([], [], 'm-', label="Accel cmd")
        self.ax_accel.grid(True, linestyle=':', alpha=0.6)

        # cross-track error
        self.ax_cte.set_title("Cross-Track Error", fontsize=11, fontweight='bold')
        self.ax_cte.set_ylabel("e_y (m)")
        self.cte_line, = self.ax_cte.plot([], [], 'c-', label="CTE")
        self.ax_cte.axhline(0.0, color='k', linewidth=0.7)
        self.ax_cte.grid(True, linestyle=':', alpha=0.6)

        # heading error
        self.ax_heading.set_title("Heading Error", fontsize=11, fontweight='bold')
        self.ax_heading.set_ylabel("e_psi (rad)")
        self.ax_heading.set_xlabel("Time (s)")
        self.heading_line, = self.ax_heading.plot([], [], 'r-', label="Heading error")
        self.ax_heading.axhline(0.0, color='k', linewidth=0.7)
        self.ax_heading.grid(True, linestyle=':', alpha=0.6)

    # ---------- Path geometry exposure ----------

    def set_path(self, route_x, route_y, route_v=None):
        """
        Set or update global path to plot.
        route_v is optional: if provided, path line is colored by reference speed.
        """
        if route_x is None or route_y is None or len(route_x) < 2:
            return

        self.route_x = np.asarray(route_x, dtype=float)
        self.route_y = np.asarray(route_y, dtype=float)

        # remove any previous collection
        if self.path_lc is not None:
            self.path_lc.remove()
            self.path_lc = None

        points = np.array([self.route_x, self.route_y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        if route_v is not None:
            route_v = np.asarray(route_v, dtype=float)
            norm = plt.Normalize(vmin=float(route_v.min()), vmax=float(route_v.max()))
            self.path_lc = LineCollection(segments, cmap='viridis', norm=norm, linewidths=2.5, alpha=0.8)
            self.path_lc.set_array(route_v)
        else:
            self.path_lc = LineCollection(segments, colors='gray', linewidths=2.5, alpha=0.8)

        self.ax_path.add_collection(self.path_lc)

        # Autoscale view with padding
        x_min, x_max = float(self.route_x.min()), float(self.route_x.max())
        y_min, y_max = float(self.route_y.min()), float(self.route_y.max())
        pad = 5.0
        self.ax_path.set_xlim(x_min - pad, x_max + pad)
        self.ax_path.set_ylim(y_min - pad, y_max + pad)

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    # ---------- Logging & Update ----------

    def log_state(self, t, x, y, yaw,
                  speed_actual, speed_ref,
                  accel_cmd, steer_cmd,
                  e_y, e_psi):
        """
        Store a snapshot to the circular buffer.
        """
        self.telemetry.append({
            "t": t,
            "x": x,
            "y": y,
            "yaw": yaw,
            "v": speed_actual,
            "v_ref": speed_ref,
            "accel": accel_cmd,
            "steer": steer_cmd,
            "e_y": e_y,
            "e_psi": e_psi,
        })

    def update_plot(self):
        """
        Redraw plots based on latest telemetry.
        Call this at ~10–20 Hz from ROS timer.
        """
        if not self.telemetry:
            return

        data_list = list(self.telemetry)

        # ----- Path / vehicle pose -----
        xs = [d["x"] for d in data_list]
        ys = [d["y"] for d in data_list]

        # history trace
        self.trace_line.set_data(xs, ys)

        # current pose
        last = data_list[-1]
        self.car_dot.set_data([last["x"]], [last["y"]])

        # heading arrow
        self.heading_quiver.set_offsets([last["x"], last["y"]])
        self.heading_quiver.set_UVC(np.cos(last["yaw"]), np.sin(last["yaw"]))

        # ----- Time axis subset for errors -----
        hist_len = min(len(data_list), ERROR_PLOT_HISTORY)
        recent = data_list[-hist_len:]

        ts     = [d["t"] for d in recent]
        v_act  = [d["v"] for d in recent]
        v_ref  = [d["v_ref"] for d in recent]
        accels = [d["accel"] for d in recent]
        steers = [d["steer"] for d in recent]
        ctes   = [d["e_y"] for d in recent]
        h_err  = [d["e_psi"] for d in recent]

        # speed tracking
        self.speed_actual_line.set_data(ts, v_act)
        self.speed_ref_line.set_data(ts, v_ref)
        if ts:
            self.ax_speed.set_xlim(ts[0], ts[-1])
            v_min = min(v_act + v_ref)
            v_max = max(v_act + v_ref)
            self.ax_speed.set_ylim(v_min - 1.0, v_max + 1.0)

        # acceleration
        self.accel_cmd_line.set_data(ts, accels)
        if ts:
            self.ax_accel.set_xlim(ts[0], ts[-1])
            a_min = min(accels)
            a_max = max(accels)
            if a_min == a_max:
                a_min -= 0.5
                a_max += 0.5
            self.ax_accel.set_ylim(a_min - 0.5, a_max + 0.5)

        # cross-track
        self.cte_line.set_data(ts, ctes)
        if ts:
            self.ax_cte.set_xlim(ts[0], ts[-1])
            e_min = min(ctes)
            e_max = max(ctes)
            if e_min == e_max:
                e_min -= 0.2
                e_max += 0.2
            self.ax_cte.set_ylim(e_min - 0.2, e_max + 0.2)

        # heading error
        self.heading_line.set_data(ts, h_err)
        if ts:
            self.ax_heading.set_xlim(ts[0], ts[-1])
            h_min = min(h_err)
            h_max = max(h_err)
            if h_min == h_max:
                h_min -= 0.1
                h_max += 0.1
            self.ax_heading.set_ylim(h_min - 0.1, h_max + 0.1)

        # finally draw
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()


# ==============================
# ROS2 TELEMETRY NODE
# ==============================

class TelemetryNode(Node):
    def __init__(self):
        super().__init__("telemetry_plotter")

        # state
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.speed = 0.0
        self.have_odom = False

        self.steer_cmd = 0.0
        self.accel_cmd = 0.0
        self.speed_ref = 0.0

        self.route_x = None
        self.route_y = None
        self.route_s = None
        self.route_len = 0.0
        self.cur_idx = 0

        self.start_time = time.perf_counter()

        # visualizer
        self.vis = TelemetryVisualizer()

        # subscriptions
        self.create_subscription(
            Odometry,
            "/ground_truth/odom",
            self.odom_cb,
            10
        )

        self.create_subscription(
            AckermannDriveStamped,
            "/cmd",
            self.cmd_cb,
            10
        )

        self.create_subscription(
            Path,
            "/planned_path",
            self.path_cb,
            10
        )

        self.create_subscription(
            Float32,
            "/desired_velocity",
            self.vel_ref_cb,
            10
        )

        # timer to update plot (e.g. 20 Hz)
        self.timer = self.create_timer(0.05, self.timer_cb)

        self.get_logger().info("Real-time telemetry plotter started.")

    # ---------- Callbacks ----------

    def odom_cb(self, msg: Odometry):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y

        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.yaw = math.atan2(siny_cosp, cosy_cosp)

        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        self.speed = math.hypot(vx, vy)

        self.have_odom = True

    def cmd_cb(self, msg: AckermannDriveStamped):
        self.steer_cmd = float(msg.drive.steering_angle)
        self.accel_cmd = float(msg.drive.acceleration)

    def path_cb(self, msg: Path):
        if len(msg.poses) < 2:
            return

        xs = [p.pose.position.x for p in msg.poses]
        ys = [p.pose.position.y for p in msg.poses]

        xs = np.asarray(xs, dtype=float) * SCALING_FACTOR
        ys = np.asarray(ys, dtype=float) * SCALING_FACTOR

        xs, ys, s, total_len = preprocess_path(xs, ys, loop=ROUTE_IS_LOOP)

        self.route_x = xs
        self.route_y = ys
        self.route_s = s
        self.route_len = total_len

        # update visualizer path (no velocity color here; can pass route_v if you have it)
        self.vis.set_path(self.route_x, self.route_y)

    def vel_ref_cb(self, msg: Float32):
        self.speed_ref = float(msg.data)

    # ---------- Timer ----------

    def timer_cb(self):
        if not self.have_odom:
            return
        if self.route_x is None or self.route_y is None or len(self.route_x) < 3:
            return

        # time since start
        t = time.perf_counter() - self.start_time

        # closest path index
        self.cur_idx = local_closest_index(
            (self.x, self.y),
            self.route_x,
            self.route_y,
            self.cur_idx
        )

        # cross-track & heading error using same math as controllers
        e_y, psi_ref = cross_track_error(
            self.x, self.y,
            self.route_x,
            self.route_y,
            self.cur_idx
        )
        e_psi = math.atan2(math.sin(psi_ref - self.yaw),
                           math.cos(psi_ref - self.yaw))

        # log into visualizer
        self.vis.log_state(
            t=t,
            x=self.x,
            y=self.y,
            yaw=self.yaw,
            speed_actual=self.speed,
            speed_ref=self.speed_ref,
            accel_cmd=self.accel_cmd,
            steer_cmd=self.steer_cmd,
            e_y=e_y,
            e_psi=e_psi,
        )

        # update plots
        self.vis.update_plot()


# ==============================
# MAIN
# ==============================

def main(args=None):
    rclpy.init(args=args)
    node = TelemetryNode()

    try:
        # You want this node to keep matplotlib GUI alive while spinning.
        # rclpy doesn't block matplotlib, so this is fine.
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
