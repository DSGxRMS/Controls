#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from std_msgs.msg import Float32
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
import numpy as np
import math


# =========================================================
#  VELOCITY PROFILER MATH CORE
# =========================================================

def compute_arc_length_from_path(path_points):
    ds_array = np.sqrt(
        np.diff(path_points[:, 0])**2 +
        np.diff(path_points[:, 1])**2
    )
    s = np.insert(np.cumsum(ds_array), 0, 0.0)
    ds = np.mean(ds_array)
    return s, ds


def compute_curvature_from_path(x, y):
    dx = np.gradient(x)
    dy = np.gradient(y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)

    kappa = np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2)**1.5
    kappa[np.isnan(kappa)] = 0.0
    kappa[np.isinf(kappa)] = 0.0
    return kappa


def compute_grip_limited_velocity(kappa, ds, ay_max, ax_max, v_max):

    v_curv = np.minimum(v_max, np.sqrt(ay_max / np.maximum(kappa, 1e-6)))

    v_fwd = np.zeros_like(v_curv)
    for i in range(len(v_fwd) - 1):
        v_fwd[i + 1] = min(
            v_curv[i + 1],
            np.sqrt(v_fwd[i]**2 + 2 * ax_max * ds)
        )

    v_bwd = np.copy(v_fwd)
    for i in range(len(v_bwd) - 1, 0, -1):
        v_bwd[i - 1] = min(
            v_bwd[i - 1],
            np.sqrt(v_bwd[i]**2 + 2 * ax_max * ds)
        )

    return v_bwd


def apply_jerk_limit(v_base, ds, ax_max, ax_min, J_max):

    n = len(v_base)
    v = np.copy(v_base)
    a = np.zeros(n)

    for i in range(1, n):
        a[i] = (v[i]**2 - v[i - 1]**2) / (2 * ds)

    for i in range(n - 2):
        dt = ds / max(v[i], 0.5)
        a[i + 1] = np.clip(a[i] + J_max * dt, ax_min, ax_max)
        v[i + 1] = np.sqrt(max(v[i]**2 + 2 * a[i + 1] * ds, 0))
        v[i + 1] = min(v[i + 1], v_base[i + 1])

    return v, a


def speed_to_color(v, v_min, v_max):
    """Blue (slow) → Yellow → Red (fast)"""
    ratio = (v - v_min) / max(v_max - v_min, 1e-6)
    ratio = np.clip(ratio, 0.0, 1.0)

    r = ratio
    g = 1.0 - abs(ratio - 0.5) * 2.0
    b = 1.0 - ratio

    return ColorRGBA(r=r, g=g, b=b, a=1.0)


# =========================================================
#  ROS2 VELOCITY PROFILER NODE
# =========================================================

class VelocityProfilerNode(Node):

    def __init__(self):
        super().__init__("velocity_profiler_node")

        #  VEHICLE LIMITS
        self.v_max = 30.0     # m/s
        self.ay_max = 10.0    # m/s^2
        self.ax_max = 6.0     # m/s^2
        self.ax_min = -8.0   # m/s^2
        self.J_max = 80.0    # m/s^3

        #  STORAGE
        self.v_profile = None
        self.a_profile = None
        self.path_received = False

        #  SUBSCRIBE TO PATH
        self.create_subscription(
            Path,
            "/planned_path",
            self.path_cb,
            10
        )

        #  OUTPUT TO CONTROLLER
        self.vel_pub = self.create_publisher(
            Float32, "/desired_velocity", 10)

        self.acc_pub = self.create_publisher(
            Float32, "/desired_acceleration", 10)

        #  RVIZ MARKERS
        self.marker_pub = self.create_publisher(
            MarkerArray, "/velocity_profile_markers", 10)

        #  TIMER (50 Hz)
        self.timer = self.create_timer(0.02, self.timer_cb)

        self.get_logger().info(" Velocity Profiler with RViz Speed Coloring + Text Started")


    # =====================================================
    #  RECEIVE PATH & BUILD SPEED PROFILE
    # =====================================================
    def path_cb(self, msg):

        if len(msg.poses) < 5:
            return

        x = np.array([p.pose.position.x for p in msg.poses])
        y = np.array([p.pose.position.y for p in msg.poses])

        path_xy = np.column_stack((x, y))

        _, ds = compute_arc_length_from_path(path_xy)
        kappa = compute_curvature_from_path(x, y)

        v_grip = compute_grip_limited_velocity(
            kappa, ds,
            self.ay_max,
            self.ax_max,
            self.v_max
        )

        self.v_profile, self.a_profile = apply_jerk_limit(
            v_grip, ds,
            self.ax_max,
            self.ax_min,
            self.J_max
        )

        self.path_received = True

        #  Publish RViz colored markers + text labels
        self.publish_velocity_markers(x, y, self.v_profile)


    # =====================================================
    #  PUBLISH DESIRED SPEED FOR CONTROLLER
    # =====================================================
    def timer_cb(self):

        if not self.path_received:
            return

        v_msg = Float32()
        a_msg = Float32()

        #  First point of local path = target
        v_msg.data = float(self.v_profile[0])
        a_msg.data = float(self.a_profile[0])

        self.vel_pub.publish(v_msg)
        self.acc_pub.publish(a_msg)


    # =====================================================
    #  RVIZ MARKERS + SPEED TEXT
    # =====================================================
    def publish_velocity_markers(self, x, y, v_profile):

        markers = MarkerArray()
        v_min = float(np.min(v_profile))
        v_max = float(np.max(v_profile))

        for i in range(len(x)):

            # ======= SPHERE MARKER (COLOR) =======
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "velocity_profile"
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD

            marker.pose.position.x = float(x[i])
            marker.pose.position.y = float(y[i])
            marker.pose.position.z = 0.2

            marker.pose.orientation.w = 1.0

            marker.scale.x = 0.25
            marker.scale.y = 0.25
            marker.scale.z = 0.25

            marker.color = speed_to_color(v_profile[i], v_min, v_max)
            markers.markers.append(marker)

            # ======= TEXT MARKER (SPEED LABEL) =======
            text = Marker()
            text.header = marker.header
            text.ns = "velocity_text"
            text.id = i + 1000
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD

            text.pose.position.x = float(x[i])
            text.pose.position.y = float(y[i])
            text.pose.position.z = 0.6

            text.scale.z = 0.3
            text.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)

            text.text = f"{v_profile[i]:.1f} m/s"

            markers.markers.append(text)

        self.marker_pub.publish(markers)


# =========================================================
#  MAIN
# =========================================================

def main(args=None):
    rclpy.init(args=args)
    node = VelocityProfilerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
