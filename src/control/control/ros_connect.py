#!/usr/bin/env python3
import math
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy

from nav_msgs.msg import Odometry
from std_msgs.msg import Float64MultiArray
from ackermann_msgs.msg import AckermannDriveStamped


class ROSInterface(Node):
    """
    Bridge between:
      - EUFSIM: /ground_truth/odom, /cmd (AckermannDriveStamped)
      - LocalPathPlanner: /local_path_points (Float64MultiArray)

    Exposes:
      - get_state()      -> (x, y, yaw, speed, have_odom)
      - get_local_path() -> (route_x, route_y) or (None, None)
      - send_command(steer, accel, speed)
    """

    def __init__(self):
        super().__init__(
            "pure_pursuit_controller",
            automatically_declare_parameters_from_overrides=True,
        )

        # ---- Parameters (can override via launch/CLI) ----
        self.declare_parameter("odom_topic", "/ground_truth/odom")
        self.declare_parameter("path_topic", "/local_path_points")
        self.declare_parameter("cmd_topic", "/cmd")
        self.declare_parameter("qos_best_effort", True)

        odom_topic = self.get_parameter("odom_topic").get_parameter_value().string_value
        path_topic = self.get_parameter("path_topic").get_parameter_value().string_value
        cmd_topic = self.get_parameter("cmd_topic").get_parameter_value().string_value
        best_effort = self.get_parameter("qos_best_effort").get_parameter_value().bool_value

        # ---- Internal state ----
        self._have_odom = False
        self._x = 0.0
        self._y = 0.0
        self._yaw = 0.0
        self._speed = 0.0

        self._route_x = None  # numpy arrays
        self._route_y = None

        # ---- QoS for ground-truth odom (match EUFSIM) ----
        odom_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=QoSReliabilityPolicy.BEST_EFFORT if best_effort
                        else QoSReliabilityPolicy.RELIABLE,
        )

        # ---- Subscriptions ----
        self.create_subscription(Odometry, odom_topic, self._odom_cb, odom_qos)
        self.create_subscription(Float64MultiArray, path_topic, self._path_cb, 10)

        # ---- Publisher to EUFSIM ----
        self._cmd_pub = self.create_publisher(AckermannDriveStamped, cmd_topic, 10)

        self.get_logger().info(
            f"[ROSInterface] odom={odom_topic}, path={path_topic}, cmd={cmd_topic}, "
            f"odom_qos={'BEST_EFFORT' if best_effort else 'RELIABLE'}"
        )

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    def _odom_cb(self, msg: Odometry):
        """Store latest vehicle pose and speed."""
        self._x = msg.pose.pose.position.x
        self._y = msg.pose.pose.position.y

        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self._yaw = math.atan2(siny_cosp, cosy_cosp)

        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        self._speed = math.hypot(vx, vy)

        self._have_odom = True

    def _path_cb(self, msg: Float64MultiArray):
        """
        msg.data layout from LocalPathPlanner:
            [x0, x1, ..., xN, y0, y1, ..., yN]
        """
        data = np.asarray(msg.data, dtype=float)
        if data.size == 0:
            self._route_x = None
            self._route_y = None
            return

        if data.size % 2 != 0:
            self.get_logger().warn(
                f"/local_path_points length {data.size} is not even; ignoring"
            )
            return

        n = data.size // 2
        self._route_x = data[:n]
        self._route_y = data[n:]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_state(self):
        """
        Returns:
            (x, y, yaw, speed, have_odom_flag)
        """
        return self._x, self._y, self._yaw, self._speed, self._have_odom

    def get_local_path(self):
        """
        Returns (route_x, route_y) as numpy arrays if available,
        otherwise (None, None).
        """
        if self._route_x is None or self._route_y is None:
            return None, None
        # Return copies so external code can't corrupt internal buffers
        return self._route_x.copy(), self._route_y.copy()

    def send_command(self, steering: float, accel: float, speed: float):
        """
        Publish AckermannDriveStamped to EUFSIM's /cmd.
        """
        msg = AckermannDriveStamped()
        msg.drive.steering_angle = float(steering)
        msg.drive.acceleration   = float(accel)
        msg.drive.speed          = float(speed)
        self._cmd_pub.publish(msg)
