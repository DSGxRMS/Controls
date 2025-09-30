import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy
import time
import math
import numpy as np
import pandas as pd
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from ackermann_msgs.msg import AckermannDriveStamped
import tf.transformations as tft
from controls_functions import *


class ControlNode(Node):
    def __init__(self):
        super().__init__('control_node', automatically_declare_parameters_from_overrides=True)

        # ---- Params ----
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('cmd_topic', '/cmd')
        self.declare_parameter('mode', 'ackermann')  # 'ackermann' or 'twist'
        self.declare_parameter('control/pathpoints.csv', 'pathpoints.csv')
        self.declare_parameter('scaling_factor', 1.0)
        self.declare_parameter('loop', False)
        self.declare_parameter('qos_best_effort', True)

        self.odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
        self.cmd_topic = self.get_parameter('cmd_topic').get_parameter_value().string_value
        self.mode = self.get_parameter('mode').get_parameter_value().string_value.lower()
        self.path_csv = self.get_parameter('path_csv').get_parameter_value().string_value
        self.scaling_factor = float(self.get_parameter('scaling_factor').get_parameter_value().double_value)
        self.loop = self.get_parameter('loop').get_parameter_value().bool_value
        self.best_effort = self.get_parameter('qos_best_effort').get_parameter_value().bool_value

        # ---- Load path and initialize controller ----
        self.controller = self.load_path(self.path_csv, self.scaling_factor, self.loop)

        # ---- QoS ----
        qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=QoSReliabilityPolicy.BEST_EFFORT if self.best_effort
                        else QoSReliabilityPolicy.RELIABLE
        )

        # ---- Subscriptions ----
        self.last_stamp = None
        self.create_subscription(Odometry, self.odom_topic, self._odom_cb, qos)

        # ---- Publisher ----
        if self.mode == 'ackermann':
            self.pub_ack = self.create_publisher(AckermannDriveStamped, self.cmd_topic, 10)
            self.pub_twist = None
        else:
            self.pub_twist = self.create_publisher(Twist, self.cmd_topic, 10)
            self.pub_ack = None

        self.get_logger().info(
            f"[control_node] odom={self.odom_topic} -> {self.mode}@{self.cmd_topic} "
            f"(path={self.path_csv}, loop={self.loop}, scaling={self.scaling_factor}, "
            f"{'BEST_EFFORT' if self.best_effort else 'RELIABLE'})"
        )

    def load_path(self, csv_path, scaling_factor, loop):
        """
        Load and preprocess path from CSV.

        Args:
            csv_path (str): Path to CSV file
            scaling_factor (float): Scaling factor
            loop (bool): Whether path is a loop

        Returns:
            ControlAlgorithm: Initialized control algorithm
        """
        df = pd.read_csv(csv_path)
        rx, ry = resample_track(df["x"].to_numpy() * scaling_factor,
                                df["y"].to_numpy() * scaling_factor)
        route_x, route_y = ry + 15.0, -rx
        return ControlAlgorithm(route_x, route_y, loop=loop)

    def _odom_cb(self, msg: Odometry):
        """
        Odometry callback: extract state, update controller, compute controls, publish.
        """
        # Extract vehicle state
        cx = msg.pose.pose.position.x
        cy = msg.pose.pose.position.y
        speed = msg.twist.twist.linear.x

        # Compute yaw
        q = msg.pose.pose.orientation
        yaw = tft.euler_from_quaternion([q.x, q.y, q.z, q.w])[2]

        now = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        if self.last_stamp is None:
            dt = 0.1  # Initial dt
        else:
            dt = now - self.last_stamp
            if dt <= 0:
                return

        self.last_stamp = now

        # Update controller state
        self.controller.update_state((cx, cy), yaw, speed, dt, now)

        # Compute controls
        controls = self.controller.compute_controls((cx, cy), yaw, speed, dt)

        # Publish
        if self.pub_ack:
            ack_msg = AckermannDriveStamped()
            ack_msg.header.stamp = msg.header.stamp
            ack_msg.drive.steering_angle = controls['steering_cmd'] * MAX_STEER_RAD
            # Map throttle/brake to acceleration
            accel = controls['throttle'] * AX_MAX - controls['brake'] * abs(AX_MIN)
            ack_msg.drive.acceleration = accel
            ack_msg.drive.speed = speed  # Current speed
            self.pub_ack.publish(ack_msg)
        else:
            twist_msg = Twist()
            twist_msg.linear.x = controls['throttle'] - controls['brake']  # Simplified
            twist_msg.angular.z = controls['steering_cmd'] * MAX_STEER_RAD
            self.pub_twist.publish(twist_msg)


def main():
    rclpy.init()
    node = ControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()