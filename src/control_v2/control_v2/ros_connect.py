import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from nav_msgs.msg import Odometry, Path
from ackermann_msgs.msg import AckermannDriveStamped
import math


class ROSInterface(Node):
    def __init__(self, odom_topic="/slam/odom", cmd_topic="/cmd", path_topic="/path_points"):
        super().__init__('ros_interface')

        self.cx, self.cy, self.yaw, self.speed = 0.0, 0.0, 0.0, 0.0
        self.have_odom = False
        self.latest_path = None

        # Best Effort QoS (typical for high-rate sensor-ish streams)
        self.qos_best_effort = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE
        )

        self.create_subscription(Odometry, odom_topic, self._odom_cb, self.qos_best_effort)
        self.create_subscription(Path, path_topic, self._path_cb, self.qos_best_effort)

        self.pub = self.create_publisher(AckermannDriveStamped, cmd_topic, 10)

    def _odom_cb(self, msg):
        self.cx = msg.pose.pose.position.x
        self.cy = msg.pose.pose.position.y

        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.yaw = math.atan2(siny_cosp, cosy_cosp)

        vx, vy = msg.twist.twist.linear.x, msg.twist.twist.linear.y
        self.speed = (vx**2 + vy**2) ** 0.5

        self.have_odom = True

    def _path_cb(self, msg):
        self.latest_path = msg

    def get_state(self):
        """Returns (x, y, yaw, speed, have_odom)"""
        return (self.cx, self.cy, self.yaw, self.speed, self.have_odom)

    def get_path(self):
        """Returns list of (x, y) path points in global coordinates"""
        if self.latest_path is None:
            return []

        return [(p.pose.position.x, p.pose.position.y) for p in self.latest_path.poses]

    def send_command(self, steering, speed=None, accel=None):
        msg = AckermannDriveStamped()
        msg.drive.steering_angle = float(steering)
        if accel is not None:
            msg.drive.acceleration = float(accel)
        if speed is not None:
            msg.drive.speed = float(speed)

        self.get_logger().info(
            f"Sending command: Steer={steering:.2f}, Speed={speed}, Accel={accel}"
        )
        self.pub.publish(msg)
