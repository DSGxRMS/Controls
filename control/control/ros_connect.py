import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
import math

class ROSInterface(Node):
    def __init__(self, odom_topic="/ground_truth/odom", cmd_topic="/cmd"):
        super().__init__('ros_interface')

        self.cx, self.cy, self.yaw, self.speed = 0.0, 0.0, 0.0, 0.0
        self.have_odom = False

        self.create_subscription(Odometry, odom_topic, self._odom_cb, 10)
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

    def get_state(self):
        """Returns (x, y, yaw, speed, have_odom)"""
        return (self.cx, self.cy, self.yaw, self.speed, self.have_odom)

    def send_command(self, steering, speed=None, accel=None):
        msg = AckermannDriveStamped()
        msg.drive.steering_angle = steering
        if accel is not None:
            msg.drive.acceleration = accel
        if speed is not None:
            msg.drive.speed = speed
        self.get_logger().info(f"Sending command: Steer={steering:.2f}, Speed={speed}, Accel={accel}")
        self.pub.publish(msg)