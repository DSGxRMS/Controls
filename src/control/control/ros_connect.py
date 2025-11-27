#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import Float64
import math


class ROSInterface(Node):
    def __init__(self, odom_topic="/gt_odom", cmd_topic="/cmd"):
        super().__init__("ros_interface")

        # Vehicle state
        self.cx, self.cy = 0.0, 0.0
        self.yaw = 0.0
        self.speed = 0.0
        self.have_odom = False

        # Subscriber for ground truth odom
        self.create_subscription(Odometry, odom_topic, self._odom_cb, 10)

        # Command publisher (primary control output)
        self.cmd_pub = self.create_publisher(AckermannDriveStamped, cmd_topic, 10)

        # Debug publishers for plotter / tuning
        self.pub_target_v = self.create_publisher(Float64, "/debug/target_speed", 10)
        self.pub_speed    = self.create_publisher(Float64, "/debug/speed", 10)
        self.pub_steer    = self.create_publisher(Float64, "/debug/steering", 10)
        self.pub_cte      = self.create_publisher(Float64, "/debug/crosstrack", 10)
        self.pub_accel    = self.create_publisher(Float64, "/debug/accel", 10)
        self.pub_brake    = self.create_publisher(Float64, "/debug/brake", 10)
        self.pub_s        = self.create_publisher(Float64, "/debug/s", 10)

        self.get_logger().info(
            f"[ROSInterface] INFO: reading odom from {odom_topic}, publishing accel+steer commands to {cmd_topic}"
        )

    # ------------------------- ODOM CALLBACK -------------------------
    def _odom_cb(self, msg):
        self.cx = msg.pose.pose.position.x
        self.cy = msg.pose.pose.position.y

        # quaternion -> yaw
        q = msg.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y*q.y + q.z*q.z)
        self.yaw = math.atan2(siny, cosy)

        vx, vy = msg.twist.twist.linear.x, msg.twist.twist.linear.y
        self.speed = math.sqrt(vx*vx + vy*vy)

        self.have_odom = True

    # ------------------------- PUBLIC API -------------------------
    def get_state(self):
        return (self.cx, self.cy, self.yaw, self.speed, self.have_odom)

    def send_command(self, steering, accel, speed=None):
        """
        Primary control: steering + acceleration.
        Speed is optional and only passed through if explicitly used by sim.
        """
        msg = AckermannDriveStamped()
        msg.drive.steering_angle = float(steering)
        msg.drive.acceleration   = float(accel)

        # OPTIONAL: keep speed for simulator recording but not for actual control policy
        if speed is not None:
            msg.drive.speed = float(speed)

        self.cmd_pub.publish(msg)
        # Debugging throttle (do not spam)
        # self.get_logger().debug(f"Cmd: steer={steering:.3f}, accel={accel:.3f}, speed(optional)={speed}")
