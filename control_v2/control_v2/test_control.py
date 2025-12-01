#!/usr/bin/env python3
"""
Test script for the improved control system.
This script can be used to test various aspects of the control system.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy
import time
import math
import numpy as np
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, Quaternion, Point, Pose, PoseStamped
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import Header
import transforms3d.euler as tft


class ControlTester(Node):
    def __init__(self):
        super().__init__('control_tester')
        
        # Parameters
        self.declare_parameter('test_duration', 50.0)  # seconds
        self.declare_parameter('acceleration', 5.0)  # m/s²
        self.declare_parameter('steering_angle', 0.1)  # radians
        
        self.test_duration = self.get_parameter('test_duration').get_parameter_value().double_value
        self.acceleration = self.get_parameter('acceleration').get_parameter_value().double_value
        self.steering_angle = self.get_parameter('steering_angle').get_parameter_value().double_value
        
        # QoS
        qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=QoSReliabilityPolicy.BEST_EFFORT
        )
        
        # Publishers
        self.odom_pub = self.create_publisher(Odometry, '/odom', qos)
        self.cmd_pub = self.create_publisher(AckermannDriveStamped, '/cmd', 10)
        
        # Subscribers to monitor control output
        self.create_subscription(AckermannDriveStamped, '/cmd', self._cmd_cb, 10)
        
        # Test state
        self.start_time = time.time()
        self.cmd_received = False
        self.last_cmd = None
        
        # Timer for publishing test odometry and commands
        self.timer = self.create_timer(0.02, self._publish_test_data)  # 50 Hz
        
        self.get_logger().info(f"Starting control test for moving mode with {self.acceleration} m/s² acceleration and {self.steering_angle} rad steering for {self.test_duration} seconds")

    def _cmd_cb(self, msg: AckermannDriveStamped):
        """Monitor control commands"""
        self.cmd_received = True
        self.last_cmd = msg
        
        self.get_logger().info(
            f"Command received: steering={msg.drive.steering_angle:.3f} rad, "
            f"accel={msg.drive.acceleration:.3f} m/s², speed={msg.drive.speed:.3f} m/s"
        )

    def _publish_test_data(self):
        """Publish test odometry data and control commands"""
        current_time = time.time()
        elapsed = current_time - self.start_time
        
        if elapsed > self.test_duration:
            self._report_results()
            rclpy.shutdown()
            return
        
        # Publish control commands
        self._publish_control_commands()
        
        # Publish test odometry for moving scenario
        self._publish_moving_odom(elapsed)

    def _publish_control_commands(self):
        """Publish acceleration and steering commands"""
        cmd_msg = AckermannDriveStamped()
        cmd_msg.header = Header()
        cmd_msg.header.stamp = self.get_clock().now().to_msg()
        cmd_msg.header.frame_id = 'base_link'
        
        # Set acceleration and steering commands
        cmd_msg.drive.acceleration = self.acceleration  # 5 m/s²
        cmd_msg.drive.steering_angle = self.steering_angle  # steering command
        cmd_msg.drive.speed = 0.0  # Let acceleration control the speed
        
        # Publish command
        self.cmd_pub.publish(cmd_msg)

    def _publish_moving_odom(self, elapsed):
        """Publish odometry data for moving scenario"""
        # Create odometry message
        odom = Odometry()
        odom.header = Header()
        odom.header.stamp = self.get_clock().now().to_msg()
        odom.header.frame_id = 'odom'
        odom.child_frame_id = 'base_link'
        
        # Vehicle moving with acceleration
        # Using kinematic equations: x = 0.5 * a * t^2 (starting from rest)
        x = 0.5 * self.acceleration * elapsed * elapsed
        y = 0.0
        yaw = 0.0
        
        # Velocity: v = a * t
        vx = self.acceleration * elapsed
        vy = 0.0
        vyaw = 0.0
        
        # Set position
        odom.pose.pose.position = Point(x=x, y=y, z=0.0)
        
        # Set orientation (quaternion from yaw)
        quat = tft.euler2quat(0, 0, yaw)
        odom.pose.pose.orientation = Quaternion(x=quat[1], y=quat[2], z=quat[3], w=quat[0])
        
        # Set velocities
        odom.twist.twist.linear.x = vx
        odom.twist.twist.linear.y = vy
        odom.twist.twist.angular.z = vyaw
        
        # Publish
        self.odom_pub.publish(odom)

    def _report_results(self):
        """Report test results"""
        self.get_logger().info("=" * 50)
        self.get_logger().info("CONTROL TEST RESULTS")
        self.get_logger().info("=" * 50)
        
        if self.cmd_received:
            self.get_logger().info("✓ Control commands were received")
            if self.last_cmd:
                self.get_logger().info(f"  Last command: steering={self.last_cmd.drive.steering_angle:.3f} rad, "
                                     f"accel={self.last_cmd.drive.acceleration:.3f} m/s², "
                                     f"speed={self.last_cmd.drive.speed:.3f} m/s")
        else:
            self.get_logger().error("✗ No control commands received - CHECK YOUR CONTROL NODE!")
        
        self.get_logger().info(f"Test mode: moving")
        self.get_logger().info(f"Acceleration command: {self.acceleration} m/s²")
        self.get_logger().info(f"Steering command: {self.steering_angle} rad")
        self.get_logger().info(f"Test duration: {self.test_duration:.1f} seconds")
        self.get_logger().info("=" * 50)


def main():
    rclpy.init()
    
    # Create and run tester
    tester = ControlTester()
    
    try:
        rclpy.spin(tester)
    except KeyboardInterrupt:
        pass
    finally:
        tester.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()