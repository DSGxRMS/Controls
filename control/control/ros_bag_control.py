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
import transforms3d.euler as tft
from control.controls_functions import *


class ControlNode(Node):
    def __init__(self):
        super().__init__('control_node', automatically_declare_parameters_from_overrides=True)

        # ---- Params ----
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('cmd_topic', '/cmd')
        self.declare_parameter('mode', 'ackermann')  # 'ackermann' or 'twist'
        self.declare_parameter('path_csv', '/root/rosws/ctrl_main/src/control/control/pathpoints.csv')
        self.declare_parameter('scaling_factor', 1.0)
        self.declare_parameter('loop', False)
        self.declare_parameter('qos_best_effort', True)
        self.declare_parameter('hz', 50.0)  # Control rate
        self.declare_parameter('path_offset_x', 0.0)  # Path offset in x
        self.declare_parameter('path_offset_y', 0.0)  # Path offset in y
        self.declare_parameter('enable_startup_mode', True)  # Enable startup assistance

        self.odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
        self.cmd_topic = self.get_parameter('cmd_topic').get_parameter_value().string_value
        self.mode = self.get_parameter('mode').get_parameter_value().string_value.lower()
        self.path_csv = self.get_parameter('path_csv').get_parameter_value().string_value
        self.scaling_factor = float(self.get_parameter('scaling_factor').get_parameter_value().double_value)
        self.loop = self.get_parameter('loop').get_parameter_value().bool_value
        self.best_effort = self.get_parameter('qos_best_effort').get_parameter_value().bool_value
        self.hz = float(self.get_parameter('hz').get_parameter_value().double_value)
        self.path_offset_x = float(self.get_parameter('path_offset_x').get_parameter_value().double_value)
        self.path_offset_y = float(self.get_parameter('path_offset_y').get_parameter_value().double_value)
        self.enable_startup_mode = self.get_parameter('enable_startup_mode').get_parameter_value().bool_value

        # ---- Load path and initialize controller ----
        try:
            self.controller = self.load_path(self.path_csv, self.scaling_factor, self.loop)
            self.get_logger().info(f"Successfully loaded path with {len(self.controller.xs)} points")
        except Exception as e:
            self.get_logger().error(f"Failed to load path: {e}")
            self.controller = None

        # ---- QoS ----
        qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=QoSReliabilityPolicy.BEST_EFFORT if self.best_effort
                        else QoSReliabilityPolicy.RELIABLE
        )

        # ---- Vehicle state ----
        self.current_state = {
            'position': (0.0, 0.0),
            'yaw': 0.0,
            'speed': 0.0,
            'has_odom': False
        }
        self.last_stamp = None
        self.startup_complete = False

        # ---- Subscriptions ----
        self.create_subscription(Odometry, self.odom_topic, self._odom_cb, qos)

        # ---- Publisher ----
        if self.mode == 'ackermann':
            self.pub_ack = self.create_publisher(AckermannDriveStamped, self.cmd_topic, 10)
            self.pub_twist = None
        else:
            self.pub_twist = self.create_publisher(Twist, self.cmd_topic, 10)
            self.pub_ack = None

        # ---- Timer for consistent publishing ----
        self.timer = self.create_timer(1.0 / max(1.0, self.hz), self._control_loop)

        self.get_logger().info(
            f"[control_node] odom={self.odom_topic} -> {self.mode}@{self.cmd_topic} "
            f"(path={self.path_csv}, loop={self.loop}, scaling={self.scaling_factor}, "
            f"offset=({self.path_offset_x}, {self.path_offset_y}), "
            f"{'BEST_EFFORT' if self.best_effort else 'RELIABLE'}, {self.hz:.1f}Hz)"
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
        
        # Apply scaling
        x_scaled = df["x"].to_numpy() * scaling_factor
        y_scaled = df["y"].to_numpy() * scaling_factor
        
        # Resample track
        rx, ry = resample_track(x_scaled, y_scaled)
        
        # Apply coordinate transformation and offsets
        # Original transformation: route_x, route_y = ry + 15.0, -rx
        # Fixed transformation to align with vehicle start position
        route_x = rx + self.path_offset_x  # Use configurable offset instead of hardcoded 15.0
        route_y = ry + self.path_offset_y
        
        self.get_logger().info(f"Path bounds: x=[{route_x.min():.2f}, {route_x.max():.2f}], "
                              f"y=[{route_y.min():.2f}, {route_y.max():.2f}]")
        
        return ControlAlgorithm(route_x, route_y, loop=loop)

    def _odom_cb(self, msg: Odometry):
        """
        Odometry callback: extract and store vehicle state.
        """
        try:
            # Extract vehicle state
            cx = msg.pose.pose.position.x
            cy = msg.pose.pose.position.y
            speed = abs(msg.twist.twist.linear.x)  # Ensure positive speed

            # Compute yaw
            q = msg.pose.pose.orientation
            yaw = tft.quat2euler([q.w, q.x, q.y, q.z])[2]  # Note: different order for transforms3d

            now = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

            if self.last_stamp is None:
                dt = 0.1  # Initial dt
            else:
                dt = now - self.last_stamp
                if dt <= 0:
                    return

            self.last_stamp = now

            # Update stored state
            self.current_state = {
                'position': (cx, cy),
                'yaw': yaw,
                'speed': speed,
                'has_odom': True
            }

            # Update controller state if available
            if self.controller is not None:
                self.controller.update_state((cx, cy), yaw, speed, dt, now)

        except Exception as e:
            self.get_logger().error(f"Error in odometry callback: {e}")

    def _control_loop(self):
        """
        Main control loop - called by timer at regular intervals.
        """
        try:
            if not self.current_state['has_odom']:
                # No odometry data yet, publish zero commands
                self._publish_zero_commands()
                return

            if self.controller is None:
                # No valid controller, publish zero commands
                self._publish_zero_commands()
                return

            # Get current state
            pos_xy = self.current_state['position']
            yaw = self.current_state['yaw']
            speed = self.current_state['speed']
            dt = 1.0 / self.hz

            # Handle startup mode
            if self.enable_startup_mode and not self.startup_complete:
                startup_throttle, startup_brake, is_started = startup_control(speed)
                if is_started:
                    self.startup_complete = True
                    self.get_logger().info("Startup phase completed, switching to path following")
                else:
                    # Publish startup commands
                    self._publish_commands(0.0, startup_throttle, startup_brake, speed)
                    self.get_logger().debug(f"Startup mode: speed={speed:.2f} m/s, throttle={startup_throttle:.3f}")
                    return

            # Compute controls using the controller
            controls = self.controller.compute_controls(pos_xy, yaw, speed, dt)

            # Extract control values
            steering_cmd = controls['steering_cmd']
            throttle = controls['throttle']
            brake = controls['brake']

            # Log control information
            self.get_logger().debug(
                f"Control: pos=({pos_xy[0]:.2f}, {pos_xy[1]:.2f}), "
                f"speed={speed:.2f} m/s, steer={steering_cmd:.3f}, "
                f"throttle={throttle:.3f}, brake={brake:.3f}, "
                f"v_err={controls.get('v_err', 0.0):.2f}"
            )

            # Publish controls
            self._publish_commands(steering_cmd, throttle, brake, speed)

        except Exception as e:
            self.get_logger().error(f"Error in control loop: {e}")
            self._publish_zero_commands()

    def _publish_commands(self, steering_cmd, throttle, brake, current_speed):
        """
        Publish control commands.
        
        Args:
            steering_cmd (float): Normalized steering command [-1, 1]
            throttle (float): Throttle command [0, 1]
            brake (float): Brake command [0, 1]
            current_speed (float): Current vehicle speed
        """
        try:
            if self.pub_ack:
                ack_msg = AckermannDriveStamped()
                ack_msg.header.stamp = self.get_clock().now().to_msg()
                ack_msg.drive.steering_angle = steering_cmd * MAX_STEER_RAD
                
                # Map throttle/brake to acceleration
                accel = throttle * AX_MAX - brake * abs(AX_MIN)
                ack_msg.drive.acceleration = accel
                ack_msg.drive.speed = current_speed  # Current speed
                
                self.pub_ack.publish(ack_msg)
                
            else:
                twist_msg = Twist()
                twist_msg.linear.x = throttle - brake  # Simplified for twist
                twist_msg.angular.z = steering_cmd * MAX_STEER_RAD
                self.pub_twist.publish(twist_msg)
                
        except Exception as e:
            self.get_logger().error(f"Error publishing commands: {e}")

    def _publish_zero_commands(self):
        """
        Publish zero control commands (safe stop).
        """
        self._publish_commands(0.0, 0.0, 0.0, 0.0)


def main():
    rclpy.init()
    node = ControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()