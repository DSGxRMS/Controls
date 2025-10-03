import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import Twist
import numpy as np
import time

from .path_manager import PathManager
from .controller import Controller

class ControlNode(Node):
    """
    Manages ROS2 interactions, including the node, parameters, topics, and the main loop.
    """
    def __init__(self):
        super().__init__('control_node')

        # Declare and get parameters
        self._declare_parameters()
        
        # Initialize PathManager and Controller
        self.path_manager = PathManager(
            csv_path=self.get_parameter('path_csv').value,
            scaling_factor=self.get_parameter('scaling_factor').value,
            loop=self.get_parameter('loop').value,
            path_offset_x=self.get_parameter('path_offset_x').value,
            path_offset_y=self.get_parameter('path_offset_y').value
        )
        
        controller_config = {
            'la_dist_min': 1.0, 'la_dist_max': 5.0,
            'la_vel_min': 2.0, 'la_vel_max': 10.0,
            'startup_v_threshold': 0.5, 'startup_throttle': 0.3
        }
        self.controller = Controller(self.path_manager, controller_config)

        # State variables
        self.current_state = {'x': 0.0, 'y': 0.0, 'yaw': 0.0, 'v': 0.0}
        self.last_time = time.time()
        self.in_startup_mode = self.get_parameter('enable_startup_mode').value

        # Setup publishers and subscribers
        self._setup_ros_comms()

    def _declare_parameters(self):
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('cmd_topic', '/cmd')
        self.declare_parameter('mode', 'ackermann')
        self.declare_parameter('path_csv', '/root/rosws/ctrl_main/src/control/control/pathpoints.csv')
        self.declare_parameter('scaling_factor', 1.0)
        self.declare_parameter('loop', False)
        self.declare_parameter('hz', 50.0)
        self.declare_parameter('path_offset_x', 0.0)
        self.declare_parameter('path_offset_y', 0.0)
        self.declare_parameter('enable_startup_mode', True)

    def _setup_ros_comms(self):
        # Subscribers
        self.create_subscription(
            Odometry,
            self.get_parameter('odom_topic').value,
            self._odom_cb,
            10
        )

        # Publishers
        self.mode = self.get_parameter('mode').value
        if self.mode == 'ackermann':
            self.cmd_pub = self.create_publisher(AckermannDriveStamped, self.get_parameter('cmd_topic').value, 10)
        else:
            self.cmd_pub = self.create_publisher(Twist, self.get_parameter('cmd_topic').value, 10)

        # Control loop timer
        self.create_timer(1.0 / self.get_parameter('hz').value, self._control_loop)

    def _odom_cb(self, msg):
        # Extract state from odometry message
        q = msg.pose.pose.orientation
        yaw = np.arctan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y**2 + q.z**2))
        
        self.current_state['x'] = msg.pose.pose.position.x
        self.current_state['y'] = msg.pose.pose.position.y
        self.current_state['yaw'] = yaw
        self.current_state['v'] = msg.twist.twist.linear.x

    def _control_loop(self):
        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time

        if dt <= 0: return

        state = self.current_state
        
        if self.in_startup_mode:
            controls = self.controller.startup_control(state['v'])
            if controls:
                steer, throttle, brake = controls
            else:
                self.in_startup_mode = False
        
        if not self.in_startup_mode:
            steer, throttle, brake = self.controller.compute_controls(
                state['x'], state['y'], state['yaw'], state['v'], dt
            )
            
        self._publish_commands(steer, throttle, brake)

    def _publish_commands(self, steer, throttle, brake):
        if self.mode == 'ackermann':
            msg = AckermannDriveStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.drive.steering_angle = steer
            msg.drive.speed = throttle
            # Ackermann messages don't typically have a separate brake field.
            # We can model it by reducing speed.
            if brake > 0:
                msg.drive.speed = 0.0
        else: # Twist
            msg = Twist()
            msg.linear.x = throttle
            if brake > 0:
                msg.linear.x = 0.0
            msg.angular.z = steer
            
        self.cmd_pub.publish(msg)