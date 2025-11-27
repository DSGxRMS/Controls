#!/usr/bin/env python3

#THIS WILL GIVE ALL THE PLOTS OF THE CAR CONTROLLER IN REAL TIME

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64
from std_msgs.msg import Float64MultiArray
from nav_msgs.msg import Odometry

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import time

class ControlPlotter(Node):
    def __init__(self):
        super().__init__("control_plotter")

        # Data buffers
        self.s_data = []                # distance along path
        self.speed_data = []            # actual vehicle speed
        self.target_v_data = []         # target speed from controller
        self.steer_data = []            # steering angle cmd
        self.cte_data = []              # cross-track error
        self.accel_data = []            # accel command
        self.brake_data = []            # brake command

        # ROS Subscribers
        self.create_subscription(Float64, "/vehicle/s", self.s_callback, 10)
        self.create_subscription(Float64, "/vehicle/speed", self.speed_callback, 10)
        self.create_subscription(Float64, "/control/target_speed", self.target_speed_callback, 10)
        self.create_subscription(Float64, "/control/steer", self.steer_callback, 10)
        self.create_subscription(Float64, "/control/crosstrack_error", self.cte_callback, 10)
        self.create_subscription(Float64, "/control/accel", self.accel_callback, 10)
        self.create_subscription(Float64, "/control/brake", self.brake_callback, 10)

        self.get_logger().info("Real-time controller plotter initialized.")

        # Setup GUI
        self.fig, self.ax = plt.subplots(5, 1, figsize=(10, 15))
        plt.tight_layout(pad=2.0)

        self.ani = animation.FuncAnimation(self.fig, self.update_plot, interval=100)
        plt.show()

    # ------------------- Callbacks ----------------------
    def s_callback(self, msg): self._append(self.s_data, msg.data)
    def speed_callback(self, msg): self._append(self.speed_data, msg.data)
    def target_speed_callback(self, msg): self._append(self.target_v_data, msg.data)
    def steer_callback(self, msg): self._append(self.steer_data, msg.data)
    def cte_callback(self, msg): self._append(self.cte_data, msg.data)
    def accel_callback(self, msg): self._append(self.accel_data, msg.data)
    def brake_callback(self, msg): self._append(self.brake_data, msg.data)

    def _append(self, arr, val, limit=5000):
        arr.append(val)
        if len(arr) > limit:
            arr.pop(0)

    # ------------------- Plot Update ----------------------
    def update_plot(self, frame):
        if len(self.s_data) < 2:
            return

        s = np.array(self.s_data)

        self.ax[0].clear()
        self.ax[0].plot(s, self.speed_data, label="Speed")
        self.ax[0].plot(s, self.target_v_data, label="Velocity Profile", alpha=0.7)
        self.ax[0].set_title("Speed vs Distance")
        self.ax[0].legend()
        self.ax[0].grid()

        self.ax[1].clear()
        self.ax[1].plot(s, self.steer_data, label="Steering Angle", color="orange")
        self.ax[1].set_title("Steering Angle vs Distance")
        self.ax[1].grid()

        self.ax[2].clear()
        self.ax[2].plot(s, self.cte_data, label="Cross Track Error", color="red")
        self.ax[2].set_title("Cross Track Error vs Distance")
        self.ax[2].grid()

        self.ax[3].clear()
        self.ax[3].plot(s, self.accel_data, label="Accel", color="green")
        self.ax[3].plot(s, self.brake_data, label="Brake", color="purple")
        self.ax[3].set_title("Acceleration & Brake Inputs")
        self.ax[3].legend()
        self.ax[3].grid()

        # throttle + brake stacked
        self.ax[4].clear()
        self.ax[4].plot(self.accel_data, label="Accel", color="green")
        self.ax[4].plot(self.brake_data, label="Brake", color="purple")
        self.ax[4].set_title("Accel vs Brake over time")
        self.ax[4].legend()
        self.ax[4].grid()

def main(args=None):
    rclpy.init(args=args)
    node = ControlPlotter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
