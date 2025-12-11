#!/usr/bin/env python3
import rclpy
import threading, time, math
import numpy as np

from std_msgs.msg import Float32
from control.ros_connect import ROSInterface
from control.control_utils import (
    PID,
    preprocess_path,
    local_closest_index,
    path_heading,
    cross_track_error,
)

# ==============================
# CONFIGURATION
# ==============================

ROUTE_IS_LOOP = False
SCALING_FACTOR = 1.0

# Vehicle geometry & limits
WHEELBASE_M = 1.5
MAX_STEER_RAD = 1.0   # rad

# Stanley parameters
STANLEY_K = 1.2
STANLEY_EPS = 0.1

# Longitudinal PID limits
PID_KP = 3.2
PID_KI = 0.0
PID_KD = 0.0

MAX_ACCEL = 3.0      # m/s^2
MAX_BRAKE = -5.0     # m/s^2

STOP_SPEED_THRESHOLD = 0.1


# ==============================
# STANLEY CONTROLLER (LATERAL)
# ==============================

def stanley_steering(cx, cy, yaw, speed, route_x, route_y, cur_idx):
    """
    delta = heading_error + atan2(k * e_lat, v + eps)
    """

    # Path heading at closest index
    psi_ref = path_heading(route_x, route_y, cur_idx)

    # Heading error wrapped to [-pi, pi]
    heading_error = psi_ref - yaw
    heading_error = math.atan2(math.sin(heading_error), math.cos(heading_error))

    # Cross track error (signed)
    e_lat, _ = cross_track_error(cx, cy, route_x, route_y, cur_idx)

    # Avoid division by zero
    v_ctrl = max(speed, STANLEY_EPS)

    # Stanley steering law
    delta = heading_error + math.atan2(STANLEY_K * e_lat, v_ctrl)

    # Saturation
    delta = max(-MAX_STEER_RAD, min(MAX_STEER_RAD, delta))

    return delta


# ==============================
# MAIN CONTROLLER NODE
# ==============================

class StanleyPIDController:

    def __init__(self):
        rclpy.init()

        # ✅ ROSInterface (odom, local path, cmd)
        self.node = ROSInterface()

        # ✅ Spin ROS in background
        executor = rclpy.executors.SingleThreadedExecutor()
        executor.add_node(self.node)
        threading.Thread(target=executor.spin, daemon=True).start()

        # ✅ Subscribe to velocity profiler output
        self.desired_velocity = 0.0
        self.node.create_subscription(
            Float32,
            "/desired_velocity",
            self.velocity_cb,
            10
        )

        # ✅ Longitudinal PID
        self.th_pid = PID(PID_KP, PID_KI, PID_KD)

        # ✅ State
        self.cur_idx = 0
        self.last_t = time.perf_counter()

        input("✅ Press Enter to start CLEAN Stanley + PID control...\n")

    def velocity_cb(self, msg):
        """Target speed from velocity profiler node"""
        self.desired_velocity = float(msg.data)

    def run(self):
        while rclpy.ok():

            # ---------------------------------
            # 1) GET VEHICLE STATE (ODOMETRY)
            # ---------------------------------
            cx, cy, yaw, speed, have_odom = self.node.get_state()
            if not have_odom:
                time.sleep(0.01)
                continue

            # ---------------------------------
            # 2) GET LOCAL PATH
            # ---------------------------------
            route_x, route_y = self.node.get_local_path()

            if route_x is None or route_y is None or len(route_x) < 3:
                self.node.send_command(0.0, accel=0.0, speed=0.0)
                time.sleep(0.01)
                continue

            route_x = np.asarray(route_x, dtype=float) * SCALING_FACTOR
            route_y = np.asarray(route_y, dtype=float) * SCALING_FACTOR

            route_x, route_y, route_s, route_len = preprocess_path(
                route_x, route_y, loop=ROUTE_IS_LOOP
            )

            # ---------------------------------
            # 3) TIME UPDATE
            # ---------------------------------
            now = time.perf_counter()
            dt = max(now - self.last_t, 0.01)
            self.last_t = now

            # ---------------------------------
            # 4) CLOSEST POINT ON PATH
            # ---------------------------------
            self.cur_idx = local_closest_index(
                (cx, cy), route_x, route_y, self.cur_idx
            )
            self.cur_idx = int(self.cur_idx)

            # ---------------------------------
            # 5) LATERAL CONTROL → STANLEY
            # ---------------------------------
            steering_angle = stanley_steering(
                cx, cy, yaw, speed,
                route_x, route_y,
                self.cur_idx
            )

            # ---------------------------------
            # 6) LONGITUDINAL CONTROL (FROM PROFILER)
            # ---------------------------------
            target_speed = self.desired_velocity
            v_err = target_speed - speed

            if v_err >= 0:
                accel_cmd = self.th_pid.update(v_err, dt)
                accel_cmd = max(0.0, min(MAX_ACCEL, accel_cmd))
            else:
                self.th_pid.reset()
                accel_cmd = max(MAX_BRAKE, v_err)

            # ---------------------------------
            # 7) SEND COMMAND TO SIMULATOR
            # ---------------------------------
            self.node.send_command(
                steering_angle,
                accel=accel_cmd,
                speed=target_speed
            )

            # ---------------------------------
            # 8) STOP CONDITION
            # ---------------------------------
            if (
                (not ROUTE_IS_LOOP)
                and self.cur_idx >= len(route_x) - 1
                and speed < STOP_SPEED_THRESHOLD
            ):
                print(" Stanley: Reached end of path and stopped.")
                break

            time.sleep(0.005)

        rclpy.shutdown()


# ==============================
# ENTRY POINT
# ==============================

def main():
    ctrl = StanleyPIDController()
    ctrl.run()



if __name__ == "__main__":
    main()