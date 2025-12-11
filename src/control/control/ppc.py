#!/usr/bin/env python3
import rclpy
import threading, time, math
import numpy as np

from std_msgs.msg import Float32
from control.ros_connect import ROSInterface
from control.control_utils import (
    PID,
    PIDRange,
    preprocess_path,
    local_closest_index,
    path_heading,
    cross_track_error,
    pure_pursuit_steer,
)

# ==============================
# CONFIGURATION
# ==============================

ROUTE_IS_LOOP = False
SCALING_FACTOR = 1.0

# Vehicle geometry
WHEELBASE_M = 1.5
MAX_STEER_RAD = 1.0  # rad

# PID for throttle
THROTTLE_KP = 3.2
THROTTLE_KI = 0.0
THROTTLE_KD = 0.0

BRAKE_GAIN = 0.7
STOP_SPEED_THRESHOLD = 0.1


# ==============================
# MAIN PPC CONTROLLER NODE
# ==============================

class PurePursuitController:

    def __init__(self):
        rclpy.init()

        # ✅ ROS Interface (odom, path, cmd)
        self.node = ROSInterface()

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

        # ✅ Controllers
        self.th_pid = PID(THROTTLE_KP, THROTTLE_KI, THROTTLE_KD)
        self.steer_pid = PIDRange(kp=0.15, ki=0.05, kd=0.20,
                                  out_min=-0.60, out_max=0.60)

        # ✅ State
        self.cur_idx = 0
        self.last_t = time.perf_counter()
        self.prev_steering = 0.0

        input("✅ Press Enter to start CLEAN Pure Pursuit Control...\n")


    def velocity_cb(self, msg):
        """Velocity target from your old velocity profiler"""
        self.desired_velocity = float(msg.data)


    def run(self):

        while rclpy.ok():

            # ---------------------------------
            # 1) GET VEHICLE STATE
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

            route_x = route_x * SCALING_FACTOR
            route_y = route_y * SCALING_FACTOR

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
            # 4) CLOSEST PATH INDEX
            # ---------------------------------
            self.cur_idx = local_closest_index(
                (cx, cy), route_x, route_y, self.cur_idx, loop=ROUTE_IS_LOOP
            )
            self.cur_idx = int(self.cur_idx)

            # ---------------------------------
            # 5) PURE PURSUIT STEERING
            # ---------------------------------
            steering_pp, _ = pure_pursuit_steer(
                (cx, cy), yaw, speed,
                route_x, route_y,
                self.cur_idx,
                route_s, route_len,
                loop=ROUTE_IS_LOOP
            )

            # Cross-track error correction
            e_lat, _ = cross_track_error(
                cx, cy, route_x, route_y,
                self.cur_idx, loop=ROUTE_IS_LOOP
            )

            steer_corr = self.steer_pid.update(e_lat, dt)
            steering_cmd = max(-1.0, min(1.0, steering_pp + steer_corr))        #steering command calculation
            steering_cmd *= MAX_STEER_RAD

            # ---------------------------------
            # 6) LONGITUDINAL CONTROL (FROM PROFILER)
            # ---------------------------------
            target_speed = self.desired_velocity
            v_err = target_speed - speed

            if v_err >= 0:
                throttle = self.th_pid.update(v_err, dt)
                brake = 0.0
            else:
                self.th_pid.reset()
                throttle = 0.0
                brake = min(1.0, -v_err * BRAKE_GAIN)

            accel_cmd = throttle - brake

            # ---------------------------------
            # 7) SEND COMMAND
            # ---------------------------------
            self.node.send_command(
                steering_cmd,
                accel=accel_cmd,
                speed=target_speed
            )

            # ---------------------------------
            # 8) STOP CONDITION
            # ---------------------------------
            if (not ROUTE_IS_LOOP and
                self.cur_idx >= len(route_x) - 2 and
                speed < STOP_SPEED_THRESHOLD):
                print("✅ PPC: End of path reached.")
                break

            time.sleep(0.005)

        rclpy.shutdown()


# ==============================
# ENTRY POINT
# ==============================

def main():
    ctrl = PurePursuitController()
    ctrl.run()
    
    
    
if __name__ == "__main__":
    main()
