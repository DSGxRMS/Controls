#!/usr/bin/env python3
import rclpy
import threading, time, math
import numpy as np

from std_msgs.msg import Float32
from control.ros_connect import ROSInterface
from control.control_utils import (
    preprocess_path,
    local_closest_index,
    path_heading,
    cross_track_error,
    compute_signed_curvature,
)

# ======================================
# CONFIGURATION (MASS BODY CAR)
# ======================================

ROUTE_IS_LOOP = False
SCALING_FACTOR = 1.0

WHEELBASE_M   = 1.5
MAX_STEER_RAD = 1.0

MAX_ACCEL = 3.0
MAX_BRAKE = -5.0

STOP_SPEED_THRESHOLD = 0.1

# ======================================
# LQR WEIGHTS  (TUNED FOR MASS-BODY CAR)
# ======================================

Q_LQR = np.diag([
    10.0,   # lateral error (e_y)
    6.0,    # heading error (e_psi)
    5.0     # velocity error (v - v_ref)
])

R_LQR = np.diag([
    0.4,    # steering effort
    0.6     # acceleration effort
])


# ======================================
# DISCRETE RICCATI SOLVER
# ======================================

def solve_dare(A, B, Q, R, iters=150, tol=1e-8):
    P = Q.copy()
    for _ in range(iters):
        BT_P = B.T @ P
        S = R + BT_P @ B
        K_term = np.linalg.solve(S, BT_P)
        Pn = A.T @ P @ A - A.T @ P @ B @ K_term + Q
        if np.max(np.abs(Pn - P)) < tol:
            P = Pn
            break
        P = Pn
    return P


def compute_lqr_gain(A, B):
    P = solve_dare(A, B, Q_LQR, R_LQR)
    BT_P = B.T @ P
    S = R_LQR + BT_P @ B
    K = np.linalg.solve(S, BT_P @ A)
    return K


# ======================================
# FULL LQR CONTROL LAW
# ======================================

def full_lqr_control(cx, cy, yaw, v, v_ref, route_x, route_y, idx, dt):

    # --- Path geometry from utilities ---
    e_y, psi_ref = cross_track_error(cx, cy, route_x, route_y, idx)
    kappa = compute_signed_curvature(route_x, route_y)[idx]

    # --- Errors ---
    e_psi = math.atan2(math.sin(psi_ref - yaw), math.cos(psi_ref - yaw))
    e_v   = v_ref - v

    x = np.array([[e_y],
                  [e_psi],
                  [e_v]])

    # --- Linearized discrete dynamics (mass-body kinematic bicycle) ---
    v_eff = max(v, 0.5)
    dt_eff = max(dt, 1e-3)

    A = np.array([
        [1.0,         v_eff * dt_eff,              0.0],
        [0.0,         1.0,          dt_eff * v_eff / WHEELBASE_M],
        [0.0,         0.0,                       1.0]
    ])

    B = np.array([
        [0.0,                         0.0],
        [dt_eff * v_eff / WHEELBASE_M, 0.0],
        [0.0,                         dt_eff]
    ])

    # --- LQR Gain ---
    K = compute_lqr_gain(A, B)   # shape (2,3)

    # --- State feedback ---
    u = -K @ x
    delta = float(u[0])
    accel = float(u[1])

    # --- Feedforward steering from curvature ---
    delta_ff = math.atan(WHEELBASE_M * kappa)
    delta += delta_ff

    # --- Actuator saturation ---
    delta = max(-MAX_STEER_RAD, min(MAX_STEER_RAD, delta))
    accel = max(MAX_BRAKE, min(MAX_ACCEL, accel))

    return delta, accel


# ======================================
# ROS2 LQR CONTROLLER NODE
# ======================================

class LQRFullController:

    def __init__(self):
        rclpy.init()
        self.node = ROSInterface()

        executor = rclpy.executors.SingleThreadedExecutor()
        executor.add_node(self.node)
        threading.Thread(target=executor.spin, daemon=True).start()

        # ✅ External velocity profiler
        self.desired_velocity = 0.0
        self.node.create_subscription(
            Float32,
            "/desired_velocity",
            self.velocity_cb,
            10
        )

        self.cur_idx = 0
        self.last_t = time.perf_counter()

        input("✅ Press Enter to start FULL LQR MASS-BODY CONTROL...\n")

    def velocity_cb(self, msg):
        self.desired_velocity = float(msg.data)

    def run(self):

        while rclpy.ok():

            # --- ODOMETRY ---
            cx, cy, yaw, speed, have_odom = self.node.get_state()
            if not have_odom:
                time.sleep(0.01)
                continue

            # --- LOCAL PATH ---
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

            # --- TIME STEP ---
            now = time.perf_counter()
            dt = max(now - self.last_t, 0.01)
            self.last_t = now

            # --- CLOSEST PATH INDEX ---
            self.cur_idx = local_closest_index(
                (cx, cy), route_x, route_y, self.cur_idx
            )
            self.cur_idx = int(self.cur_idx)

            # --- FULL LQR CONTROL ---
            v_ref = self.desired_velocity
            steer, accel = full_lqr_control(
                cx, cy, yaw, speed,
                v_ref,
                route_x, route_y,
                self.cur_idx, dt
            )

            # --- SEND COMMAND (PURE LQR ACCEL + STEER) ---
            self.node.send_command(
                steer,
                accel=accel,
                speed=0.0   # let velocity evolve via physics
            )

            # --- STOP CONDITION ---
            if (
                (not ROUTE_IS_LOOP)
                and self.cur_idx >= len(route_x) - 2
                and speed < STOP_SPEED_THRESHOLD
            ):
                print("✅ LQR: End of path reached.")
                break

            time.sleep(0.005)

        rclpy.shutdown()


# ======================================
# ENTRY POINT
# ======================================

if __name__ == "__main__":
    ctrl = LQRFullController()
    ctrl.run()
