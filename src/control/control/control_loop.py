#!/usr/bin/env python3
import rclpy
import threading, time, math
import numpy as np

from control.ros_connect import ROSInterface
from Controls.src.control.control.control_utils import *

# Path / scenario constants
ROUTE_IS_LOOP = False
scaling_factor = 1.0  # kept in case you want to scale incoming path

# Controller / vehicle params
WHEELBASE_M = 1.5
MAX_STEER_RAD = 1.0  # 10 degrees

PROFILE_WINDOW_M = 80.0
PROFILE_HZ = 20
BRAKE_GAIN = 0.7
STOP_SPEED_THRESHOLD = 0.1   # m/s

# Jerk-limited velocity profile params
V_MIN = 4.0          # Lower minimum for tight turns
V_MAX = 8.0          # m/s
A_MAX = 15.0         # m/s^2
D_MAX = 20.0         # m/s^2
J_MAX = 70.0         # m/s^3

# Pure pursuit velocity limiting
STEER_SPEED_LIMIT_FACTOR = 0.5   # Factor to reduce speed based on steering angle
MAX_STEER_FOR_SPEED_LIMIT = 0.3  # Radians where speed limiting starts
STEER_RATE_THRESHOLD = 0.5       # Steering rate (rad/s) where additional speed limiting starts
STEER_RATE_FACTOR = 0.2          # How much to reduce speed based on steering rate

# Multi-point curvature parameters
CURVATURE_SAMPLE_POINTS = 8       # Number of points to sample for curvature
CURVATURE_LOOKAHEAD_FACTOR = 0.3  # How far ahead to look (fraction of profile window)


def main():
    rclpy.init()
    node = ROSInterface()

    # spin ROS in background
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True).start()

    # -------------------- Controllers --------------------
    th_pid = PID(3.2, 0, 0)
    steer_pid = PIDRange(kp=0.15, ki=0.05, kd=0.20,
                         out_min=-0.60, out_max=0.60)

    # -------------------- Control Loop --------------------
    cur_idx = 0
    last_t = time.perf_counter()
    last_profile_t = last_t
    prev_steering = 0.0  # For steering rate calculation

    input("Press Enter to start...")

    while rclpy.ok():
        # ---- 1) Get vehicle state ----
        cx, cy, yaw, speed, have_odom = node.get_state()
        if not have_odom:
            time.sleep(0.01)
            continue

        # ---- 2) Get local path from path-planner node ----
        # This must be implemented in ROSInterface (see section 2 below)
        route_x, route_y = node.get_local_path()  # NEW: dynamic path from another node

        if route_x is None or route_y is None or len(route_x) < 3:
            # No path yet or too few points to do pure pursuit safely
            # You can optionally send zero command here
            time.sleep(0.01)
            continue

        # Optional scaling / transform hook (if your planner is in a different frame)
        route_x = route_x * scaling_factor
        route_y = route_y * scaling_factor

        # Preprocess local path (non-loop)
        route_x, route_y, route_s, route_len = preprocess_path(
            route_x, route_y, loop=ROUTE_IS_LOOP
        )
        seg_ds = segment_distances(route_x, route_y, ROUTE_IS_LOOP)

        # Curvature + speed limits for *this* local segment
        kappa_signed = compute_signed_curvature(route_x, route_y)
        v_limit_global = ackermann_curv_speed_limit(
            kappa_signed, wheelbase=WHEELBASE_M, v_max=V_MAX, d_max=D_MAX
        )
        route_v = v_limit_global.copy()

        # ---- 3) Time update ----
        now = time.perf_counter()
        dt = now - last_t
        last_t = now

        # ---- 4) Index of closest point in the local segment ----
        # Since the planner is already giving a short window around the car,
        # just search from 0 each time (simpler & safe).
        cur_idx = local_closest_index((cx, cy), route_x, route_y, 0, loop=ROUTE_IS_LOOP)

        # ---- 5) Velocity profile refresh (periodic) ----
        if now - last_profile_t >= (1.0 / PROFILE_HZ):
            last_profile_t = now

            # Get pure pursuit lookahead point index
            Ld = calc_lookahead(speed)
            tgt_idx = forward_index_by_distance(cur_idx, Ld, route_s, route_len, ROUTE_IS_LOOP)

            # Pure pursuit geometry
            dx = route_x[tgt_idx] - cx
            dy = route_y[tgt_idx] - cy
            look_ahead_distance = math.sqrt(dx * dx + dy * dy)

            e_lat, _ = cross_track_error(cx, cy, route_x, route_y, cur_idx, loop=ROUTE_IS_LOOP)
            path_yaw = path_heading(route_x, route_y, cur_idx, loop=ROUTE_IS_LOOP)

            heading_error = yaw - path_yaw
            heading_error = math.atan2(math.sin(heading_error), math.cos(heading_error))

            # Pure pursuit curvature approximation
            if look_ahead_distance > 0.1:
                pp_curvature = 2.0 * e_lat / (look_ahead_distance * look_ahead_distance)
                pp_curvature = max(-CURVATURE_MAX, min(CURVATURE_MAX, pp_curvature))
            else:
                pp_curvature = 0.0

            # --- Multi-Point Curvature Calculation (on local path only) ---
            curvature_sample_points = []
            curvature_lookahead = min(PROFILE_WINDOW_M * CURVATURE_LOOKAHEAD_FACTOR, PROFILE_WINDOW_M)

            for i in range(CURVATURE_SAMPLE_POINTS):
                fraction = i / (CURVATURE_SAMPLE_POINTS - 1) if CURVATURE_SAMPLE_POINTS > 1 else 0.0
                sample_dist = fraction * curvature_lookahead
                sample_idx = forward_index_by_distance(cur_idx, sample_dist, route_s, route_len, ROUTE_IS_LOOP)
                curvature_sample_points.append(sample_idx)

            conservative_speed_limit = V_MAX
            for idx in curvature_sample_points:
                if idx != cur_idx:
                    # Crosstrack at sample point
                    e_lat_sample, _ = cross_track_error(
                        route_x[idx], route_y[idx], route_x, route_y, cur_idx, loop=ROUTE_IS_LOOP
                    )

                    sample_lookahead = calc_lookahead(speed)
                    if sample_lookahead > 0.1:
                        pp_curvature_sample = 2.0 * e_lat_sample / (sample_lookahead * sample_lookahead)
                        pp_curvature_sample = max(-CURVATURE_MAX, min(CURVATURE_MAX, pp_curvature_sample))
                    else:
                        pp_curvature_sample = 0.0

                    pp_safe_speed = ackermann_curv_speed_limit(
                        np.array([pp_curvature_sample]),
                        wheelbase=WHEELBASE_M, v_max=V_MAX, d_max=D_MAX
                    )[0]

                    path_curvature = kappa_signed[idx]
                    path_safe_speed = ackermann_curv_speed_limit(
                        np.array([path_curvature]),
                        wheelbase=WHEELBASE_M, v_max=V_MAX, d_max=D_MAX
                    )[0]

                    point_speed_limit = min(pp_safe_speed, path_safe_speed)
                    conservative_speed_limit = min(conservative_speed_limit, point_speed_limit)

            # Velocity profile window – limited by local path length
            end_idx = forward_index_by_distance(cur_idx, PROFILE_WINDOW_M, route_s, route_len, ROUTE_IS_LOOP)

            if ROUTE_IS_LOOP:
                if end_idx >= cur_idx:
                    idxs = np.arange(cur_idx, end_idx + 1)
                else:
                    idxs = np.concatenate((np.arange(cur_idx, len(route_x)), np.arange(0, end_idx + 1)))
            else:
                idxs = np.arange(cur_idx, min(end_idx + 1, len(route_x)))

            if len(idxs) == 0:
                # no valid window, skip this cycle
                time.sleep(0.005)
                continue

            if ROUTE_IS_LOOP:
                ds_win_list = [0.0]
                for ii in range(len(idxs) - 1):
                    ds_win_list.append(seg_ds[idxs[ii]])
                ds_win = np.asarray(ds_win_list, dtype=float)
            else:
                if len(idxs) > 0:
                    ds_win = np.concatenate(([0.0], seg_ds[idxs[:-1]]))
                else:
                    ds_win = np.asarray([0.0], dtype=float)

            v_lim_win = np.minimum(v_limit_global[idxs], conservative_speed_limit)
            v0 = speed
            # For a short local path, just use last point’s limit as vf
            vf = float(v_lim_win[-1])

            v_prof_win = jerk_limited_velocity_profile(
                v_lim_win, ds_win, v0, vf, V_MIN, V_MAX, A_MAX, D_MAX, J_MAX
            )

            if ROUTE_IS_LOOP and end_idx < cur_idx:
                n1 = len(route_x) - cur_idx
                route_v[cur_idx:] = v_prof_win[:n1]
                route_v[:end_idx + 1] = v_prof_win[n1:]
            else:
                route_v[cur_idx:cur_idx + len(v_prof_win)] = v_prof_win

        # ---- 6) Steering control ----
        steering_ppc, tgt_idx = pure_pursuit_steer(
            (cx, cy), yaw, speed,
            route_x, route_y, cur_idx,
            route_s, route_len, loop=ROUTE_IS_LOOP
        )

        e_lat, _ = cross_track_error(cx, cy, route_x, route_y, cur_idx, loop=ROUTE_IS_LOOP)
        steer_correction = steer_pid.update(e_lat, dt)
        steering_cmd = max(-1.0, min(1.0, steering_ppc + steer_correction))

        # ---- 7) Steering Rate Consideration ----
        steering_rate = abs(steering_cmd * MAX_STEER_RAD - prev_steering) / max(dt, 0.001)
        prev_steering = steering_cmd * MAX_STEER_RAD

        steering_rate_factor = 1.0
        if steering_rate > STEER_RATE_THRESHOLD:
            rate_reduction = (steering_rate - STEER_RATE_THRESHOLD) * STEER_RATE_FACTOR
            steering_rate_factor = max(0.1, 1.0 - rate_reduction)

        abs_steering = abs(steering_cmd * MAX_STEER_RAD)
        if abs_steering > MAX_STEER_FOR_SPEED_LIMIT:
            steering_factor = max(0.1, 1.0 - (abs_steering - MAX_STEER_FOR_SPEED_LIMIT) * STEER_SPEED_LIMIT_FACTOR)
            angle_factor = steering_factor
        else:
            angle_factor = 1.0

        speed_limit_factor = min(steering_rate_factor, angle_factor)
        limited_speed = route_v[cur_idx] * speed_limit_factor

        # ---- 8) Longitudinal control ----
        v_err = limited_speed - speed
        if v_err >= 0:
            throttle = th_pid.update(v_err, dt)
            brake = 0.0
        else:
            th_pid.reset()
            throttle, brake = 0.0, min(1.0, -v_err * BRAKE_GAIN)

        # ---- 9) Send command to ROS ----
        node.send_command(
         steering_cmd * MAX_STEER_RAD,
         accel=throttle - brake,
         speed=limited_speed)  # optional passthrough for logging or GUI


        time.sleep(0.005)

    rclpy.shutdown()


if __name__ == '__main__':
    main()
