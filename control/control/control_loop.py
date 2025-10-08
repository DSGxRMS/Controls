#!/usr/bin/env python3
import rclpy
import threading, time, math
import numpy as np
import pandas as pd
from control.ros_connect import ROSInterface
from control.control_utils import *

# Path / scenario constants (keep these here if you like)

PATHPOINTS_CSV = "/root/rosws/ctrl_main/src/control/control/pathpoints.csv"
ROUTE_IS_LOOP = False
scaling_factor = 1

# Controller / vehicle params (also mirrored in control_utils for safety)

WHEELBASE_M = 1.5
MAX_STEER_RAD = 1 # 10 degrees

PROFILE_WINDOW_M = 100.0
PROFILE_HZ = 10
BRAKE_GAIN = 0.7
STOP_SPEED_THRESHOLD = 0.1   # m/s, vehicle considered stopped

# Jerk-limited velocity profile params (passed into the utility)

V_MIN = 2.0          # Lower minimum for tight turns
V_MAX = 8.0          # m/s
A_MAX = 15.0         # m/s^2
D_MAX = 20.0         # m/s^2 (max decel)
J_MAX = 70.0         # m/s^3

# Pure pursuit velocity limiting
STEER_SPEED_LIMIT_FACTOR = 0.5  # Factor to reduce speed based on steering angle
MAX_STEER_FOR_SPEED_LIMIT = 0.3  # Maximum steering angle where speed limiting starts

def main():
    rclpy.init()
    node = ROSInterface()

    # spin ROS in background
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True).start()

    # -------------------- Load path --------------------
    df = pd.read_csv(PATHPOINTS_CSV)
    rx, ry = resample_track(df["x"].to_numpy() * scaling_factor,
                            df["y"].to_numpy() * scaling_factor)
    route_x, route_y = ry + 15.0, -rx
    route_x, route_y, route_s, route_len = preprocess_path(route_x, route_y, loop=ROUTE_IS_LOOP)
    seg_ds = segment_distances(route_x, route_y, ROUTE_IS_LOOP)

    kappa_signed = compute_signed_curvature(route_x, route_y)
    v_limit_global = ackermann_curv_speed_limit(kappa_signed, wheelbase=WHEELBASE_M, v_max=V_MAX, d_max=D_MAX)
    route_v = v_limit_global.copy()

    # -------------------- Controllers --------------------
    th_pid = PID(3.2, 0, 0)
    steer_pid = PIDRange(kp=0.15, ki=0.05, kd=0.20,
                         out_min=-0.60, out_max=0.60)

    # -------------------- Control Loop --------------------
    cur_idx = 0
    last_t = time.perf_counter()
    last_profile_t = last_t

    input("Press Enter to start...")

    while rclpy.ok():
        cx, cy, yaw, speed, have_odom = node.get_state()
        if not have_odom:
            time.sleep(0.01)
            continue

        now = time.perf_counter()
        dt = now - last_t
        last_t = now

        # update index
        cur_idx = local_closest_index((cx, cy), route_x, route_y, cur_idx, loop=ROUTE_IS_LOOP)

        # refresh velocity profile (periodically)
        if now - last_profile_t >= (1.0 / PROFILE_HZ):
            last_profile_t = now
            
            # Get pure pursuit lookahead point
            Ld = calc_lookahead(speed)
            tgt_idx = forward_index_by_distance(cur_idx, Ld, route_s, route_len, ROUTE_IS_LOOP)
            
            # Calculate pure pursuit path curvature from current position to lookahead point
            # We'll use the arc that pure pursuit would follow
            dx = route_x[tgt_idx] - cx
            dy = route_y[tgt_idx] - cy
            look_ahead_distance = math.sqrt(dx*dx + dy*dy)
            
            # Calculate the curvature of the pure pursuit arc
            # This is an approximation based on the cross-track error and look-ahead distance
            _, path_yaw = cross_track_error(cx, cy, route_x, route_y, cur_idx, loop=ROUTE_IS_LOOP)
            path_yaw = path_heading(route_x, route_y, cur_idx, loop=ROUTE_IS_LOOP)  # Get actual path heading
            
            # Calculate vehicle heading relative to path
            heading_error = yaw - path_yaw
            # Normalize to [-pi, pi]
            heading_error = math.atan2(math.sin(heading_error), math.cos(heading_error))
            
            # Calculate cross-track error
            e_lat, _ = cross_track_error(cx, cy, route_x, route_y, cur_idx, loop=ROUTE_IS_LOOP)
            
            # Pure pursuit curvature approximation: kappa = 2 * cross_track_error / (look_ahead_distance^2)
            if look_ahead_distance > 0.1:  # Avoid division by zero
                pp_curvature = 2.0 * e_lat / (look_ahead_distance * look_ahead_distance)
                pp_curvature = max(-CURVATURE_MAX, min(CURVATURE_MAX, pp_curvature))
            else:
                pp_curvature = 0.0
            
            # Calculate safe speed based on pure pursuit path curvature
            pp_safe_speed = ackermann_curv_speed_limit(np.array([pp_curvature]), wheelbase=WHEELBASE_M, v_max=V_MAX, d_max=D_MAX)[0]
            
            # Also consider the original path curvature
            path_curvature = kappa_signed[cur_idx]
            path_safe_speed = ackermann_curv_speed_limit(np.array([path_curvature]), wheelbase=WHEELBASE_M, v_max=V_MAX, d_max=D_MAX)[0]
            
            # Use the more conservative speed (minimum of both)
            conservative_speed_limit = min(pp_safe_speed, path_safe_speed)
            
            # Create a window for velocity profile calculation based on pure pursuit considerations
            end_idx = forward_index_by_distance(cur_idx, PROFILE_WINDOW_M, route_s, route_len, ROUTE_IS_LOOP)

            if ROUTE_IS_LOOP:
                if end_idx >= cur_idx:
                    idxs = np.arange(cur_idx, end_idx + 1)
                else:
                    idxs = np.concatenate((np.arange(cur_idx, len(route_x)), np.arange(0, end_idx + 1)))
            else:
                idxs = np.arange(cur_idx, end_idx + 1)

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

            # Use the conservative speed limit for this window
            v_lim_win = np.minimum(v_limit_global[idxs], conservative_speed_limit)
            v0 = speed
            vf = 0.0 if (not ROUTE_IS_LOOP and len(idxs) > 0 and idxs[-1] == len(route_x) - 1) else float(v_lim_win[-1])

            v_prof_win = jerk_limited_velocity_profile(
                v_lim_win, ds_win, v0, vf, V_MIN, V_MAX, A_MAX, D_MAX, J_MAX
            )

            if ROUTE_IS_LOOP and end_idx < cur_idx:
                n1 = len(route_x) - cur_idx
                route_v[cur_idx:] = v_prof_win[:n1]
                route_v[:end_idx + 1] = v_prof_win[n1:]
            else:
                route_v[cur_idx:end_idx + 1] = v_prof_win

        # --- Steering control ---
        steering_ppc, tgt_idx = pure_pursuit_steer((cx, cy), yaw, speed,
                                                   route_x, route_y, cur_idx,
                                                   route_s, route_len, loop=ROUTE_IS_LOOP)
        e_lat, _ = cross_track_error(cx, cy, route_x, route_y, cur_idx, loop=ROUTE_IS_LOOP)
        steer_correction = steer_pid.update(e_lat, dt)
        steering_cmd = max(-1.0, min(1.0, steering_ppc + steer_correction))
        
        # Apply steering-based velocity limiting
        abs_steering = abs(steering_cmd * MAX_STEER_RAD)
        if abs_steering > MAX_STEER_FOR_SPEED_LIMIT:
            # Reduce speed based on steering angle
            steering_factor = max(0.1, 1.0 - (abs_steering - MAX_STEER_FOR_SPEED_LIMIT) * STEER_SPEED_LIMIT_FACTOR)
            limited_speed = route_v[cur_idx] * steering_factor
        else:
            limited_speed = route_v[cur_idx]

        # --- Longitudinal control ---
        v_err = limited_speed - speed
        if v_err >= 0:
            throttle = th_pid.update(v_err, dt)
            brake = 0.0
        else:
            th_pid.reset()
            throttle, brake = 0.0, min(1.0, -v_err * BRAKE_GAIN)

        # --- Send command to ROS --- (steering in radians, speed in m/s)
        desired_speed = limited_speed
        # If you want to send acceleration instead of speed modify ROSInterface send_command to accept accel.
        node.send_command(steering_cmd * MAX_STEER_RAD, speed=desired_speed, accel=throttle - brake)

        # --- Exit condition ---
        if (not ROUTE_IS_LOOP) and cur_idx >= len(route_x) - 1 and speed < STOP_SPEED_THRESHOLD:
            print("Reached end of route and stopped. Exiting loop.")
            break

        time.sleep(0.02)

    rclpy.shutdown()

if __name__ == '__main__':
    main()