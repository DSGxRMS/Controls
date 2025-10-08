#!/usr/bin/env python3
import rclpy
import threading, time, math
import numpy as np
import pandas as pd
from control.ros_connect import ROSInterface
from control.control_utils import *
from pathlib import Path
# Path / scenario constants
PATHPOINTS_CSV = Path(__file__).parent / "pathpoints.csv"
ROUTE_IS_LOOP = False
scaling_factor = 1

# Controller / vehicle params
WHEELBASE_M = 1.5
MAX_STEER_RAD = 1 # 10 degrees

PROFILE_WINDOW_M = 100.0
PROFILE_HZ = 10
BRAKE_GAIN = 0.7
STOP_SPEED_THRESHOLD = 0.1   # m/s

# Jerk-limited velocity profile params
V_MIN = 4.0          # Lower minimum for tight turns
V_MAX = 8.0          # m/s
A_MAX = 15.0         # m/s^2
D_MAX = 20.0         # m/s^2
J_MAX = 70.0         # m/s^3

# Pure pursuit velocity limiting
STEER_SPEED_LIMIT_FACTOR = 0.5  # Factor to reduce speed based on steering angle
MAX_STEER_FOR_SPEED_LIMIT = 0.3  # Radians where speed limiting starts
STEER_RATE_THRESHOLD = 0.5       # Steering rate (rad/s) where additional speed limiting starts
STEER_RATE_FACTOR = 0.2          # How much to reduce speed based on steering rate

# Multi-point curvature parameters
CURVATURE_SAMPLE_POINTS = 8      # Number of points to sample for curvature
CURVATURE_LOOKAHEAD_FACTOR = 0.3 # How far ahead to look (fraction of profile window)

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
    prev_steering = 0.0  # For steering rate calculation
    
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
            dx = route_x[tgt_idx] - cx
            dy = route_y[tgt_idx] - cy
            look_ahead_distance = math.sqrt(dx*dx + dy*dy)
            
            # Calculate the curvature of the pure pursuit arc
            # This is an approximation based on the cross-track error and look-ahead distance
            e_lat, _ = cross_track_error(cx, cy, route_x, route_y, cur_idx, loop=ROUTE_IS_LOOP)
            path_yaw = path_heading(route_x, route_y, cur_idx, loop=ROUTE_IS_LOOP)
            
            # Calculate heading error
            heading_error = yaw - path_yaw
            heading_error = math.atan2(math.sin(heading_error), math.cos(heading_error))
            
            # Calculate pure pursuit curvature approximation
            if look_ahead_distance > 0.1:
                pp_curvature = 2.0 * e_lat / (look_ahead_distance * look_ahead_distance)
                pp_curvature = max(-CURVATURE_MAX, min(CURVATURE_MAX, pp_curvature))
            else:
                pp_curvature = 0.0
            
            # --- Multi-Point Curvature Calculation ---
            # Sample multiple points along the path to get a more accurate speed limit
            curvature_sample_points = []
            curvature_lookahead = min(PROFILE_WINDOW_M * CURVATURE_LOOKAHEAD_FACTOR, PROFILE_WINDOW_M)
            
            # Get indices for sampling points
            for i in range(CURVATURE_SAMPLE_POINTS):
                fraction = i / (CURVATURE_SAMPLE_POINTS - 1)
                sample_dist = fraction * curvature_lookahead
                sample_idx = forward_index_by_distance(cur_idx, sample_dist, route_s, route_len, ROUTE_IS_LOOP)
                curvature_sample_points.append(sample_idx)
            
            # Find the most conservative speed limit from these points
            conservative_speed_limit = V_MAX
            for idx in curvature_sample_points:
                # Calculate speed limit for pure pursuit path curvature at this point
                if idx != cur_idx:
                    # Get cross track error at this point
                    e_lat_sample, _ = cross_track_error(route_x[idx], route_y[idx], route_x, route_y, cur_idx, loop=ROUTE_IS_LOOP)
                    # Calculate look ahead distance for this point (simplified)
                    sample_lookahead = calc_lookahead(speed)
                    if sample_lookahead > 0.1:
                        pp_curvature_sample = 2.0 * e_lat_sample / (sample_lookahead * sample_lookahead)
                        pp_curvature_sample = max(-CURVATURE_MAX, min(CURVATURE_MAX, pp_curvature_sample))
                    else:
                        pp_curvature_sample = 0.0
                    
                    pp_safe_speed = ackermann_curv_speed_limit(np.array([pp_curvature_sample]), 
                                                              wheelbase=WHEELBASE_M, v_max=V_MAX, d_max=D_MAX)[0]
                    
                    # Also consider the original path curvature
                    path_curvature = kappa_signed[idx]
                    path_safe_speed = ackermann_curv_speed_limit(np.array([path_curvature]), 
                                                               wheelbase=WHEELBASE_M, v_max=V_MAX, d_max=D_MAX)[0]
                    
                    # Take the minimum speed limit
                    point_speed_limit = min(pp_safe_speed, path_safe_speed)
                    conservative_speed_limit = min(conservative_speed_limit, point_speed_limit)
            
            # Create a window for velocity profile calculation
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
        
        # --- Steering Rate Consideration ---
        # Calculate how fast steering is changing
        steering_rate = abs(steering_cmd * MAX_STEER_RAD - prev_steering) / max(dt, 0.001)
        prev_steering = steering_cmd * MAX_STEER_RAD  # Update for next iteration
        
        # Apply steering rate-based speed limiting
        steering_rate_factor = 1.0
        if steering_rate > STEER_RATE_THRESHOLD:
            # Reduce speed more aggressively for higher steering rates
            rate_reduction = (steering_rate - STEER_RATE_THRESHOLD) * STEER_RATE_FACTOR
            steering_rate_factor = max(0.1, 1.0 - rate_reduction)
        
        # Apply steering angle-based velocity limiting
        abs_steering = abs(steering_cmd * MAX_STEER_RAD)
        if abs_steering > MAX_STEER_FOR_SPEED_LIMIT:
            # Reduce speed based on steering angle
            steering_factor = max(0.1, 1.0 - (abs_steering - MAX_STEER_FOR_SPEED_LIMIT) * STEER_SPEED_LIMIT_FACTOR)
            angle_factor = steering_factor
        else:
            angle_factor = 1.0
        
        # Combine both limiting factors
        speed_limit_factor = min(steering_rate_factor, angle_factor)
        limited_speed = route_v[cur_idx] * speed_limit_factor

        # --- Longitudinal control ---
        v_err = limited_speed - speed
        if v_err >= 0:
            throttle = th_pid.update(v_err, dt)
            brake = 0.0
        else:
            th_pid.reset()
            throttle, brake = 0.0, min(1.0, -v_err * BRAKE_GAIN)

        # --- Send command to ROS ---
        node.send_command(steering_cmd * MAX_STEER_RAD, speed=limited_speed, accel=throttle - brake)

        # --- Exit condition ---
        if (not ROUTE_IS_LOOP) and cur_idx >= len(route_x) - 1 and speed < STOP_SPEED_THRESHOLD:
            print("Reached end of route and stopped. Exiting loop.")
            break

        time.sleep(0.02)

    rclpy.shutdown()

if __name__ == '__main__':
    main()