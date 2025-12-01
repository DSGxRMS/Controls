#!/usr/bin/env python3
import rclpy
import threading, time, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from control_v2.ros_connect import ROSInterface
from control_v2.control_utils import *
from control_v2.telemetryplot import TelemetryVisualizer, generate_turning_arc

# ================================
# Control Constants
# ================================
MAX_VELOCITY = 4.0
VEL_LIMIT_FACTOR = 0.3
LOOK_AHEAD_UPDATE_INTERVAL =1.2  # Seconds between lookahead point updates
ROUTE_IS_LOOP = False
STOP_SPEED_THRESHOLD = 0.1
WHEELBASE_M = 1.5  # Critical for valid Arc visualization

# Updated Steering Constant based on your PPC function
MAX_STEER_RAD = math.pi / 2  # 90 degrees

# ================================
# Visualization Constants
# ================================
VIZ_UPDATE_HZ = 20  # Limit plot updates to 20 FPS to prevent main-thread blocking

def main():
    rclpy.init()
    node = ROSInterface()
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)
    
    # Run ROS spin in a separate thread to keep the buffer fresh
    threading.Thread(target=executor.spin, daemon=True).start()

    # Controllers
    th_pid = PID(3.2, 0, 0)
    
    # ---------------------------------------------------------
    # Initialization Phase
    # ---------------------------------------------------------
    print("⏳ Waiting for first odometry message...")
    while True:
        cx, cy, yaw, speed, have_odom = node.get_state()
        if have_odom:
            print(f"✅ First position received: x={cx:.2f}, y={cy:.2f}, speed={speed:.2f}")
            break
        time.sleep(0.1)

    input("Press Enter to start control loop...")

    # Logic Variables
    last_lookahead_update = 0
    last_control_time = time.perf_counter()
    cur_idx = 0
    
    # Lookahead State (Persisted between updates)
    look_ahead_x, look_ahead_y = None, None
    look_ahead_dist = 0.0

    # Visualization State
    viz = None
    last_viz_update = time.perf_counter()

    # Enable interactive mode for live plotting
    plt.ion()

    # ---------------------------------------------------------
    # Main Control Loop
    # ---------------------------------------------------------
    while rclpy.ok():
        now = time.perf_counter()
        dt = now - last_control_time
        last_control_time = now

        # 1. Perception & Path
        path_points = node.get_path()

        if not path_points:
            time.sleep(0.05)
            continue

        path_points = np.array(path_points)
        route_x, route_y = path_points[:, 0], path_points[:, 1]
        
        # Lazy Initialization of Visualizer (Needs valid path data first)
        if viz is None and len(route_x) > 0:
            # Create a dummy max velocity profile for the heatmap if a real one isn't available
            dummy_route_v = np.full_like(route_x, MAX_VELOCITY)
            viz = TelemetryVisualizer(route_x, route_y, dummy_route_v)
            plt.show()

        # 2. State Estimation
        curve = compute_signed_curvature(route_x, route_y)
        cx, cy, yaw, speed, have_odom = node.get_state()
        cur_idx = local_closest_index((cx, cy), route_x, route_y, cur_idx, loop=ROUTE_IS_LOOP)
        
        # Ensure cur_idx is a scalar integer to prevent array ambiguity errors
        if np.ndim(cur_idx) > 0:
            cur_idx = cur_idx[0]
        cur_idx = int(cur_idx)
        
        # 3. Lookahead Logic (Updated periodically as per control-v2 logic)
        if now - last_lookahead_update >= LOOK_AHEAD_UPDATE_INTERVAL:
            last_lookahead_update = now
            _, _, s, route_len = preprocess_path(route_x, route_y, loop=ROUTE_IS_LOOP)
            # Unpack all return values directly. calc_lookahead_point returns (tx, ty, Ld, idx)
            look_ahead_x, look_ahead_y, look_ahead_dist, look_ahead_idx = calc_lookahead_point(speed, route_x, route_y, cur_idx, s, route_len, loop=ROUTE_IS_LOOP)
           
        # 4. Steering Control (Pure Pursuit)
        if look_ahead_x is not None:
            # Function now returns NORMALIZED steering (-1.0 to 1.0)
            steering_norm = pure_pursuit_steer((cx, cy), yaw, look_ahead_x, look_ahead_y, look_ahead_dist)
        else:
            steering_norm = 0.0

        # Calculate the actual physical steering angle in radians
        # This ensures the plot and the simulator receive the exact same value
        actual_steering_rad = steering_norm * MAX_STEER_RAD

        # 5. Speed Control (Curvature-based)
        # Prevent index out of bounds if path changes
        safe_idx = min(cur_idx, len(curve) - 1)
        target_speed = MAX_VELOCITY * (1 - VEL_LIMIT_FACTOR * abs(curve[safe_idx]))
        
        speed_error = target_speed - speed
        # Use .update() instead of .compute() to match PID class definition
        accel_cmd = th_pid.update(speed_error, dt=dt)
        accel_cmd = max(-3.0, min(2.0, accel_cmd))

        # 6. Visualization Logging
        if viz is not None:
            # Update global path visualization (in case path changes)
            dummy_route_v = np.full_like(route_x, MAX_VELOCITY)
            viz.update_path_data(route_x, route_y, dummy_route_v)

            # Prepare data
            viz_lookahead = (look_ahead_x, look_ahead_y) if look_ahead_x is not None else None
            
            # Generate Arc: Uses RADIANS (physics)
            # Use the actual calculated radians, not just the normalized value
            arc_pts = generate_turning_arc(cx, cy, yaw, actual_steering_rad, WHEELBASE_M)

            viz.log_state(
                x=cx, y=cy, yaw=yaw, speed=speed,
                steering_cmd=steering_norm,  # Log normalized value (-1 to 1) directly
                lookahead_pt=viz_lookahead,
                future_pts=[],              # Feature not used in this version
                arc_pts=arc_pts,
                target_speed=target_speed
            )

            # Throttled Rendering
            if now - last_viz_update >= (1.0 / VIZ_UPDATE_HZ):
                viz.update_plot_manual()
                plt.pause(0.001)
                last_viz_update = now

        # 7. Actuation
        node.send_command(
            steering=actual_steering_rad,  # Send the actual radians to the simulator
            speed=target_speed,
            accel=accel_cmd
         )

        # 8. Termination Condition
        if (not ROUTE_IS_LOOP) and cur_idx >= len(route_x) - 1 and speed < STOP_SPEED_THRESHOLD:
            print("✅ Reached end of route and stopped. Exiting.")
            break

        time.sleep(0.05)

    rclpy.shutdown()
    plt.ioff()
    plt.show()

if __name__ == '__main__':
    main()