import numpy as np
import math
from .control_utils import (
    PID, PIDRange, local_closest_index, pure_pursuit_steer, cross_track_error,
    jerk_limited_velocity_profile, forward_index_by_distance, segment_distances,
    ackermann_curv_speed_limit, compute_signed_curvature,
    V_MIN, V_MAX, A_MAX, D_MAX, J_MAX, PROFILE_HZ, PROFILE_WINDOW_M, BRAKE_GAIN,
    MAX_STEER_RAD, WHEELBASE_M
)

class ControlAlgorithm:
    """
    Encapsulates path data, PID controllers, and control computation methods.
    """
    def __init__(self, path_manager):
        """
        Initialize with a PathManager instance.

        Args:
            path_manager (PathManager): An instance of the PathManager class.
        """
        self.path_manager = path_manager
        self.loop = path_manager.loop
        self.xs = path_manager.xs
        self.ys = path_manager.ys
        self.s = path_manager.s
        self.total_len = path_manager.total_len

        self.seg_ds = segment_distances(self.xs, self.ys, self.loop)
        self.kappa_signed = compute_signed_curvature(self.xs, self.ys)
        self.v_limit_global = ackermann_curv_speed_limit(self.kappa_signed)
        self.route_v = self.v_limit_global.copy()

        # Controllers
        self.th_pid = PID(3.2, 0.5, 0.134)
        self.steer_pid = PIDRange(kp=0.15, ki=0.05, kd=0.20, out_min=-0.60, out_max=0.60)

        # State
        self.cur_idx = 0
        self.last_profile_t = 0.0

    def update_state(self, pos_xy, yaw, speed, dt, now):
        """
        Update internal state with current vehicle state.

        Args:
            pos_xy (tuple): (x, y) position
            yaw (float): Yaw angle
            speed (float): Speed
            dt (float): Time step
            now (float): Current time
        """
        self.cur_idx = local_closest_index(pos_xy, self.xs, self.ys, self.cur_idx, self.loop)

        # Refresh velocity profile if needed
        if now - self.last_profile_t >= (1.0 / PROFILE_HZ):
            self.last_profile_t = now
            end_idx = forward_index_by_distance(self.cur_idx, PROFILE_WINDOW_M, self.s, self.total_len, self.loop)

            if self.loop:
                if end_idx >= self.cur_idx:
                    idxs = np.arange(self.cur_idx, end_idx + 1)
                else:
                    idxs = np.concatenate((np.arange(self.cur_idx, len(self.xs)), np.arange(0, end_idx + 1)))
            else:
                idxs = np.arange(self.cur_idx, end_idx + 1)

            if self.loop:
                ds_win_list = [0.0]
                for ii in range(len(idxs) - 1):
                    ds_win_list.append(self.seg_ds[idxs[ii]])
                ds_win = np.asarray(ds_win_list, dtype=float)
            else:
                ds_win = np.concatenate(([0.0], self.seg_ds[idxs[:-1]]))

            v_lim_win = self.v_limit_global[idxs]
            v0 = speed
            vf = 0.0 if (not self.loop and idxs[-1] == len(self.xs) - 1) else float(v_lim_win[-1])

            v_prof_win = jerk_limited_velocity_profile(
                v_lim_win, ds_win, v0, vf, V_MIN, V_MAX, A_MAX, D_MAX, J_MAX
            )

            if self.loop and end_idx < self.cur_idx:
                n1 = len(self.xs) - self.cur_idx
                self.route_v[self.cur_idx:] = v_prof_win[:n1]
                self.route_v[:end_idx + 1] = v_prof_win[n1:]
            else:
                self.route_v[self.cur_idx:end_idx + 1] = v_prof_win

    def compute_controls(self, pos_xy, yaw, speed, dt):
        """
        Compute steering and throttle controls.

        Args:
            pos_xy (tuple): (x, y) position
            yaw (float): Yaw angle
            speed (float): Speed
            dt (float): Time step

        Returns:
            dict: Control outputs and logging data
        """
        # Pure Pursuit feedforward steering
        steering_ppc, tgt_idx = pure_pursuit_steer(pos_xy, yaw, speed, self.xs, self.ys, self.cur_idx, self.s, self.total_len, self.loop)

        # Lateral error feedback
        e_lat, theta_ref = cross_track_error(pos_xy[0], pos_xy[1], self.xs, self.ys, self.cur_idx, self.loop)
        steer_correction = self.steer_pid.update(e_lat, dt)

        
        steering_cmd = max(-1.0, min(1.0, steering_ppc + steer_correction))

        # Longitudinal control
        v_err = self.route_v[self.cur_idx] - speed
        if v_err >= 0:
            throttle = self.th_pid.update(v_err, dt)
            brake = 0.0
        else:
            self.th_pid.reset()
            throttle = 0.0
            brake = min(1.0, -v_err * BRAKE_GAIN)

        # Logging data
        tgt_x, tgt_y = self.xs[tgt_idx], self.ys[tgt_idx]
        
        return {
            'steering_cmd': steering_cmd,
            'throttle': throttle,
            'brake': brake,
            'v_err': v_err,
            'e_lat': e_lat,
        }