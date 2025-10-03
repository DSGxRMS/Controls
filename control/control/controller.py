import numpy as np
from . import control_utils as utils

class PID:
    """
    A simple PID controller.
    """
    def __init__(self, Kp, Ki, Kd, integral_limit=1.0, output_limit=1.0):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.integral_limit = integral_limit
        self.output_limit = output_limit
        self.last_error = 0.0
        self.integral = 0.0

    def __call__(self, error, dt):
        self.integral += error * dt
        self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)
        
        derivative = (error - self.last_error) / dt
        self.last_error = error
        
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        return np.clip(output, -self.output_limit, self.output_limit)

class PIDRange(PID):
    """
    A PID controller with a specified output range.
    """
    def __init__(self, Kp, Ki, Kd, integral_limit=1.0, low=0.0, high=1.0):
        super().__init__(Kp, Ki, Kd, integral_limit)
        self.low = low
        self.high = high

    def __call__(self, error, dt):
        output = super().__call__(error, dt)
        return np.clip(output, self.low, self.high)

class Controller:
    """
    Core control logic, including state management and control algorithm.
    """
    def __init__(self, path_manager, config):
        self.path_manager = path_manager
        self.config = config
        
        # Initialize PID controllers
        self.pid_throttle = PID(Kp=1.0, Ki=0.1, Kd=0.05)
        self.pid_brake = PIDRange(Kp=1.0, Ki=0.1, Kd=0.05, low=0.0, high=1.0)
        
        # Controller state
        self.cur_idx = 0
        self.v_target = 1.0  # Initial target velocity

    def update_state(self, x, y, v):
        """
        Update the controller's internal state.
        """
        self.cur_idx = utils.local_closest_index(
            cur_pos=(x, y),
            xs=self.path_manager.xs,
            ys=self.path_manager.ys,
            cur_idx=self.cur_idx
        )
        
        # Update target velocity based on path curvature
        self.v_target = self.path_manager.v_limit_global[self.cur_idx]

    def compute_controls(self, x, y, yaw, v, dt):
        """
        Compute steering, throttle, and brake commands.
        """
        # Update state first
        self.update_state(x, y, v)
        
        # Calculate lookahead distance
        la_dist = utils.calc_lookahead(
            v,
            self.config['la_dist_min'], self.config['la_dist_max'],
            self.config['la_vel_min'], self.config['la_vel_max']
        )
        
        # Calculate steering angle
        steer = utils.pure_pursuit_steer(
            x, y, yaw,
            self.path_manager.xs, self.path_manager.ys,
            self.path_manager.s, self.path_manager.total_len,
            self.cur_idx, la_dist
        )
        
        # Calculate throttle and brake
        error_v = self.v_target - v
        
        if error_v > 0:
            throttle = self.pid_throttle(error_v, dt)
            brake = 0.0
        else:
            throttle = 0.0
            brake = self.pid_brake(-error_v, dt)
            
        return steer, throttle, brake

    def startup_control(self, v):
        """
        Control logic for the initial startup sequence.
        """
        if v < self.config['startup_v_threshold']:
            return 0.0, self.config['startup_throttle'], 0.0  # Gentle acceleration
        return None  # End of startup mode