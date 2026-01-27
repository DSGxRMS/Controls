#!/usr/bin/env python3
"""
Gymnasium-compatible RL Environment for Path Following Training
Uses kinematic bicycle model and HJB-inspired reward function
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import math
import pandas as pd
from typing import Tuple, Optional


class PathFollowingEnv(gym.Env):
    """
    RL Environment for autonomous vehicle path following.
    
    State Space (9D):
        - Lateral error (e_lat) [m]
        - Heading error (e_psi) [rad]
        - Velocity (v) [m/s]
        - Distance to next 3 waypoints [m, m, m]
        - Curvature at current position (κ) [1/m]
        - Previous steering command (δ_prev) [rad]
        - Previous throttle command (a_prev) [m/s²]
    
    Action Space (2D - Continuous):
        - Steering command: [-0.52, 0.52] rad
        - Throttle/brake: [-3.0, 3.0] m/s²
    """
    
    metadata = {'render_modes': ['human'], 'render_fps': 20}
    
    def __init__(
        self,
        path_csv: str,
        dt: float = 0.05,
        wheelbase: float = 1.5,
        max_steps: int = 2000,
        max_deviation: float = 2.0,
        target_velocity: float = 5.0
    ):
        super().__init__()
        
        # Load reference path
        self.path_data = pd.read_csv(path_csv)
        self.path_x = self.path_data['x'].values
        self.path_y = self.path_data['y'].values
        self.path_length = len(self.path_x)
        
        # Precompute path features
        self._precompute_path_features()
        
        # Vehicle parameters
        self.dt = dt
        self.wheelbase = wheelbase
        self.max_deviation = max_deviation
        self.target_velocity = target_velocity
        self.max_steps = max_steps
        
        # Action space: [steering, throttle]
        self.action_space = spaces.Box(
            low=np.array([-0.52, -3.0], dtype=np.float32),
            high=np.array([0.52, 3.0], dtype=np.float32),
            dtype=np.float32
        )
        
        # State space: [e_lat, e_psi, v, d1, d2, d3, kappa, δ_prev, a_prev]
        self.observation_space = spaces.Box(
            low=np.array([-3.0, -np.pi, 0.0, 0.0, 0.0, 0.0, -2.0, -0.52, -3.0], dtype=np.float32),
            high=np.array([3.0, np.pi, 15.0, 20.0, 20.0, 20.0, 2.0, 0.52, 3.0], dtype=np.float32),
            dtype=np.float32
        )
        
        # Reward weights (HJB cost function)
        self.w_lat = 10.0      # Lateral error weight
        self.w_psi = 5.0       # Heading error weight
        self.w_v = 1.0         # Velocity tracking weight
        self.w_steer = 0.1     # Steering effort weight
        self.w_accel = 0.05    # Acceleration effort weight
        self.w_jerk_steer = 2.0  # Steering jerk weight
        self.w_jerk_accel = 0.5  # Acceleration jerk weight
        
        # State variables
        self.reset()
    
    def _precompute_path_features(self):
        """Precompute path headings and curvatures"""
        # Path headings
        dx = np.gradient(self.path_x)
        dy = np.gradient(self.path_y)
        self.path_headings = np.arctan2(dy, dx)
        
        # Curvature (signed)
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)
        denom = np.power(dx**2 + dy**2, 1.5) + 1e-12
        self.path_curvatures = (dx * ddy - dy * ddx) / denom
        self.path_curvatures = np.clip(self.path_curvatures, -2.0, 2.0)
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        # Random start position along path (first 10% to allow warmup)
        self.current_idx = np.random.randint(0, max(1, self.path_length // 10))
        
        # Vehicle state: [x, y, yaw, v]
        self.x = self.path_x[self.current_idx]
        self.y = self.path_y[self.current_idx]
        self.yaw = self.path_headings[self.current_idx]
        self.v = 0.0  # Start from rest
        
        # Previous actions
        self.prev_steering = 0.0
        self.prev_accel = 0.0
        
        # Episode tracking
        self.steps = 0
        self.cumulative_reward = 0.0
        
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info
    
    def _get_closest_path_index(self) -> int:
        """Find closest path point to current position"""
        dx = self.path_x - self.x
        dy = self.path_y - self.y
        distances = dx**2 + dy**2
        
        # Search locally around current index for efficiency
        search_start = max(0, self.current_idx - 20)
        search_end = min(self.path_length, self.current_idx + 50)
        
        local_dists = distances[search_start:search_end]
        local_min_idx = np.argmin(local_dists)
        
        return search_start + local_min_idx
    
    def _compute_lateral_error(self, idx: int) -> float:
        """Compute signed lateral error (perpendicular distance to path)"""
        path_heading = self.path_headings[idx]
        dx = self.x - self.path_x[idx]
        dy = self.y - self.path_y[idx]
        
        # Project onto perpendicular axis
        e_lat = -math.sin(path_heading) * dx + math.cos(path_heading) * dy
        return e_lat
    
    def _compute_heading_error(self, idx: int) -> float:
        """Compute heading error (normalized to [-π, π])"""
        path_heading = self.path_headings[idx]
        e_psi = self.yaw - path_heading
        # Normalize to [-π, π]
        e_psi = math.atan2(math.sin(e_psi), math.cos(e_psi))
        return e_psi
    
    def _get_waypoint_distances(self, idx: int) -> np.ndarray:
        """Get distances to next 3 waypoints"""
        distances = []
        for i in range(1, 4):
            target_idx = min(idx + i * 5, self.path_length - 1)
            dx = self.path_x[target_idx] - self.x
            dy = self.path_y[target_idx] - self.y
            dist = math.sqrt(dx**2 + dy**2)
            distances.append(dist)
        return np.array(distances, dtype=np.float32)
    
    def _get_observation(self) -> np.ndarray:
        """Construct observation vector"""
        idx = self._get_closest_path_index()
        
        e_lat = self._compute_lateral_error(idx)
        e_psi = self._compute_heading_error(idx)
        waypoint_dists = self._get_waypoint_distances(idx)
        kappa = self.path_curvatures[idx]
        
        obs = np.array([
            e_lat,
            e_psi,
            self.v,
            waypoint_dists[0],
            waypoint_dists[1],
            waypoint_dists[2],
            kappa,
            self.prev_steering,
            self.prev_accel
        ], dtype=np.float32)
        
        return obs
    
    def _get_info(self) -> dict:
        """Additional info for logging"""
        idx = self._get_closest_path_index()
        return {
            'path_progress': idx / self.path_length,
            'lateral_error': self._compute_lateral_error(idx),
            'heading_error': self._compute_heading_error(idx),
            'velocity': self.v,
            'current_idx': idx
        }
    
    def _compute_reward(self, obs: np.ndarray, action: np.ndarray) -> float:
        """
        HJB-inspired reward function (negative cost-to-go)
        
        Reward = -(Q(s,a)) where Q represents immediate cost
        """
        e_lat = obs[0]
        e_psi = obs[1]
        v = obs[2]
        delta = action[0]  # steering
        a_cmd = action[1]   # throttle
        
        # State costs
        lat_cost = self.w_lat * e_lat**2
        heading_cost = self.w_psi * e_psi**2
        vel_cost = self.w_v * (v - self.target_velocity)**2
        
        # Action costs
        steer_cost = self.w_steer * delta**2
        accel_cost = self.w_accel * a_cmd**2
        
        # Jerk costs (change in actions)
        steer_jerk = self.w_jerk_steer * (delta - self.prev_steering)**2
        accel_jerk = self.w_jerk_accel * (a_cmd - self.prev_accel)**2
        
        # Total cost
        total_cost = (lat_cost + heading_cost + vel_cost + 
                     steer_cost + accel_cost + steer_jerk + accel_jerk)
        
        # Return negative cost as reward (maximize reward = minimize cost)
        reward = -total_cost
        
        return reward
    
    def _update_vehicle_dynamics(self, steering: float, accel: float):
        """
        Update vehicle state using kinematic bicycle model
        Matches the predict_bicycle_state from control_utils
        """
        # Clamp acceleration
        dv = accel * self.dt
        self.v = max(0.0, min(15.0, self.v + dv))  # Speed limits
        
        # Kinematic bicycle model
        if abs(steering) < 1e-4:
            # Straight line motion
            self.x += self.v * math.cos(self.yaw) * self.dt
            self.y += self.v * math.sin(self.yaw) * self.dt
        else:
            # Curved motion
            tan_delta = math.tan(steering)
            turn_radius = self.wheelbase / tan_delta
            angular_velocity = self.v / turn_radius
            
            # Update heading
            dyaw = angular_velocity * self.dt
            
            # Center of rotation
            cx = self.x - turn_radius * math.sin(self.yaw)
            cy = self.y + turn_radius * math.cos(self.yaw)
            
            # New state
            self.yaw += dyaw
            self.x = cx + turn_radius * math.sin(self.yaw)
            self.y = cy - turn_radius * math.cos(self.yaw)
        
        # Normalize yaw
        self.yaw = math.atan2(math.sin(self.yaw), math.cos(self.yaw))
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one environment step"""
        steering = float(action[0])
        accel = float(action[1])
        
        # Get observation before action
        obs = self._get_observation()
        
        # Update vehicle dynamics
        self._update_vehicle_dynamics(steering, accel)
        
        # Update closest path index
        self.current_idx = self._get_closest_path_index()
        
        # Compute reward
        reward = self._compute_reward(obs, action)
        
        # Update previous actions
        self.prev_steering = steering
        self.prev_accel = accel
        
        # Get new observation
        obs_new = self._get_observation()
        info = self._get_info()
        
        # Check termination conditions
        self.steps += 1
        self.cumulative_reward += reward
        
        # Truncation: max steps reached
        truncated = self.steps >= self.max_steps
        
        # Termination: large deviation or path completion
        lateral_error = abs(info['lateral_error'])
        path_complete = info['path_progress'] > 0.95
        too_far = lateral_error > self.max_deviation
        
        terminated = too_far or path_complete
        
        # Penalty for going off track
        if too_far:
            reward -= 100.0
            info['termination_reason'] = 'deviation'
        elif path_complete:
            reward += 50.0  # Bonus for completion
            info['termination_reason'] = 'success'
        
        return obs_new, reward, terminated, truncated, info
    
    def render(self):
        """Render environment (optional, for visualization)"""
        if self.render_mode == 'human':
            # Could implement matplotlib visualization here
            pass


if __name__ == "__main__":
    # Test environment
    import os
    
    # Use relative path for testing
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path_csv = os.path.join(script_dir, 'resource', 'pathpoints_shifted.csv')
    
    env = PathFollowingEnv(path_csv=path_csv)
    
    print("Testing PathFollowingEnv...")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # Test reset
    obs, info = env.reset()
    print(f"\nInitial observation: {obs}")
    print(f"Initial info: {info}")
    
    # Test a few random steps
    print("\nTesting random steps...")
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i+1}: reward={reward:.3f}, progress={info['path_progress']:.3f}, "
              f"lat_err={info['lateral_error']:.3f}")
        
        if terminated or truncated:
            print(f"Episode ended: {info.get('termination_reason', 'truncated')}")
            break
    
    print("\n✓ Environment test completed successfully!")
