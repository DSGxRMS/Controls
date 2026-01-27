#!/usr/bin/env python3
"""
ROS2 HJB-RL Controller Node
Deploys trained SAC policy for real-time path following
"""

import rclpy
from rclpy.node import Node
import numpy as np
import torch
import math
import threading
import time
import os
from pathlib import Path

from std_msgs.msg import Float32
from nav_msgs.msg import Path as PathMsg

from control_v2.ros_connect import ROSInterface
from control_v2.hjb_networks import GaussianPolicy


class HJBRLController(Node):
    """
    ROS2 node for HJB-based RL controller
    
    Loads pre-trained SAC actor network and performs inference
    for real-time steering and throttle control
    """
    
    def __init__(self,model_path: str = None, device: str = 'auto'):
        """
        Args:
            model_path: Path to trained model (.pth file)
            device: 'auto', 'cuda', or 'cpu'
        """
        super().__init__('hjb_rl_controller')
        
        # Device setup
        if device == 'auto':
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.get_logger().info(f"🎯 HJB-RL Controller initialized on {self.device}")
        
        # Load model
        if model_path is None:
            # Default to final trained model
            script_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(script_dir, '../../models/hjb_controller/final.pth')
        
        self.policy = self._load_policy(model_path)
        self.policy.eval()  # Set to evaluation mode
        
        # ROS Interface
        self.ros_interface = ROSInterface()
        
        # Spin ROS in background thread
        self.executor = rclpy.executors.SingleThreadedExecutor()
        self.executor.add_node(self.ros_interface)
        threading.Thread(target=self.executor.spin, daemon=True).start()
        
        # Vehicle parameters (match training environment)
        self.wheelbase = 1.5
        self.max_steer = 0.52
        self.max_accel = 3.0
        self.target_velocity = 5.0
        
        # State tracking
        self.current_idx = 0
        self.prev_steering = 0.0
        self.prev_accel = 0.0
        
        # Path data
        self.path_x = None
        self.path_y = None
        self.path_headings = None
        self.path_curvatures = None
        
        # Fallback controller flag
        self.use_fallback = False
        
        self.get_logger().info("✅ HJB-RL Controller ready!")
    
    def _load_policy(self, model_path: str):
        """Load trained policy network"""
        if not os.path.exists(model_path):
            self.get_logger().error(f"❌ Model file not found: {model_path}")
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Create policy network
        state_dim = 9
        action_dim = 2
        
        # Mock action space for policy initialization
        class MockActionSpace:
            low = np.array([-0.52, -3.0])
            high = np.array([0.52, 3.0])
        
        policy = GaussianPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=256,
            action_space=MockActionSpace()
        ).to(self.device)
        
        # Load weights
        policy.load_state_dict(checkpoint['policy_state_dict'])
        
        self.get_logger().info(f"📂 Loaded policy from {model_path}")
        return policy
    
    def _precompute_path_features(self):
        """Precompute path headings and curvatures"""
        dx = np.gradient(self.path_x)
        dy = np.gradient(self.path_y)
        self.path_headings = np.arctan2(dy, dx)
        
        # Curvature
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)
        denom = np.power(dx**2 + dy**2, 1.5) + 1e-12
        self.path_curvatures = (dx * ddy - dy * ddx) / denom
        self.path_curvatures = np.clip(self.path_curvatures, -2.0, 2.0)
    
    def _get_closest_path_index(self, x, y):
        """Find closest path point"""
        if self.path_x is None:
            return 0
        
        dx = self.path_x - x
        dy = self.path_y - y
        distances = dx**2 + dy**2
        
        # Local search
        search_start = max(0, self.current_idx - 20)
        search_end = min(len(self.path_x), self.current_idx + 50)
        
        local_dists = distances[search_start:search_end]
        local_min_idx = np.argmin(local_dists)
        
        return search_start + local_min_idx
    
    def _compute_observation(self, x, y, yaw, v):
        """
        Construct observation vector for RL policy
        Matches training environment state space
        """
        if self.path_x is None or len(self.path_x) < 3:
            return None
        
        idx = self._get_closest_path_index(x, y)
        self.current_idx = idx
        
        # Lateral error
        path_heading = self.path_headings[idx]
        dx = x - self.path_x[idx]
        dy = y - self.path_y[idx]
        e_lat = -math.sin(path_heading) * dx + math.cos(path_heading) * dy
        
        # Heading error
        e_psi = yaw - path_heading
        e_psi = math.atan2(math.sin(e_psi), math.cos(e_psi))
        
        # Waypoint distances (next 3 waypoints)
        waypoint_dists = []
        for i in range(1, 4):
            target_idx = min(idx + i * 5, len(self.path_x) - 1)
            dx = self.path_x[target_idx] - x
            dy = self.path_y[target_idx] - y
            dist = math.sqrt(dx**2 + dy**2)
            waypoint_dists.append(dist)
        
        # Curvature
        kappa = self.path_curvatures[idx]
        
        # Construct observation
        obs = np.array([
            e_lat,
            e_psi,
            v,
            waypoint_dists[0],
            waypoint_dists[1],
            waypoint_dists[2],
            kappa,
            self.prev_steering,
            self.prev_accel
        ], dtype=np.float32)
        
        return obs
    
    def _get_action_from_policy(self, obs):
        """
        Get action from trained policy (deterministic)
        """
        try:
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                _, _, action = self.policy.sample(obs_tensor)  # Use mean (deterministic)
                action = action.cpu().numpy()[0]
            return action
        except Exception as e:
            self.get_logger().error(f"❌ Policy inference failed: {e}")
            self.use_fallback = True
            return None
    
    def _pure_pursuit_fallback(self, x, y, yaw, v):
        """Fallback to Pure Pursuit if RL fails"""
        if self.path_x is None or len(self.path_x) < 3:
            return 0.0, 0.0
        
        # Simple pure pursuit
        lookahead = 3.0 + 0.6 * v
        idx = self.current_idx
        
        # Find lookahead point
        target_dist = 0.0
        while target_dist < lookahead and idx < len(self.path_x) - 1:
            idx += 1
            dx = self.path_x[idx] - x
            dy = self.path_y[idx] - y
            target_dist = math.sqrt(dx**2 + dy**2)
        
        # Compute steering
        tx, ty = self.path_x[idx], self.path_y[idx]
        dx = tx - x
        dy = ty - y
        
        # Transform to vehicle frame
        lx = dx * math.cos(yaw) + dy * math.sin(yaw)
        ly = -dx * math.sin(yaw) + dy * math.cos(yaw)
        
        # Curvature
        dist_sq = max(lx**2 + ly**2, 0.1)
        kappa = 2.0 * ly / dist_sq
        steering = math.atan(self.wheelbase * kappa)
        steering = max(-self.max_steer, min(self.max_steer, steering))
        
        # Simple velocity control
        accel = 1.0 if v < self.target_velocity else 0.0
        
        return steering, accel
    
    def run(self):
        """Main control loop"""
        self.get_logger().info("🏁 Starting control loop...")
        
        rate = self.create_rate(20)  # 20 Hz
        
        while rclpy.ok():
            # Get vehicle state
            x, y, yaw, v, have_odom = self.ros_interface.get_state()
            if not have_odom:
                time.sleep(0.05)
                continue
            
            # Get path
            path_x, path_y = self.ros_interface.get_local_path()
            
            if path_x is None or path_y is None or len(path_x) < 3:
                self.ros_interface.send_command(0.0, accel=0.0, speed=0.0)
                time.sleep(0.05)
                continue
            
            # Update path if changed
            if self.path_x is None or len(path_x) != len(self.path_x):
                self.path_x = np.array(path_x, dtype=np.float32)
                self.path_y = np.array(path_y, dtype=np.float32)
                self._precompute_path_features()
                self.current_idx = 0
            
            # Get observation
            obs = self._compute_observation(x, y, yaw, v)
            
            if obs is None:
                time.sleep(0.05)
                continue
            
            # Get action from policy
            if not self.use_fallback:
                action = self._get_action_from_policy(obs)
                
                if action is not None:
                    steering = float(action[0])
                    accel = float(action[1])
                else:
                    # Fallback triggered
                    steering, accel = self._pure_pursuit_fallback(x, y, yaw, v)
                    self.get_logger().warn("⚠️  Using fallback Pure Pursuit controller")
            else:
                steering, accel = self._pure_pursuit_fallback(x, y, yaw, v)
            
            # Update previous actions
            self.prev_steering = steering
            self.prev_accel = accel
            
            # Send command
            self.ros_interface.send_command(
                steering=steering,
                accel=accel,
                speed=self.target_velocity
            )
            
            rate.sleep()


def main(args=None):
    rclpy.init(args=args)
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None,
                       help='Path to trained model file')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='Device to run inference on')
    
    args, unknown = parser.parse_known_args()
    
    try:
        controller = HJBRLController(model_path=args.model, device=args.device)
        controller.run()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
