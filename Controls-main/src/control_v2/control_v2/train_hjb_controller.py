#!/usr/bin/env python3
"""
Training Script for HJB-based RL Controller
Trains SAC agent on ideal path following task
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json
from pathlib import Path

# Add parent directory to path for imports
import torch

from control_v2.rl_hjb_env import PathFollowingEnv
from control_v2.hjb_agent import SACAgent


class TrainingLogger:
    """Logs training metrics and generates plots"""
    
    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.episode_rewards = []
        self.episode_lengths = []
        self.lateral_errors = []
        self.heading_errors = []
        self.success_rate = []
        
        self.critic_losses = []
        self.policy_losses = []
        self.alphas = []
    
    def log_episode(self, reward, length, lat_error, heading_error, success):
        """Log episode statistics"""
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        self.lateral_errors.append(lat_error)
        self.heading_errors.append(heading_error)
        self.success_rate.append(1.0 if success else 0.0)
    
    def log_training(self, metrics):
        """Log training metrics"""
        if metrics is not None:
            self.critic_losses.append(metrics['critic_loss'])
            self.policy_losses.append(metrics['policy_loss'])
            self.alphas.append(metrics['alpha'])
    
    def save_plots(self):
        """Generate and save training curves"""
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        
        # Episode rewards
        axes[0, 0].plot(self.episode_rewards, alpha=0.6, label='Episode Reward')
        axes[0, 0].plot(self._moving_average(self.episode_rewards, 50), 
                       label='50-ep Average', linewidth=2)
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Total Reward')
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Lateral error
        axes[0, 1].plot(self.lateral_errors, alpha=0.6, label='Average Lateral Error')
        axes[0, 1].plot(self._moving_average(self.lateral_errors, 50),
                       label='50-ep Average', linewidth=2)
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Lateral Error (m)')
        axes[0, 1].set_title('Path Tracking Error')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Success rate
        success_ma = self._moving_average(self.success_rate, 50)
        axes[1, 0].plot(success_ma, linewidth=2, color='green')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Success Rate')
        axes[1, 0].set_title('Success Rate (50-episode average)')
        axes[1, 0].set_ylim([0, 1.1])
        axes[1, 0].grid(True, alpha=0.3)
        
        # Critic loss
        if len(self.critic_losses) > 0:
            axes[1, 1].plot(self.critic_losses, alpha=0.6, label='Critic Loss')
            axes[1, 1].plot(self._moving_average(self.critic_losses, 100),
                           label='100-step Average', linewidth=2)
            axes[1, 1].set_xlabel('Update Step')
            axes[1, 1].set_ylabel('MSE Loss')
            axes[1, 1].set_title('Critic Loss')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        # Policy loss
        if len(self.policy_losses) > 0:
            axes[2, 0].plot(self.policy_losses, alpha=0.6, label='Policy Loss')
            axes[2, 0].plot(self._moving_average(self.policy_losses, 100),
                           label='100-step Average', linewidth=2)
            axes[2, 0].set_xlabel('Update Step')
            axes[2, 0].set_ylabel('Loss')
            axes[2, 0].set_title('Policy Loss')
            axes[2, 0].legend()
            axes[2, 0].grid(True, alpha=0.3)
        
        # Alpha (temperature)
        if len(self.alphas) > 0:
            axes[2, 1].plot(self.alphas, linewidth=2, color='purple')
            axes[2, 1].set_xlabel('Update Step')
            axes[2, 1].set_ylabel('Alpha')
            axes[2, 1].set_title('Entropy Temperature (α)')
            axes[2, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.log_dir / 'training_curves.png'
        plt.savefig(save_path, dpi=150)
        print(f"📊 Training curves saved to {save_path}")
        plt.close()
    
    def save_stats(self):
        """Save training statistics to JSON"""
        stats = {
            'final_avg_reward': float(np.mean(self.episode_rewards[-100:])),
            'final_avg_lateral_error': float(np.mean(self.lateral_errors[-100:])),
            'final_success_rate': float(np.mean(self.success_rate[-100:])),
            'total_episodes': len(self.episode_rewards),
            'best_episode_reward': float(np.max(self.episode_rewards)),
            'final_alpha': float(self.alphas[-1]) if len(self.alphas) > 0 else 0.0
        }
        
        save_path = self.log_dir / 'training_stats.json'
        with open(save_path, 'w') as f:
            json.dump(stats, f, indent=4)
        print(f"📈 Training stats saved to {save_path}")
        
        return stats
    
    @staticmethod
    def _moving_average(data, window):
        """Compute moving average"""
        if len(data) < window:
            return data
        return np.convolve(data, np.ones(window)/window, mode='valid')


def train(
    path_csv: str,
    episodes: int = 1000,
    max_steps: int = 2000,
    save_freq: int = 100,
    eval_freq: int = 50,
    log_dir: str = 'results/hjb_training',
    model_dir: str = 'models/hjb_controller'
):
    """
    Train SAC agent on path following task
    
    Args:
        path_csv: Path to ideal path CSV file
        episodes: Number of training episodes
        max_steps: Maximum steps per episode
        save_freq: Model checkpoint save frequency
        eval_freq: Evaluation frequency
        log_dir: Directory for logs and plots
        model_dir: Directory for model checkpoints
    """
    # Create directories
    log_dir = Path(log_dir)
    model_dir = Path(model_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("🚗 Training HJB-based RL Controller for Path Following")
    print("=" * 80)
    print(f"📍 Path: {path_csv}")
    print(f"🎯 Episodes: {episodes}")
    print(f"💾 Model dir: {model_dir}")
    print(f"📊 Log dir: {log_dir}")
    print(f"🖥️  Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    print("=" * 80)
    
    # Initialize environment
    env = PathFollowingEnv(
        path_csv=path_csv,
        dt=0.05,
        max_steps=max_steps,
        target_velocity=5.0
    )
    
    # Initialize agent
    agent = SACAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        action_space=env.action_space,
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
        lr=3e-4,
        hidden_dim=256,
        buffer_size=100000,
        batch_size=256,
        device='auto'
    )
    
    # Initialize logger
    logger = TrainingLogger(log_dir)
    
    # Training loop
    print("\n🏁 Starting training...")
    
    for episode in range(episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_steps = 0
        episode_lat_errors = []
        episode_heading_errors = []
        
        done = False
        truncated = False
        
        while not (done or truncated):
            # Select action
            action = agent.select_action(obs, evaluate=False)
            
            # Execute action
            next_obs, reward, done, truncated, info = env.step(action)
            
            # Store transition
            agent.memory.push(obs, action, reward, next_obs, done or truncated)
            
            # Update metrics
            episode_reward += reward
            episode_steps += 1
            episode_lat_errors.append(abs(info['lateral_error']))
            episode_heading_errors.append(abs(info['heading_error']))
            
            obs = next_obs
            
            # Update agent (start after some exploration)
            if len(agent.memory) > agent.batch_size:
                metrics = agent.update_parameters(updates=1)
                logger.log_training(metrics)
        
        # Log episode
        avg_lat_error = np.mean(episode_lat_errors)
        avg_heading_error = np.mean(episode_heading_errors)
        success = info.get('termination_reason') == 'success'
        
        logger.log_episode(
            episode_reward, episode_steps, avg_lat_error, avg_heading_error, success
        )
        
        # Print progress
        if (episode + 1) % 10 == 0:
            recent_rewards = np.mean(logger.episode_rewards[-10:])
            recent_lat_err = np.mean(logger.lateral_errors[-10:])
            recent_success = np.mean(logger.success_rate[-10:])
            
            print(f"Episode {episode+1}/{episodes} | "
                  f"Reward: {recent_rewards:.2f} | "
                  f"Lat Error: {recent_lat_err:.3f}m | "
                  f"Success: {recent_success:.2%} | "
                  f"Buffer: {len(agent.memory)}")
        
        # Save checkpoint
        if (episode + 1) % save_freq == 0:
            checkpoint_path = model_dir / f'checkpoint_ep{episode+1}.pth'
            agent.save(str(checkpoint_path))
        
        # Evaluation run (deterministic policy)
        if (episode + 1) % eval_freq == 0:
            eval_reward, eval_lat_err = evaluate_agent(env, agent, num_episodes=5)
            print(f"📊 Evaluation | Avg Reward: {eval_reward:.2f} | "
                  f"Avg Lat Error: {eval_lat_err:.3f}m")
    
    # Save final model
    final_path = model_dir / 'final.pth'
    agent.save(str(final_path))
    
    # Generate plots and stats
    logger.save_plots()
    stats = logger.save_stats()
    
    print("\n" + "=" * 80)
    print("✅ Training completed!")
    print("=" * 80)
    print(f"📊 Final Statistics:")
    print(f"   Average Reward (last 100 ep): {stats['final_avg_reward']:.2f}")
    print(f"   Average Lateral Error: {stats['final_avg_lateral_error']:.3f} m")
    print(f"   Success Rate: {stats['final_success_rate']:.2%}")
    print(f"   Best Episode Reward: {stats['best_episode_reward']:.2f}")
    print("=" * 80)
    
    return agent, stats


def evaluate_agent(env, agent, num_episodes=5):
    """Evaluate agent with deterministic policy"""
    total_reward = 0
    total_lat_error = 0
    
    for _ in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        lat_errors = []
        
        done = False
        truncated = False
        
        while not (done or truncated):
            action = agent.select_action(obs, evaluate=True)  # Deterministic
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            lat_errors.append(abs(info['lateral_error']))
        
        total_reward += episode_reward
        total_lat_error += np.mean(lat_errors)
    
    return total_reward / num_episodes, total_lat_error / num_episodes


def main():
    parser = argparse.ArgumentParser(description='Train HJB-based RL Controller')
    parser.add_argument('--path', type=str, 
                       default='resource/pathpoints_shifted.csv',
                       help='Path to CSV file with ideal path')
    parser.add_argument('--episodes', type=int, default=1000,
                       help='Number of training episodes')
    parser.add_argument('--max-steps', type=int, default=2000,
                       help='Maximum steps per episode')
    parser.add_argument('--save-freq', type=int, default=100,
                       help='Model checkpoint save frequency')
    parser.add_argument('--log-dir', type=str, default='results/hjb_training',
                       help='Directory for logs and plots')
    parser.add_argument('--model-dir', type=str, default='models/hjb_controller',
                       help='Directory for model checkpoints')
    
    args = parser.parse_args()
    
    # Convert relative path to absolute
    if not os.path.isabs(args.path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        args.path = os.path.join(script_dir, args.path)
    
    # Train
    train(
        path_csv=args.path,
        episodes=args.episodes,
        max_steps=args.max_steps,
        save_freq=args.save_freq,
        log_dir=args.log_dir,
        model_dir=args.model_dir
    )


if __name__ == '__main__':
    main()
