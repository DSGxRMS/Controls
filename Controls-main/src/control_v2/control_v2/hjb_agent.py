#!/usr/bin/env python3
"""
Soft Actor-Critic (SAC) Agent for HJB-based RL Controller
Implements discrete-time HJB equation via Bellman backups
"""

import torch
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
from collections import deque
import random
from typing import Tuple

from control_v2.hjb_networks import QNetwork, GaussianPolicy


class ReplayBuffer:
    """
    Experience Replay Buffer for off-policy learning
    Stores transitions: (s, a, r, s', done)
    """
    
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Add transition to buffer"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int):
        """Sample random batch of transitions"""
        batch = random.sample(self.buffer, batch_size)
        
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.float32),
            np.array(rewards, dtype=np.float32).reshape(-1, 1),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32).reshape(-1, 1)
        )
    
    def __len__(self):
        return len(self.buffer)


class SACAgent:
    """
    Soft Actor-Critic Agent
    
    Implements discrete-time HJB via:
    Q(s,a) = r(s,a) + γ * E[V(s') - α*log π(a'|s')]
    
    where V is the soft value function with entropy regularization
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        action_space,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        lr: float = 3e-4,
        hidden_dim: int = 256,
        buffer_size: int = 100000,
        batch_size: int = 256,
        device: str = 'auto'
    ):
        """
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            action_space: Gym action space (for bounds)
            gamma: Discount factor for HJB
            tau: Soft update coefficient for target networks
            alpha: Entropy temperature (exploration)
            lr: Learning rate
            hidden_dim: Hidden layer size
            buffer_size: Replay buffer capacity
            batch_size: Training batch size
            device: 'auto', 'cuda', or 'cpu'
        """
        # Device setup
        if device == 'auto':
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"🎯 SAC Agent initialized on device: {self.device}")
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        
        # Automatic entropy tuning
        self.target_entropy = -action_dim  # -dim(A)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha = alpha
        self.alpha_optim = Adam([self.log_alpha], lr=lr)
        
        # Networks
        self.critic = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        self.policy = GaussianPolicy(
            state_dim, action_dim, hidden_dim, action_space
        ).to(self.device)
        
        # Optimizers
        self.critic_optim = Adam(self.critic.parameters(), lr=lr)
        self.policy_optim = Adam(self.policy.parameters(), lr=lr)
        
        # Replay buffer
        self.memory = ReplayBuffer(buffer_size)
        
        # Training statistics
        self.updates = 0
    
    def select_action(self, state: np.ndarray, evaluate: bool = False):
        """
        Select action from policy
        
        Args:
            state: Current state observation
            evaluate: If True, use deterministic policy (no exploration)
        
        Returns:
            action: numpy array
        """
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        if evaluate:
            _, _, action = self.policy.sample(state)
        else:
            action, _, _ = self.policy.sample(state)
        
        return action.detach().cpu().numpy()[0]
    
    def update_parameters(self, updates: int = 1):
        """
        Update critic and actor networks via HJB Bellman backup
        
        Returns:
            Dictionary of training metrics
        """
        if len(self.memory) < self.batch_size:
            return None
        
        metrics = {
            'critic_loss': 0.0,
            'policy_loss': 0.0,
            'alpha_loss': 0.0,
            'alpha': self.alpha
        }
        
        for _ in range(updates):
            # Sample batch from replay buffer
            states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
            
            states = torch.FloatTensor(states).to(self.device)
            actions = torch.FloatTensor(actions).to(self.device)
            rewards = torch.FloatTensor(rewards).to(self.device)
            next_states = torch.FloatTensor(next_states).to(self.device)
            dones = torch.FloatTensor(dones).to(self.device)
            
            # ========================================
            # Update Critic (Q-function)
            # ========================================
            with torch.no_grad():
                # Sample next actions from current policy
                next_actions, next_log_probs, _ = self.policy.sample(next_states)
                
                # Compute target Q-values (double Q-learning)
                target_q1, target_q2 = self.critic_target(next_states, next_actions)
                target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_probs
                
                # HJB Bellman backup: Q(s,a) = r + γ * (1-done) * V(s')
                target_q = rewards + (1 - dones) * self.gamma * target_q
            
            # Current Q estimates
            current_q1, current_q2 = self.critic(states, actions)
            
            # Critic loss (MSE)
            critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
            
            # Optimize critic
            self.critic_optim.zero_grad()
            critic_loss.backward()
            self.critic_optim.step()
            
            # ========================================
            # Update Actor (Policy)
            # ========================================
            # Sample actions from current policy
            pi_actions, log_probs, _ = self.policy.sample(states)
            
            # Compute Q-values for sampled actions
            q1_pi, q2_pi = self.critic(states, pi_actions)
            min_q_pi = torch.min(q1_pi, q2_pi)
            
            # Policy loss (maximize Q - α*log π)
            # Equivalent to minimizing negative objective
            policy_loss = (self.alpha * log_probs - min_q_pi).mean()
            
            # Optimize policy
            self.policy_optim.zero_grad()
            policy_loss.backward()
            self.policy_optim.step()
            
            # ========================================
            # Update Temperature (α) - Automatic Entropy Tuning
            # ========================================
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            
            self.alpha = self.log_alpha.exp().item()
            
            # ========================================
            # Soft Update Target Networks
            # ========================================
            for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_param.data.copy_(
                    target_param.data * (1.0 - self.tau) + param.data * self.tau
                )
            
            # Accumulate metrics
            metrics['critic_loss'] += critic_loss.item()
            metrics['policy_loss'] += policy_loss.item()
            metrics['alpha_loss'] += alpha_loss.item()
            self.updates += 1
        
        # Average metrics
        metrics['critic_loss'] /= updates
        metrics['policy_loss'] /= updates
        metrics['alpha_loss'] /= updates
        
        return metrics
    
    def save(self, filepath: str):
        """Save model weights"""
        torch.save({
            'critic_state_dict': self.critic.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'policy_state_dict': self.policy.state_dict(),
            'critic_optim_state_dict': self.critic_optim.state_dict(),
            'policy_optim_state_dict': self.policy_optim.state_dict(),
            'alpha_optim_state_dict': self.alpha_optim.state_dict(),
            'log_alpha': self.log_alpha,
            'updates': self.updates
        }, filepath)
        print(f"💾 Model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load model weights"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        
        self.critic_optim.load_state_dict(checkpoint['critic_optim_state_dict'])
        self.policy_optim.load_state_dict(checkpoint['policy_optim_state_dict'])
        self.alpha_optim.load_state_dict(checkpoint['alpha_optim_state_dict'])
        
        self.log_alpha = checkpoint['log_alpha']
        self.alpha = self.log_alpha.exp().item()
        self.updates = checkpoint['updates']
        
        print(f"📂 Model loaded from {filepath} (updates: {self.updates})")


if __name__ == "__main__":
    # Test SAC agent
    print("Testing SAC Agent...")
    
    # Mock action space
    class MockActionSpace:
        def __init__(self):
            self.low = np.array([-0.52, -3.0])
            self.high = np.array([0.52, 3.0])
    
    action_space = MockActionSpace()
    
    # Initialize agent
    agent = SACAgent(
        state_dim=9,
        action_dim=2,
        action_space=action_space,
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
        lr=3e-4,
        batch_size=32,  # Small batch for testing
        device='auto'
    )
    
    print(f"\n1. Testing action selection...")
    test_state = np.random.randn(9)
    action = agent.select_action(test_state, evaluate=False)
    print(f"   Sampled action: {action}")
    assert action.shape == (2,), "Action shape mismatch"
    print("   ✓ Action selection test passed!")
    
    print(f"\n2. Testing replay buffer...")
    for i in range(100):
        s = np.random.randn(9)
        a = np.random.randn(2)
        r = np.random.randn()
        s_next = np.random.randn(9)
        done = False
        agent.memory.push(s, a, r, s_next, done)
    
    print(f"   Buffer size: {len(agent.memory)}")
    assert len(agent.memory) == 100, "Buffer size mismatch"
    print("   ✓ Replay buffer test passed!")
    
    print(f"\n3. Testing network updates...")
    metrics = agent.update_parameters(updates=5)
    print(f"   Critic loss: {metrics['critic_loss']:.4f}")
    print(f"   Policy loss: {metrics['policy_loss']:.4f}")
    print(f"   Alpha: {metrics['alpha']:.4f}")
    print("   ✓ Network update test passed!")
    
    print(f"\n4. Testing save/load...")
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
        temp_path = f.name
    
    agent.save(temp_path)
    agent.load(temp_path)
    
    import os
    os.remove(temp_path)
    print("   ✓ Save/load test passed!")
    
    print("\n✅ All SAC agent tests passed successfully!")
