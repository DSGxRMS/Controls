#!/usr/bin/env python3
"""
PyTorch Neural Networks for HJB-based RL Controller
Implements Actor-Critic architecture for Soft Actor-Critic (SAC) algorithm
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np


# Initialize weights with Xavier initialization
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class QNetwork(nn.Module):
    """
    Critic Network: Q(s, a) → Scalar
    Estimates action-value function (Q-function) for HJB
    Uses double Q-learning trick (two Q-networks)
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(QNetwork, self).__init__()
        
        # Q1 architecture
        self.q1_fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.q1_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q1_fc3 = nn.Linear(hidden_dim, 128)
        self.q1_out = nn.Linear(128, 1)
        
        # Q2 architecture (independent for stability)
        self.q2_fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.q2_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q2_fc3 = nn.Linear(hidden_dim, 128)
        self.q2_out = nn.Linear(128, 1)
        
        self.apply(weights_init_)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor):
        """
        Forward pass for both Q-networks
        Returns: (Q1(s,a), Q2(s,a))
        """
        xu = torch.cat([state, action], dim=1)
        
        # Q1 forward
        x1 = F.relu(self.q1_fc1(xu))
        x1 = F.relu(self.q1_fc2(x1))
        x1 = F.relu(self.q1_fc3(x1))
        q1 = self.q1_out(x1)
        
        # Q2 forward
        x2 = F.relu(self.q2_fc1(xu))
        x2 = F.relu(self.q2_fc2(x2))
        x2 = F.relu(self.q2_fc3(x2))
        q2 = self.q2_out(x2)
        
        return q1, q2


class GaussianPolicy(nn.Module):
    """
    Actor Network: π(s) → (μ, σ)
    Outputs Gaussian distribution parameters for continuous actions
    Implements squashed Gaussian policy with tanh
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        action_space=None,
        log_std_min: float = -20,
        log_std_max: float = 2
    ):
        super(GaussianPolicy, self).__init__()
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.epsilon = 1e-6
        
        # Network architecture
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 128)
        
        # Mean and log_std heads
        self.mean = nn.Linear(128, action_dim)
        self.log_std = nn.Linear(128, action_dim)
        
        self.apply(weights_init_)
        
        # Action rescaling (from [-1, 1] to actual bounds)
        if action_space is None:
            self.action_scale = torch.tensor(1.0)
            self.action_bias = torch.tensor(0.0)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.0
            )
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.0
            )
    
    def forward(self, state: torch.Tensor):
        """
        Forward pass through network
        Returns: (μ, log_σ)
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std
    
    def sample(self, state: torch.Tensor):
        """
        Sample action from policy distribution
        Returns: (action, log_prob, mean)
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        # Sample from Gaussian
        normal = Normal(mean, std)
        x_t = normal.rsample()  # Reparameterization trick
        
        # Apply tanh squashing
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        
        # Compute log probability with change of variables
        log_prob = normal.log_prob(x_t)
        
        # Enforcing action bounds (tanh correction)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + self.epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        
        return action, log_prob, mean
    
    def to(self, device):
        """Override to method to also move action scaling tensors"""
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)


class DeterministicPolicy(nn.Module):
    """
    Deterministic Actor Network: π(s) → a
    Used for evaluation (no exploration noise)
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        action_space=None
    ):
        super(DeterministicPolicy, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 128)
        self.mean = nn.Linear(128, action_dim)
        
        self.apply(weights_init_)
        
        # Action rescaling
        if action_space is None:
            self.action_scale = 1.0
            self.action_bias = 0.0
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.0
            )
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.0
            )
    
    def forward(self, state: torch.Tensor):
        """Forward pass to get deterministic action"""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        mean = torch.tanh(self.mean(x)) * self.action_scale + self.action_bias
        return mean
    
    def to(self, device):
        """Override to method to also move action scaling tensors"""
        if isinstance(self.action_scale, torch.Tensor):
            self.action_scale = self.action_scale.to(device)
            self.action_bias = self.action_bias.to(device)
        return super(DeterministicPolicy, self).to(device)


if __name__ == "__main__":
    # Test networks
    print("Testing HJB Neural Networks...")
    
    state_dim = 9
    action_dim = 2
    batch_size = 32
    
    # Check GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Test Q-Network
    print("\n1. Testing Q-Network...")
    q_net = QNetwork(state_dim, action_dim, hidden_dim=256).to(device)
    
    state = torch.randn(batch_size, state_dim).to(device)
    action = torch.randn(batch_size, action_dim).to(device)
    
    q1, q2 = q_net(state, action)
    print(f"   Q1 output shape: {q1.shape}, Q2 output shape: {q2.shape}")
    print(f"   Q1 sample values: {q1[0].item():.4f}, Q2 sample values: {q2[0].item():.4f}")
    assert q1.shape == (batch_size, 1), "Q1 shape mismatch"
    assert q2.shape == (batch_size, 1), "Q2 shape mismatch"
    print("   ✓ Q-Network test passed!")
    
    # Test Gaussian Policy
    print("\n2. Testing Gaussian Policy...")
    
    # Create mock action space
    class MockActionSpace:
        def __init__(self):
            self.low = np.array([-0.52, -3.0])
            self.high = np.array([0.52, 3.0])
    
    action_space = MockActionSpace()
    policy = GaussianPolicy(state_dim, action_dim, hidden_dim=256, action_space=action_space).to(device)
    
    action, log_prob, mean = policy.sample(state)
    print(f"   Action shape: {action.shape}, Log prob shape: {log_prob.shape}")
    print(f"   Sample action: {action[0].cpu().numpy()}")
    print(f"   Sample log_prob: {log_prob[0].item():.4f}")
    
    # Check action bounds
    assert torch.all(action[:, 0] >= -0.52) and torch.all(action[:, 0] <= 0.52), "Steering out of bounds"
    assert torch.all(action[:, 1] >= -3.0) and torch.all(action[:, 1] <= 3.0), "Throttle out of bounds"
    print("   ✓ Gaussian Policy test passed!")
    
    # Test Deterministic Policy
    print("\n3. Testing Deterministic Policy...")
    det_policy = DeterministicPolicy(state_dim, action_dim, hidden_dim=256, action_space=action_space).to(device)
    
    det_action = det_policy(state)
    print(f"   Deterministic action shape: {det_action.shape}")
    print(f"   Sample deterministic action: {det_action[0].cpu().numpy()}")
    
    assert torch.all(det_action[:, 0] >= -0.52) and torch.all(det_action[:, 0] <= 0.52), "Steering out of bounds"
    assert torch.all(det_action[:, 1] >= -3.0) and torch.all(det_action[:, 1] <= 3.0), "Throttle out of bounds"
    print("   ✓ Deterministic Policy test passed!")
    
    # Test gradient flow
    print("\n4. Testing gradient flow...")
    loss = q1.mean() + log_prob.mean()
    loss.backward()
    
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in q_net.parameters())
    assert has_grad, "No gradients in Q-network"
    print("   ✓ Gradient flow test passed!")
    
    print("\n✅ All neural network tests passed successfully!")
    print(f"📊 Total parameters:")
    print(f"   Q-Network: {sum(p.numel() for p in q_net.parameters()):,}")
    print(f"   Policy: {sum(p.numel() for p in policy.parameters()):,}")
