import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist

class ActorNetwork(nn.Module):
    """
    Actor network for the PPO agent.

    Parameters:
    -----------
    state_dim: int
        Dimension of the state space.
    action_dim: array
        Dimension of the action space.
    """
    def __init__(self, state_dim, action_dim):
        # Run the constructor of the parent class (nn.Module):
        super().__init__()

        # 隱藏層設定
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)

        # 輸出層: action_dim 個 logits（用於 Categorical 分布）
        self.output_layer = nn.Linear(128, action_dim)

        # Example:
        num_gate_types = action_dim
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.output_layer = nn.Linear(128, num_gate_types)

    def forward(self, state):
        """
        Forward pass.
        """

        # 保證 state 是 float tensor
        if not torch.is_floating_point(state):
            state = state.float()

        # 前向運算
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        # 輸出 logits（未經 softmax）
        gate_logits = self.output_layer(x)

        # 用 logits 建立離散分布（內建 softmax）
        gate_dist = dist.Categorical(logits=gate_logits)

        return gate_dist
        
        # Example:
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        # Gate Type Distribution:
        gate_logits = self.output_layer(x) # Raw logits.
        gate_dist = dist.Categorical(logits=gate_logits) # Categorical internally applies Softmax.
        return gate_dist

class CriticNetwork(nn.Module):
    """
    Critic network for the PPO agent.

    Parameters:
    -----------
    state_dim: int
        Dimension of the state space.
    fc1_dims: int
        Number of neurons in the first hidden layer.
    fc2_dims: int
        Number of neurons in the second hidden layer.
    """
    def __init__(self, state_dim, fc1_dims=256, fc2_dims=256):
        # Run the constructor of the parent class (nn.Module):
        super().__init__()

        # Neural network layers:
        self.fcnn = nn.Sequential(
                nn.Linear(state_dim, fc1_dims),
                nn.ReLU(),
                nn.Linear(fc1_dims, fc2_dims),
                nn.ReLU(),
                nn.Linear(fc2_dims, 1)
        )

    def forward(self, state):
        """
        Forward pass.
        """
        value = self.fcnn(state)
        return value
