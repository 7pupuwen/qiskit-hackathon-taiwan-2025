# Importing libraries: 
import os # For file system access.
import torch # For tensor operations.
from torch.optim import Adam, SGD # For gradient update.
from typing import Tuple # For typing annotation.

# Importing custom modules:
from src.actor_critic_networks import ActorNetwork, CriticNetwork
from src.memory import PPOMemory

class PPOAgent:
    '''
    Class for the PPO agent.

    Parameters:
    -----------
    state_dim: int
        Dimension of the state space.
    action_dim: array
        Dimension of the action space.
    learning_rate: float
        Learning rate for the optimizer.
    gamma: float
        Discount factor for future rewards.
    gae_lambda: float
        Lambda parameter for Generalized Advantage Estimation (GAE).
    policy_clip: float
        Clipping parameter for the policy loss.
    batch_size: int
        Batch size for training.
    num_epochs: int
        Number of epochs for training.
    optimizer_option: str
        Choice of optimizer ('Adam' or 'SGD').
    chkpt_dir: str
        Directory to save the model checkpoint.
    '''
    def __init__(self,
                 state_dim,
                 action_dim,
                 learning_rate=0.0003,
                 gamma=0.99,
                 gae_lambda=0.95,
                 policy_clip=0.2,
                 batch_size=64,
                 num_epochs=10,
                 optimizer_option="Adam",
                 chkpt_dir='model/ppo'):

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.policy_clip = policy_clip
        self.num_epochs = num_epochs

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # Instantiate the Actor and Critic networks:
        self.actor = ActorNetwork(state_dim=state_dim, action_dim=action_dim).to(self.device)
        self.critic = CriticNetwork(state_dim=state_dim).to(self.device)

        # Define a dictionary for optimizers:
        optimizers = {
            "Adam": Adam,
            "SGD": SGD
            }
        OptimizerClass = optimizers.get(optimizer_option, Adam) # Default to Adam if not found.
        
        # Define optimizers:
        self.actor_optimizer = OptimizerClass(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = OptimizerClass(self.critic.parameters(), lr=learning_rate)

        # Buffer for storing transitions:
        self.memory_buffer = PPOMemory(batch_size)

        # Checkpoint paths:
        self.actor_chkpt = os.path.join(chkpt_dir, 'actor_net_torch_ppo')
        self.critic_chkpt = os.path.join(chkpt_dir, 'critic_net_torch_ppo')
        
    def sample_action(self, observation: torch.tensor) -> Tuple[list, list, float]:
        """
        Sample actions from the policy network given the current state (observation).

        Args:
            observation (torch.tensor): the state representation.

        Returns:
            action (list): list of action(s).
            probs (list): list of probability distribution(s) over action(s).
            value (float): the value from the Critic network.
        """
        observation = observation.to(self.device)
    
        # 用 Actor 得到動作分布
        gate_dist = self.actor(observation)
    
        # 從概率分布取樣一個動作
        action = gate_dist.sample()
    
        # 動作的 log 機率（用於 PPO 計算 loss）
        probs = gate_dist.log_prob(action)
    
        # 用 Critic 得到 state value
        value = self.critic(observation)
    
        # 回傳時轉成普通 Python 類型（list / float），方便保存到 memory
        return action.detach().cpu().tolist(), probs.detach().cpu().tolist(), value.item()


    def store_transitions(self, state, action, reward, probs, vals, done):
        """
        This method stores transitions in the memory buffer.
        """
        self.memory_buffer.store_memory(state, action, reward, probs, vals, done)

    def learn(self):
        """
        This method implements the learning step.
        """

        device = self.device
        memory = self.memory_buffer
    
        for _ in range(self.num_epochs):
            states, actions, rewards, old_probs, values, dones = memory.generate_batches()
    
            states = torch.tensor(states, dtype=torch.float).to(device)
            actions = torch.tensor(actions).to(device)
            rewards = torch.tensor(rewards).to(device)
            old_probs = torch.tensor(old_probs).to(device)
            values = torch.tensor(values).to(device)
            dones = torch.tensor(dones).to(device)
    
            # 計算折扣回報和優勢
            returns = []
            discounted_reward = 0
            for reward, is_terminal in zip(reversed(rewards), reversed(dones)):
                if is_terminal:
                    discounted_reward = 0
                discounted_reward = reward + self.gamma * discounted_reward
                returns.insert(0, discounted_reward)
            returns = torch.tensor(returns).to(device)
            advantages = returns - values
    
            gate_dists = self.actor(states)
            new_probs = gate_dists.log_prob(actions)
    
            prob_ratio = (new_probs - old_probs).exp()
    
            weighted_probs = advantages * prob_ratio
            clipped_probs = advantages * torch.clamp(prob_ratio, 1 - self.policy_clip, 1 + self.policy_clip)
            actor_loss = -torch.min(weighted_probs, clipped_probs).mean()
    
            critic_loss = F.mse_loss(self.critic(states).squeeze(), returns)
    
            total_loss = actor_loss + 0.5 * critic_loss
    
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            total_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()
    
        pass

    def save_models(self):
        print("Saving models...")
        torch.save(self.actor.state_dict(), self.actor_chkpt)
        torch.save(self.critic.state_dict(), self.critic_chkpt)

    def load_models(self):
        print("Loading models...")
        self.actor.load_state_dict(torch.load(self.actor_chkpt))
        self.critic.load_state_dict(torch.load(self.critic_chkpt))
