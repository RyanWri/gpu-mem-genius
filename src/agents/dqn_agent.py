import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from src.replay_buffer import ReplayBuffer
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=input_dim, out_channels=32, kernel_size=8, stride=4
        )
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.fc4 = nn.Linear(7 * 7 * 64, 512)
        self.head = nn.Linear(512, action_dim)

    def forward(self, x):
        x = x.float() / 255  # Normalize input
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc4(x.view(x.size(0), -1)))
        return self.head(x)


class DQNAgent:
    def __init__(self, state_dim, action_dim, config):
        """
        Initialize the DQN Agent with the given configuration.
        Args:
            config (dict): Contains hyperparameters and settings.
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = config.get("gamma")
        self.epsilon_start = config.get("epsilon_start")
        self.epsilon_decay = config.get("epsilon_decay")
        self.epsilon_min = config.get("epsilon_min")
        self.update_epsilon_greedy(step=0)
        self.lr = config.get("learning_rate")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize networks
        self.q_network = DQN(self.state_dim, self.action_dim).to(self.device)
        self.target_network = DQN(self.state_dim, self.action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # Optimizer
        self.optimizer = optim.RMSprop(
            self.q_network.parameters(),
            lr=self.lr,
            alpha=0.95,  # RMSProp decay factor
            eps=0.01,  # Epsilon for numerical stability
            momentum=0.0,  # Momentum term, as suggested by the paper
        )
        self.loss_fn = nn.SmoothL1Loss()

    def select_action(self, state, step):
        """
        Epsilon-greedy action selection.
        Expects state to have shape (channels, height, width). We add a batch dimension.
        """
        if np.random.rand() < self.epsilon or step < 50000:
            return np.random.randint(self.action_dim)

        else:
            state_tensor = (
                torch.tensor(state, dtype=torch.uint8).unsqueeze(0).to(self.device)
            )
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            return torch.argmax(q_values).item()

    def train(self, replay_buffer: ReplayBuffer, batch_size) -> float:
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
        q_values = self.q_network(states).gather(1, actions)
        with torch.no_grad():
            best_actions = self.q_network(next_states).argmax(1, keepdim=True)
            max_next_q_values = self.target_network(next_states).gather(1, best_actions)
            target_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values

        loss = self.loss_fn(q_values, target_q_values)
        # Track Q-value statistics
        mean_q = q_values.mean().item()
        max_q = q_values.max().item()
        min_q = q_values.min().item()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        return loss.item()

    def update_target_network(self):
        """
        Update the target network weights.
        """
        self.target_network.load_state_dict(self.q_network.state_dict())

    def get_hidden_model(self):
        return self.q_network

    def update_epsilon_greedy(self, step: int):
        decay_steps = 1_000_000  # 1 million steps before reaching epsilon_min
        self.epsilon = max(
            self.epsilon_min,
            self.epsilon_start
            - (step / decay_steps) * (self.epsilon_start - self.epsilon_min),
        )

    def get_exploration_rate(self):
        return self.epsilon
