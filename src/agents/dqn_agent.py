import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from src.replay_buffer import ReplayBuffer


class DQN(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(DQN, self).__init__()
        c, h, w = input_dim  # Channels, Height, Width

        # Two convolutional layers as described in the paper:
        self.conv_layers = nn.Sequential(
            nn.Conv2d(c, 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
        )

        # Dynamically compute the size of the output from conv layers:
        with torch.no_grad():
            dummy_input = torch.zeros(1, c, h, w)
            conv_out = self.conv_layers(dummy_input)
            conv_out_size = conv_out.view(1, -1).size(1)

        # Fully connected layers: one hidden layer with 256 units, then output layer.
        self.fc_layers = nn.Sequential(
            nn.Linear(conv_out_size, 256), nn.ReLU(), nn.Linear(256, action_dim)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten the conv output
        return self.fc_layers(x)


class DQNAgent:
    def __init__(self, state_dim, action_dim, config):
        """
        Initialize the DQN Agent with the given configuration.
        Args:
            config (dict): Contains hyperparameters and settings.
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = config.get("gamma", 0.99)
        self.epsilon = config.get("epsilon", 1.0)
        self.epsilon_decay = config.get("epsilon_decay", 0.9995)
        self.epsilon_min = config.get("epsilon_min", 0.1)
        self.lr = config.get("lr", 1e-4)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize networks
        self.q_network = DQN(self.state_dim, self.action_dim).to(self.device)
        self.target_network = DQN(self.state_dim, self.action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # Optimizer
        self.optimizer = optim.RMSprop(
            self.q_network.parameters(), lr=self.lr, alpha=0.95, eps=0.01
        )
        self.loss_fn = nn.SmoothL1Loss()

    def select_action(self, state):
        """
        Epsilon-greedy action selection.
        Expects state to have shape (channels, height, width). We add a batch dimension.
        """
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        state_tensor = (
            torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        )
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return torch.argmax(q_values).item()

    def train(self, batch_size, replay_buffer: ReplayBuffer):
        """
        Train the Q-network using a batch from the replay buffer.
        Expects that the replay buffer returns tensors of correct shape:
            - states: (batch_size, channels, height, width)
            - actions: (batch_size, 1)
            - rewards: (batch_size, 1)
            - next_states: (batch_size, channels, height, width)
            - dones: (batch_size, 1)
        """
        if replay_buffer.get_current_size() < batch_size:
            return

        # Sample a batch from the replay buffer
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
        # The tensors are already on the appropriate device from ReplayBuffer,
        # but we can re-ensure that here.
        states = states.to(self.device)
        actions = actions.to(
            self.device
        ).long()  # Already unsqueezed in the ReplayBuffer
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # Compute Q-values for current states
        q_values = self.q_network(states).gather(1, actions)

        # Compute target Q-values using the target network
        with torch.no_grad():
            max_next_q_values = self.target_network(next_states).max(1)[0].unsqueeze(1)
            target_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values

        # Compute loss and perform optimization
        loss = self.loss_fn(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def update_target_network(self):
        """
        Update the target network weights.
        """
        self.target_network.load_state_dict(self.q_network.state_dict())

    def get_hidden_model(self):
        return self.q_network
