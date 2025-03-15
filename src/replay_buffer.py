import torch
import numpy as np
from collections import deque


class ReplayBuffer:
    def __init__(self, config: dict):
        buffer_size = config["buffer_size"]
        batch_size = config["batch_size"]
        frame_stack_size = 4  # Number of frames per state
        self.buffer = deque(maxlen=buffer_size)
        self.frame_stack = deque(maxlen=frame_stack_size)
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, action, reward, next_state, done):
        """Stack frames before adding to replay buffer"""
        if len(self.frame_stack) < self.frame_stack.maxlen:
            for _ in range(self.frame_stack.maxlen - len(self.frame_stack)):
                self.frame_stack.append(state)  # Fill with initial frame
        self.frame_stack.append(state)
        stacked_state = np.array(self.frame_stack)

        self.frame_stack.append(next_state)
        stacked_next_state = np.array(self.frame_stack)

        self.buffer.append((stacked_state, action, reward, stacked_next_state, done))

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(
            *(self.buffer[i] for i in indices)
        )
        # Convert to tensors
        t_states = torch.tensor(np.array(states), dtype=torch.float32)
        t_actions = torch.tensor(np.array(actions), dtype=torch.int32).unsqueeze(1)
        t_rewards = torch.tensor(np.array(rewards), dtype=torch.float32).unsqueeze(1)
        t_next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
        t_dones = torch.tensor(np.array(dones), dtype=torch.float32).unsqueeze(1)

        return (t_states, t_actions, t_rewards, t_next_states, t_dones)

    def usage(self):
        return round(len(self.buffer) / self.buffer_size, 3)

    def usage_percent(self):
        return len(self.buffer) / self.buffer_size * 100

    def get_current_size(self):
        return len(self.buffer)
