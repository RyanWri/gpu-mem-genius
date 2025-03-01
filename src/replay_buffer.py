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
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
        )

    def usage(self):
        return round(len(self.buffer) / self.buffer_size, 3)

    def usage_percent(self):
        return len(self.buffer) / self.buffer_size * 100

    def get_current_size(self):
        return len(self.buffer)
