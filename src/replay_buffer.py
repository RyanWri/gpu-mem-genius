import gc
import torch
import numpy as np
from collections import deque
import logging

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("training.log")],
)


class ReplayBuffer:
    def __init__(self, config: dict):
        self.buffer_size = config["buffer_size"]
        self.buffer = deque(maxlen=self.buffer_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gpu_buffer = None
        self.gpu_buffer_size = 0
        self.transfer_interval = 10000
        self.cpu_memory_threshold = 85

    def add(self, state, action, reward, next_state, done, step):
        reward = np.clip(reward, -1, 1)
        self.buffer.append((state, action, reward, next_state, done))
        if (
            self.device.type == "cuda"
            and step % self.transfer_interval == 0
            and len(self.buffer) >= self.transfer_interval
        ):
            logging.info("transferring buffer to gpu")
            self.transfer_to_gpu()

    def transfer_to_gpu(self):
        """Transfers experiences to GPU, prioritizing last 50,000 samples if memory is limited."""

        if len(self.buffer) == 0:
            logging.info("[WARNING] Buffer is empty, nothing to transfer.")
            return

        torch.cuda.synchronize()
        # Clear GPU buffer before transferring to avoid stacking
        self.clear_gpu_buffer()
        transfer_size = len(self.buffer)
        logging.info(f"[INFO] Transferring {transfer_size} experiences to GPU...")

        # Fetch the last `transfer_size` experiences
        sampled_experiences = list(self.buffer)[-transfer_size:]

        # Convert to tensors and move to GPU
        states, actions, rewards, next_states, dones = zip(*sampled_experiences)
        self.gpu_buffer = (
            torch.tensor(np.array(states), dtype=torch.uint8).to(self.device),
            torch.tensor(np.array(actions), dtype=torch.int64)
            .unsqueeze(1)
            .to(self.device),
            torch.tensor(np.array(rewards), dtype=torch.float32)
            .unsqueeze(1)
            .to(self.device),
            torch.tensor(np.array(next_states), dtype=torch.uint8).to(self.device),
            torch.tensor(np.array(dones), dtype=torch.float32)
            .unsqueeze(1)
            .to(self.device),
        )

        self.gpu_buffer_size = transfer_size
        logging.info(f"[INFO] Successfully transferred {transfer_size} samples to GPU.")

    def clear_gpu_buffer(self):
        """Clears the GPU buffer to free memory if needed."""
        self.gpu_buffer = None
        self.gpu_buffer_size = 0
        torch.cuda.empty_cache()

    def sample(self, batch_size):
        if (
            self.device.type == "cuda"
            and self.gpu_buffer is not None
            and self.gpu_buffer_size > 0
        ):
            return self.sample_gpu_buffer(batch_size)

        return self.sample_cpu_buffer(batch_size)

    def sample_cpu_buffer(self, batch_size):
        """Sample batch from replay buffer and stack frames correctly"""
        valid_indices = [
            i for i in range(len(self.buffer) - 4) if not self.buffer[i + 3][4]
        ]  # Ensure episode continuity
        indices = np.random.choice(valid_indices, batch_size, replace=False)
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for idx in indices:
            # Stack 4 consecutive frames to form a state
            state_stack = np.stack(
                [self.buffer[idx + i][0] for i in range(4)], axis=0
            )  # shape (4, 84, 84)

            next_state_stack = np.stack(
                [self.buffer[idx + i][3] for i in range(4)], axis=0
            )  # shape (4, 84, 84)

            states.append(state_stack)
            actions.append(self.buffer[idx][1])
            rewards.append(self.buffer[idx][2])
            next_states.append(next_state_stack)
            dones.append(self.buffer[idx][4])

        return (
            torch.tensor(np.array(states), dtype=torch.uint8).to(self.device),
            torch.tensor(np.array(actions), dtype=torch.int64)
            .unsqueeze(1)
            .to(self.device),
            torch.tensor(np.array(rewards), dtype=torch.float32)
            .unsqueeze(1)
            .to(self.device),
            torch.tensor(np.array(next_states), dtype=torch.uint8).to(self.device),
            torch.tensor(np.array(dones), dtype=torch.float32)
            .unsqueeze(1)
            .to(self.device),
        )

    def sample_gpu_buffer(self, batch_size):
        """Samples a batch from the GPU buffer efficiently using tensor indexing.

        Parameters:
            batch_size (int): Number of samples to retrieve.

        Returns:
            Tuple of Tensors (states, actions, rewards, next_states, dones) on GPU.
        """
        if self.gpu_buffer is None or self.gpu_buffer_size == 0:
            logging.info("[WARNING] GPU buffer is empty. Cannot sample.")
            return None

        if batch_size > self.gpu_buffer_size:
            batch_size = self.gpu_buffer_size  # Prevent oversampling

        # Generate random indices for sampling
        indices = torch.randint(
            0, self.gpu_buffer_size - 3, (batch_size,), device=self.device
        )

        # Stack 4 consecutive frames to form a state
        states = torch.stack(
            [self.gpu_buffer[0][indices + i] for i in range(4)], dim=1
        )  # (batch_size, 4, 84, 84)
        next_states = torch.stack(
            [self.gpu_buffer[3][indices + i] for i in range(4)], dim=1
        )  # (batch_size, 4, 84, 84)

        actions = self.gpu_buffer[1][indices]
        rewards = self.gpu_buffer[2][indices]
        dones = self.gpu_buffer[4][indices]

        return states, actions, rewards, next_states, dones

    def get_stacked_state(self, current_state):
        frame_history = [current_state]  # Start with the latest frame

        # Retrieve the last 3 states from the buffer
        for i in range(3):
            if len(self.buffer) > i:
                frame_history.append(
                    self.buffer[-(i + 1)][0]
                )  # Get the last stored frame
            else:
                frame_history.append(
                    current_state
                )  # Repeat the current frame if buffer is not full

        # Stack frames in correct order: oldest -> newest
        stacked_state = np.stack(
            frame_history[::-1], axis=0
        )  # Reverse to maintain order

        return stacked_state  # Shape: (4, 84, 84)

    def usage(self):
        return round(len(self.buffer) / self.buffer_size, 3)

    def usage_percent(self):
        return len(self.buffer) / self.buffer_size * 100

    def get_current_size(self):
        return len(self.buffer)

    def clear(self):
        """Clears the buffer and frees memory."""
        self.buffer.clear()
        gc.collect()
        torch.cuda.empty_cache()

    def memory_usage(self):
        buffer_current_size_bytes = sum(
            state.nbytes
            + next_state.nbytes
            + np.array(action).nbytes
            + np.array(reward).nbytes
            + np.array(done).nbytes
            for state, action, reward, next_state, done in self.buffer
        )

        buffer_max_size_bytes = (
            self.buffer_size * buffer_current_size_bytes / max(len(self.buffer), 1)
        )

        return {
            "buffer_current_usage_mb": buffer_current_size_bytes / (1024**2),
            "buffer_max_usage_mb": buffer_max_size_bytes / (1024**2),
            "buffer_usage_percent": (len(self.buffer) / self.buffer_size) * 100,
            "buffer_current_size": len(self.buffer),
            "buffer_max_size": self.buffer_size,
        }
