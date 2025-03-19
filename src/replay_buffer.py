import gc
import torch
import numpy as np
from collections import deque
import psutil  # For checking memory usage


class ReplayBuffer:
    def __init__(self, config: dict):
        self.buffer_size = config["buffer_size"]
        self.buffer = deque(maxlen=self.buffer_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gpu_buffer = None  # Holds experiences on GPU
        self.gpu_buffer_size = 0  # Tracks how many experiences are in GPU
        self.transfer_interval = 50000  # Transfer every 50k steps
        self.max_gpu_samples = 300000  # Limit GPU buffer to ~6GB usage
        self.cpu_memory_threshold = 85  # Reduce memory usage when CPU exceeds 85%

    def add(self, state, action, reward, next_state, done, step):
        """Store a transition (state, action, reward, next_state, done) in the CPU buffer."""
        reward = np.clip(reward, -1, 1)
        self.buffer.append((state, action, reward, next_state, done))

        # Every 50k steps, transfer 50k samples to GPU
        if (
            step % self.transfer_interval == 0
            and len(self.buffer) >= self.transfer_interval
        ):
            if self.device == "cuda":
                self.transfer_to_gpu()

        # Monitor CPU memory and free resources if usage exceeds 85%
        self.check_cpu_memory()

    def transfer_to_gpu(self):
        """Move a batch of 50k transitions from CPU to GPU if memory allows."""
        if len(self.buffer) < self.transfer_interval:
            return  # Not enough samples to transfer

        print("[INFO] Checking GPU memory before transfer...")
        torch.cuda.synchronize()
        free_mem = torch.cuda.memory_allocated() / 1e9  # Convert to GB
        print(f"[INFO] Free GPU Memory: {free_mem:.2f} GB")

        # Check if we have enough VRAM for transfer (~1GB per 50k samples)
        if free_mem > 9:  # Keep ~3GB free for computations
            print("[WARNING] GPU memory is running low, clearing some GPU buffer...")
            self.clear_gpu_buffer()

        # Transfer 50k samples from CPU to GPU
        print("[INFO] Transferring 50k experiences to GPU...")
        sample_indices = np.random.choice(len(self.buffer), 50000, replace=False)
        sampled_experiences = [self.buffer[i] for i in sample_indices]

        # Convert to tensors and move to GPU
        states, actions, rewards, next_states, dones = zip(*sampled_experiences)
        self.gpu_buffer = (
            torch.tensor(np.array(states), dtype=torch.float32).to(self.device),
            torch.tensor(np.array(actions), dtype=torch.int64)
            .unsqueeze(1)
            .to(self.device),
            torch.tensor(np.array(rewards), dtype=torch.float32)
            .unsqueeze(1)
            .to(self.device),
            torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device),
            torch.tensor(np.array(dones), dtype=torch.float32)
            .unsqueeze(1)
            .to(self.device),
        )
        self.gpu_buffer_size = len(sample_indices)

    def clear_gpu_buffer(self):
        """Clears the GPU buffer to free memory if needed."""
        self.gpu_buffer = None
        self.gpu_buffer_size = 0
        torch.cuda.empty_cache()
        print("[INFO] Cleared GPU buffer to free memory.")

    def check_cpu_memory(self):
        """Monitor CPU memory usage and reduce memory if usage exceeds 85%."""
        cpu_usage = psutil.virtual_memory().percent
        if cpu_usage > self.cpu_memory_threshold:
            print(f"[WARNING] CPU memory is at {cpu_usage}%, freeing resources...")
            self.reduce_memory_usage()

    def reduce_memory_usage(self):
        """Free unused CPU memory by reducing buffer size and running garbage collection."""
        num_to_remove = int(len(self.buffer) * 0.5)
        for _ in range(num_to_remove):
            self.buffer.popleft()
        # Run garbage collection
        gc.collect()

        # Free unused PyTorch memory
        if self.device == "cuda":
            torch.cuda.empty_cache()
            self.transfer_to_gpu()

        print("[INFO] Reduced buffer size and cleared cache. CPU memory freed.")

    def sample(self, batch_size):
        """Sample batch from replay buffer and stack frames correctly"""
        indices = np.random.choice(len(self.buffer) - 3, batch_size, replace=False)
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
        print("[INFO] Cleared entire buffer and freed memory.")
