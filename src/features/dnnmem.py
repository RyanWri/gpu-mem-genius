import psutil
import torch
from src.replay_buffer import ReplayBuffer


# 1) Pseudocode to estimate model size in PyTorch:
# function estimate_model_size(model):
# This function sums up the memory taken by each parameter in the model.
# Optionally, consider activation memory and optimizer states if a more comprehensive
# memory estimation is needed.
def estimate_model_size(model):
    total_memory = 0
    for param in model.parameters():
        num_elements = param.numel()
        bytes_per_element = param.element_size()
        total_memory += num_elements * bytes_per_element
    return total_memory


# 3) How to get the model parameters or size in PyTorch:
# - Use model.parameters() to iterate over all parameters.
# - param.numel() gives number of elements in the tensor.
# - param.element_size() gives the size in bytes of each element.
# - Multiply them to get the total bytes used by that parameter.
# - Sum over all parameters to get total parameter memory usage.
def estimate_model_mb(model):
    model_size_bytes = estimate_model_size(model)
    model_size_mb = model_size_bytes / (1024**2)
    return model_size_mb


def free_memory_if_needed(replay_buffer: ReplayBuffer, gpu_flush: bool) -> None:
    if psutil.virtual_memory().percent > 80:
        replay_buffer.clear()
        if gpu_flush:
            torch.cuda.empty_cache()
