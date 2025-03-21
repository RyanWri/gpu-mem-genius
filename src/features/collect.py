import torch
from src.replay_buffer import ReplayBuffer
from src.features.dnnmem import estimate_model_mb
import psutil


# need to use dnnmem to calculate layers and weights
def calc_agent_memory(agent):
    estimated_size = estimate_model_mb(agent)
    return estimated_size


def get_gpu_metrics(device=0):
    if not torch.cuda.is_available():
        return {}
    allocated_mb = torch.cuda.memory_allocated(device) / (1024**2)
    reserved_mb = torch.cuda.memory_reserved(device) / (1024**2)
    total_mb = torch.cuda.get_device_properties(device).total_memory / (1024**2)
    return {
        "gpu_allocated_mb": allocated_mb,
        "gpu_reserved_mb": reserved_mb,
        "gpu_total_mb": total_mb,
        "gpu_usage_percent": 100.0 * reserved_mb / total_mb,
    }


def get_cpu_metrics():
    return {
        "cpu_percent": psutil.cpu_percent(interval=1),
        "cpu_memory_used_mb": psutil.virtual_memory().used / (1024**2),
        "cpu_memory_percent": psutil.virtual_memory().percent,
    }


def collect_resources(replay_buffer: ReplayBuffer, training_step: int):
    cpu_stats = get_cpu_metrics()
    gpu_stats = get_gpu_metrics()
    buffer_stats = replay_buffer.memory_usage()

    return {
        "step": training_step,
        **cpu_stats,
        **gpu_stats,
        **buffer_stats,
    }
