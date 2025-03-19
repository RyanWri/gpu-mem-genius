from collections import Counter
import torch
from scipy.stats import entropy
from src.features.dnnmem import estimate_model_mb


# need to use dnnmem to calculate layers and weights
def calc_agent_memory(agent):
    estimated_size = estimate_model_mb(agent)
    return estimated_size


# learn distribution of states
def get_state_entropy(states):
    state_freq = Counter(states)
    freq_list = list(state_freq.values())
    return entropy(freq_list, base=2)


def calculate_complexity_scores(game_id):
    """
    need to implement game complexity based on environment and rules
    """
    rules_complexity = calculate_rules_complexity(game_id)
    pass


def calculate_rules_complexity(game_id):
    # This is a simplistic manual mapping, and you should adjust it based on your game analysis
    complexity_mapping = {
        "ALE/Pong-v5": 1,
        "ALE/MontezumasRevenge-v5": 5,
        "ALE/MsPacman-v5": 3,
        "ALE/Hero-v5": 4,
        # Add more games as needed
    }
    return complexity_mapping.get(game_id, 1)  # Default to 1 if game not listed


def collect_gpu_metrics():
    allocated = torch.cuda.memory_allocated() / 1e6
    reserved = torch.cuda.memory_reserved() / 1e6
    max_allocated = torch.cuda.max_memory_allocated() / 1e6
