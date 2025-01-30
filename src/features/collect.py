from collections import Counter
import math
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


# understand how well our agent explore his environment
def get_exploration_rate(episode_number, epsilon_config):
    epsilon_min, epsilon_max, decay_rate = (
        epsilon_config["epsilon_min"],
        epsilon_config["epsilon"],
        epsilon_config["epsilon_decay"],
    )
    return epsilon_min + (epsilon_max - epsilon_min) * math.exp(
        -decay_rate * episode_number
    )


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
