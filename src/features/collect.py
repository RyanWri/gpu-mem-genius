from collections import Counter
import math
from scipy.stats import entropy
from src.atari_env import AtariEnv
from src.replay_buffer import ReplayBuffer


def get_network_architecture(dqn_agent):
    # need to use dnnmem to calculate layers and weights
    return None


def environment_info(env):
    return {
        "game_name": env.unwrapped.spec.id,
        "action_space": env.action_space.n,
        "observation_space": env.observation_space.shape,
    }


def collect_static_features(
    env: AtariEnv, network_architecture, replay_buffer: ReplayBuffer
):
    state_dim, action_dim = env.get_dimensions().values()
    dqn_agent_mem = get_network_architecture(network_architecture)
    return {
        "agent_memory": dqn_agent_mem,
        "replay_buffer_size": replay_buffer.get_buffer_size(),
        "state_dimension": state_dim,
        "action_dimension": action_dim,
    }


def collect_dynamic_features(
    reward, num_steps, states, episode_duration, episode_number
):
    # needs to add states_entropy = get_state_entropy(states)
    epsilon_config = (0.1, 0.01, 0.0001)
    exploration_rate = get_exploration_rate(episode_number, epsilon_config)

    return {
        "episode_reward": reward,
        "episode_steps": num_steps,
        "episode_duration": episode_duration,
        "episode_exploration_rate": exploration_rate,
        "episode_states_entropy": 0,
    }


def get_state_entropy(states):
    state_freq = Counter(states)
    freq_list = list(state_freq.values())
    return entropy(freq_list, base=2)


def get_exploration_rate(episode_number, epsilon_config):
    epsilon_min, epsilon_max, decay_rate = (
        epsilon_config["epsilon_min"],
        epsilon_config["epsilon"],
        epsilon_config["epsilon_decay"],
    )
    return epsilon_min + (epsilon_max - epsilon_min) * math.exp(
        -decay_rate * episode_number
    )
