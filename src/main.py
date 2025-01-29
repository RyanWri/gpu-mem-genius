import pandas as pd
import psutil
import torch
import time
import gymnasium as gym
import ale_py
import numpy as np
from src.features.collect import get_exploration_rate
from src.dqn_agent import DQNAgent
from src.replay_buffer import ReplayBuffer
from src.loaders import load_config

# load configuration
config = load_config("src/configurations/experiment_poc.yaml")
episodes = config["environment"]["episodes"]

# register atari game
gym.register_envs(ale_py)
env = gym.make(
    id=config["environment"]["game_name"],
    render_mode=config["environment"]["render_mode"],
)

# build agent and replay buffer
# remember to pass channel first for state dim
agent = DQNAgent(
    state_dim=(
        env.observation_space.shape[2],
        env.observation_space.shape[0],
        env.observation_space.shape[1],
    ),
    action_dim=env.action_space.n,
    config=config["agent"],
)
replay_buffer = ReplayBuffer(config["replay_buffer"])

# hyperparameters
batch_size = config["environment"]["batch_size"]
target_update_frequency = config["environment"]["target_update_frequency"]


# collect static features
static_features = {
    "game_name": config["environment"]["game_name"],
    "state_dim": env.observation_space.shape,
    "action_dim": env.action_space.n,
}

dataset = []

# main loop
for episode in range(episodes):
    # first step of an episode
    state, info = env.reset()
    state = np.transpose(state, (2, 0, 1))  # Convert to channel-first
    total_reward = 0

    # measure episode time
    start_time = time.time()

    # run episode till truncated or terminated
    done = False
    while not done:
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        next_state = np.transpose(next_state, (2, 0, 1))

        # Add transition to replay buffer
        replay_buffer.add(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

    episode_time = time.time() - start_time
    # dynamic features collected at the end of each episode should be inserted here
    dynamic_features = {
        "episode_reward": total_reward,
        "episode_length": episode_time,
        "exploration_rate": get_exploration_rate(episode, config["agent"]),
    }

    # Convert to MB
    gpu_before_train = (
        torch.cuda.memory_allocated() / 1e6 if torch.cuda.is_available() else 0
    )

    # Train the agent at the end of the episode
    agent.train(batch_size, replay_buffer)

    # Convert to MB
    gpu_after_train = (
        torch.cuda.memory_allocated() / 1e6 if torch.cuda.is_available() else 0
    )

    # CPU memory
    cpu_usage = psutil.virtual_memory().used / 1e6  # MB

    # Update the target network periodically
    if episode % target_update_frequency == 0:
        agent.update_target_network()

    memory_features = {
        "gpu_before_train": gpu_before_train,
        "gpu_after_train": gpu_after_train,
        "cpu_usage": cpu_usage,
    }

    # Merge static and dynamic features
    episode_features = {**static_features, **dynamic_features, **memory_features}
    dataset.append(episode_features)


df = pd.DataFrame(dataset)
df.to_csv("dataset_v1.csv")
