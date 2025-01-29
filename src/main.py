import torch
import time
import gymnasium as gym
import ale_py
import numpy as np
from src.dqn_agent import DQNAgent
from src.replay_buffer import ReplayBuffer
from src.loaders import load_config

config = load_config("src/configurations/experiment_poc.yaml")
episodes = config["environment"]["episodes"]
gym.register_envs(ale_py)
env = gym.make(
    id=config["environment"]["game_name"],
    render_mode=config["environment"]["render_mode"],
)

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
batch_size = config["environment"]["batch_size"]
target_update_frequency = config["environment"]["target_update_frequency"]


for episode in range(episodes):
    state, info = env.reset()
    state = np.transpose(state, (2, 0, 1))  # Convert to channel-first
    total_reward = 0
    start_time = time.time()

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
    # Train the agent at the end of the episode
    agent.train(batch_size, replay_buffer)

    # Update the target network periodically
    if episode % target_update_frequency == 0:
        agent.update_target_network()

    # Log GPU usage
    if torch.cuda.is_available():
        gpu_usage = torch.cuda.memory_allocated() / 1e6  # MB
    else:
        gpu_usage = 0

    end_time = time.time()
    print(
        f"Episode {episode + 1}/{episodes}, Reward: {total_reward}, Epsilon: {agent.epsilon:.3f}, GPU Usage: {gpu_usage:.2f} MB, Time: {end_time - start_time:.2f}s"
    )
