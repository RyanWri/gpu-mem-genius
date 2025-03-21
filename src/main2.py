import random
import psutil
import torch
import time
import logging
from datetime import datetime
from src.features.collect import calc_agent_memory
from src.atari_env import make_env
from src.agents.dqn_agent import DQNAgent
from src.replay_buffer import ReplayBuffer
from src.loaders import (
    load_checkpoint,
    load_config,
    save_checkpoint,
    save_list_of_dicts_to_dataframe,
)

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("training.log")],
)


def initialize_training(game_id="pong"):
    """Initialize environment, agent, replay buffer, and configurations."""
    logging.info("Loading configuration...")
    experiment_game = "pong"
    config_filename = f"src/configurations/experiment_{experiment_game}.yaml"
    config = load_config(config_filename)

    env = make_env(
        config["environment"]["game_name"],
        config["environment"]["render_mode"],
        config["environment"]["obs_type"],
    )

    agent = DQNAgent(
        state_dim=4,
        action_dim=env.action_space.n,
        config=config["agent"],
    )

    if config["checkpoints"]["load_checkpoint"]:
        logging.info("Loading checkpoint ...")
        load_checkpoint(agent, config["checkpoints"]["checkpoint_path"])

    replay_buffer = ReplayBuffer(config["replay_buffer"])
    return env, agent, replay_buffer, config


def train_step(agent, replay_buffer, batch_size, step):
    """Perform a single training step, manage memory, and log loss."""
    if step % 4 == 0:
        loss = agent.train(replay_buffer, batch_size)
        if random.random() < 0.001:
            print(f"Loss at step {step}: {loss}")
    agent.update_epsilon_greedy(step)


def log_step(
    episode, total_reward, agent, step, episode_time, cpu_usage, gpu_before, gpu_after
):
    """Handles structured logging per episode."""
    logging.info(
        f"Episode: {episode}, Reward: {total_reward}, Steps: {step}, Time: {episode_time:.2f}s"
    )
    logging.info(
        f"CPU Usage: {cpu_usage}MB, GPU Before: {gpu_before:.2f}MB, GPU After: {gpu_after:.2f}MB"
    )
    logging.info(f"Exploration Rate: {agent.get_exploration_rate():.4f}")


def checkpoint_manager(agent, episode, checkpoints, save_options, dt):
    """Saves agent checkpoints periodically."""
    if episode % checkpoints["frequency"] == 0:
        save_checkpoint(agent, episode, save_options, dt)


def training_loop(env, agent, replay_buffer, config):
    """Main training loop handling episodes and logging."""
    episodes = config["environment"]["episodes"]
    batch_size = config["environment"]["batch_size"]
    target_update_frequency = config["environment"]["target_update_frequency"]
    checkpoints = config["checkpoints"]
    save_options = config["save_options"]
    replay_start_size = 50000  # Minimum samples before training

    dataset = []
    dt = datetime.now().strftime("%Y%m%d_%H%M%S")
    # collect static features
    static_features = {
        "game_name": config["environment"]["game_name"],
        "state_dim": env.observation_space.shape,
        "action_dim": env.action_space.n,
        "agent_hidden_model_size": calc_agent_memory(agent.get_hidden_model()),
    }

    step = 0
    for episode in range(episodes + 1):
        logging.info(f"Starting episode {episode}/{episodes+1}")

        checkpoint_manager(agent, episode, checkpoints, save_options, dt)

        state, info = env.reset()
        total_reward = 0
        start_time = time.time()
        done = False

        gpu_before_train = (
            torch.cuda.memory_allocated() / 1e6 if torch.cuda.is_available() else 0
        )

        while not done:
            action = agent.select_action(replay_buffer.get_stacked_state(state), step)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            if replay_buffer.get_current_size() >= replay_start_size:
                train_step(agent, replay_buffer, batch_size, step)
                if step % target_update_frequency == 0:
                    agent.update_target_network()

            replay_buffer.add(state, action, reward, next_state, done, step)
            state = next_state
            total_reward += reward
            step += 1

        episode_time = time.time() - start_time
        gpu_after_train = (
            torch.cuda.memory_allocated() / 1e6 if torch.cuda.is_available() else 0
        )
        cpu_usage = psutil.virtual_memory().used / 1e6  # Convert to MB

        log_step(
            episode,
            total_reward,
            agent,
            step,
            episode_time,
            cpu_usage,
            gpu_before_train,
            gpu_after_train,
        )

        # Append to dataset
        episode_features = collect_and_log_features(
            episode, total_reward, agent, start_time, gpu_before_train, static_features
        )
        dataset.append(episode_features)

        if episode % checkpoints["data"] == 0:
            logging.info("Saving dataset...")
            save_list_of_dicts_to_dataframe(dataset, save_options, dt)

    logging.info("Training complete.")


def collect_and_log_features(
    episode, total_reward, agent, start_time, gpu_before_train, static_features
):
    """Collects dynamic training features and appends them to the dataset."""
    episode_time = time.time() - start_time

    # Dynamic episode features
    dynamic_features = {
        "episode_reward": total_reward,
        "episode_length": episode_time,
        "exploration_rate": agent.get_exploration_rate(),
    }

    # Convert GPU memory usage to MB
    gpu_after_train = (
        torch.cuda.memory_allocated() / 1e6 if torch.cuda.is_available() else 0
    )

    # CPU memory usage in MB
    cpu_usage = psutil.virtual_memory().used / 1e6

    # Memory usage features
    memory_features = {
        "gpu_before_train": gpu_before_train,
        "gpu_after_train": gpu_after_train,
        "cpu_usage": cpu_usage,
    }

    # Merge all features (static + dynamic + memory)
    episode_features = {**static_features, **dynamic_features, **memory_features}
    return episode_features


if __name__ == "__main__":
    env, agent, replay_buffer, config = initialize_training(game_id="pong")
    training_loop(env, agent, replay_buffer, config)
