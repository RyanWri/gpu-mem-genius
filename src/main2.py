import time
import logging
from datetime import datetime
from src.features.collect import calc_agent_memory, collect_resources
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
    agent.update_epsilon_greedy(step)


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
    replay_start_size = 10000  # Minimum samples before training

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

            if step % 1000 == 0:
                logging.info(f"Step {step}: collecting resources...")
                duration = time.time()
                features = collect_and_log_features(
                    step,
                    agent,
                    duration=duration - start_time,
                    static_features=static_features,
                )
                dataset.append(features)
                start_time = duration
            step += 1

        if episode % checkpoints["data"] == 0:
            logging.info("Saving dataset...")
            save_list_of_dicts_to_dataframe(dataset, save_options, dt)

    logging.info("Training complete.")


def collect_and_log_features(step, agent, duration, static_features):
    resources_metrics = collect_resources(replay_buffer, step)
    dynamic_features = {
        "duration": duration,
        "exploration_rate": agent.get_exploration_rate(),
    }
    # Merge all features (static + dynamic + memory)
    features = {**static_features, **dynamic_features, **resources_metrics}
    return features


if __name__ == "__main__":
    env, agent, replay_buffer, config = initialize_training(game_id="pong")
    training_loop(env, agent, replay_buffer, config)
