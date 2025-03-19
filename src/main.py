import csv
from datetime import datetime
import psutil
import torch
import time
import logging
from src.atari_env import make_env
from src.features.collect import calc_agent_memory
from src.agents.dqn_agent import DQNAgent
from src.replay_buffer import ReplayBuffer
from src.loaders import (
    load_checkpoint,
    load_config,
    save_checkpoint,
    save_list_of_dicts_to_dataframe,
    write_loss_logs,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("training.log")],
)

# load configuration
logging.info("Loading configuration...")
experiment_game = "pong"
config_filename = f"src/configurations/experiment_{experiment_game}.yaml"
config = load_config(config_filename)
episodes = config["environment"]["episodes"]
save_options = config["save_options"]
checkpoints = config["checkpoints"]

# register atari game
logging.info(f"Registering environment: {config['environment']['game_name']}")
env = make_env(
    config["environment"]["game_name"],
    config["environment"]["render_mode"],
    config["environment"]["obs_type"],
)

# build agent and replay buffer
# remember to pass channel first for state dim
agent = DQNAgent(
    state_dim=env.observation_space.shape,
    action_dim=env.action_space.n,
    config=config["agent"],
)

# load checkpoint if necessary
if checkpoints["load_checkpoint"]:
    logging.info("Loading checkpoint ...")
    load_checkpoint(agent, checkpoints["checkpoint_path"])

replay_buffer = ReplayBuffer(config["replay_buffer"])

# hyperparameters
batch_size = config["environment"]["batch_size"]
target_update_frequency = config["environment"]["target_update_frequency"]

agent_hidden_model_memory = calc_agent_memory(agent.get_hidden_model())

# collect static features
static_features = {
    "game_name": config["environment"]["game_name"],
    "state_dim": env.observation_space.shape,
    "action_dim": env.action_space.n,
    "agent_hidden_model_size": agent_hidden_model_memory,
}

dataset = []
dt = datetime.now().strftime("%Y%m%d_%H%M%S")


# handle loss
loss_log_file = "loss_log.csv"

# Open file & write header only once
with open(loss_log_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Episode", "Step", "Loss"])  # Headers


replay_start_size = 50000  # Minimum samples before training
train_frequency = 4  # Train every 4 steps

# main loop
step = 0
for episode in range(episodes + 1):
    logging.info(f"Starting episode {episode}/{episodes+1}")

    # Modify Training Loop in `main.py`
    if episode % checkpoints["frequency"] == 0:
        save_checkpoint(agent, episode, save_options, dt)
    if episode % 5 == 0:
        print(f"Replay Buffer Size: {len(replay_buffer.buffer)}")
    # first step of an episode
    state, info = env.reset()
    total_reward = 0
    # measure episode time
    start_time = time.time()
    # run episode till truncated or terminated
    done = False

    # Convert to MB
    gpu_before_train = (
        torch.cuda.memory_allocated() / 1e6 if torch.cuda.is_available() else 0
    )

    while not done:
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        # Add transition to replay buffer
        replay_buffer.add(state, action, reward, next_state, done, step)
        state = next_state
        total_reward += reward
        step += 1

        # Train only if we have enough samples
        if step > replay_start_size and step % train_frequency == 0:
            batch = replay_buffer.sample(32)
            loss = agent.train(batch)
            write_loss_logs(loss_log_file, episode, step, loss)
            agent.update_epsilon_greedy(step)
            if step % target_update_frequency == 0:
                agent.update_target_network()

    episode_time = time.time() - start_time

    # dynamic features collected at the end of each episode should be inserted here
    dynamic_features = {
        "episode_reward": total_reward,
        "episode_length": episode_time,
        "exploration_rate": agent.get_exploration_rate(),
    }

    # Convert to MB
    gpu_after_train = (
        torch.cuda.memory_allocated() / 1e6 if torch.cuda.is_available() else 0
    )

    # CPU memory
    cpu_usage = psutil.virtual_memory().used / 1e6  # MB

    memory_features = {
        "gpu_before_train": gpu_before_train,
        "gpu_after_train": gpu_after_train,
        "cpu_usage": cpu_usage,
    }

    # Merge static and dynamic features
    episode_features = {**static_features, **dynamic_features, **memory_features}
    dataset.append(episode_features)

    if episode % checkpoints["data"] == 0:
        # save dataset as dataframe to disk
        logging.info("Saving dataset...")
        save_list_of_dicts_to_dataframe(dataset, save_options, dt=dt)

logging.info("Training complete.")
