import csv
import yaml
import pandas as pd
import torch
import os
from src.agents.dqn_agent import DQNAgent


def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def save_list_of_dicts_to_dataframe(dict_list: list[dict], save_options: dict, dt: str):
    if not dict_list:
        raise ValueError("The list of dictionaries is empty.")
    if not isinstance(dict_list, list):
        raise TypeError("The first parameter must be a list of dictionaries.")
    if not isinstance(dict_list[0], dict):
        raise TypeError("The list must contain dictionary elements.")

    df = pd.DataFrame(dict_list)
    folder, format = save_options.values()
    filename = f"data.{format}"
    save_path = os.path.join(folder, dt, filename)
    if format == "csv":
        df.to_csv(save_path, index=False)
    else:
        # by default save parquet unless otherwise
        df.to_parquet(save_path)


def save_checkpoint(agent, episode: int, save_options: str, dt: str):
    folder, format = save_options.values()
    save_folder = os.path.join(folder, dt)
    os.makedirs(save_folder, exist_ok=True)
    torch.save(
        agent.q_network.state_dict(),
        os.path.join(save_folder, f"dqn_checkpoint_{episode}.pth"),
    )


def load_checkpoint(agent: DQNAgent, checkpoint_path: str):
    agent.q_network.load_state_dict(torch.load(checkpoint_path))


def write_loss_logs(loss_log_file, episode, step, loss_value):
    with open(loss_log_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([episode, step, loss_value])  # Save loss
