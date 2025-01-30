import yaml
import pandas as pd
import os
from datetime import datetime


def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def save_list_of_dicts_to_dataframe(dict_list: list[dict], save_options: dict):
    if not dict_list:
        raise ValueError("The list of dictionaries is empty.")
    if not isinstance(dict_list, list):
        raise TypeError("The first parameter must be a list of dictionaries.")
    if not isinstance(dict_list[0], dict):
        raise TypeError("The list must contain dictionary elements.")

    df = pd.DataFrame(dict_list)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder, format = save_options.values()
    filename = f"{timestamp}.{format}"
    save_path = os.path.join(folder, filename)
    if format == "csv":
        df.to_csv(save_path)
    else:
        # by default save parquet unless otherwise
        df.to_parquet(save_path)
