import joblib
import pandas as pd
from estimation_model.utils import parse_state_dim
from itertools import product


def load_model(model_path: str):
    return joblib.load(model_path)


def load_data():
    input_file = "/home/ran/datasets/gpu-mem-genius/all_data_combined.csv"
    df = pd.read_csv(input_file)
    df["state_height"], df["state_width"] = zip(*df["state_dim"].map(parse_state_dim))
    df_clean = df.dropna(subset=["state_height", "state_width"])
    return df_clean


def get_static_features(gpu: bool):
    target = "gpu_allocated_mb" if gpu else "cpu_memory_used_mb"
    categorical_features = ["game_name"]
    numeric_features = [
        "state_height",
        "state_width",
        "action_dim",
        "agent_hidden_model_size",
        "exploration_rate",
        "step",
    ]
    feature_cols = numeric_features + categorical_features
    return feature_cols, target


def get_dynamic_features(gpu: bool):
    categorical_features = ["game_name"]
    numeric_features = [
        "state_height",
        "state_width",
        "action_dim",
        "agent_hidden_model_size",
        "exploration_rate",
        "step",
    ]
    buffer_arguments = "buffer_current_usage_mb,buffer_max_usage_mb,buffer_usage_percent,buffer_current_size,buffer_max_size".split(
        ","
    )
    if gpu:
        gpu_arguments = [
            "duration",
            "gpu_reserved_mb",
            "gpu_free_mb",
            "gpu_usage_percent",
        ]
        numeric_features = numeric_features + gpu_arguments + buffer_arguments
    else:
        cpu_arguments = ["duration", "cpu_memory_percent"]
        numeric_features = numeric_features + cpu_arguments + buffer_arguments

    target = "gpu_allocated_mb" if gpu else "cpu_memory_used_mb"
    feature_cols = numeric_features + categorical_features
    return feature_cols, target


def get_model_label(version: int, is_gpu: bool) -> str:
    model_name = "dynamic" if version == 1 else "apriori"
    target_name = "gpu" if is_gpu else "cpu"
    print(f"Running model: ({version}, {is_gpu}) → {model_name}_{target_name}")
    model_label = f"{model_name}_{target_name}"
    return model_label


def main(df: pd.DataFrame, version: int, gpu: bool):
    model_label = get_model_label(version, gpu)
    model_path = f"/home/ran/models/gpu-mem-genius/{model_label}.joblib"
    model = load_model(model_path)
    if version == 0:
        feature_columns, target = get_static_features(gpu=gpu)
    elif version == 1:
        feature_columns, target = get_dynamic_features(gpu=gpu)

    df_features = df.dropna(subset=feature_columns)
    X = df_features[feature_columns]
    y_true = df_features[target]
    y_pred = model.predict(X)
    results_df = df_features[["game_name", "step"]].copy()
    if gpu:
        actual_col, pred_col = "actual_gpu_memory", "predicted_gpu_memory"
    else:
        actual_col, pred_col = "actual_cpu_memory", "predicted_cpu_memory"
    results_df[actual_col] = y_true.values
    results_df[pred_col] = y_pred
    output_path = (
        f"/home/ran/models/gpu-mem-genius/results/predictions/{model_label}.csv"
    )
    results_df.to_csv(output_path, index=False)
    print(f"Saved full-frame predictions to {output_path}")


if __name__ == "__main__":
    df = load_data()
    for version, is_gpu in product([0, 1], [False, True]):
        main(df, version, is_gpu)
