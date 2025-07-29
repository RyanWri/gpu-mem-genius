import joblib
import pandas as pd
from estimation_model.utils import parse_state_dim


def load_model(model_path: str):
    return joblib.load(model_path)


def main(model_label: str):
    input_file = "/home/ran/datasets/gpu-mem-genius/all_data_combined.csv"
    model_path = "/home/ran/models/gpu-mem-genius/apriori_cpu.joblib"

    # Load model and data
    model = load_model(model_path)
    df = pd.read_csv(input_file)

    # Parse and clean state dimensions
    df["state_height"], df["state_width"] = zip(*df["state_dim"].map(parse_state_dim))
    df_clean = df.dropna(subset=["state_height", "state_width"])

    # Define features
    categorical_features = ["game_name"]
    numeric_features = [
        "state_height",
        "state_width",
        "action_dim",
        "agent_hidden_model_size",
        "exploration_rate",
        "step",
    ]
    feature_columns = numeric_features + categorical_features

    # Drop rows with missing feature values
    df_features = df_clean.dropna(subset=feature_columns)

    # Predict
    X = df_features[feature_columns]
    y_true = df_features["cpu_memory_used_mb"]
    y_pred = model.predict(X)

    # Combine for output
    results_df = df_features[["game_name", "step"]].copy()
    results_df["actual_cpu_memory"] = y_true.values
    results_df["predicted_cpu_memory"] = y_pred

    # Save to CSV
    output_path = (
        f"/home/ran/models/gpu-mem-genius/results/predictions/{model_label}.csv"
    )
    results_df.to_csv(output_path, index=False)
    print(f"Saved full-frame predictions to {output_path}")


if __name__ == "__main__":
    for model_version, gpu in zip([0, 1], [False, True]):
        model_name = "apriori" if model_version == 0 else "dynamic"
        model_label = f"{model_name}_gpu" if gpu else f"{model_name}_cpu"
        main(model_label=model_label)
