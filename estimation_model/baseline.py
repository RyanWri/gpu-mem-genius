import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from estimation_model.utils import parse_state_dim, create_model, scorers


def setup(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df["state_height"], df["state_width"] = zip(*df["state_dim"].map(parse_state_dim))
    df_clean = df.dropna(subset=["state_height", "state_width"])
    return df_clean


# estimation based on apriori data only
def static_arguments(gpu: bool = False):
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
    return numeric_features, categorical_features, target


# improved model with data from warmup training steps
def dynamic_arguments(gpu: bool = False):
    numeric_features, categorical_features, target = static_arguments(gpu)
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

    return numeric_features, categorical_features, target


def run_model(numeric_features, categorical_features, target):
    filepath = "/home/ran/datasets/gpu-mem-genius/all_data_combined.csv"
    df = setup(filepath)
    X = df[numeric_features + categorical_features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = create_model(numeric_features, categorical_features)
    model.fit(X_train, y_train)
    neg_mae, neg_rmse, r2_scorer = scorers()
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    mae_scores = cross_val_score(model, X, y, cv=kf, scoring=neg_mae)
    rmse_scores = cross_val_score(model, X, y, cv=kf, scoring=neg_rmse)
    r2_scores = cross_val_score(model, X, y, cv=kf, scoring=r2_scorer)

    print(f"MAE: Mean={-mae_scores.mean():.3f} MB, Std={mae_scores.std():.3f}")
    print(f"RMSE: Mean={-rmse_scores.mean():.3f} MB, Std={rmse_scores.std():.3f}")
    print(f"R²: Mean={r2_scores.mean():.6f}, Std={r2_scores.std():.6f}")
    return mae_scores, rmse_scores, r2_scores


def main(version: int, gpu: bool = False):
    model_version = {0: "apriori", 1: "dynamic"}
    if model_version[version] not in ["apriori", "dynamic"]:
        raise ValueError(f"model must be either apriori or dynamic, got {version}")

    numeric_features, categorical_features, target = (
        dynamic_arguments(gpu) if version == 1 else static_arguments(gpu)
    )
    run_model(numeric_features, categorical_features, target)


if __name__ == "__main__":
    print("running apriori model for cpu")
    main(version=0, gpu=False)
    print("running apriori model for gpu")
    main(version=0, gpu=True)
    print("running dynamic model for cpu")
    main(version=1, gpu=False)
    print("running dynamic model for gpu")
    main(version=1, gpu=True)
