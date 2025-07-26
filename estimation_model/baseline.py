import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from estimation_model.utils import parse_state_dim, create_model, scorers

def setup(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df["state_height"], df["state_width"] = zip(*df["state_dim"].map(parse_state_dim))
    df_clean = df.dropna(subset=["state_height", "state_width"])
    return df_clean

def arguments(gpu : bool = False):
    target = "gpu_allocated_mb" if gpu else "cpu_memory_used_mb" 
    categorical_features = ["game_name"]
    numeric_features = [
        "state_height",
        "state_width",
        "action_dim",
        "agent_hidden_model_size",
        "exploration_rate",
        "buffer_max_size",
        "step"
    ]
    return numeric_features, categorical_features, target



def run_model(gpu: bool = False):
    filepath = "/home/ran/datasets/gpu-mem-genius/all_data_combined.csv"
    df = setup(filepath)
    df['step'] = df['step']
    numeric_features, categorical_features, target = arguments()
    X = df[numeric_features + categorical_features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = create_model(numeric_features, categorical_features)
    model.fit(X_train, y_train)
    neg_mae,neg_rmse, r2_scorer = scorers()
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    mae_scores = cross_val_score(model, X, y, cv=kf, scoring=neg_mae)
    rmse_scores = cross_val_score(model, X, y, cv=kf, scoring=neg_rmse)
    r2_scores = cross_val_score(model, X, y, cv=kf, scoring=r2_scorer)

    print(f"MAE: Mean={-mae_scores.mean():.3f} MB, Std={mae_scores.std():.3f}")
    print(f"RMSE: Mean={-rmse_scores.mean():.3f} MB, Std={rmse_scores.std():.3f}")
    print(f"R²: Mean={r2_scores.mean():.6f}, Std={r2_scores.std():.6f}")
    return mae_scores, rmse_scores, r2_scores

if __name__ == "__main__":
    run_model(gpu=False)
    run_model(gpu=True)
