import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    r2_score,
    make_scorer,
)
from estimation_model.utils import parse_state_dim, split, create_model

def setup(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df["state_height"], df["state_width"] = zip(*df["state_dim"].map(parse_state_dim))
    df_clean = df.dropna(subset=["state_height", "state_width"])
    return df_clean


def main():
    filepath = "/home/linuxu/datasets/atari-mem/all_data_combined.csv"
    df = setup(filepath)
    x_train, x_test, y_train, y_test = split(df)
    model = create_model(numeric_features, categorical_features)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)






# 3. Define input features and target variable
#    Features are only those known before training begins
features = [
    "game_name",  # categorical: game/environment name
    "state_height",  # numeric: height dimension of the state
    "state_width",  # numeric: width dimension of the state
    "action_dim",  # numeric: number of possible actions
    "agent_hidden_model_size",  # numeric: size of hidden layers in the agent model
    "exploration_rate",  # numeric: exploration rate hyperparameter
    "buffer_max_size",  # numeric: max replay buffer size configured
    "step"
]
target = "cpu_memory_used_mb"  # the CPU memory footprint to predict


# 5. Set up preprocessing pipelines for categorical and numeric features
categorical_features = ["game_name"]
numeric_features = [
    "state_height",
    "state_width",
    "action_dim",
    "agent_hidden_model_size",
    "exploration_rate",
    "buffer_max_size",
]



def 


# Compute cross-validated scores
mae_scores = cross_val_score(model, X, y, cv=kf, scoring=neg_mae)
rmse_scores = cross_val_score(model, X, y, cv=kf, scoring=neg_rmse)
r2_scores = cross_val_score(model, X, y, cv=kf, scoring=r2_scorer)

print(f"MAE: Mean={-mae_scores.mean():.3f} MB, Std={mae_scores.std():.3f}")
print(f"RMSE: Mean={-rmse_scores.mean():.3f} MB, Std={rmse_scores.std():.3f}")
print(f"R²: Mean={r2_scores.mean():.6f}, Std={r2_scores.std():.6f}")

with open("estimation_model/model_results.txt", "w") as f:
    f.write(f"MAE: Mean={-mae_scores.mean():.3f} MB, Std={mae_scores.std():.3f}\n")
    f.write(f"RMSE: Mean={-rmse_scores.mean():.3f} MB, Std={rmse_scores.std():.3f}\n")
    f.write(f"R²: Mean={r2_scores.mean():.6f}, Std={r2_scores.std():.6f}\n")
