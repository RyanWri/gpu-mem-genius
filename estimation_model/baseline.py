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


# 1. Load your dataset (adjust the path as needed)
file_path = "/home/linuxu/datasets/atari-mem/all_data_combined.csv"
df = pd.read_csv(file_path)


# 2. Parse 'state_dim' feature into numeric 'state_height' and 'state_width'
def parse_state_dim(dim_str):
    try:
        dims = dim_str.strip("()").split(",")
        return int(dims[0]), int(dims[1])
    except Exception:
        # In case parsing fails, return NaN
        return np.nan, np.nan


df["state_height"], df["state_width"] = zip(*df["state_dim"].map(parse_state_dim))

# Drop rows where parsing failed
df_clean = df.dropna(subset=["state_height", "state_width"])

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
]
target = "cpu_memory_used_mb"  # the CPU memory footprint to predict

X = df_clean[features]
y = df_clean[target]

# 4. Split dataset into training and test sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

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

# OneHotEncoder for categorical, StandardScaler for numeric features
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", StandardScaler(), numeric_features),
    ]
)

# 6. Create the regression pipeline
#    Using RandomForestRegressor for robust regression
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(random_state=42, n_estimators=100)),
    ]
)

# 7. Train the model on training data
model.fit(X_train, y_train)

# 8. Predict on test data
y_pred = model.predict(X_test)


def mae_scorer(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)


def rmse_scorer(y_true, y_pred):
    return root_mean_squared_error(y_true, y_pred)


# Wrap scorers for cross_val_score (needs higher is better, so negate errors)
neg_mae = make_scorer(mae_scorer, greater_is_better=False)
neg_rmse = make_scorer(rmse_scorer, greater_is_better=False)
r2_scorer = make_scorer(r2_score)

# Prepare k-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

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
