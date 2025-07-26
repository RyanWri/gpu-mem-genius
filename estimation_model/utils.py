import numpy as np
from sklearn.model_selection import cross_val_score, KFold
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


def parse_state_dim(dim_str):
    try:
        dims = dim_str.strip("()").split(",")
        return int(dims[0]), int(dims[1])
    except Exception:
        # In case parsing fails, return NaN
        return np.nan, np.nan


def create_model(numeric_features, categorical_features):
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", StandardScaler(), numeric_features),
        ]
    )
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", RandomForestRegressor(random_state=42, n_estimators=100)),
        ]
    )
    return model


def scorers():
    neg_mae = make_scorer(mean_absolute_error, greater_is_better=False)
    neg_rmse = make_scorer(root_mean_squared_error, greater_is_better=False)
    r2_scorer = make_scorer(r2_score)
    return neg_mae, neg_rmse, r2_scorer


def cross_validation(model, X, y, neg_mae, neg_rmse, r2_scorer):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    mae_scores = cross_val_score(model, X, y, cv=kf, scoring=neg_mae)
    rmse_scores = cross_val_score(model, X, y, cv=kf, scoring=neg_rmse)
    r2_scores = cross_val_score(model, X, y, cv=kf, scoring=r2_scorer)

    print(f"MAE: Mean={-mae_scores.mean():.3f} MB, Std={mae_scores.std():.3f}")
    print(f"RMSE: Mean={-rmse_scores.mean():.3f} MB, Std={rmse_scores.std():.3f}")
    print(f"R²: Mean={r2_scores.mean():.6f}, Std={r2_scores.std():.6f}")
    return mae_scores, rmse_scores, r2_scores
