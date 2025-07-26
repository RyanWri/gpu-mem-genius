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
def parse_state_dim(dim_str):
    try:
        dims = dim_str.strip("()").split(",")
        return int(dims[0]), int(dims[1])
    except Exception:
        # In case parsing fails, return NaN
        return np.nan, np.nan


def split(df: pd.DataFrame, features: list[str], target: str):
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

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

def eval(y_true, y_pred):
    def mae_scorer(y_true, y_pred):
        return mean_absolute_error(y_true, y_pred)
    def rmse_scorer(y_true, y_pred):
        return root_mean_squared_error(y_true, y_pred)


    neg_mae = make_scorer(mae_scorer, greater_is_better=False)
    neg_rmse = make_scorer(rmse_scorer, greater_is_better=False)
    r2_scorer = make_scorer(r2_score)
    
