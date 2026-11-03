"""Model/pipeline definitions shared by training and the on-the-fly rebuild."""
import numpy as np
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier, XGBRegressor

from . import config as C


def make_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)),
    ])
    return ColumnTransformer([
        ("num", numeric, C.NUMERIC_FEATURES),
        ("cat", categorical, C.CATEGORICAL_FEATURES),
    ])


def classifiers() -> dict:
    return {
        "LogisticRegression": LogisticRegression(max_iter=1000, C=1.0, random_state=C.RANDOM_STATE),
        "RandomForest": RandomForestClassifier(
            n_estimators=60, max_depth=14, min_samples_leaf=10, max_features=0.3, max_samples=0.3,
            n_jobs=-1, random_state=C.RANDOM_STATE),
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
            tree_method="hist", eval_metric="logloss", random_state=C.RANDOM_STATE),
    }


def regressors() -> dict:
    # Future price is multiplicative (Price * (1+r)^5) -> models are fitted on log(price)
    def wrap(model):
        return TransformedTargetRegressor(regressor=model, func=np.log, inverse_func=np.exp)

    return {
        "LinearRegression": wrap(LinearRegression()),
        "RandomForest": wrap(RandomForestRegressor(
            n_estimators=60, max_depth=14, min_samples_leaf=10, max_features=0.3, max_samples=0.3,
            n_jobs=-1, random_state=C.RANDOM_STATE)),
        "XGBoost": wrap(XGBRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
            tree_method="hist", random_state=C.RANDOM_STATE)),
    }
