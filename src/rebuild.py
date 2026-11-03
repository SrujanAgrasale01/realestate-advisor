"""Rebuild the two champion models with the libraries installed on THIS machine.

Pickled sklearn / XGBoost models are tied to the library versions they were
trained with.  If loading models/*.joblib fails (e.g. different Python or
xgboost version) the app calls rebuild_champions() automatically (~30 s).
Manual run:  python -m src.rebuild
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from . import config as C
from .modeling import classifiers, make_preprocessor, regressors


def rebuild_champions() -> None:
    d = pd.read_csv(C.DATA_PROCESSED)
    for col in C.CATEGORICAL_FEATURES:
        d[col] = d[col].astype(object)
    X = d[C.MODEL_FEATURES]
    y_clf = d[C.CLF_TARGET].astype(int)
    y_reg = d[C.REG_TARGET].astype(float)
    idx_tr, _ = train_test_split(np.arange(len(d)), test_size=C.TEST_SIZE, stratify=y_clf,
                                 random_state=C.RANDOM_STATE)
    clf = Pipeline([("pre", make_preprocessor()), ("model", classifiers()["XGBoost"])])
    reg = Pipeline([("pre", make_preprocessor()), ("model", regressors()["LinearRegression"])])
    clf.fit(X.iloc[idx_tr], y_clf.iloc[idx_tr])
    reg.fit(X.iloc[idx_tr], y_reg.iloc[idx_tr])
    C.MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(clf, C.CLF_PATH, compress=3)
    joblib.dump(reg, C.REG_PATH, compress=3)


if __name__ == "__main__":
    rebuild_champions()
    print("Models rebuilt for the installed library versions.")
