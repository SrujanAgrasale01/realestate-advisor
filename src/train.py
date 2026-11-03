"""Steps 3 & 4 - Model development and MLflow experiment tracking.

Trains three classifiers (Good_Investment) and three regressors (Future_Price_5Y),
logs every run (parameters, metrics, plots, model artefact) to MLflow, registers
the best model of each task in the MLflow Model Registry and saves the winners to
models/*.joblib for the Streamlit app.

Usage:  python -m src.train            (rebuilds the MLflow store from scratch)
        python -m src.train --keep-mlflow   (append to an existing MLflow store)
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
import warnings

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import mlflow.sklearn  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
import xgboost  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor  # noqa: E402
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.linear_model import LinearRegression, LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score, confusion_matrix, f1_score, mean_absolute_error, mean_squared_error,
    precision_score, r2_score, recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # noqa: E402
from xgboost import XGBClassifier, XGBRegressor  # noqa: E402

from . import config as C  # noqa: E402
from .modeling import classifiers, make_preprocessor, regressors  # noqa: E402

warnings.filterwarnings("ignore")

CLF_EXPERIMENT = "RealEstate_Good_Investment_Classification"
REG_EXPERIMENT = "RealEstate_Future_Price_Regression"
CLF_REGISTERED_NAME = "RealEstate_GoodInvestment_Classifier"
REG_REGISTERED_NAME = "RealEstate_FuturePrice_Regressor"


# ----------------------------------------------------------------------------
# data / pipelines
# ----------------------------------------------------------------------------
def load_data() -> pd.DataFrame:
    d = pd.read_csv(C.DATA_PROCESSED)
    if os.environ.get("RE_QUICK"):  # smoke-test mode: tiny sample, same code path
        d = d.sample(int(os.environ["RE_QUICK"]), random_state=C.RANDOM_STATE).reset_index(drop=True)
    for col in C.CATEGORICAL_FEATURES:
        d[col] = d[col].astype(object)
    return d


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _simple_params(estimator) -> dict:
    out = {}
    for k, v in estimator.get_params().items():
        if isinstance(v, (int, float, str, bool)) and v is not None:
            out[k] = v
    return out


def feature_importance(pipe: Pipeline, X: pd.DataFrame, y, scoring: str, n_rows: int = 8000) -> pd.DataFrame:
    """Permutation importance on the original (pre one-hot) features.

    Importance = drop in the test score when a feature's values are shuffled,
    normalised to sum to 1.  Model-agnostic, so it is comparable across all
    models and is not biased towards high-cardinality one-hot features.
    """
    idx = np.random.default_rng(C.RANDOM_STATE).choice(len(X), size=min(n_rows, len(X)), replace=False)
    res = permutation_importance(pipe, X.iloc[idx], np.asarray(y)[idx], scoring=scoring,
                                 n_repeats=3, random_state=C.RANDOM_STATE, n_jobs=1)
    df = pd.DataFrame({"feature": list(X.columns), "importance": np.clip(res.importances_mean, 0, None)})
    total = df["importance"].sum()
    df["importance"] = df["importance"] / total if total > 0 else 0.0
    return df.sort_values("importance", ascending=False).reset_index(drop=True)


def _save_fig(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def plot_confusion(cm: np.ndarray, title: str, path) -> None:
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Not good", "Good"],
           yticklabels=["Not good", "Good"], xlabel="Predicted", ylabel="Actual", title=title)
    _save_fig(fig, path)


def plot_importance(imp: pd.DataFrame, title: str, path, top: int = 15) -> None:
    t = imp.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(t["feature"], t["importance"] * 100, color="#3b6ea5")
    ax.set(title=title, xlabel="Permutation importance (share of total, %)")
    _save_fig(fig, path)


# ----------------------------------------------------------------------------
# training loops
# ----------------------------------------------------------------------------
def set_experiment(name: str) -> None:
    if mlflow.get_experiment_by_name(name) is None:
        mlflow.create_experiment(name, artifact_location=(C.MLFLOW_ARTIFACTS / name).as_uri())
    mlflow.set_experiment(name)


def run_classification(X_tr, X_te, y_tr, y_te) -> dict:
    set_experiment(CLF_EXPERIMENT)
    results, fitted = {}, {}
    for name, model in classifiers().items():
        pipe = Pipeline([("pre", make_preprocessor()), ("model", model)])
        with mlflow.start_run(run_name=name) as run:
            t0 = time.time()
            pipe.fit(X_tr, y_tr)
            fit_s = time.time() - t0
            proba = pipe.predict_proba(X_te)[:, 1]
            pred = (proba >= 0.5).astype(int)
            cm = confusion_matrix(y_te, pred)
            m = {
                "accuracy": accuracy_score(y_te, pred),
                "precision": precision_score(y_te, pred),
                "recall": recall_score(y_te, pred),
                "f1": f1_score(y_te, pred),
                "roc_auc": roc_auc_score(y_te, proba),
            }
            imp = feature_importance(pipe, X_te, y_te, "roc_auc")
            cm_path = C.REPORTS_DIR / f"confusion_matrix_{name}.png"
            plot_confusion(cm, f"{name} - confusion matrix", cm_path)
            imp_csv = C.REPORTS_DIR / f"_tmp_importance_clf_{name}.csv"
            imp.to_csv(imp_csv, index=False)

            mlflow.set_tags({"task": "classification", "target": C.CLF_TARGET, "library": type(model).__module__.split(".")[0]})
            mlflow.log_params({"model": name, **{f"p_{k}": v for k, v in _simple_params(model).items()},
                               "n_train": len(X_tr), "n_test": len(X_te), "n_features": len(C.MODEL_FEATURES),
                               "label_threshold": C.GOOD_INVESTMENT_THRESHOLD})
            mlflow.log_metrics({**m, "train_seconds": fit_s})
            mlflow.log_artifact(str(cm_path), "plots")
            mlflow.log_artifact(str(imp_csv), "feature_importance")
            info = mlflow.sklearn.log_model(pipe, name="model", input_example=X_te.head(3), serialization_format="cloudpickle")
            imp_csv.unlink()

            results[name] = {**{k: float(v) for k, v in m.items()}, "train_seconds": float(fit_s),
                             "confusion_matrix": cm.tolist(), "run_id": run.info.run_id,
                             "model_uri": info.model_uri}
            fitted[name] = (pipe, proba, imp)
            print(f"[clf] {name:<20} acc={m['accuracy']:.4f} f1={m['f1']:.4f} auc={m['roc_auc']:.4f} ({fit_s:.0f}s)")
    return {"results": results, "fitted": fitted}


def run_regression(X_tr, X_te, y_tr, y_te) -> dict:
    set_experiment(REG_EXPERIMENT)
    results, fitted = {}, {}
    for name, model in regressors().items():
        pipe = Pipeline([("pre", make_preprocessor()), ("model", model)])
        with mlflow.start_run(run_name=name) as run:
            t0 = time.time()
            pipe.fit(X_tr, y_tr)
            fit_s = time.time() - t0
            pred = pipe.predict(X_te)
            m = {
                "rmse": float(np.sqrt(mean_squared_error(y_te, pred))),
                "mae": float(mean_absolute_error(y_te, pred)),
                "r2": float(r2_score(y_te, pred)),
                "mape_pct": float(np.mean(np.abs((y_te - pred) / y_te)) * 100),
            }
            imp = feature_importance(pipe, X_te, y_te, "r2")
            imp_csv = C.REPORTS_DIR / f"_tmp_importance_reg_{name}.csv"
            imp.to_csv(imp_csv, index=False)
            fig_path = C.REPORTS_DIR / f"pred_vs_actual_{name}.png"
            idx = np.random.default_rng(C.RANDOM_STATE).choice(len(y_te), size=min(5000, len(y_te)), replace=False)
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.scatter(np.asarray(y_te)[idx], pred[idx], s=3, alpha=0.3, color="#e76f51")
            lim = [min(y_te.min(), pred.min()), max(y_te.max(), pred.max())]
            ax.plot(lim, lim, "k--", lw=1)
            ax.set(title=f"{name} - predicted vs actual", xlabel="Actual (lakhs)", ylabel="Predicted (lakhs)")
            _save_fig(fig, fig_path)

            mlflow.set_tags({"task": "regression", "target": C.REG_TARGET, "log_target": "true",
                             "library": type(model.regressor).__module__.split(".")[0]})
            mlflow.log_params({"model": name, **{f"p_{k}": v for k, v in _simple_params(model.regressor).items()},
                               "n_train": len(X_tr), "n_test": len(X_te), "n_features": len(C.MODEL_FEATURES),
                               "horizon_years": C.HORIZON_YEARS})
            mlflow.log_metrics({**m, "train_seconds": fit_s})
            mlflow.log_artifact(str(fig_path), "plots")
            mlflow.log_artifact(str(imp_csv), "feature_importance")
            info = mlflow.sklearn.log_model(pipe, name="model", input_example=X_te.head(3), serialization_format="cloudpickle")
            imp_csv.unlink()

            results[name] = {**m, "train_seconds": float(fit_s), "run_id": run.info.run_id,
                             "model_uri": info.model_uri}
            fitted[name] = (pipe, pred, imp)
            print(f"[reg] {name:<20} rmse={m['rmse']:.3f} mae={m['mae']:.3f} r2={m['r2']:.5f} ({fit_s:.0f}s)")
    return {"results": results, "fitted": fitted}


def register_best(name: str, model_uri: str, registered_name: str, metrics: dict) -> dict:
    client = MlflowClient()
    mv = mlflow.register_model(model_uri, registered_name)
    client.set_registered_model_alias(registered_name, "champion", mv.version)
    client.set_model_version_tag(registered_name, mv.version, "selected_model", name)
    for k, v in metrics.items():
        if isinstance(v, float):
            client.set_model_version_tag(registered_name, mv.version, f"test_{k}", f"{v:.6f}")
    return {"registered_name": registered_name, "version": str(mv.version), "alias": "champion"}


# ----------------------------------------------------------------------------
def main(keep_mlflow: bool = False) -> None:
    C.MODELS_DIR.mkdir(exist_ok=True)
    C.REPORTS_DIR.mkdir(exist_ok=True)
    if not keep_mlflow:
        if C.MLFLOW_DB.exists():
            C.MLFLOW_DB.unlink()
        shutil.rmtree(C.MLFLOW_ARTIFACTS, ignore_errors=True)
    mlflow.set_tracking_uri(f"sqlite:///{C.MLFLOW_DB.as_posix()}")

    d = load_data()
    X = d[C.MODEL_FEATURES]
    y_clf = d[C.CLF_TARGET].astype(int)
    y_reg = d[C.REG_TARGET].astype(float)
    idx_tr, idx_te = train_test_split(
        np.arange(len(d)), test_size=C.TEST_SIZE, stratify=y_clf, random_state=C.RANDOM_STATE)
    X_tr, X_te = X.iloc[idx_tr], X.iloc[idx_te]
    print(f"Train {len(X_tr):,} / Test {len(X_te):,} | positive rate {y_clf.mean():.3f}")

    clf = run_classification(X_tr, X_te, y_clf.iloc[idx_tr], y_clf.iloc[idx_te])
    reg = run_regression(X_tr, X_te, y_reg.iloc[idx_tr], y_reg.iloc[idx_te])

    best_clf = max(clf["results"], key=lambda k: (clf["results"][k]["f1"], clf["results"][k]["roc_auc"]))
    best_reg = min(reg["results"], key=lambda k: reg["results"][k]["rmse"])
    print(f"Best classifier: {best_clf} | Best regressor: {best_reg}")

    set_experiment(CLF_EXPERIMENT)
    reg_clf = register_best(best_clf, clf["results"][best_clf]["model_uri"], CLF_REGISTERED_NAME,
                            clf["results"][best_clf])
    set_experiment(REG_EXPERIMENT)
    reg_reg = register_best(best_reg, reg["results"][best_reg]["model_uri"], REG_REGISTERED_NAME,
                            reg["results"][best_reg])

    # --- persist winners for the app ---------------------------------------
    clf_pipe, clf_proba, clf_imp = clf["fitted"][best_clf]
    reg_pipe, reg_pred, reg_imp = reg["fitted"][best_reg]
    joblib.dump(clf_pipe, C.CLF_PATH, compress=3)
    joblib.dump(reg_pipe, C.REG_PATH, compress=3)
    clf_imp.to_csv(C.CLF_IMPORTANCE_CSV, index=False)
    reg_imp.to_csv(C.REG_IMPORTANCE_CSV, index=False)
    plot_importance(clf_imp, f"Classifier feature importance ({best_clf})", C.REPORTS_DIR / "feature_importance_classifier.png")
    plot_importance(reg_imp, f"Regressor feature importance ({best_reg})", C.REPORTS_DIR / "feature_importance_regressor.png")

    y_te_c = y_clf.iloc[idx_te].to_numpy()
    fpr, tpr, _ = roc_curve(y_te_c, clf_proba)
    keep = np.linspace(0, len(fpr) - 1, min(300, len(fpr))).astype(int)
    y_te_r = y_reg.iloc[idx_te].to_numpy()
    samp = np.random.default_rng(C.RANDOM_STATE).choice(len(y_te_r), size=min(2000, len(y_te_r)), replace=False)

    def strip(res):
        return {k: {kk: vv for kk, vv in v.items() if kk != "model_uri"} for k, v in res.items()}

    summary = {
        "dataset": {"rows": int(len(d)), "train_rows": int(len(X_tr)), "test_rows": int(len(X_te)),
                    "good_investment_rate": float(y_clf.mean()), "n_features": len(C.MODEL_FEATURES)},
        "classification": strip(clf["results"]),
        "regression": strip(reg["results"]),
        "best": {"classifier": best_clf, "regressor": best_reg},
        "registry": {"classifier": reg_clf, "regressor": reg_reg},
        "roc_curve": {"fpr": fpr[keep].round(5).tolist(), "tpr": tpr[keep].round(5).tolist()},
        "pred_vs_actual": {"actual": y_te_r[samp].round(2).tolist(), "predicted": reg_pred[samp].round(2).tolist()},
        "versions": {"sklearn": sklearn.__version__, "xgboost": xgboost.__version__,
                     "mlflow": mlflow.__version__, "pandas": pd.__version__, "numpy": np.__version__},
    }
    C.METRICS_JSON.write_text(json.dumps(summary, indent=2))
    pd.concat([
        pd.DataFrame(summary["classification"]).T.assign(task="classification"),
        pd.DataFrame(summary["regression"]).T.assign(task="regression"),
    ]).drop(columns=["confusion_matrix", "run_id"]).to_csv(C.REPORTS_DIR / "model_comparison.csv")
    print("Saved models, metrics and MLflow registry entries.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep-mlflow", action="store_true", help="append to existing MLflow store")
    main(keep_mlflow=ap.parse_args().keep_mlflow)
