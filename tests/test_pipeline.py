"""Automated checks: data, features, models, serving and the Streamlit app."""
import json

import numpy as np
import pandas as pd
import pytest

from src import config as C
from src.features import clean_data, compute_reference, engineer_features
from src.predict import Advisor


@pytest.fixture(scope="module")
def processed():
    return pd.read_csv(C.DATA_PROCESSED)


@pytest.fixture(scope="module")
def advisor():
    return Advisor()


def test_clean_data_fixes_floors_and_duplicates():
    raw = pd.read_csv(C.DATA_RAW, nrows=500)
    raw = pd.concat([raw, raw.iloc[:5]], ignore_index=True)           # inject duplicates
    raw.loc[3, "Price_in_Lakhs"] = np.nan                              # inject a missing value
    clean, report = clean_data(raw)
    assert report["duplicate_ids_removed"] == 5
    assert clean["Price_in_Lakhs"].notna().all()
    assert (clean["Floor_No"] <= clean["Total_Floors"]).all()


def test_processed_dataset_is_model_ready(processed):
    assert len(processed) == 250_000
    assert processed[C.MODEL_FEATURES].notna().all().all()
    assert set(processed[C.CLF_TARGET].unique()) == {0, 1}
    assert 0.2 < processed[C.CLF_TARGET].mean() < 0.6
    assert (processed[C.REG_TARGET] > processed["Price_in_Lakhs"]).mean() > 0.99
    assert (processed["Floor_No"] <= processed["Total_Floors"]).all()


def test_feature_engineering_is_reproducible(processed):
    raw = pd.read_csv(C.DATA_RAW, nrows=2000)
    clean, _ = clean_data(raw)
    ref = json.loads(C.ARTIFACTS_JSON.read_text())["reference"]
    feats = engineer_features(clean, ref)
    for col in C.MODEL_FEATURES:
        assert col in feats.columns
    merged = processed.set_index("ID").loc[clean["ID"]]
    for col in ["Price_per_SqFt_INR", "Price_vs_City_Median", "Infrastructure_Score", "Amenity_Count"]:
        np.testing.assert_allclose(feats[col].to_numpy(), merged[col].to_numpy(), rtol=1e-3, atol=1e-3)


def _inputs_from_row(row):
    return {
        "State": row.State, "City": row.City, "Property_Type": row.Property_Type, "BHK": int(row.BHK),
        "Size_in_SqFt": float(row.Size_in_SqFt), "Price_in_Lakhs": float(row.Price_in_Lakhs),
        "Year_Built": int(row.Year_Built), "Furnished_Status": row.Furnished_Status,
        "Floor_No": int(row.Floor_No), "Total_Floors": int(row.Total_Floors),
        "Nearby_Schools": int(row.Nearby_Schools), "Nearby_Hospitals": int(row.Nearby_Hospitals),
        "Public_Transport_Accessibility": row.Public_Transport_Accessibility,
        "Parking_Space": row.Parking_Space, "Security": row.Security, "Facing": row.Facing,
        "Owner_Type": row.Owner_Type, "Availability_Status": row.Availability_Status,
        "Amenities": row.Amenities if isinstance(row.Amenities, str) else "",
    }


def test_single_prediction_matches_batch_prediction(processed, advisor):
    sample = processed.sample(40, random_state=1)
    X = sample[C.MODEL_FEATURES].copy()
    for col in C.CATEGORICAL_FEATURES:
        X[col] = X[col].astype(object)
    batch = advisor.predict_features(X.assign())
    for (idx, row), p, f in zip(sample.iterrows(), batch["prob_good"], batch["future_price"]):
        res = advisor.predict(_inputs_from_row(row))
        assert res["prob_good"] == pytest.approx(p, abs=1e-3)
        assert res["future_price"] == pytest.approx(f, rel=1e-3)


def test_prediction_output_is_sensible(advisor):
    res = advisor.predict({
        "State": "Maharashtra", "City": "Mumbai", "Property_Type": "Apartment", "BHK": 3,
        "Size_in_SqFt": 2500, "Price_in_Lakhs": 150.0, "Year_Built": 2015, "Furnished_Status": "Semi-furnished",
        "Floor_No": 4, "Total_Floors": 12, "Nearby_Schools": 8, "Nearby_Hospitals": 7,
        "Public_Transport_Accessibility": "High", "Parking_Space": "Yes", "Security": "Yes", "Facing": "East",
        "Owner_Type": "Builder", "Availability_Status": "Ready_to_Move", "Amenities": ["Gym", "Pool", "Garden"]})
    assert 0 <= res["prob_good"] <= 1
    assert 150 < res["future_price"] < 150 * 2.0
    assert 3 < res["cagr_pct"] < 15
    assert res["is_good"] is True


def test_invalid_inputs_are_rejected(advisor):
    base = {
        "State": "Maharashtra", "City": "Pune", "Property_Type": "Villa", "BHK": 2, "Size_in_SqFt": 1200,
        "Price_in_Lakhs": 80.0, "Year_Built": 2000, "Furnished_Status": "Unfurnished", "Floor_No": 2,
        "Total_Floors": 5, "Nearby_Schools": 3, "Nearby_Hospitals": 3,
        "Public_Transport_Accessibility": "Low", "Parking_Space": "No", "Security": "No", "Facing": "North",
        "Owner_Type": "Owner", "Availability_Status": "Under_Construction", "Amenities": []}
    with pytest.raises(ValueError):
        advisor.predict({**base, "Floor_No": 9})
    with pytest.raises(ValueError):
        advisor.predict({**base, "Price_in_Lakhs": 0})
    broken = dict(base)
    broken.pop("City")
    with pytest.raises(ValueError):
        advisor.predict(broken)


def test_model_quality_thresholds():
    m = json.loads(C.METRICS_JSON.read_text())
    clf = m["classification"][m["best"]["classifier"]]
    reg = m["regression"][m["best"]["regressor"]]
    assert clf["accuracy"] > 0.90 and clf["f1"] > 0.88 and clf["roc_auc"] > 0.95
    assert reg["r2"] > 0.99 and reg["rmse"] < 25


def test_mlflow_registry_has_champions():
    import mlflow
    from mlflow.tracking import MlflowClient
    mlflow.set_tracking_uri(f"sqlite:///{C.MLFLOW_DB.as_posix()}")
    client = MlflowClient()
    names = {m.name for m in client.search_registered_models()}
    assert {"RealEstate_GoodInvestment_Classifier", "RealEstate_FuturePrice_Regressor"} <= names
    runs = client.search_runs([e.experiment_id for e in client.search_experiments()])
    assert len(runs) >= 6


def test_registered_champion_models_load_and_predict(processed, advisor):
    import mlflow
    mlflow.set_tracking_uri(f"sqlite:///{C.MLFLOW_DB.as_posix()}")
    sample = processed.sample(200, random_state=3)
    X = sample[C.MODEL_FEATURES].copy()
    for col in C.CATEGORICAL_FEATURES:
        X[col] = X[col].astype(object)
    clf = mlflow.sklearn.load_model("models:/RealEstate_GoodInvestment_Classifier@champion")
    reg = mlflow.sklearn.load_model("models:/RealEstate_FuturePrice_Regressor@champion")
    np.testing.assert_allclose(clf.predict_proba(X)[:, 1], advisor.clf.predict_proba(X)[:, 1], atol=1e-6)
    np.testing.assert_allclose(reg.predict(X), advisor.reg.predict(X), rtol=1e-6)


@pytest.mark.parametrize("page", ["🔮 Investment Advisor", "🔎 Property Explorer", "📊 Market Insights", "🧪 Model Performance"])
def test_streamlit_pages_render_without_errors(page):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(C.ROOT / "app" / "streamlit_app.py"), default_timeout=180).run()
    assert not at.exception, at.exception
    at.sidebar.radio[0].set_value(page).run()
    assert not at.exception, at.exception


def test_streamlit_prediction_flow():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(C.ROOT / "app" / "streamlit_app.py"), default_timeout=180).run()
    at.button[0].click().run()
    assert not at.exception, at.exception
    assert len(at.metric) >= 4
    assert any("Investment" in s.value for s in list(at.success) + list(at.warning))


def test_streamlit_rejects_bad_floor():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(C.ROOT / "app" / "streamlit_app.py"), default_timeout=180).run()
    floor_inputs = [n for n in at.number_input if n.label == "Floor number"]
    floor_inputs[0].set_value(60)
    at.button[0].click().run()
    assert not at.exception
    assert len(at.error) == 1
