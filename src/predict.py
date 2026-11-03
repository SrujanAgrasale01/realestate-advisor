"""Inference helpers shared by the Streamlit app and the tests."""
from __future__ import annotations

import json

import joblib
import pandas as pd

from . import config as C
from .features import engineer_features

REQUIRED_FIELDS = [
    "State", "City", "Property_Type", "BHK", "Size_in_SqFt", "Price_in_Lakhs", "Year_Built",
    "Furnished_Status", "Floor_No", "Total_Floors", "Nearby_Schools", "Nearby_Hospitals",
    "Public_Transport_Accessibility", "Parking_Space", "Security", "Facing", "Owner_Type",
    "Availability_Status",
]


def format_lakhs(value: float) -> str:
    """Pretty-print an amount given in lakhs of INR."""
    if value >= 100:
        return f"₹ {value / 100:,.2f} Cr"
    return f"₹ {value:,.2f} Lakh"


class Advisor:
    """Loads the saved preprocessing artefacts and the two champion models."""

    def __init__(self) -> None:
        artifacts = json.loads(C.ARTIFACTS_JSON.read_text())
        self.ref: dict = artifacts["reference"]
        self.options: dict = artifacts["options"]
        try:
            self.clf = joblib.load(C.CLF_PATH)
            self.reg = joblib.load(C.REG_PATH)
        except Exception:  # models pickled with other library versions -> rebuild locally
            from .rebuild import rebuild_champions
            rebuild_champions()
            self.clf = joblib.load(C.CLF_PATH)
            self.reg = joblib.load(C.REG_PATH)

    # ------------------------------------------------------------------
    def build_features(self, inputs: dict) -> pd.DataFrame:
        missing = [f for f in REQUIRED_FIELDS if f not in inputs]
        if missing:
            raise ValueError(f"Missing fields: {', '.join(missing)}")
        if inputs["Price_in_Lakhs"] <= 0 or inputs["Size_in_SqFt"] <= 0 or inputs["BHK"] <= 0:
            raise ValueError("Price, size and BHK must be positive.")
        if inputs["Floor_No"] > inputs["Total_Floors"]:
            raise ValueError("Floor number cannot exceed the total number of floors.")

        amenities = inputs.get("Amenities", [])
        if isinstance(amenities, (list, tuple, set)):
            amenities = ", ".join(sorted(amenities))
        row = {f: inputs[f] for f in REQUIRED_FIELDS}
        row["Amenities"] = amenities
        df = pd.DataFrame([row])
        for col in C.CATEGORICAL_FEATURES:
            df[col] = df[col].astype(object)
        return engineer_features(df, self.ref)

    def predict_features(self, feats: pd.DataFrame) -> pd.DataFrame:
        """Run both models on an already engineered feature frame."""
        X = feats[C.MODEL_FEATURES]
        proba = self.clf.predict_proba(X)[:, 1]
        future = self.reg.predict(X)
        out = pd.DataFrame({"prob_good": proba, "future_price": future}, index=feats.index)
        out["is_good"] = out["prob_good"] >= 0.5
        return out

    def predict(self, inputs: dict) -> dict:
        feats = self.build_features(inputs)
        res = self.predict_features(feats).iloc[0]
        price = float(inputs["Price_in_Lakhs"])
        future = float(res["future_price"])
        prob = float(res["prob_good"])
        city_median = self.ref["city_median_ppsf"].get(inputs["City"], self.ref["global_median_ppsf"])
        return {
            "is_good": bool(res["is_good"]),
            "prob_good": prob,
            "confidence": max(prob, 1 - prob),
            "future_price": future,
            "appreciation_pct": (future / price - 1) * 100,
            "cagr_pct": ((future / price) ** (1 / C.HORIZON_YEARS) - 1) * 100,
            "price_per_sqft": float(feats["Price_per_SqFt_INR_Raw"].iloc[0]),
            "city_median_ppsf": float(city_median),
            "price_vs_city_median": float(feats["Price_vs_City_Median"].iloc[0]),
            "infrastructure_score": float(feats["Infrastructure_Score"].iloc[0]),
            "age": int(feats["Age_of_Property"].iloc[0]),
        }
