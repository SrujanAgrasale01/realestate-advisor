"""Step 1 - Data preprocessing.

Reads the raw CSV, cleans it, engineers features, derives both targets and
writes:
  * data/processed/housing_processed.csv
  * models/preprocessing_artifacts.json  (reference statistics + app options)
  * reports/data_quality_report.json
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import config as C
from .features import add_targets, clean_data, compute_reference, engineer_features


def build_app_options(df: pd.DataFrame) -> dict:
    """Values the Streamlit form needs (dropdown choices and slider ranges)."""
    states = {
        state: sorted(group["City"].unique().tolist())
        for state, group in df.groupby("State")
    }
    return {
        "states": states,
        "property_types": sorted(df["Property_Type"].unique().tolist()),
        "furnished_status": sorted(df["Furnished_Status"].unique().tolist()),
        "facing": sorted(df["Facing"].unique().tolist()),
        "owner_types": sorted(df["Owner_Type"].unique().tolist()),
        "availability": sorted(df["Availability_Status"].unique().tolist()),
        "transport": ["Low", "Medium", "High"],
        "amenities": C.AMENITY_LIST,
        "ranges": {
            "bhk": [int(df["BHK"].min()), int(df["BHK"].max())],
            "size": [int(df["Size_in_SqFt"].min()), int(df["Size_in_SqFt"].max())],
            "price": [float(df["Price_in_Lakhs"].min()), float(df["Price_in_Lakhs"].max())],
            "year_built": [int(df["Year_Built"].min()), int(C.CURRENT_YEAR)],
            "floors": [int(df["Total_Floors"].min()), int(df["Total_Floors"].max())],
            "schools": [int(df["Nearby_Schools"].min()), int(df["Nearby_Schools"].max())],
            "hospitals": [int(df["Nearby_Hospitals"].min()), int(df["Nearby_Hospitals"].max())],
        },
        "defaults": {
            "size": int(df["Size_in_SqFt"].median()),
            "price": float(round(df["Price_in_Lakhs"].median(), 1)),
            "year_built": int(df["Year_Built"].median()),
        },
    }


def run() -> pd.DataFrame:
    C.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    C.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    C.DATA_PROCESSED.parent.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(C.DATA_RAW)
    print(f"Raw data: {raw.shape}")

    clean, report = clean_data(raw)
    ref = compute_reference(clean)
    feats = engineer_features(clean, ref)
    rng = np.random.default_rng(C.RANDOM_STATE)
    processed = add_targets(feats, rng)

    report["outliers_flagged_price_per_sqft"] = int(processed["PPSF_Outlier"].sum())
    report["good_investment_rate"] = float(processed[C.CLF_TARGET].mean())
    report["good_investment_threshold"] = C.GOOD_INVESTMENT_THRESHOLD
    report["avg_annual_growth_rate"] = float(processed["Annual_Growth_Rate"].mean())
    report["columns_out"] = int(processed.shape[1])

    float_cols = processed.select_dtypes("float").columns
    processed[float_cols] = processed[float_cols].round(4)
    processed.to_csv(C.DATA_PROCESSED, index=False)

    artifacts = {"reference": ref, "options": build_app_options(processed)}
    C.ARTIFACTS_JSON.write_text(json.dumps(artifacts, indent=2))
    (C.REPORTS_DIR / "data_quality_report.json").write_text(json.dumps(report, indent=2))

    print(json.dumps(report, indent=2))
    print(f"Saved processed data -> {C.DATA_PROCESSED}")
    return processed


if __name__ == "__main__":
    run()
