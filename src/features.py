"""Data cleaning, feature engineering and domain-rule targets.

The same functions are used by the offline pipeline (preprocess.py) and by the
Streamlit app (predict.py), so training and serving can never drift apart.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

STRING_COLUMNS = [
    "State", "City", "Locality", "Property_Type", "Furnished_Status",
    "Public_Transport_Accessibility", "Parking_Space", "Security", "Amenities",
    "Facing", "Owner_Type", "Availability_Status",
]
NUMERIC_COLUMNS = [
    "BHK", "Size_in_SqFt", "Price_in_Lakhs", "Year_Built", "Floor_No",
    "Total_Floors", "Nearby_Schools", "Nearby_Hospitals",
]


# ----------------------------------------------------------------------------
# 1. Cleaning
# ----------------------------------------------------------------------------
def _strip(value):
    return value.strip() if isinstance(value, str) else value


def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Remove duplicates, impute missing values, fix inconsistencies."""
    report: dict = {"rows_in": int(len(df))}
    df = df.copy()

    # tidy strings (object dtype keeps missing values as NaN)
    for col in STRING_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype(object).map(_strip)
            df[col] = df[col].replace("", np.nan)

    # duplicates
    if "ID" in df.columns:
        before = len(df)
        df = df.drop_duplicates(subset="ID", keep="first")
        report["duplicate_ids_removed"] = int(before - len(df))
    before = len(df)
    df = df.drop_duplicates(subset=[c for c in df.columns if c != "ID"], keep="first")
    report["duplicate_rows_removed"] = int(before - len(df))

    # missing values: median for numeric, mode for categorical
    missing = df.isna().sum()
    report["missing_before"] = {k: int(v) for k, v in missing.items() if v > 0}
    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    for col in STRING_COLUMNS:
        if col == "Amenities":
            df[col] = df[col].fillna("")
        elif df[col].isna().any():
            df[col] = df[col].fillna(df[col].mode().iloc[0])

    # invalid records
    before = len(df)
    valid = (df["Price_in_Lakhs"] > 0) & (df["Size_in_SqFt"] > 0) & (df["BHK"] > 0)
    df = df[valid]
    report["invalid_rows_removed"] = int(before - len(df))

    # Floor_No cannot exceed Total_Floors -> the two values were swapped
    bad_floor = df["Floor_No"] > df["Total_Floors"]
    report["floor_inconsistencies_fixed"] = int(bad_floor.sum())
    floor, total = df["Floor_No"].copy(), df["Total_Floors"].copy()
    df["Floor_No"] = np.where(bad_floor, total, floor)
    df["Total_Floors"] = np.where(bad_floor, floor, total)

    # integer-like columns
    for col in ["BHK", "Year_Built", "Floor_No", "Total_Floors",
                "Nearby_Schools", "Nearby_Hospitals"]:
        df[col] = df[col].round().astype(int)

    report["rows_out"] = int(len(df))
    return df.reset_index(drop=True), report


# ----------------------------------------------------------------------------
# 2. Reference statistics (fit once on the cleaned data)
# ----------------------------------------------------------------------------
def compute_reference(df: pd.DataFrame) -> dict:
    ppsf = df["Price_in_Lakhs"] * 1e5 / df["Size_in_SqFt"]
    q1, q3 = ppsf.quantile(0.25), ppsf.quantile(0.75)
    iqr = q3 - q1
    return {
        "city_median_ppsf": (
            ppsf.groupby(df["City"]).median().round(2).to_dict()
        ),
        "global_median_ppsf": float(round(ppsf.median(), 2)),
        "ppsf_lower": float(max(0.0, q1 - 1.5 * iqr)),
        "ppsf_upper": float(q3 + 1.5 * iqr),
    }


# ----------------------------------------------------------------------------
# 3. Feature engineering
# ----------------------------------------------------------------------------
def engineer_features(df: pd.DataFrame, ref: dict) -> pd.DataFrame:
    d = df.copy()

    raw_ppsf = d["Price_in_Lakhs"] * 1e5 / d["Size_in_SqFt"]
    d["Price_per_SqFt_INR_Raw"] = raw_ppsf
    d["PPSF_Outlier"] = ((raw_ppsf < ref["ppsf_lower"]) | (raw_ppsf > ref["ppsf_upper"])).astype(int)
    d["Price_per_SqFt_INR"] = raw_ppsf.clip(ref["ppsf_lower"], ref["ppsf_upper"])

    city_median = d["City"].map(ref["city_median_ppsf"]).fillna(ref["global_median_ppsf"])
    d["Price_vs_City_Median"] = d["Price_per_SqFt_INR"] / city_median
    d["Log_Price"] = np.log(d["Price_in_Lakhs"])
    d["Size_per_BHK"] = d["Size_in_SqFt"] / d["BHK"].clip(lower=1)
    d["Floor_Ratio"] = d["Floor_No"] / d["Total_Floors"].clip(lower=1)
    d["Age_of_Property"] = (C.CURRENT_YEAR - d["Year_Built"]).clip(lower=0)

    d["Transport_Score"] = d["Public_Transport_Accessibility"].map(C.TRANSPORT_SCORE).fillna(1)
    d["School_Density_Score"] = (d["Nearby_Schools"] / 10).clip(0, 1)
    hospitals = (d["Nearby_Hospitals"] / 10).clip(0, 1)
    d["Infrastructure_Score"] = 10 * (
        0.30 * d["School_Density_Score"]
        + 0.25 * hospitals
        + 0.25 * d["Transport_Score"] / 2
        + 0.10 * (d["Security"] == "Yes")
        + 0.10 * (d["Parking_Space"] == "Yes")
    )

    amen = d["Amenities"].fillna("").astype(str)
    for name in C.AMENITY_LIST:
        d[f"Has_{name}"] = amen.str.contains(name, case=False, regex=False).astype(int)
    d["Amenity_Count"] = d[[f"Has_{a}" for a in C.AMENITY_LIST]].sum(axis=1)
    return d


# ----------------------------------------------------------------------------
# 4. Domain-rule targets
# ----------------------------------------------------------------------------
def _city_base_growth(city: pd.Series) -> pd.Series:
    base = pd.Series(C.CITY_BASE_GROWTH["tier3"], index=city.index, dtype=float)
    base[city.isin(C.TIER2)] = C.CITY_BASE_GROWTH["tier2"]
    base[city.isin(C.TIER1)] = C.CITY_BASE_GROWTH["tier1"]
    return base


def annual_growth_rate(d: pd.DataFrame, rng: np.random.Generator | None = None) -> pd.Series:
    """Feature-based annual appreciation rate r (decimal)."""
    r = (
        _city_base_growth(d["City"])
        + d["Property_Type"].map(C.TYPE_ADJ).fillna(0.0)
        + d["Availability_Status"].map(C.AVAILABILITY_ADJ).fillna(0.0)
        + d["Public_Transport_Accessibility"].map(C.TRANSPORT_ADJ).fillna(0.0)
        + C.INFRA_ADJ_MAX * (d["Infrastructure_Score"] - 5) / 5
        - C.AGE_ADJ_MAX * (d["Age_of_Property"] / 35).clip(0, 1)
    )
    if rng is not None:
        r = r + rng.normal(0.0, C.GROWTH_NOISE_SD, size=len(d))
    return r.clip(*C.GROWTH_CLIP)


def investment_score(d: pd.DataFrame) -> pd.Series:
    """Multi-factor score in [0, 100] used to derive the Good_Investment label."""
    ratio = d["Price_vs_City_Median"]
    value = np.select([ratio <= 0.8, ratio <= 1.0, ratio <= 1.2], [30, 20, 8], default=0)
    g = d["Annual_Growth_Rate"]
    growth = np.select([g >= 0.09, g >= 0.08, g >= 0.07], [20, 14, 8], default=0)
    bhk = 10 * (d["BHK"] >= 3)
    ready = 10 * (d["Availability_Status"] == "Ready_to_Move")
    transport = d["Public_Transport_Accessibility"].map({"High": 10, "Medium": 5, "Low": 0}).fillna(5)
    infra = (d["Nearby_Schools"] + d["Nearby_Hospitals"]) / 20 * 10
    security = 3 * (d["Security"] == "Yes")
    parking = 2 * (d["Parking_Space"] == "Yes")
    amenities = d["Amenity_Count"]
    return pd.Series(
        value + growth + bhk + ready + transport + infra + security + parking + amenities,
        index=d.index, dtype=float,
    )


def add_targets(d: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    d = d.copy()
    d["Annual_Growth_Rate"] = annual_growth_rate(d, rng)
    d[C.REG_TARGET] = d["Price_in_Lakhs"] * (1 + d["Annual_Growth_Rate"]) ** C.HORIZON_YEARS
    d["Appreciation_Pct_5Y"] = (d[C.REG_TARGET] / d["Price_in_Lakhs"] - 1) * 100
    d["Investment_Score"] = investment_score(d)
    d[C.CLF_TARGET] = (d["Investment_Score"] >= C.GOOD_INVESTMENT_THRESHOLD).astype(int)
    return d
