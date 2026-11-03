# Project Documentation — Real Estate Investment Advisor

## 1. Problem statement
Help investors decide (a) whether a property is a **Good Investment** and (b) what its **price will be after 5 years**,
using the supplied `india_housing_prices.csv` (250,000 listings, 42 cities, 20 states, 23 columns), and deliver the
answer through a Streamlit app with MLflow experiment tracking.

## 2. Methodology

### 2.1 Data preprocessing (`src/features.py`, `src/preprocess.py`)
| Step | What was done | Result on this data |
|---|---|---|
| Duplicates | removed duplicate IDs and duplicate records | 0 found |
| Missing values | median (numeric) / mode (categorical) imputation; imputers are also part of the model pipeline so the app can handle gaps | 0 found |
| Invalid records | price, size, BHK must be positive | 0 removed |
| Floor consistency | `Floor_No > Total_Floors` means the two values were swapped, so they are swapped back | **116,304 rows (46.5 %) fixed** |
| Age | recomputed as `2025 − Year_Built` (matches the source column) | consistent |
| Outliers | IQR rule on price per sq ft; flag `PPSF_Outlier`; winsorise at the fences for modelling | 19,723 flagged (7.9 %); no size/price outliers |
| Scaling / encoding | `StandardScaler` for numerics, `OneHotEncoder(handle_unknown="ignore")` for State, City, Property_Type, Furnished_Status, Transport, Parking, Security, Facing, Owner_Type, Availability | inside the sklearn `Pipeline` (no leakage from the test set) |

`Locality` (500 anonymous labels) is used in EDA and filters but not in the models (high cardinality, no signal).

### 2.2 Feature engineering
* `Price_per_SqFt_INR` — recomputed in rupees (`price × 1e5 / size`); the source column is in lakhs per sq ft (≈0.13). Winsorised.
* `Price_vs_City_Median` — price per sq ft ÷ city median (value indicator).
* `School_Density_Score` (schools ÷ 10), `Infrastructure_Score` (0–10: schools 30 %, hospitals 25 %, transport 25 %, security 10 %, parking 10 %).
* `Transport_Score`, amenity flags (`Has_Gym`, …) and `Amenity_Count` (parsed from the 325 free-text combinations).
* `Floor_Ratio`, `Size_per_BHK`, `Log_Price`, `Age_of_Property`.

### 2.3 Target creation (domain rules)
The dataset contains neither a future price nor an investment label, so both are derived (seeded, reproducible):

**Future price (regression)** — *feature-based* option of the brief:
`Future_Price_5Y = Price × (1 + r)^5`, where
`r = city-tier base (8.5 % / 7.5 % / 6.5 %) + property-type adj (0 / +0.4 / +0.8 pp) + availability adj (+1 pp if under construction) + transport adj (0–0.5 pp) + infrastructure adj (±0.5 pp) − age penalty (≤ 0.3 pp) + N(0, 0.5 pp)`, clipped to 3–15 %. Mean r = 8.4 %.

**Good Investment (classification)** — *multi-factor score* option of the brief (0–100, label = 1 if score ≥ 60; 40.1 % positives):

| Component | Points |
|---|---|
| Price per sq ft vs city median: ≤0.8× / ≤1.0× / ≤1.2× | 30 / 20 / 8 |
| Modelled annual growth: ≥9 % / ≥8 % / ≥7 % | 20 / 14 / 8 |
| BHK ≥ 3 | 10 |
| Ready to move | 10 |
| Public transport High / Medium | 10 / 5 |
| Schools + hospitals nearby | 0–10 |
| Security / parking | 3 / 2 |
| Amenities (count) | 0–5 |

*RERA status* from the brief's example is not in the data and was not used. The label uses the noisy growth rate, while
`Future_Price_5Y`, the growth rate and the score itself are **excluded** from the model inputs, so there is no direct leakage.

### 2.4 Exploratory data analysis (`reports/eda_findings.md`, `reports/eda/`)
All 20 questions of the brief plus: price trends by city, area vs investment return, a safety proxy (**no crime-rate column exists**;
security + infrastructure used instead) and infrastructure vs resale value.

### 2.5 Modelling & tracking (`src/train.py`)
* 80/20 split (200,000 / 50,000), stratified on the investment label.
* 3 classifiers and 3 regressors, each a single `Pipeline(preprocessor → model)`; regressors use `TransformedTargetRegressor` (log target) because future price is multiplicative.
* **MLflow** (SQLite backend, so the Model Registry works): experiments `RealEstate_Good_Investment_Classification` and
  `RealEstate_Future_Price_Regression`; each run logs parameters, metrics, confusion-matrix / predicted-vs-actual plots,
  permutation-importance CSV and the full model with input example/signature. The best model per task (F1, RMSE) is registered as
  `RealEstate_GoodInvestment_Classifier` and `RealEstate_FuturePrice_Regressor` with alias `champion`.
* Feature importance = permutation importance on the test set (aggregates one-hot columns back to the original feature).

## 3. Findings

### 3.1 Data
* The raw data is clean (no nulls/duplicates) but its **raw features are essentially independent of price**: price vs size r = −0.003; city
  average prices only span 251–258 lakhs; furnishing, parking, facing, schools, hospitals, amenities and transport all move price per sq ft by < 1–2 %.
  Price and size are both ~uniform. This is typical of synthetic data; no real-market conclusions should be drawn from these relationships.
* Because price per sq ft = price ÷ size, large properties automatically look "cheap per sq ft" — which is why size correlates with the
  Good-Investment label (r = +0.38) but not with appreciation (r = +0.001).
* 46.5 % of rows had floor and total floors reversed.

### 3.2 Models (test set, 50,000 rows)
| Model | Metrics |
|---|---|
| Logistic Regression | Acc 0.905 · Prec 0.879 · F1 0.882 · AUC 0.967 |
| Random Forest | Acc 0.926 · F1 0.909 · AUC 0.983 |
| **XGBoost** | **Acc 0.952 · Prec 0.938 · Rec 0.942 · F1 0.940 · AUC 0.992** (confusion matrix: TN 28,693 · FP 1,248 · FN 1,171 · TP 18,888) |

| Model | RMSE (lakhs) | MAE | R² | MAPE |
|---|---|---|---|---|
| **Linear Regression (log target)** | **10.10** | 7.05 | **0.9978** | 1.85 % |
| XGBoost (log target) | 10.29 | 7.22 | 0.9977 | 1.97 % |
| Random Forest (log target) | 17.42 | 12.49 | 0.9933 | 3.37 % |

* Classifier drivers: price per sq ft relative to the city median (≈73 % of permutation importance), BHK, infrastructure score, city.
* Regressor driver: current (log) price (≈99 %). Residual error is the 0.5 pp noise injected in the growth rate; both Linear Regression and XGBoost reach it.
* The remaining classification errors (≈4.8 %) come from the same noise, which moves growth across the 7/8/9 % score steps.

## 4. Business use cases
* **Investors** — quick screen of a listing: verdict + confidence, 5-year value, comparison with a fixed 8 % benchmark.
* **Buyers** — filter by city, BHK, size and budget in the Explorer to shortlist high-scoring properties.
* **Real-estate platforms** — batch-score listings with `Advisor.predict_features()` and show a data-backed "investment score".
* **Analysts** — heatmaps and trend charts by city/state and property type.

## 5. Limitations & how to improve with real data
* Targets are rule-derived; replace `add_targets` with actual historical sale prices / appreciation and re-run `python run_pipeline.py`.
* Growth rates, city tiers and score weights are assumptions stored in `src/config.py` and are easy to tune.
* No crime, RERA, location coordinates or listing dates exist in the data; adding them would give richer location features and true price trends.
* City medians used in `Price_vs_City_Median` are computed on the full dataset (a small aggregate-level leak, negligible with ~6,000 rows per city).
* The app warns when inputs fall outside the training ranges (price 10–500 lakhs, size 500–5,000 sq ft).

## 6. Reproducibility
Fixed seed (42); `python run_pipeline.py` reproduced identical metrics on repeated runs. 15 automated tests (`tests/`) check data quality,
feature reproducibility, single-vs-batch prediction parity, input validation, model quality thresholds, the MLflow registry
(champions load and match the joblib models) and every Streamlit page including the prediction flow and error handling.
