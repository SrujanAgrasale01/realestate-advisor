# 🏠 Real Estate Investment Advisor — Predicting Property Profitability & Future Value

An end-to-end machine-learning project that tells an investor
1. **whether a property is a "Good Investment"** (classification), and
2. **what it will be worth after 5 years** (regression),

served through an interactive **Streamlit** app with experiments tracked in **MLflow**.

| | Result (50,000-row hold-out test set) |
|---|---|
| **Classifier** (XGBoost) | Accuracy **95.2%** · Precision 93.8% · Recall 94.2% · F1 **0.940** · ROC-AUC **0.992** |
| **Regressor** (Linear Regression on log-price) | RMSE **10.1 lakhs** · MAE 7.0 lakhs · R² **0.9978** · MAPE 1.85% |

> **Please read:** the supplied dataset has no future price and no investment label, so both targets are
> *engineered from documented domain rules* (see `docs/Project_Documentation.md`). The metrics therefore show
> how well the models learn those rules, not real-market forecasting accuracy. The raw data also behaves like a
> synthetic dataset (price is unrelated to size, city, amenities…), which the EDA reports honestly.

---

## 1. Quick start

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app/streamlit_app.py                      # trained models are already included
```

Other commands

| Task | Command |
|---|---|
| Re-run everything (preprocess → EDA → train + MLflow) | `python run_pipeline.py` (≈ 5 min) |
| Individual steps | `python -m src.preprocess` · `python -m src.eda` · `python -m src.train` |
| Run the 15 automated tests | `python -m pytest tests -q` |
| Open the MLflow UI | `python -m src.relocate_mlflow` *(once, after moving the folder)* then `mlflow ui --backend-store-uri sqlite:///mlflow.db` → http://localhost:5000 |

Tested with Python 3.12 (versions pinned in `requirements.txt`; the saved models must be loaded with the same
scikit-learn / xgboost versions).

**Troubleshooting – `XGBoostError: input stream corrupted` / pickle errors:** the saved models are tied to the
library versions they were trained with. The app now detects this and rebuilds the two models automatically on
first start (~30 s). You can also do it manually: `python -m src.rebuild` (models only) or `python -m src.train`
(models + MLflow). If `pip install -r requirements.txt` fails on a newer Python (e.g. 3.14), install the latest
versions of the same packages instead.

## 2. Project structure

```
real_estate_advisor/
├── app/streamlit_app.py         # Streamlit UI (4 pages)
├── src/
│   ├── config.py                # paths, constants, domain rules, feature lists
│   ├── features.py              # cleaning, feature engineering, growth model, investment score
│   ├── preprocess.py            # Step 1  -> data/processed + models/preprocessing_artifacts.json
│   ├── eda.py                   # Step 2  -> reports/eda/*.png + reports/eda_findings.md
│   ├── train.py                 # Steps 3-4: 6 models, MLflow tracking + model registry
│   ├── predict.py               # inference helper used by the app and tests
│   └── relocate_mlflow.py       # fixes MLflow paths after moving the folder
├── data/raw/india_housing_prices.csv
├── data/processed/housing_processed.csv      # cleaned + engineered dataset (deliverable)
├── models/                      # classifier.joblib, regressor.joblib, preprocessing_artifacts.json
├── reports/                     # eda/, eda_findings.md, metrics.json, model_comparison.csv, importances, plots
├── mlflow.db + mlartifacts/     # MLflow experiment logs + registered models
├── tests/test_pipeline.py       # data, features, models, registry and Streamlit app tests
├── docs/Project_Documentation.md  # methodology, findings, use cases, limitations
└── run_pipeline.py
```

## 3. Requirement checklist

| Requirement | Where |
|---|---|
| Handle missing values & duplicates | `features.clean_data` (median / mode imputation, ID + row de-duplication; also imputers inside the model pipeline) |
| Normalise / scale numeric features | `StandardScaler` in `train.make_preprocessor` |
| Encode Location, Property_Type etc. | `OneHotEncoder` (State, City, Property_Type, …) |
| New features (Price per Sq Ft, School Density Score, …) | `features.engineer_features` (price/sqft in ₹, price vs city median, school density, infrastructure score, amenity flags, floor ratio, size per BHK, log price…) |
| Binary "Good Investment" label from domain rules | `features.investment_score` — multi-factor score ≥ 60 |
| Future price after 5 years | `features.annual_growth_rate` — feature-based growth: `Future = Price × (1 + r)^5` |
| Outlier detection | IQR rule, flag `PPSF_Outlier`, winsorised price per sq ft |
| 20 EDA questions + 4 Step-2 analyses | `src/eda.py` → `reports/eda_findings.md` and 24 charts |
| Classification models + metrics | Logistic Regression, Random Forest, XGBoost — Accuracy, Precision, Recall, F1, ROC-AUC, confusion matrix |
| Regression models + metrics | Linear Regression, Random Forest, XGBoost — RMSE, MAE, R² (+ MAPE) |
| MLflow: runs, params, metrics, artefacts, registry | `src/train.py` — 2 experiments, 6 runs, 2 registered models with alias `champion` |
| Streamlit: input form, filters, verdict, 5-year price, confidence, feature importance, heatmaps, trend charts | `app/streamlit_app.py` |
| Cleaned dataset, scripts, MLflow logs, app, documentation | all included |

## 4. The Streamlit app

1. **🔮 Investment Advisor** – State/City + property form (type, BHK, size, price, year built, floors, schools,
   hospitals, transport, parking, security, amenities…) → *Good / Not good* verdict with confidence and gauge,
   estimated price after 5 years, implied annual growth, price per sq ft vs city median, 5-year projection vs a fixed 8 %
   benchmark, and feature importance.
2. **🔎 Property Explorer** – filter all 250 k listings by state, city, type, BHK, area, price, availability;
   KPIs, sortable table, CSV download, charts.
3. **📊 Market Insights** – location-wise heatmaps (city/state × type/BHK/…), city trend charts, growth ranking,
   correlation matrix.
4. **🧪 Model Performance** – metric tables, confusion matrix, ROC curve, predicted-vs-actual, feature
   importance, how targets are defined, MLflow instructions.

**Deploying to Streamlit Community Cloud:** push the folder to GitHub (all files are < 100 MB), choose
`app/streamlit_app.py` as the entry point. `mlflow` and `pytest` can be removed from `requirements.txt` for
deployment because the app does not import them.

## 5. Models & results

| Task | Model | Key metrics (test set) |
|---|---|---|
| Classification | Logistic Regression | Acc 0.905 · F1 0.882 · AUC 0.967 |
| Classification | Random Forest | Acc 0.926 · F1 0.909 · AUC 0.983 |
| Classification | **XGBoost (champion)** | **Acc 0.952 · F1 0.940 · AUC 0.992** |
| Regression | **Linear Regression, log target (champion)** | **RMSE 10.10 · MAE 7.05 · R² 0.9978** |
| Regression | Random Forest | RMSE 17.42 · MAE 12.49 · R² 0.9933 |
| Regression | XGBoost | RMSE 10.29 · MAE 7.22 · R² 0.9977 |

The regressors are fitted on `log(Future_Price)`; the noise injected into the growth rate sets a floor of roughly
10 lakhs RMSE, which Linear Regression and XGBoost both reach.

Full details: `docs/Project_Documentation.md`, `reports/eda_findings.md`, `reports/model_comparison.csv`.
