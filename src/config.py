"""Central configuration: paths, constants, domain rules."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw" / "india_housing_prices.csv"
DATA_PROCESSED = ROOT / "data" / "processed" / "housing_processed.csv"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
EDA_DIR = REPORTS_DIR / "eda"
ARTIFACTS_JSON = MODELS_DIR / "preprocessing_artifacts.json"
CLF_PATH = MODELS_DIR / "classifier.joblib"
REG_PATH = MODELS_DIR / "regressor.joblib"
METRICS_JSON = REPORTS_DIR / "metrics.json"
CLF_IMPORTANCE_CSV = REPORTS_DIR / "feature_importance_classifier.csv"
REG_IMPORTANCE_CSV = REPORTS_DIR / "feature_importance_regressor.csv"
MLFLOW_DB = ROOT / "mlflow.db"
MLFLOW_ARTIFACTS = ROOT / "mlartifacts"

RANDOM_STATE = 42
CURRENT_YEAR = 2025          # the dataset's Age_of_Property = 2025 - Year_Built
HORIZON_YEARS = 5
TEST_SIZE = 0.20

# ----------------------------------------------------------------------------
# Domain rules for the engineered targets
# ----------------------------------------------------------------------------
# Feature-based growth model:  r = city_base + type_adj + availability_adj
#                                  + transport_adj + infrastructure_adj
#                                  + age_adj + noise
TIER1 = ["Mumbai", "New Delhi", "Bangalore", "Hyderabad", "Chennai", "Pune",
         "Kolkata", "Ahmedabad", "Gurgaon", "Noida"]
TIER2 = ["Dwarka", "Faridabad", "Jaipur", "Lucknow", "Kochi", "Indore", "Surat",
         "Nagpur", "Coimbatore", "Bhubaneswar", "Trivandrum", "Vishakhapatnam",
         "Mysore", "Dehradun", "Bhopal", "Guwahati", "Vijayawada", "Mangalore"]
CITY_BASE_GROWTH = {"tier1": 0.085, "tier2": 0.075, "tier3": 0.065}
TYPE_ADJ = {"Apartment": 0.000, "Independent House": 0.004, "Villa": 0.008}
AVAILABILITY_ADJ = {"Ready_to_Move": 0.000, "Under_Construction": 0.010}
TRANSPORT_ADJ = {"Low": 0.000, "Medium": 0.0025, "High": 0.005}
INFRA_ADJ_MAX = 0.005       # +/- applied around infrastructure score of 5/10
AGE_ADJ_MAX = 0.003         # penalty for the oldest buildings
GROWTH_NOISE_SD = 0.005
GROWTH_CLIP = (0.03, 0.15)

# Multi-factor investment score (0-100); Good Investment = score >= threshold
GOOD_INVESTMENT_THRESHOLD = 60
TRANSPORT_SCORE = {"Low": 0, "Medium": 1, "High": 2}
AMENITY_LIST = ["Clubhouse", "Garden", "Gym", "Playground", "Pool"]

# ----------------------------------------------------------------------------
# Model feature lists
# ----------------------------------------------------------------------------
NUMERIC_FEATURES = [
    "BHK", "Size_in_SqFt", "Price_in_Lakhs", "Log_Price", "Age_of_Property", "Floor_No",
    "Total_Floors", "Floor_Ratio", "Nearby_Schools", "Nearby_Hospitals",
    "Price_per_SqFt_INR", "Price_vs_City_Median", "Size_per_BHK",
    "Transport_Score", "School_Density_Score", "Infrastructure_Score",
    "Amenity_Count",
] + [f"Has_{a}" for a in AMENITY_LIST]
CATEGORICAL_FEATURES = [
    "State", "City", "Property_Type", "Furnished_Status",
    "Public_Transport_Accessibility", "Parking_Space", "Security", "Facing",
    "Owner_Type", "Availability_Status",
]
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

CLF_TARGET = "Good_Investment"
REG_TARGET = "Future_Price_5Y"
