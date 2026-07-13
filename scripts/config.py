import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_PATH = os.path.join(DATA_DIR, "creditcard.csv")
PLOTS_DIR = os.path.join(PROJECT_ROOT, "plots")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")

for _dir in (DATA_DIR, PLOTS_DIR, MODELS_DIR, REPORTS_DIR):
    os.makedirs(_dir, exist_ok=True)

RANDOM_STATE = 2018

TEST_SIZE = 0.20
VALID_SIZE = 0.20  
TARGET_COL = "Class"

N_SPLITS = 5


RESAMPLING_METHODS = [
    "none",
    "random_undersample",
    "cluster_centroids",
    "smote",
    "adasyn",
    "smoteenn",
]

MODEL_NAMES = [
    "logistic_regression",
    "knn",
    "random_forest",
    "extra_trees",
    "balanced_random_forest",
    "xgboost",
    "lightgbm",
    "catboost",
]


SLOW_MODELS = set()

RESAMPLING_CV_SPLITS = 3
COST_SENSITIVE_CV_SPLITS = 5