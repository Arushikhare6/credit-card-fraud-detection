from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    VotingClassifier,
)
from imblearn.ensemble import BalancedRandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
 
from config import RANDOM_STATE
 
 
def compute_scale_pos_weight(y_train) -> float:
   
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    return n_neg / max(n_pos, 1)
 
 
def build_models(y_train_for_weight=None, fast: bool = False):
    
    spw = compute_scale_pos_weight(y_train_for_weight) if y_train_for_weight is not None else 1.0
    n_estimators = 150 if fast else 400
 
    models = {
        "logistic_regression": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=RANDOM_STATE,
        ),
        "knn": KNeighborsClassifier(
            n_neighbors=5,
            weights="distance",
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=n_estimators,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=n_estimators,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "balanced_random_forest": BalancedRandomForestClassifier(
            n_estimators=n_estimators,
            sampling_strategy="all",
            replacement=True,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "xgboost": XGBClassifier(
            n_estimators=n_estimators,
            max_depth=6,
            learning_rate=0.05,
            scale_pos_weight=spw,
            eval_metric="aucpr",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "lightgbm": LGBMClassifier(
            n_estimators=n_estimators,
            max_depth=6,
            learning_rate=0.05,
            scale_pos_weight=spw,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=-1,
        ),
        "catboost": CatBoostClassifier(
            iterations=n_estimators,
            depth=6,
            learning_rate=0.05,
            scale_pos_weight=spw,
            random_state=RANDOM_STATE,
            verbose=False,
        ),
    }
    return models
 
 
def train_model(name: str, X_train, y_train, X_valid=None, y_valid=None, fast: bool = False):
    
    models = build_models(y_train_for_weight=y_train, fast=fast)
    if name not in models:
        raise ValueError(f"Unknown model: {name}. Options: {list(models.keys())}")
 
    model = models[name]
 
    if name == "xgboost" and X_valid is not None and y_valid is not None:
        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
    else:
        model.fit(X_train, y_train)
 
    return model
 
 
def train_all_models(X_train, y_train, X_valid=None, y_valid=None, fast: bool = False,
                      skip: set = None):
    
    skip = skip or set()
    fitted = {}
    for name in build_models(y_train_for_weight=y_train, fast=fast).keys():
        if name in skip:
            continue
        print(f"Training {name}...")
        fitted[name] = train_model(name, X_train, y_train, X_valid, y_valid, fast=fast)
    return fitted
 
 
def build_voting_classifier(fitted_models: dict, top_n_names: list):
    
    from sklearn.frozen import FrozenEstimator
 
    estimators = [
        (name.replace("__", "_"), FrozenEstimator(fitted_models[name]))
        for name in top_n_names if name in fitted_models
    ]
    if len(estimators) < 2:
        raise ValueError("Need at least 2 fitted models to build a voting classifier.")
 
    voting_clf = VotingClassifier(estimators=estimators, voting="soft", n_jobs=-1)
    return voting_clf
 
 
def fit_voting_classifier(voting_clf, X_train, y_train):
    voting_clf.fit(X_train, y_train)
    return voting_clf
 