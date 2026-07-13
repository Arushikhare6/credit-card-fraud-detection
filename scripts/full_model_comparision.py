import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import make_scorer, f1_score, average_precision_score
from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler, ClusterCentroids
from imblearn.combine import SMOTEENN
from sklearn.cluster import MiniBatchKMeans

from config import RANDOM_STATE, RESAMPLING_CV_SPLITS, COST_SENSITIVE_CV_SPLITS, SLOW_MODELS
from model_training import build_models


PRAUC_SCORER = make_scorer(average_precision_score, response_method="predict_proba")
F1_MACRO_SCORER = make_scorer(f1_score, average="macro")


def get_samplers() -> dict:
    
    return {
        "none": None,
        "random_undersample": RandomUnderSampler(random_state=RANDOM_STATE),
        "cluster_centroids": ClusterCentroids(
            estimator=MiniBatchKMeans(n_init=1, random_state=RANDOM_STATE),
            random_state=RANDOM_STATE,
        ),
        "smote": SMOTE(random_state=RANDOM_STATE),
        "adasyn": ADASYN(random_state=RANDOM_STATE),
        "smoteenn": SMOTEENN(random_state=RANDOM_STATE),
    }


def run_full_resampling_matrix(X_train, y_train, skip_models: set = None,
                                n_splits: int = RESAMPLING_CV_SPLITS,
                                fast: bool = True) -> pd.DataFrame:
    
    skip_models = skip_models if skip_models is not None else SLOW_MODELS
    samplers = get_samplers()
    models = build_models(y_train_for_weight=y_train, fast=fast)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    rows = []
    total = sum(1 for m in models if m not in skip_models) * len(samplers)
    done = 0

    for sampler_name, sampler in samplers.items():
        for model_name, model in models.items():
            if model_name in skip_models:
                continue

            done += 1
            print(f"[{done}/{total}] sampler={sampler_name:<20} model={model_name}")

            if sampler is None:
                pipeline = model
            else:
                pipeline = make_pipeline(sampler, model)

            try:
                f1_scores = cross_val_score(pipeline, X_train, y_train, cv=cv,
                                             scoring=F1_MACRO_SCORER, n_jobs=1)
                prauc_scores = cross_val_score(pipeline, X_train, y_train, cv=cv,
                                                scoring=PRAUC_SCORER, n_jobs=1)
                rows.append({
                    "sampler": sampler_name,
                    "model": model_name,
                    "f1_macro_mean": round(f1_scores.mean(), 4),
                    "f1_macro_std": round(f1_scores.std(), 4),
                    "prauc_mean": round(prauc_scores.mean(), 4),
                    "prauc_std": round(prauc_scores.std(), 4),
                })
            except Exception as e:
                print(f"  -> FAILED ({e}); skipping this combination")
                rows.append({
                    "sampler": sampler_name,
                    "model": model_name,
                    "f1_macro_mean": None,
                    "f1_macro_std": None,
                    "prauc_mean": None,
                    "prauc_std": None,
                })

    results = pd.DataFrame(rows).sort_values("prauc_mean", ascending=False).reset_index(drop=True)
    return results


def run_cost_sensitive_comparison(X_train, y_train, skip_models: set = None,
                                   n_splits: int = COST_SENSITIVE_CV_SPLITS,
                                   fast: bool = True) -> pd.DataFrame:
    
    skip_models = skip_models if skip_models is not None else SLOW_MODELS
    models = build_models(y_train_for_weight=y_train, fast=fast)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    rows = []
    for model_name, model in models.items():
        if model_name in skip_models:
            continue
        print(f"Cost-sensitive CV: {model_name}")
        try:
            f1_scores = cross_val_score(model, X_train, y_train, cv=cv,
                                         scoring=F1_MACRO_SCORER, n_jobs=1)
            prauc_scores = cross_val_score(model, X_train, y_train, cv=cv,
                                            scoring=PRAUC_SCORER, n_jobs=1)
            rows.append({
                "model": model_name,
                "f1_macro_mean": round(f1_scores.mean(), 4),
                "f1_macro_std": round(f1_scores.std(), 4),
                "prauc_mean": round(prauc_scores.mean(), 4),
                "prauc_std": round(prauc_scores.std(), 4),
            })
        except Exception as e:
            print(f"  -> FAILED ({e}); skipping")

    results = pd.DataFrame(rows).sort_values("prauc_mean", ascending=False).reset_index(drop=True)
    return results


def pick_best_overall(resampling_df: pd.DataFrame, cost_sensitive_df: pd.DataFrame, top_n: int = 4):
    
    cs = cost_sensitive_df.copy()
    cs["sampler"] = "cost_sensitive"

    combined = pd.concat([resampling_df, cs], ignore_index=True)
    combined = combined.dropna(subset=["prauc_mean"]).sort_values("prauc_mean", ascending=False)
    return combined.head(top_n).reset_index(drop=True)