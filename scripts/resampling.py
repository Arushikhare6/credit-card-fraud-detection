from imblearn.under_sampling import RandomUnderSampler, ClusterCentroids
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTEENN
from sklearn.model_selection import StratifiedKFold

from config import RANDOM_STATE, N_SPLITS


def get_resampler(method: str):
    
    if method == "none":
        return None
    if method == "random_undersample":
        return RandomUnderSampler(random_state=RANDOM_STATE)
    if method == "cluster_centroids":
        return ClusterCentroids(random_state=RANDOM_STATE)
    if method == "smote":
        return SMOTE(random_state=RANDOM_STATE)
    if method == "adasyn":
        return ADASYN(random_state=RANDOM_STATE)
    if method == "smoteenn":
        return SMOTEENN(random_state=RANDOM_STATE)

    raise ValueError(f"Unknown resampling method: {method}")


def resample_train_only(X_train, y_train, method: str):
    
    resampler = get_resampler(method)
    if resampler is None:
        return X_train, y_train
    return resampler.fit_resample(X_train, y_train)


def leak_safe_cv_scores(model, X, y, method: str, scoring_fn, n_splits: int = N_SPLITS):
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    scores = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Resample ONLY the fold's training portion
        X_tr_res, y_tr_res = resample_train_only(X_tr, y_tr, method)

        fold_model = model.__class__(**model.get_params())
        fold_model.fit(X_tr_res, y_tr_res)

        y_val_proba = fold_model.predict_proba(X_val)[:, 1]
        score = scoring_fn(y_val, y_val_proba)
        scores.append(score)
        print(f"  [{method}] fold {fold_idx}/{n_splits}: {score:.4f}")

    return scores