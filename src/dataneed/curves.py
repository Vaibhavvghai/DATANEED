from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def preprocessing_for(X: pd.DataFrame) -> ColumnTransformer:
    numeric = list(X.select_dtypes(include=np.number).columns)
    categorical = [c for c in X.columns if c not in numeric]
    transformers = []
    if numeric:
        transformers.append(("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), numeric))
    if categorical:
        transformers.append(("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical))
    return ColumnTransformer(transformers)


def score_predictions(y_true, y_pred, metric: str) -> float:
    if metric == "accuracy":
        return float(accuracy_score(y_true, y_pred))
    average = "binary" if pd.Series(y_true).nunique() == 2 else "weighted"
    return float(f1_score(y_true, y_pred, average=average, zero_division=0))


def generate_learning_curve(
    X: pd.DataFrame,
    y: pd.Series,
    estimator,
    fractions=(0.10, 0.20, 0.35, 0.50, 0.70, 0.85),
    seeds=(11, 22, 33),
    metric="f1",
) -> pd.DataFrame:
    rows = []
    min_per_class = int(y.value_counts().min())
    if min_per_class < 2:
        raise ValueError("Every class needs at least two observations")

    for fraction in fractions:
        f1_scores, accuracy_scores, used_sizes = [], [], []
        for seed in seeds:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.20, stratify=y, random_state=seed
            )
            requested = max(int(len(X_train) * fraction), y.nunique() * 4)
            requested = min(requested, len(X_train))
            if requested < len(X_train):
                X_sub, _, y_sub, _ = train_test_split(
                    X_train,
                    y_train,
                    train_size=requested,
                    stratify=y_train,
                    random_state=seed + 1000,
                )
            else:
                X_sub, y_sub = X_train, y_train

            pipeline = Pipeline([
                ("preprocess", preprocessing_for(X)),
                ("model", clone(estimator)),
            ])
            pipeline.fit(X_sub, y_sub)
            predictions = pipeline.predict(X_test)
            f1_scores.append(score_predictions(y_test, predictions, "f1"))
            accuracy_scores.append(score_predictions(y_test, predictions, "accuracy"))
            used_sizes.append(len(X_sub))

        rows.append({
            "train_size": int(round(np.mean(used_sizes))),
            "mean_score": float(np.mean(f1_scores)),
            "std_score": float(np.std(f1_scores, ddof=1)) if len(f1_scores) > 1 else 0.0,
            "mean_accuracy": float(np.mean(accuracy_scores)),
            "std_accuracy": float(np.std(accuracy_scores, ddof=1)) if len(accuracy_scores) > 1 else 0.0,
            "metric": "f1",
        })

    return pd.DataFrame(rows).drop_duplicates("train_size").sort_values("train_size")


def required_sample_size(curve: pd.DataFrame, target: float) -> float:
    """First target crossing that remains above target at the next point."""
    ordered = curve.sort_values("train_size").reset_index(drop=True)
    for i in range(len(ordered) - 1):
        if ordered.loc[i, "mean_score"] >= target and ordered.loc[i + 1, "mean_score"] >= target:
            return float(ordered.loc[i, "train_size"])
    return float("nan")

