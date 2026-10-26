from __future__ import annotations

import numpy as np
import pandas as pd


def dataset_meta_features(X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    counts = y.value_counts(dropna=False)
    probabilities = counts / counts.sum()
    entropy = float(-(probabilities * np.log2(probabilities + 1e-12)).sum())
    minority_share = float(probabilities.min())

    numeric = X.select_dtypes(include=np.number)
    categorical_count = X.shape[1] - numeric.shape[1]
    missing_rate = float(X.isna().to_numpy().mean())

    if numeric.shape[1] >= 2:
        corr = numeric.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        mean_abs_corr = float(upper.stack().mean()) if not upper.stack().empty else 0.0
    else:
        mean_abs_corr = 0.0

    return {
        "n_current": float(len(X)),
        "n_features": float(X.shape[1]),
        "n_numeric": float(numeric.shape[1]),
        "n_categorical": float(categorical_count),
        "n_classes": float(y.nunique()),
        "minority_share": minority_share,
        "class_entropy": entropy,
        "missing_rate": missing_rate,
        "feature_sample_ratio": float(X.shape[1] / max(len(X), 1)),
        "mean_abs_correlation": mean_abs_corr,
    }


def curve_features(curve: pd.DataFrame, pilot_points: int = 3) -> dict[str, float]:
    early = curve.sort_values("train_size").head(pilot_points)
    if len(early) < 2:
        raise ValueError("At least two learning-curve points are required")

    sizes = early["train_size"].to_numpy(dtype=float)
    scores = early["mean_score"].to_numpy(dtype=float)
    log_sizes = np.log(sizes)
    slope = float(np.polyfit(log_sizes, scores, 1)[0])
    gain = float(scores[-1] - scores[-2])
    curvature = float(np.diff(scores, n=2)[-1]) if len(scores) >= 3 else 0.0

    return {
        "pilot_n": float(sizes[-1]),
        "score_initial": float(scores[0]),
        "score_current": float(scores[-1]),
        "recent_gain": gain,
        "log_curve_slope": slope,
        "curve_curvature": curvature,
        "score_std_current": float(early["std_score"].iloc[-1]),
    }

