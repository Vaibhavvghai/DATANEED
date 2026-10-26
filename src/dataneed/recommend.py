from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .experiment import design_matrix


def calibration_quantiles(evaluation_path: str | Path) -> tuple[float, float]:
    """Return log-residual quantiles from leave-dataset-out predictions."""
    path = Path(evaluation_path)
    if not path.exists():
        return float(np.log(0.80)), float(np.log(1.25))
    evaluation = pd.read_csv(path)
    prediction_column = "random_forest_meta_model_n"
    required = {"required_n", prediction_column}
    if not required.issubset(evaluation.columns):
        return float(np.log(0.80)), float(np.log(1.25))
    valid = evaluation[list(required)].replace([np.inf, -np.inf], np.nan).dropna()
    valid = valid[(valid["required_n"] > 0) & (valid[prediction_column] > 0)]
    if len(valid) < 10:
        return float(np.log(0.80)), float(np.log(1.25))
    residuals = np.log(valid["required_n"].to_numpy()) - np.log(valid[prediction_column].to_numpy())
    return float(np.quantile(residuals, 0.10)), float(np.quantile(residuals, 0.90))


def predict_requirement(
    bundle: dict,
    row: pd.DataFrame,
    current_n: int,
    cost_per_label: float,
    residual_low: float,
    residual_high: float,
) -> dict:
    """Produce point and conservative sample recommendations."""
    X = design_matrix(row, expected_columns=bundle["columns"])
    point_log = float(bundle["model"].predict(X)[0])
    point_n = max(1, int(round(np.exp(point_log))))

    calibrated_low = int(round(np.exp(point_log + residual_low)))
    calibrated_high = int(round(np.exp(point_log + max(residual_high, 0.0))))

    # A meta-model can occasionally predict fewer rows than are already
    # available, even when the pilot curve is still below the requested F1.
    # In that case, enforce a learning-curve extrapolation guard. This avoids
    # telling a user that their present data are sufficient when the measured
    # pilot performance says otherwise.
    values = row.iloc[0]
    target_score = float(values.get("target_score", np.nan))
    observed_score = float(values.get("score_current", np.nan))
    pilot_n = max(float(values.get("pilot_n", current_n)), 1.0)
    slope = float(values.get("log_curve_slope", np.nan))
    target_gap = target_score - observed_score
    extrapolation_guard = int(current_n)
    guard_applied = False
    reliable_growth = True
    if np.isfinite(target_gap) and target_gap > 0.01:
        guard_applied = True
        if np.isfinite(slope) and slope > 0.005:
            log_required = np.log(pilot_n) + (target_gap / slope)
            extrapolation_guard = int(
                np.ceil(np.clip(np.exp(min(log_required, np.log(1_000_000))),
                                current_n + 1, 1_000_000))
            )
        else:
            # A flat or falling pilot curve cannot support a trustworthy
            # numerical extrapolation. Keep it out of the winner selection.
            extrapolation_guard = 1_000_000
            reliable_growth = False

    conservative_total = max(int(current_n), calibrated_high)
    conservative_total = max(conservative_total, extrapolation_guard)
    point_total = max(int(current_n), point_n)
    interval_low = max(int(current_n), calibrated_low)
    interval_high = max(conservative_total, interval_low)
    additional = max(conservative_total - int(current_n), 0)

    X_accuracy = X.copy()
    X_accuracy["log_recommended_n"] = np.log(max(conservative_total, 1))
    X_accuracy = X_accuracy.reindex(columns=bundle["accuracy_columns"], fill_value=0.0)
    expected_accuracy = float(
        np.clip(bundle["accuracy_model"].predict(X_accuracy)[0], 0.0, 1.0)
    )
    return {
        "point_total": point_total,
        "conservative_total": conservative_total,
        "additional_samples": additional,
        "estimated_cost": additional * float(cost_per_label),
        "interval_low": interval_low,
        "interval_high": interval_high,
        "expected_accuracy": expected_accuracy,
        "pilot_f1": observed_score,
        "target_reached_in_pilot": bool(
            np.isfinite(observed_score) and observed_score >= target_score
        ),
        "extrapolation_guard_applied": guard_applied,
        "reliable_growth": reliable_growth,
    }

