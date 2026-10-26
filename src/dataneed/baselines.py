from __future__ import annotations

import numpy as np
import pandas as pd


def inverse_power_prediction(curve: pd.DataFrame, target: float, pilot_points: int = 3) -> float:
    """Predict required N from early points using error = a * N^(-b)."""
    early = curve.sort_values("train_size").head(pilot_points)
    if len(early) < 2 or not 0.0 < target < 1.0:
        return float("nan")
    sizes = early["train_size"].to_numpy(dtype=float)
    errors = np.clip(1.0 - early["mean_score"].to_numpy(dtype=float), 1e-4, 0.9999)
    slope, intercept = np.polyfit(np.log(sizes), np.log(errors), 1)
    b = float(-slope)
    if not np.isfinite(b) or b <= 1e-4:
        return float("nan")
    a = float(np.exp(intercept))
    predicted = (a / max(1.0 - target, 1e-4)) ** (1.0 / b)
    if not np.isfinite(predicted):
        return float("nan")
    return float(np.clip(predicted, sizes[-1], 1_000_000))

