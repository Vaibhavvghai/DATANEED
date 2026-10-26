import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dataneed.curves import required_sample_size
from dataneed.baselines import inverse_power_prediction
from dataneed.features import curve_features, dataset_meta_features


def test_confirmed_target_crossing():
    curve = pd.DataFrame({
        "train_size": [100, 200, 400, 800],
        "mean_score": [0.70, 0.82, 0.86, 0.87],
        "std_score": [0.03, 0.02, 0.02, 0.01],
    })
    assert required_sample_size(curve, 0.85) == 400
    assert pd.isna(required_sample_size(curve, 0.90))


def test_feature_extractors_return_finite_values():
    X = pd.DataFrame({"a": [1, 2, 3, 4], "b": [4, 3, 2, 1]})
    y = pd.Series([0, 0, 1, 1])
    meta = dataset_meta_features(X, y)
    assert meta["n_features"] == 2
    assert meta["minority_share"] == 0.5

    curve = pd.DataFrame({
        "train_size": [50, 100, 200],
        "mean_score": [0.60, 0.70, 0.76],
        "std_score": [0.04, 0.03, 0.02],
    })
    features = curve_features(curve)
    assert features["pilot_n"] == 200
    assert features["recent_gain"] > 0


def test_inverse_power_prediction_is_finite():
    curve = pd.DataFrame({
        "train_size": [100, 200, 400],
        "mean_score": [0.60, 0.70, 0.78],
        "std_score": [0.03, 0.02, 0.02],
    })
    prediction = inverse_power_prediction(curve, 0.82)
    assert prediction >= 400

