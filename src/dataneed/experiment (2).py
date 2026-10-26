from __future__ import annotations

from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict

from .baselines import inverse_power_prediction
from .curves import generate_learning_curve, required_sample_size
from .datasets import dataset_catalog, load_dataset
from .features import curve_features, dataset_meta_features


TARGETS = (0.70, 0.75, 0.80, 0.85, 0.90)

warnings.filterwarnings(
    "ignore",
    message=".*sklearn.utils.parallel.delayed.*",
    category=UserWarning,
)


def base_models() -> dict:
    models = {
        "logistic_regression": LogisticRegression(max_iter=2500),
        "random_forest": RandomForestClassifier(n_estimators=180, random_state=42, n_jobs=-1),
    }
    try:
        from xgboost import XGBClassifier
        models["xgboost"] = XGBClassifier(
            n_estimators=180, max_depth=5, learning_rate=0.06,
            subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
            random_state=42, n_jobs=-1,
        )
    except ImportError:
        warnings.warn("xgboost is unavailable; running the remaining models")
    return models


def _curve_key(dataset: str, algorithm: str) -> str:
    return f"{dataset}::{algorithm}"


def build_meta_dataset(
    profile: str = "quick", targets=TARGETS, checkpoint_path: Path | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    checkpoint_rows = pd.DataFrame()
    if checkpoint_path is not None and checkpoint_path.exists():
        checkpoint_rows = pd.read_csv(checkpoint_path)
    curve_rows, meta_rows = [], []
    completed = set()
    if not checkpoint_rows.empty:
        completed = {
            _curve_key(str(d), str(a))
            for d, a in checkpoint_rows[["dataset", "algorithm"]].drop_duplicates().itertuples(index=False)
        }

    models = base_models()
    for spec in dataset_catalog(profile):
        print(f"Loading dataset: {spec.name}", flush=True)
        X, y = load_dataset(spec)
        data_features = dataset_meta_features(X, y)
        for model_name, model in models.items():
            key = _curve_key(spec.name, model_name)
            if key in completed:
                curve = checkpoint_rows[
                    (checkpoint_rows["dataset"] == spec.name)
                    & (checkpoint_rows["algorithm"] == model_name)
                ].copy()
                print(f"  Reusing checkpoint: {model_name}", flush=True)
            else:
                print(f"  Training curves: {model_name}", flush=True)
                curve = generate_learning_curve(X, y, model)
                curve.insert(0, "dataset", spec.name)
                curve.insert(1, "algorithm", model_name)
                checkpoint_rows = pd.concat([checkpoint_rows, curve], ignore_index=True)
                if checkpoint_path is not None:
                    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                    checkpoint_rows.to_csv(checkpoint_path, index=False)
            curve_rows.append(curve)
            early = curve_features(curve, pilot_points=3)
            for target in targets:
                required_n = required_sample_size(curve, target)
                meta_rows.append({
                    "dataset": spec.name,
                    "algorithm": model_name,
                    "target_score": float(target),
                    **data_features,
                    **early,
                    "power_law_n": inverse_power_prediction(curve, target, pilot_points=3),
                    "accuracy_at_required_n": (
                        float(curve.loc[curve["train_size"] == int(required_n), "mean_accuracy"].iloc[0])
                        if np.isfinite(required_n) else np.nan
                    ),
                    "required_n": required_n,
                    "target_reached": int(np.isfinite(required_n)),
                })
    return pd.concat(curve_rows, ignore_index=True), pd.DataFrame(meta_rows)


def feature_columns(frame: pd.DataFrame) -> list[str]:
    excluded = {
        "dataset", "required_n", "target_reached", "power_law_n",
        "accuracy_at_required_n",
    }
    return [c for c in frame.columns if c not in excluded]


def design_matrix(frame: pd.DataFrame, expected_columns=None) -> pd.DataFrame:
    X = pd.get_dummies(frame[feature_columns(frame)], columns=["algorithm"], dtype=float)
    if expected_columns is not None:
        X = X.reindex(columns=expected_columns, fill_value=0.0)
    return X


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    valid = np.isfinite(actual) & np.isfinite(predicted)
    actual, predicted = actual[valid], predicted[valid]
    if len(actual) == 0:
        return {"rows": 0, "mae": np.nan, "median_ape": np.nan,
                "underestimation_rate": np.nan, "target_attainment_rate": np.nan}
    return {
        "rows": int(len(actual)),
        "mae": float(mean_absolute_error(actual, predicted)),
        "median_ape": float(np.median(np.abs(predicted - actual) / actual)),
        "underestimation_rate": float(np.mean(predicted < actual)),
        "target_attainment_rate": float(np.mean(predicted >= actual)),
    }


def train_and_evaluate(meta: pd.DataFrame, artifact_dir: Path):
    reached = meta[meta["target_reached"] == 1].dropna(subset=["required_n"]).copy()
    if reached["dataset"].nunique() < 2:
        raise ValueError("Need at least two datasets with reached targets")
    X = design_matrix(reached)
    y = np.log(reached["required_n"].to_numpy(dtype=float))
    groups = reached["dataset"].to_numpy()
    cv = LeaveOneGroupOut()
    models = {
        "linear_meta_model": Ridge(alpha=1.0),
        "random_forest_meta_model": RandomForestRegressor(
            n_estimators=500, min_samples_leaf=2, random_state=42, n_jobs=-1
        ),
    }
    evaluation = reached[[
        "dataset", "algorithm", "target_score", "required_n",
        "power_law_n", "accuracy_at_required_n",
    ]].copy()
    actual = evaluation["required_n"].to_numpy(dtype=float)
    summary_rows = [{
        "method": "inverse_power_law",
        **_metrics(actual, evaluation["power_law_n"].to_numpy(dtype=float)),
    }]
    for name, model in models.items():
        predicted = np.exp(cross_val_predict(model, X, y, groups=groups, cv=cv))
        evaluation[f"{name}_n"] = predicted
        summary_rows.append({"method": name, **_metrics(actual, predicted)})

    final_model = models["random_forest_meta_model"]
    final_model.fit(X, y)

    accuracy_rows = reached.dropna(subset=["accuracy_at_required_n"]).copy()
    X_accuracy = design_matrix(accuracy_rows, expected_columns=list(X.columns))
    X_accuracy["log_recommended_n"] = np.log(accuracy_rows["required_n"].to_numpy(dtype=float))
    y_accuracy = accuracy_rows["accuracy_at_required_n"].to_numpy(dtype=float)
    accuracy_groups = accuracy_rows["dataset"].to_numpy()
    accuracy_model = RandomForestRegressor(
        n_estimators=500, min_samples_leaf=2, random_state=43, n_jobs=-1
    )
    accuracy_cv = cross_val_predict(
        accuracy_model, X_accuracy, y_accuracy,
        groups=accuracy_groups, cv=LeaveOneGroupOut(),
    )
    accuracy_evaluation = accuracy_rows[[
        "dataset", "algorithm", "target_score", "required_n", "accuracy_at_required_n"
    ]].copy()
    accuracy_evaluation["predicted_accuracy"] = np.clip(accuracy_cv, 0.0, 1.0)
    accuracy_evaluation["absolute_accuracy_error"] = np.abs(
        accuracy_evaluation["predicted_accuracy"] - accuracy_evaluation["accuracy_at_required_n"]
    )
    accuracy_model.fit(X_accuracy, y_accuracy)
    bundle = {
        "model": final_model,
        "columns": list(X.columns),
        "accuracy_model": accuracy_model,
        "accuracy_columns": list(X_accuracy.columns),
        "training_datasets": sorted(reached["dataset"].unique().tolist()),
        "training_n_range": [float(reached["n_current"].min()), float(reached["n_current"].max())],
    }
    joblib.dump(bundle, artifact_dir / "phase2_dual_metric_model.joblib")
    return bundle, evaluation, pd.DataFrame(summary_rows), accuracy_evaluation


def run_pipeline(artifact_dir: str | Path = "artifacts", profile: str = "quick") -> dict:
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = artifact_dir / f"{profile}_dual_metric_curves_checkpoint.csv"
    curves, meta = build_meta_dataset(profile=profile, checkpoint_path=checkpoint)
    curves.to_csv(artifact_dir / f"{profile}_dual_metric_learning_curves.csv", index=False)
    meta.to_csv(artifact_dir / f"{profile}_dual_metric_meta_dataset.csv", index=False)
    bundle, evaluation, comparison, accuracy_evaluation = train_and_evaluate(meta, artifact_dir)
    evaluation.to_csv(artifact_dir / f"{profile}_dual_metric_evaluation.csv", index=False)
    comparison.to_csv(artifact_dir / f"{profile}_dual_metric_method_comparison.csv", index=False)
    accuracy_evaluation.to_csv(
        artifact_dir / f"{profile}_accuracy_evaluation.csv", index=False
    )
    return {"curves": curves, "meta": meta, "bundle": bundle,
            "evaluation": evaluation, "comparison": comparison,
            "accuracy_evaluation": accuracy_evaluation}

