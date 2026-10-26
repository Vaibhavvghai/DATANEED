from __future__ import annotations

import pandas as pd


def dataset_health_checks(frame: pd.DataFrame, target_column: str) -> list[dict[str, str]]:
    """Return concise, user-facing data quality checks for a classification CSV."""
    checks: list[dict[str, str]] = []
    rows = max(len(frame), 1)
    target = frame[target_column]
    features = frame.drop(columns=[target_column])

    missing_rate = float(frame.isna().to_numpy().mean())
    if missing_rate == 0:
        checks.append({"status": "Good", "check": "Missing values", "detail": "No missing values detected."})
    elif missing_rate <= 0.10:
        checks.append({"status": "Warning", "check": "Missing values", "detail": f"{missing_rate:.1%} of all cells are missing; imputation will be applied."})
    else:
        checks.append({"status": "Risk", "check": "Missing values", "detail": f"{missing_rate:.1%} of all cells are missing. Review the affected columns before trusting the estimate."})

    duplicates = int(frame.duplicated().sum())
    checks.append({
        "status": "Good" if duplicates == 0 else "Warning",
        "check": "Duplicate rows",
        "detail": "No duplicate rows detected." if duplicates == 0 else f"{duplicates:,} duplicate rows ({duplicates / rows:.1%}) may overstate performance.",
    })

    valid_target = target.dropna().astype(str)
    counts = valid_target.value_counts()
    minority_share = float(counts.min() / counts.sum()) if len(counts) else 0.0
    if minority_share >= 0.20:
        imbalance_status = "Good"
    elif minority_share >= 0.10:
        imbalance_status = "Warning"
    else:
        imbalance_status = "Risk"
    checks.append({
        "status": imbalance_status,
        "check": "Class balance",
        "detail": f"Smallest class contains {minority_share:.1%} of labelled rows.",
    })

    target_missing = int(target.isna().sum())
    checks.append({
        "status": "Good" if target_missing == 0 else "Warning",
        "check": "Missing target",
        "detail": "Every row has a target label." if target_missing == 0 else f"{target_missing:,} rows have no target and are excluded.",
    })

    id_like = [
        str(column) for column in features.columns
        if features[column].nunique(dropna=True) / rows >= 0.98
    ]
    checks.append({
        "status": "Good" if not id_like else "Warning",
        "check": "Identifier-like features",
        "detail": "No near-unique feature detected." if not id_like else "Consider removing: " + ", ".join(id_like[:5]),
    })
    return checks


def manual_health_checks(
    current_n: int,
    n_features: int,
    minority_share: float,
    missing_rate: float,
    pilot_n: int,
) -> list[dict[str, str]]:
    checks: list[dict[str, str]] = []
    checks.append({
        "status": "Good" if current_n >= 200 else "Warning",
        "check": "Dataset size",
        "detail": f"{current_n:,} labelled rows are currently available.",
    })
    checks.append({
        "status": "Good" if minority_share >= 0.20 else ("Warning" if minority_share >= 0.10 else "Risk"),
        "check": "Class balance",
        "detail": f"Minority-class share is {minority_share:.1%}.",
    })
    checks.append({
        "status": "Good" if missing_rate <= 0.10 else "Risk",
        "check": "Missing values",
        "detail": f"Reported missing-value rate is {missing_rate:.1%}.",
    })
    checks.append({
        "status": "Good" if n_features / max(current_n, 1) <= 0.10 else "Warning",
        "check": "Feature-to-sample ratio",
        "detail": f"{n_features} features for {current_n:,} rows.",
    })
    checks.append({
        "status": "Good" if pilot_n <= current_n else "Risk",
        "check": "Pilot size",
        "detail": "Pilot size is within the available data." if pilot_n <= current_n else "Pilot size exceeds the current labelled sample count.",
    })
    return checks


def algorithm_explanation(winner: pd.Series) -> str:
    algorithm = str(winner["Algorithm"])
    base = {
        "Random Forest": "Random Forest can model non-linear relationships and feature interactions with limited preprocessing.",
        "XGBoost": "XGBoost builds boosted trees sequentially and often performs strongly on mixed tabular data.",
        "Logistic Regression": "Logistic Regression is a stable, interpretable baseline and works well when class boundaries are approximately linear.",
    }.get(algorithm, f"{algorithm} produced the strongest pilot result.")
    conservative = int(winner["Conservative total"])
    pilot = float(winner["Pilot F1"])
    target = float(winner["Target F1"])
    reason = (
        f" It was selected because it has the lowest conservative requirement among algorithms "
        f"with a usable learning trend ({conservative:,} total samples). Its pilot F1 is "
        f"{pilot:.1%} against a target of {target:.1%}."
    )
    return base + reason

