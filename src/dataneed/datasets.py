from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml, load_breast_cancer, load_iris, load_wine
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    loader: Optional[Callable] = None
    openml_id: Optional[int] = None
    max_samples: Optional[int] = 15_000


def quick_catalog() -> list[DatasetSpec]:
    """Offline catalog used to verify installation in a few minutes."""
    return [
        DatasetSpec("breast_cancer", loader=load_breast_cancer, max_samples=None),
        DatasetSpec("wine", loader=load_wine, max_samples=None),
        DatasetSpec("iris", loader=load_iris, max_samples=None),
    ]


def full_catalog() -> list[DatasetSpec]:
    """Ten public classification tasks for the Phase 2 experiment."""
    return quick_catalog() + [
        DatasetSpec("credit_g", openml_id=31, max_samples=None),
        DatasetSpec("diabetes", openml_id=37, max_samples=None),
        DatasetSpec("spambase", openml_id=44, max_samples=None),
        DatasetSpec("bank_marketing", openml_id=1461, max_samples=15_000),
        DatasetSpec("blood_transfusion", openml_id=1464, max_samples=None),
        DatasetSpec("phoneme", openml_id=1489, max_samples=None),
        DatasetSpec("adult", openml_id=1590, max_samples=15_000),
    ]


def dataset_catalog(profile: str = "quick") -> list[DatasetSpec]:
    if profile == "quick":
        return quick_catalog()
    if profile == "full":
        return full_catalog()
    raise ValueError("profile must be 'quick' or 'full'")


def _clean_target(y) -> tuple[pd.Series, np.ndarray]:
    raw = pd.Series(y, name="target")
    valid = ~raw.isna()
    raw = raw.loc[valid].reset_index(drop=True)
    codes, _ = pd.factorize(raw.astype(str), sort=True)
    return pd.Series(codes, name="target", dtype=int), valid.to_numpy()


def _stratified_limit(
    X: pd.DataFrame, y: pd.Series, max_samples: Optional[int]
) -> tuple[pd.DataFrame, pd.Series]:
    if max_samples is None or len(X) <= max_samples:
        return X.reset_index(drop=True), y.reset_index(drop=True)
    X_keep, _, y_keep, _ = train_test_split(
        X, y, train_size=max_samples, stratify=y, random_state=2026
    )
    return X_keep.reset_index(drop=True), y_keep.reset_index(drop=True)


def load_dataset(spec: DatasetSpec) -> tuple[pd.DataFrame, pd.Series]:
    if spec.loader is not None:
        bundle = spec.loader(as_frame=True)
    elif spec.openml_id is not None:
        # The pandas ARFF parser can fail when an OpenML categorical value is
        # stored as an integer (observed with Spambase on pandas 3 / Python 3.14).
        # liac-arff handles those categories consistently across environments.
        bundle = fetch_openml(
            data_id=spec.openml_id,
            as_frame=True,
            parser="liac-arff",
        )
    else:
        raise ValueError(f"No loader configured for {spec.name}")

    X = bundle.data.copy()
    y, valid_mask = _clean_target(bundle.target)
    X = X.loc[valid_mask].reset_index(drop=True)
    for column in X.columns:
        if isinstance(X[column].dtype, pd.CategoricalDtype) or X[column].dtype == object:
            X[column] = X[column].astype("object")
            X[column] = X[column].where(X[column].notna(), np.nan)
    numeric_columns = list(X.select_dtypes(include=np.number).columns)
    X[numeric_columns] = X[numeric_columns].replace([np.inf, -np.inf], np.nan)
    return _stratified_limit(X, y, spec.max_samples)

