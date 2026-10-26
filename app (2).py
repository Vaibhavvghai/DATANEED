from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from dataneed.experiment import design_matrix

st.set_page_config(page_title="DataNeed AI", page_icon="📊", layout="wide")
st.title("DataNeed AI")
st.caption("Estimate the labelled sample requirement from pilot data and early learning behaviour")

model_path = ROOT / "artifacts" / "phase2_dual_metric_model.joblib"
if not model_path.exists():
    st.warning("Run `python scripts/run_phase2.py --profile quick` before opening the app.")
    st.stop()

bundle = joblib.load(model_path)

with st.sidebar:
    st.header("Task settings")
    algorithm = st.selectbox("Algorithm", ["logistic_regression", "random_forest", "xgboost"])
    target_score = st.slider("Target F1", 0.50, 0.99, 0.85, 0.01)
    cost_per_label = st.number_input("Cost per labelled sample (₹)", min_value=0.0, value=8.0)

left, right = st.columns(2)
with left:
    st.subheader("Dataset properties")
    n_current = st.number_input("Current labelled samples", min_value=50, value=1000)
    n_features = st.number_input("Features", min_value=1, value=20)
    n_numeric = st.number_input("Numerical features", min_value=0, value=20)
    n_classes = st.number_input("Classes", min_value=2, value=2)
    minority_share = st.slider("Minority-class share", 0.01, 0.50, 0.30, 0.01)
    missing_rate = st.slider("Missing-value rate", 0.0, 0.90, 0.0, 0.01)

with right:
    st.subheader("Early learning curve")
    pilot_n = st.number_input("Largest pilot size", min_value=20, value=500)
    score_initial = st.slider("Initial F1", 0.0, 1.0, 0.65, 0.01)
    score_current = st.slider("Current F1", 0.0, 1.0, 0.80, 0.01)
    recent_gain = st.number_input("Gain at latest step", value=0.04, step=0.01)
    curve_curvature = st.number_input("Curve curvature", value=-0.01, step=0.01)
    score_std_current = st.number_input("Current score standard deviation", min_value=0.0, value=0.02, step=0.01)

if st.button("Estimate requirement", type="primary"):
    n_categorical = max(float(n_features - n_numeric), 0.0)
    class_entropy = float(-minority_share * np.log2(minority_share) - (1-minority_share) * np.log2(1-minority_share)) if n_classes == 2 else 1.0
    log_curve_slope = float((score_current - score_initial) / max(np.log(max(pilot_n, 2)) - np.log(max(pilot_n / 4, 1)), 1e-6))
    row = pd.DataFrame([{
        "algorithm": algorithm,
        "target_score": target_score,
        "n_current": float(n_current),
        "n_features": float(n_features),
        "n_numeric": float(n_numeric),
        "n_categorical": n_categorical,
        "n_classes": float(n_classes),
        "minority_share": minority_share,
        "class_entropy": class_entropy,
        "missing_rate": missing_rate,
        "feature_sample_ratio": float(n_features / n_current),
        "mean_abs_correlation": 0.0,
        "pilot_n": float(pilot_n),
        "score_initial": score_initial,
        "score_current": score_current,
        "recent_gain": recent_gain,
        "log_curve_slope": log_curve_slope,
        "curve_curvature": curve_curvature,
        "score_std_current": score_std_current,
        "target_reached": 0,
        "dataset": "new_dataset",
        "required_n": np.nan,
    }])
    X = design_matrix(row, expected_columns=bundle["columns"])
    estimate = int(round(float(np.exp(bundle["model"].predict(X)[0]))))
    estimate = max(estimate, int(n_current))
    extra = max(estimate - int(n_current), 0)
    if hasattr(bundle["model"], "estimators_"):
        tree_estimates = np.exp(np.array([tree.predict(X)[0] for tree in bundle["model"].estimators_]))
        low = max(int(n_current), int(np.quantile(tree_estimates, 0.10)))
        high = max(estimate, int(np.quantile(tree_estimates, 0.90)))
    else:
        low = max(int(n_current), int(estimate * 0.80))
        high = max(estimate, int(estimate * 1.25))

    X_accuracy = X.copy()
    X_accuracy["log_recommended_n"] = np.log(max(estimate, 1))
    X_accuracy = X_accuracy.reindex(columns=bundle["accuracy_columns"], fill_value=0.0)
    expected_accuracy = float(np.clip(bundle["accuracy_model"].predict(X_accuracy)[0], 0.0, 1.0))

    a, b, c = st.columns(3)
    a.metric("Estimated total", f"{estimate:,}")
    b.metric("Additional samples", f"{extra:,}")
    c.metric("Estimated labelling cost", f"₹{extra * cost_per_label:,.0f}")
    d, e, f = st.columns(3)
    d.metric("Expected F1", f"{target_score * 100:.1f}%")
    e.metric("Estimated accuracy", f"{expected_accuracy * 100:.1f}%")
    f.metric("Model-spread range", f"{low:,}–{high:,}")
    chart = pd.DataFrame({
        "Pilot samples": [max(int(pilot_n / 4), 1), max(int(pilot_n / 2), 2), int(pilot_n)],
        "F1": [float(score_initial), float((score_initial + score_current) / 2), float(score_current)],
    }).set_index("Pilot samples")
    st.subheader("Early learning curve")
    st.line_chart(chart)
    if extra == 0:
        st.success("The pilot suggests that the target may be achievable with the current data. Confirm this using a held-out validation set before stopping data collection.")
    st.info("This is a research estimate. Use the full 10-dataset model before reporting final results. The displayed range measures variation across the Random Forest ensemble and is not yet a calibrated confidence interval.")

