from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from dataneed.curves import generate_learning_curve
from dataneed.diagnostics import (
    algorithm_explanation,
    dataset_health_checks,
    manual_health_checks,
)
from dataneed.experiment import base_models
from dataneed.features import curve_features, dataset_meta_features
from dataneed.recommend import calibration_quantiles, predict_requirement
from dataneed.reporting import build_pdf_report


ALGORITHM_LABELS = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}

st.set_page_config(page_title="DataNeed AI", page_icon="📊", layout="wide")
st.title("DataNeed AI")
st.caption("Estimate training-data requirements from a CSV or from manually entered pilot information")

with st.expander("How DataNeed AI works"):
    st.markdown(
        """
        1. Extract dataset properties and build small pilot learning curves.
        2. Compare Logistic Regression, Random Forest and XGBoost.
        3. Predict the required sample count with a benchmark-trained meta-model.
        4. Apply a conservative 90th-percentile calibration and a target-gap safety guard.
        5. Estimate final accuracy and labelling cost. Validate the result on an untouched test set.
        """
    )

model_path = ROOT / "artifacts" / "phase2_dual_metric_model.joblib"
if not model_path.exists():
    st.warning("Run `python scripts/run_phase2.py --profile full` before opening the app.")
    st.stop()

bundle = joblib.load(model_path)
evaluation_path = ROOT / "artifacts" / "full_dual_metric_evaluation.csv"
if not evaluation_path.exists():
    evaluation_path = ROOT / "artifacts" / "quick_dual_metric_evaluation.csv"
residual_low, residual_high = calibration_quantiles(evaluation_path)


def result_row(algorithm: str, target_score: float, current_n: int, cost: float,
               data_features: dict, early_features: dict) -> dict:
    row = pd.DataFrame([{
        "algorithm": algorithm,
        "target_score": float(target_score),
        **data_features,
        **early_features,
    }])
    result = predict_requirement(
        bundle, row, current_n, cost, residual_low, residual_high
    )
    return {
        "Algorithm": ALGORITHM_LABELS.get(algorithm, algorithm),
        "Point estimate": result["point_total"],
        "Conservative total": result["conservative_total"],
        "Additional samples": result["additional_samples"],
        "Target F1": float(target_score),
        "Pilot F1": result["pilot_f1"],
        "Estimated accuracy": result["expected_accuracy"],
        "Estimated cost (₹)": result["estimated_cost"],
        "Range low": result["interval_low"],
        "Range high": result["interval_high"],
        "Reliable growth": result["reliable_growth"],
        "Extrapolation used": result["extrapolation_guard_applied"],
    }


def display_single_result(result: dict, target_score: float, curve: pd.DataFrame | None = None) -> None:
    first = st.columns(4)
    first[0].metric("Point estimate", f"{result['Point estimate']:,}")
    first[1].metric("Conservative recommendation", f"{result['Conservative total']:,}")
    first[2].metric("Additional samples", f"{result['Additional samples']:,}")
    first[3].metric("Estimated cost", f"₹{result['Estimated cost (₹)']:,.0f}")
    second = st.columns(3)
    second[0].metric("Target F1", f"{target_score * 100:.1f}%")
    second[1].metric("Estimated accuracy", f"{result['Estimated accuracy'] * 100:.1f}%")
    second[2].metric(
        "Calibrated range", f"{result['Range low']:,}–{result['Range high']:,}"
    )
    if curve is not None:
        chart = curve[["train_size", "mean_score", "mean_accuracy"]].copy()
        chart = chart.rename(columns={
            "train_size": "Training samples",
            "mean_score": "F1",
            "mean_accuracy": "Accuracy",
        }).set_index("Training samples")
        st.subheader("Pilot learning curve")
        st.line_chart(chart)
    if result["Additional samples"] == 0:
        st.success("The conservative estimate indicates that the current labelled data may be sufficient. Confirm this on a separate held-out test set.")
    st.info("The conservative recommendation applies a 90th-percentile correction from leave-dataset-out residuals. Estimated accuracy has a benchmark MAE of about 6.52 percentage points.")


with st.sidebar:
    st.header("Analysis settings")
    mode = st.radio("Choose input method", ["Upload CSV", "Manual input"])
    target_score = st.slider("Target F1", 0.50, 0.99, 0.85, 0.01)
    cost_per_label = st.number_input(
        "Cost per labelled sample (₹)", min_value=0.0, value=8.0
    )


if mode == "Upload CSV":
    st.subheader("Automatic dataset analysis")
    uploaded = st.file_uploader("Upload a classification dataset", type=["csv"])
    if uploaded is None:
        st.info("Upload a CSV file to calculate dataset properties and compare all three algorithms. If you do not have a CSV, choose Manual input in the sidebar.")
        st.stop()

    try:
        frame = pd.read_csv(uploaded)
    except Exception as exc:
        st.error(f"The CSV could not be read: {exc}")
        st.stop()
    if frame.empty or frame.shape[1] < 2:
        st.error("The CSV must contain at least two columns and one data row.")
        st.stop()

    left, right = st.columns(2)
    with left:
        target_column = st.selectbox("Target column", list(frame.columns), index=len(frame.columns) - 1)
    with right:
        st.metric("Uploaded shape", f"{len(frame):,} rows × {frame.shape[1]:,} columns")

    valid = frame[target_column].notna()
    X_full = frame.loc[valid].drop(columns=[target_column]).copy()
    target_raw = frame.loc[valid, target_column].astype(str)
    codes, labels = pd.factorize(target_raw, sort=True)
    y_full = pd.Series(codes, name="target", dtype=int)
    X_full = X_full.reset_index(drop=True)

    class_counts = y_full.value_counts()
    if y_full.nunique() < 2:
        st.error("The selected target needs at least two classes.")
        st.stop()
    if y_full.nunique() > 20:
        st.error("The selected target has more than 20 classes. Phase 2 supports classification targets with 2–20 classes.")
        st.stop()
    if class_counts.min() < 5:
        st.error("Every target class needs at least five rows for stratified pilot testing.")
        st.stop()

    class_preview = pd.DataFrame({
        "Class": [str(labels[i]) for i in range(len(labels))],
        "Rows": [int(class_counts.get(i, 0)) for i in range(len(labels))],
    })
    with st.expander("Dataset preview and class distribution"):
        st.dataframe(frame.head(10), use_container_width=True)
        st.dataframe(class_preview, use_container_width=True, hide_index=True)

    health_checks = dataset_health_checks(frame, target_column)
    st.subheader("Dataset health")
    health_frame = pd.DataFrame(health_checks)
    health_frame.columns = ["Status", "Check", "Finding"]
    st.dataframe(health_frame, use_container_width=True, hide_index=True)
    if any(item["status"] == "Risk" for item in health_checks):
        st.warning("The dataset has at least one high-risk quality issue. Review it before treating recommendations as final.")

    if st.button("Analyse CSV and compare algorithms", type="primary"):
        analysis_X, analysis_y = X_full, y_full
        if len(analysis_X) > 5_000:
            analysis_X, _, analysis_y, _ = train_test_split(
                analysis_X, analysis_y, train_size=5_000,
                stratify=analysis_y, random_state=2026,
            )
            analysis_X = analysis_X.reset_index(drop=True)
            analysis_y = analysis_y.reset_index(drop=True)
            st.warning("For interactive speed, pilot curves use a reproducible 5,000-row stratified sample. The recommendation still records the full uploaded row count.")

        data_features = dataset_meta_features(analysis_X, analysis_y)
        data_features["n_current"] = float(len(X_full))
        data_features["feature_sample_ratio"] = float(
            X_full.shape[1] / max(len(X_full), 1)
        )
        results, curves = [], []
        progress = st.progress(0, text="Building pilot learning curves")
        models = base_models()
        for index, (algorithm, estimator) in enumerate(models.items(), start=1):
            curve = generate_learning_curve(
                analysis_X, analysis_y, estimator,
                fractions=(0.20, 0.35, 0.50), seeds=(11, 22), metric="f1",
            )
            early = curve_features(curve, pilot_points=3)
            results.append(result_row(
                algorithm, target_score, len(X_full), cost_per_label,
                data_features, early,
            ))
            plot_curve = curve.copy()
            plot_curve["Algorithm"] = ALGORITHM_LABELS.get(algorithm, algorithm)
            curves.append(plot_curve)
            progress.progress(index / len(models), text=f"Completed {ALGORITHM_LABELS.get(algorithm, algorithm)}")
        progress.empty()

        comparison = pd.DataFrame(results).sort_values(
            ["Reliable growth", "Conservative total", "Estimated accuracy"],
            ascending=[False, True, False]
        ).reset_index(drop=True)
        winner = comparison.iloc[0]
        if bool(winner["Reliable growth"]):
            st.success(
                f"Recommended algorithm: {winner['Algorithm']}. It has the lowest conservative sample requirement among algorithms with a usable pilot trend."
            )
        else:
            st.warning(
                "None of the pilot curves shows reliable improvement toward the target. Collect more pilot data or reduce the target before treating a sample estimate as reliable."
            )
        summary = st.columns(4)
        summary[0].metric("Recommended algorithm", winner["Algorithm"])
        summary[1].metric("Conservative total", f"{int(winner['Conservative total']):,}")
        summary[2].metric("Target F1", f"{target_score * 100:.1f}%")
        summary[3].metric("Estimated accuracy", f"{winner['Estimated accuracy'] * 100:.1f}%")

        explanation = algorithm_explanation(winner)
        st.subheader("Why this algorithm?")
        st.write(explanation)
        if float(winner["Pilot F1"]) + 0.01 < target_score:
            st.warning(
                f"The selected algorithm reached {float(winner['Pilot F1']):.1%} pilot F1, below the {target_score:.1%} target. The sample recommendation is an extrapolation, not proof that the target will be achieved."
            )

        display = comparison.copy()
        display["Target F1"] = (display["Target F1"] * 100).map(lambda x: f"{x:.1f}%")
        display["Pilot F1"] = (display["Pilot F1"] * 100).map(lambda x: f"{x:.1f}%")
        display["Estimated accuracy"] = (display["Estimated accuracy"] * 100).map(lambda x: f"{x:.1f}%")
        for column in ["Point estimate", "Conservative total", "Additional samples", "Range low", "Range high"]:
            display[column] = display[column].map(lambda x: f"{int(x):,}")
        display["Estimated cost (₹)"] = display["Estimated cost (₹)"].map(lambda x: f"₹{x:,.0f}")
        st.subheader("Algorithm comparison")
        st.dataframe(display, use_container_width=True, hide_index=True)

        if bool(comparison["Extrapolation used"].any()):
            st.warning(
                "The target F1 was not reached by at least one pilot curve. Its conservative total therefore includes a learning-curve extrapolation guard; large values mean the target is difficult or the current trend is weak."
            )

        all_curves = pd.concat(curves, ignore_index=True)
        curve_chart = all_curves.pivot_table(
            index="train_size", columns="Algorithm", values="mean_score"
        )
        st.subheader("Pilot F1 comparison")
        st.line_chart(curve_chart)
        report_comparison = comparison.rename(columns={"Estimated cost (₹)": "Estimated cost (INR)"})
        report_winner = report_comparison.iloc[0].to_dict()
        pdf_bytes = build_pdf_report(
            input_mode="CSV upload",
            dataset_name=getattr(uploaded, "name", "Uploaded CSV"),
            target_name=str(target_column),
            dataset_summary={
                "Rows": f"{len(frame):,}",
                "Features": f"{X_full.shape[1]:,}",
                "Classes": f"{y_full.nunique():,}",
                "Missing cells": f"{frame.isna().to_numpy().mean():.1%}",
            },
            winner=report_winner,
            comparison=report_comparison,
            health_checks=health_checks,
            explanation=explanation,
        )
        download_columns = st.columns(2)
        download_columns[0].download_button(
            "Download comparison CSV",
            comparison.to_csv(index=False).encode("utf-8"),
            file_name="dataneed_algorithm_comparison.csv",
            mime="text/csv",
        )
        download_columns[1].download_button(
            "Download PDF report",
            pdf_bytes,
            file_name="dataneed_analysis_report.pdf",
            mime="application/pdf",
            type="primary",
        )
        st.caption("Automatic analysis supports tabular classification. It does not currently support regression, images, text transformers or time-series forecasting.")

else:
    with st.sidebar:
        algorithm = st.selectbox(
            "Algorithm", list(ALGORITHM_LABELS),
            format_func=lambda value: ALGORITHM_LABELS[value],
        )
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
        score_std_current = st.number_input(
            "Current score standard deviation", min_value=0.0, value=0.02, step=0.01
        )

    if st.button("Estimate requirement", type="primary"):
        n_categorical = max(float(n_features - n_numeric), 0.0)
        class_entropy = (
            float(-minority_share * np.log2(minority_share) -
                  (1 - minority_share) * np.log2(1 - minority_share))
            if n_classes == 2 else 1.0
        )
        log_curve_slope = float(
            (score_current - score_initial) /
            max(np.log(max(pilot_n, 2)) - np.log(max(pilot_n / 4, 1)), 1e-6)
        )
        data_features = {
            "n_current": float(n_current), "n_features": float(n_features),
            "n_numeric": float(n_numeric), "n_categorical": n_categorical,
            "n_classes": float(n_classes), "minority_share": float(minority_share),
            "class_entropy": class_entropy, "missing_rate": float(missing_rate),
            "feature_sample_ratio": float(n_features / n_current),
            "mean_abs_correlation": 0.0,
        }
        early_features = {
            "pilot_n": float(pilot_n), "score_initial": float(score_initial),
            "score_current": float(score_current), "recent_gain": float(recent_gain),
            "log_curve_slope": log_curve_slope,
            "curve_curvature": float(curve_curvature),
            "score_std_current": float(score_std_current),
        }
        result = result_row(
            algorithm, target_score, int(n_current), cost_per_label,
            data_features, early_features,
        )
        manual_curve = pd.DataFrame({
            "train_size": [max(int(pilot_n / 4), 1), max(int(pilot_n / 2), 2), int(pilot_n)],
            "mean_score": [float(score_initial), float((score_initial + score_current) / 2), float(score_current)],
            "mean_accuracy": [np.nan, np.nan, result["Estimated accuracy"]],
        })
        display_single_result(result, target_score, manual_curve)
        manual_checks = manual_health_checks(
            int(n_current), int(n_features), float(minority_share),
            float(missing_rate), int(pilot_n),
        )
        st.subheader("Dataset health")
        manual_health_frame = pd.DataFrame(manual_checks)
        manual_health_frame.columns = ["Status", "Check", "Finding"]
        st.dataframe(manual_health_frame, use_container_width=True, hide_index=True)

        manual_winner = pd.Series(result)
        explanation = algorithm_explanation(manual_winner)
        st.subheader("Why this algorithm?")
        st.write(explanation)
        if float(result["Pilot F1"]) + 0.01 < target_score:
            st.warning(
                f"The entered pilot F1 is {float(result['Pilot F1']):.1%}, below the {target_score:.1%} target. The recommendation relies on extrapolation and must be validated experimentally."
            )

        manual_comparison = pd.DataFrame([result]).rename(
            columns={"Estimated cost (₹)": "Estimated cost (INR)"}
        )
        pdf_bytes = build_pdf_report(
            input_mode="Manual input",
            dataset_name="Manual dataset description",
            target_name="User-defined classification target",
            dataset_summary={
                "Current labelled rows": f"{int(n_current):,}",
                "Features": f"{int(n_features):,}",
                "Classes": f"{int(n_classes):,}",
                "Missing values": f"{float(missing_rate):.1%}",
            },
            winner=manual_comparison.iloc[0].to_dict(),
            comparison=manual_comparison,
            health_checks=manual_checks,
            explanation=explanation,
        )
        st.download_button(
            "Download PDF report",
            pdf_bytes,
            file_name="dataneed_manual_analysis_report.pdf",
            mime="application/pdf",
            type="primary",
        )

