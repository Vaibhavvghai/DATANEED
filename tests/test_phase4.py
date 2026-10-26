import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dataneed.diagnostics import algorithm_explanation, dataset_health_checks
from dataneed.reporting import build_pdf_report


def _comparison() -> pd.DataFrame:
    return pd.DataFrame([{
        "Algorithm": "Random Forest",
        "Conservative total": 300,
        "Additional samples": 200,
        "Target F1": 0.85,
        "Pilot F1": 0.76,
        "Estimated accuracy": 0.83,
        "Estimated cost (INR)": 1600.0,
    }])


def test_dataset_health_checks_cover_core_risks():
    frame = pd.DataFrame({
        "identifier": range(10),
        "feature": [1, 2] * 5,
        "target": ["yes"] * 7 + ["no"] * 3,
    })
    checks = dataset_health_checks(frame, "target")
    names = {item["check"] for item in checks}
    assert {"Missing values", "Duplicate rows", "Class balance"}.issubset(names)


def test_pdf_report_has_pdf_signature():
    comparison = _comparison()
    winner = comparison.iloc[0]
    pdf = build_pdf_report(
        input_mode="CSV upload",
        dataset_name="example.csv",
        target_name="target",
        dataset_summary={"Rows": "100", "Features": "4", "Classes": "2"},
        winner=winner.to_dict(),
        comparison=comparison,
        health_checks=[{"status": "Good", "check": "Missing values", "detail": "None detected."}],
        explanation=algorithm_explanation(winner),
    )
    assert pdf.startswith(b"%PDF")
    assert len(pdf) > 1000

