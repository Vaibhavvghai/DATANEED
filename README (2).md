# DataNeed AI — Phase 2

This version expands the working prototype into the main research experiment.

## Phase 2 additions

- Ten classification datasets: three built into scikit-learn and seven pinned OpenML tasks
- Six training sizes and three random seeds
- Logistic Regression, Random Forest, and XGBoost
- Target scores: 0.70, 0.75, 0.80, 0.85, and 0.90
- Resumable learning-curve checkpoints
- Inverse power-law baseline
- Linear and Random Forest meta-models
- Leave-one-dataset-out evaluation
- MAE, median percentage error, underestimation rate, and target-attainment rate
- Improved UI range that never falls below the available sample count
- Early learning-curve chart in the application
- Final output shows both expected F1 and estimated classification accuracy

## Dataset list

1. Breast Cancer
2. Wine
3. Iris
4. German Credit (`credit-g`, OpenML 31)
5. Diabetes (OpenML 37)
6. Spambase (OpenML 44)
7. Bank Marketing (OpenML 1461)
8. Blood Transfusion (OpenML 1464)
9. Phoneme (OpenML 1489)
10. Adult Income (OpenML 1590)

The two largest datasets are sampled reproducibly to 15,000 records to keep a student-laptop run practical. Dataset IDs are pinned for reproducibility.

## Windows setup

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

The dependency list includes `liac-arff`. Phase 2 forces this OpenML parser to
avoid a pandas parser failure seen with integer categories in Spambase on
Python 3.14.

## Step 1: quick verification

This uses three offline datasets and confirms that the upgraded code works:

```powershell
python scripts/run_phase2.py --profile quick
```

## Step 2: full research run

This downloads the seven OpenML datasets on the first run and starts the ten-dataset experiment:

```powershell
python scripts/run_phase2.py --profile full
```

Keep the laptop connected to power and the internet. Runtime depends on the CPU because the design contains:

```text
10 datasets × 3 algorithms × 6 sizes × 3 seeds = 540 model fits
```

`artifacts/full_dual_metric_curves_checkpoint.csv` updates after every dataset-and-algorithm curve. If the run stops, execute the same command again and completed curves will be reused.

## Step 3: open the application

```powershell
streamlit run app.py
```

## Main outputs

- `artifacts/full_dual_metric_learning_curves.csv`
- `artifacts/full_dual_metric_meta_dataset.csv`
- `artifacts/full_dual_metric_evaluation.csv`
- `artifacts/full_dual_metric_method_comparison.csv`
- `artifacts/full_accuracy_evaluation.csv`
- `artifacts/phase2_dual_metric_model.joblib`

Use `full_method_comparison.csv` for the final results table. The research hypothesis receives support only if the hybrid model performs better than the inverse power-law baseline on held-out datasets.

## Definition of required samples

For a target score, `required_n` is the first measured training size where mean validation F1 reaches the target and stays above it at the next measured size. Unreached targets remain unknown instead of receiving an invented value.

## Important interpretation

The quick profile checks software operation. Only the full profile is suitable for the report, and its conclusions apply only to the tabular classification tasks and algorithms included in the experiment.

