from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dataneed.experiment import run_pipeline


if __name__ == "__main__":
    result = run_pipeline(ROOT / "artifacts")
    evaluation = result["evaluation"]
    print("DataNeed MVP completed")
    print(f"Learning-curve rows: {len(result['curves'])}")
    print(f"Meta-dataset rows: {len(result['meta'])}")
    print(f"Evaluated rows: {len(evaluation)}")
    print(f"Leave-dataset-out MAE: {evaluation.attrs['mae']:.1f} samples")

