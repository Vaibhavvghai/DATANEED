from argparse import ArgumentParser
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dataneed.experiment import run_pipeline


def main() -> None:
    parser = ArgumentParser(description="Run the DataNeed Phase 2 experiment")
    parser.add_argument("--profile", choices=("quick", "full"), default="quick")
    args = parser.parse_args()
    result = run_pipeline(ROOT / "artifacts", profile=args.profile)
    print("\nDataNeed Phase 2 completed")
    print(f"Profile: {args.profile}")
    print(f"Datasets: {result['curves']['dataset'].nunique()}")
    print(f"Learning-curve rows: {len(result['curves'])}")
    print(f"Meta-dataset rows: {len(result['meta'])}")
    print("\nMethod comparison")
    print(result["comparison"].to_string(index=False))
    accuracy_mae = result["accuracy_evaluation"]["absolute_accuracy_error"].mean()
    print(f"\nEstimated-accuracy MAE: {accuracy_mae:.4f} ({accuracy_mae * 100:.2f} percentage points)")


if __name__ == "__main__":
    main()

