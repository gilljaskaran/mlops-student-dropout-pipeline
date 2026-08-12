"""
Data drift monitoring with EvidentlyAI (Week 9: Monitoring and Observability).

Compares the training distribution (data/processed/train.csv, "reference")
against a "current" batch and writes an HTML drift report plus a small JSON
summary that src/check_retrain_trigger.py reads for the data-based
retraining trigger (Week 10).

Usage:
    python src/monitor_drift.py
    python src/monitor_drift.py --current data/processed/test.csv
    python src/monitor_drift.py --simulate-drift   # demo: perturbs a few
                                                    # columns so the report
                                                    # actually shows drift

Note on evidently versions: the classic Report / metric_preset API moved to
evidently.legacy.* as of evidently>=0.7 (a breaking rewrite mid-2025). This
script tries the new location first and falls back to the pre-0.7 top-level
import, so `pip install evidently` keeps working either way.
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from evidently.legacy.metric_preset import DataDriftPreset
    from evidently.legacy.report import Report
except ImportError:  # evidently < 0.7
    from evidently.metric_preset import DataDriftPreset
    from evidently.report import Report

REFERENCE_PATH = Path("data/processed/train.csv")
REPORT_DIR = Path("monitoring/reports")


def load_features(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df.drop(columns=["Target"], errors="ignore")


def simulate_drift(df: pd.DataFrame, rng: np.random.RandomState) -> pd.DataFrame:
    """Shift a couple of features to mimic real drift, for the demo / to
    prove the pipeline actually catches something (Week 9: seasonal
    patterns, user behaviour changes, new cohorts all show up like this)."""
    df = df.copy()
    if "Age at enrollment" in df.columns:
        df["Age at enrollment"] = df["Age at enrollment"] + rng.randint(5, 15, size=len(df))
    if "Unemployment rate" in df.columns:
        df["Unemployment rate"] = df["Unemployment rate"] * 1.6
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--current",
        default="data/processed/test.csv",
        help="CSV to compare against the reference (train.csv). Defaults to "
        "the held-out test set; in production this would be recent "
        "logged requests (see app/main.py's logs/predictions.csv).",
    )
    parser.add_argument(
        "--simulate-drift",
        action="store_true",
        help="Perturb a couple of columns to demonstrate drift detection (for the live demo).",
    )
    args = parser.parse_args()

    if not REFERENCE_PATH.exists():
        raise SystemExit(
            f"{REFERENCE_PATH} not found. Run `dvc repro` first -- see docs/deployment.md."
        )
    current_path = Path(args.current)
    if not current_path.exists():
        raise SystemExit(f"{current_path} not found.")

    reference = load_features(REFERENCE_PATH)
    current = load_features(current_path)

    if args.simulate_drift:
        current = simulate_drift(current, np.random.RandomState(42))

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    html_path = REPORT_DIR / f"drift_report_{timestamp}.html"
    report.save_html(str(html_path))

    result = report.as_dict()
    drift_metric = next(
        m for m in result["metrics"] if m["metric"] == "DatasetDriftMetric"
    )["result"]
    summary = {
        "timestamp": timestamp,
        "reference": str(REFERENCE_PATH),
        "current": str(current_path),
        "simulated_drift": args.simulate_drift,
        "dataset_drift": drift_metric["dataset_drift"],
        "share_of_drifted_columns": drift_metric["share_of_drifted_columns"],
        "number_of_drifted_columns": drift_metric["number_of_drifted_columns"],
        "number_of_columns": drift_metric["number_of_columns"],
        "report_html": str(html_path),
    }
    (REPORT_DIR / "latest_drift_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"[monitor_drift] report saved to {html_path}")
    print(
        f"[monitor_drift] dataset_drift={summary['dataset_drift']} "
        f"share_of_drifted_columns={summary['share_of_drifted_columns']:.2f} "
        f"({summary['number_of_drifted_columns']}/{summary['number_of_columns']} columns)"
    )


if __name__ == "__main__":
    main()
