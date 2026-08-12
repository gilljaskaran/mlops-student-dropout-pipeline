"""
Retraining trigger logic (Week 10: Retraining & Continuous Learning Systems).

Combines two of the three trigger types from the lecture -- the third
(time-based) is the cron schedule in .github/workflows/retrain.yml itself.
"Use multiple triggers, don't rely on just one" (Week 10 best practice).

  1. Performance-based: current accuracy vs the baseline in metrics.json,
     using the same 5% drop threshold the lecture uses.
  2. Data-based: the dataset_drift flag from src/monitor_drift.py's latest
     run (monitoring/reports/latest_drift_summary.json).

This project has no model registry, so "baseline" is the accuracy recorded
in the currently-committed metrics.json (i.e. whatever the last merged
`train`+`evaluate` run produced) rather than a live "production model"
lookup -- see docs/model_card.md for why.

Usage:
    python src/check_retrain_trigger.py
Exit codes:
    0 = no retrain needed
    1 = retrain triggered (performance drop and/or drift detected)
"""
import json
import sys
from pathlib import Path

METRICS_PATH = Path("metrics.json")
DRIFT_SUMMARY_PATH = Path("monitoring/reports/latest_drift_summary.json")
PERFORMANCE_DROP_RATIO = 0.95  # retrain if current < 95% of baseline (Week 10)


def check_performance_trigger(baseline_accuracy: float) -> bool:
    if not METRICS_PATH.exists():
        print("[trigger] metrics.json not found -- skipping performance trigger")
        return False
    metrics = json.loads(METRICS_PATH.read_text())
    current_accuracy = metrics["accuracy"]
    threshold = baseline_accuracy * PERFORMANCE_DROP_RATIO
    triggered = current_accuracy < threshold
    print(
        f"[trigger] performance: current={current_accuracy:.4f} "
        f"baseline={baseline_accuracy:.4f} threshold={threshold:.4f} "
        f"-> {'TRIGGER' if triggered else 'ok'}"
    )
    return triggered


def check_drift_trigger() -> bool:
    if not DRIFT_SUMMARY_PATH.exists():
        print(
            "[trigger] no drift summary found (run src/monitor_drift.py first) "
            "-- skipping drift trigger"
        )
        return False
    summary = json.loads(DRIFT_SUMMARY_PATH.read_text())
    triggered = bool(summary.get("dataset_drift", False))
    print(
        f"[trigger] drift: share_of_drifted_columns="
        f"{summary.get('share_of_drifted_columns')} "
        f"dataset_drift={summary.get('dataset_drift')} -> "
        f"{'TRIGGER' if triggered else 'ok'}"
    )
    return triggered


def main():
    # From README's Results table -- the RandomForest committed as the
    # `train` stage's model. Update this if a new model is promoted.
    baseline_accuracy = 0.7605

    performance_triggered = check_performance_trigger(baseline_accuracy)
    drift_triggered = check_drift_trigger()
    should_retrain = performance_triggered or drift_triggered

    print(f"[trigger] retrain decision: {'YES' if should_retrain else 'no'}")
    sys.exit(1 if should_retrain else 0)


if __name__ == "__main__":
    main()
