"""
Compare the freshly retrained metrics.json against the version already
committed on HEAD (Week 10: "Model comparison and promotion" -- only
promote a retrained model if it's at least as good as what's deployed).

Used by .github/workflows/retrain.yml after `dvc repro` regenerates
metrics.json, to decide whether to open a PR with the new model at all.

Usage:
    python src/compare_metrics.py
Exit codes:
    0 = new model is >= old model on macro_f1 (safe to open a PR)
    1 = new model is worse (discard, do not open a PR)
"""
import json
import subprocess
import sys
from pathlib import Path


def main():
    new_metrics = json.loads(Path("metrics.json").read_text())
    new_f1 = new_metrics["macro_f1"]

    old_raw = subprocess.run(
        ["git", "show", "HEAD:metrics.json"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    old_f1 = json.loads(old_raw)["macro_f1"] if old_raw else 0.0

    improved = new_f1 >= old_f1
    print(f"[compare_metrics] new macro_f1={new_f1:.4f} old macro_f1={old_f1:.4f}")
    print(f"[compare_metrics] {'IMPROVED (or equal)' if improved else 'WORSE -- discarding'}")
    sys.exit(0 if improved else 1)


if __name__ == "__main__":
    main()
