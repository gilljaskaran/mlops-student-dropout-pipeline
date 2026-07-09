# Capturing the MLflow UI screenshots

The 7 experiment runs (baseline logistic regression + 3-config random
forest grid + 3-config XGBoost grid) are already logged to the local
`mlruns/` tracking store once you run `python src/train_mlflow.py`. That
part doesn't need a screenshot tool -- just a browser, since the MLflow UI
itself has to render the screenshots we need (any automated capture would
just be recreating a fake version of the same page, which defeats the
point for a grading submission).

Steps:

1. From the repo root, with the venv from `requirements.txt` active:
   ```
   mlflow ui --backend-store-uri ./mlruns --port 5000
   ```
2. Open http://localhost:5000 and select the
   `student-dropout-classification` experiment.
3. Screenshot 1 -- **experiment comparison table**: on the runs list view,
   add the `accuracy` and `macro_f1` columns (Columns button, top right)
   and sort by `macro_f1` descending so all 7 runs are visible with their
   metrics. Save as `docs/screenshots/experiment_comparison.png`.
4. Screenshot 2 -- **run detail view**: click into the best run
   (`xgb_lr0.05_d6` as of the last local run, but check for yourself --
   ties can shift depending on your machine's BLAS/threading) and
   screenshot the run page showing params, metrics, and the logged model
   artifact. Save as `docs/screenshots/run_detail.png`.
5. Reference both images from the README's Results section (a placeholder
   is already there -- just swap the paths in).

Takes about two minutes end to end.
