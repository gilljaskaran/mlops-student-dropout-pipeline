# Model Card: Student Dropout & Academic Success Predictor

Following the Model Cards framework (Mitchell et al., 2018 / Google
Research) as covered in Week 11 (MLOps Governance and Risk Management).

## Model Details

| | |
|---|---|
| Developer | Group 1 -- Jaskaran Singh, Eric Prakashbhai Rathod (MAI201 MLOps) |
| Version | v1.0.0 (RandomForest, `dvc.lock` model hash `69fd35c9`) |
| Type | Multi-class classification (3 classes) |
| Framework | scikit-learn `RandomForestClassifier` |
| Date | 2026-08-02 (Phase 2) |
| Repository | https://github.com/gilljaskaran/mlops-student-dropout-pipeline |

## Intended Use

**Primary use case:** early-warning triage tool for academic advisors --
flag students at higher predicted risk of dropout at enrollment time (and
after 1st/2nd semester results are in) so advisors can prioritize outreach
and support resources.

**Out-of-scope uses:**
- Fully automated decisions with no human review (e.g. auto-revoking
  funding, auto-flagging for expulsion). This model informs outreach
  priority, it does not make administrative decisions.
- Any institution other than the one that produced this dataset -- it's a
  single Portuguese higher-education institution (see docs/dataset.md);
  the feature set includes country-specific macro indicators (Portugal's
  unemployment/inflation/GDP at time of enrollment) that don't transfer to
  other education systems without retraining on local data.
- Individual-level high-stakes decisions without a human in the loop.

## Training & Data

| | |
|---|---|
| Source | UCI ML Repository, dataset ID 697 ("Predict Students' Dropout and Academic Success") |
| Size | 4,424 students, 36 features, CC BY 4.0 |
| Split | Stratified 80/20 train/test (`src/prepare.py`, `random_state=42`) |
| Preprocessing | Column-name cleanup, median/mode imputation (defensive -- source data has no missing values), integer-coded categoricals kept as-is (see docs/dataset.md for why one-hot wasn't used) |
| Features | Full list in [`docs/dataset.md`](dataset.md) -- enrollment/academic path, demographics, socio-economic flags, macro indicators, 1st/2nd semester performance |

## Evaluation

- **Test set:** held-out 20% (~885 students), stratified by class
- **Methodology:** single train/test split (not cross-validated); model
  selection was informed by 7 MLflow runs comparing baseline logistic
  regression, a random forest grid, and an XGBoost grid (see README
  Results section / `experiment-tracking` branch)
- **Primary metric:** macro-F1 (not accuracy) because of class imbalance
  -- a model that only ever predicts "Graduate" would still score ~50%
  accuracy

## Performance Metrics

| Metric | Value |
|---|---:|
| Accuracy | 0.7605 |
| Macro F1 | 0.7005 |

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Dropout | 0.818 | 0.711 | 0.761 | 284 |
| Enrolled | 0.503 | 0.478 | 0.490 | 159 |
| Graduate | 0.811 | 0.894 | 0.850 | 442 |

## Limitations

- **`Enrolled` is the weak class** -- roughly a third the size of
  `Graduate` in training data, and both precision and recall sit around
  0.50 (essentially a coin flip conditioned on the model having already
  narrowed it down to "not obviously Dropout or Graduate"). Confusion is
  roughly symmetric with both other classes (see
  `docs/confusion_matrix.png`). Treat "Enrolled" predictions as lower
  confidence than the other two classes.
- **Single institution, single country.** All 4,424 students came from one
  Portuguese institution. Macro-economic features (unemployment, inflation,
  GDP) are tied to Portugal's economy during the collection period --
  applying this model anywhere else, or even to a much later cohort at the
  same institution, is out of the validated distribution.
- **No formal fairness/bias audit.** The feature set includes demographic
  and socio-economic attributes (nationality, parents' education/
  occupation, gender, special-needs flag, scholarship/debtor status) that
  could encode structural bias -- e.g. a low-income student flagged as
  "high dropout risk" partly via a debtor flag could see that used to
  justify reduced support instead of more. We have not run subgroup
  performance comparisons (e.g. precision/recall by gender or nationality)
  to check for disparate error rates. This is a real gap, not a "future
  work" footnote -- it should be closed before this model is used to
  influence any actual advising decisions.
- **Single train/test split.** No k-fold cross-validation was run, so the
  reported metrics carry some split-dependent variance not captured here.

## Ethical & Governance Controls

- **Bias assessment:** not performed (see Limitations above) -- flagged as
  an open item, not silently omitted.
- **Privacy:** dataset is already de-identified / aggregated by UCI's
  publication process; no additional PII is collected by this project's
  pipeline or API.
- **Recommendation:** predictions should support, not replace, an
  advisor's judgment. Do not use `Enrolled`-class predictions as a sole
  basis for action given their low precision/recall.

## Monitoring & Safeguards

- **Drift detection:** `src/monitor_drift.py` (EvidentlyAI, Week 9) compares
  incoming feature distributions against the training set; run weekly and
  on-demand via `.github/workflows/retrain.yml`.
- **Retraining triggers:** performance-based (accuracy drop > 5% vs the
  baseline in `metrics.json`) and data-based (drift detected), implemented
  in `src/check_retrain_trigger.py` -- see Week 10.
- **Human oversight:** retrained models are opened as a PR, never
  auto-merged (`docs/branch-protection.md` requires 1 approving review) --
  this is the model-promotion gate from the Week 10 CT pipeline.
- **Rollback:** every promoted model version is a git commit; reverting to
  a previous `models/model.pkl` is a normal `git revert`.

## Review Information

| | |
|---|---|
| Approval | Jaskaran Singh, Eric Rathod  |
| Last Review | 2026-08-08 |
| Next Review | Before next academic term, or immediately if a drift/performance alert fires |
| Live API | https://student-dropout-api-mo0v.onrender.com |

**Where this card is stored:** this GitHub repository (version-controlled
alongside the model), per Week 11's "Model Card in Practice" guidance. It
should be updated whenever the model version changes, new data is added,
or a new limitation is discovered.
