# Dataset: Predict Students' Dropout and Academic Success

- **Source:** UCI Machine Learning Repository, dataset ID 697
  https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success
- **License:** CC BY 4.0
- **Citation:** Realinho, V., Vieira Martins, M., Machado, J., & Baptista, L.
  (2021). Predict Students' Dropout and Academic Success [Dataset]. UCI
  Machine Learning Repository. https://doi.org/10.24432/C5MC89.

## Size

- 4,424 rows (students), 36 feature columns + 1 target column
- No missing values in the raw file (UCI's own preprocessing already
  removed anomalies/outliers before publishing it — see their dataset card)

## Features

36 columns covering four groups:

- **Enrollment/academic path** — marital status, application mode/order,
  course, attendance regime, previous qualification (+ grade), admission
  grade
- **Demographics** — nationality, parents' qualification/occupation, age at
  enrollment, gender, displaced/international status, special needs
- **Socio-economic** — debtor flag, tuition-fees-up-to-date flag,
  scholarship holder flag, unemployment rate, inflation rate, GDP (macro
  indicators for the student's admission year)
- **1st/2nd semester performance** — units credited/enrolled/evaluated/
  approved, average grade, units without evaluation (x2 semesters)

Most columns are integer-coded categoricals (see UCI variable table for the
full code books, e.g. `Marital status`: 1=single, 2=married, ... 6=legally
separated). We keep them as integers rather than one-hot encoding the whole
thing — with ~30 categorical columns some of which have 20+ levels (e.g.
`Course`, `Nacionality`), one-hot would blow the feature space up a lot for
not much benefit with tree-based models. Only genuinely non-ordinal low-
cardinality columns get special handling in `prepare.py` (see comments
there); everything else is treated as already-numeric since that's how UCI
distributes it.

## Target distribution (imbalanced)

| Class    | Count | % of total |
|----------|------:|-----------:|
| Graduate | 2,209 | 49.9%      |
| Dropout  | 1,421 | 32.1%      |
| Enrolled |   794 | 17.9%      |

`Enrolled` is roughly a third the size of `Graduate`, which is why we use
macro-F1 (not accuracy) as the primary evaluation metric, and why
`class_weight="balanced"` / equivalent shows up in a couple of the models
in the MLflow experiments (see `experiment-tracking` branch).

## Known quality issues

- **Class imbalance** as above — a model that only ever predicts `Graduate`
  would still score ~50% accuracy, so accuracy alone is misleading here.
- **No missing values**, but a handful of engineered columns (e.g.
  `Curricular units 1st sem (grade)`) are 0 for students with 0 approved
  units that semester — that's a legitimate zero, not a missing value, but
  worth flagging since it looks like a placeholder at first glance.
- **Categorical codes are dataset-specific** (e.g. `Course` is a
  Portuguese-institution course catalog number) — they carry no ordinal
  meaning even though they're stored as integers.
- Column names in the raw CSV have a couple of quirks (a stray tab in
  `"Daytime/evening attendance\t"`, `Nacionality` misspelled) — normalized
  in `prepare.py` on load rather than editing the raw file, so the DVC-
  tracked source stays byte-identical to what UCI publishes.
