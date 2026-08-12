# Applying this Phase 2 delivery to your repo

This zip contains every new/changed file for Phase 2, in the same folder
structure as `mlops-student-dropout-pipeline`. Extract it over your local
clone (it won't touch anything not listed below).

## 1. Copy the files in
```bash
cd /path/to/mlops-student-dropout-pipeline
unzip -o /path/to/phase2_delivery.zip -d .
```
(5 files are modifications to existing Phase 1 files: `README.md`,
`dvc.yaml`, and the three `.gitignore` files under `data/`/`models/` --
this will overwrite them with the Phase 2 versions.)

## 2. Regenerate + commit the data/model artifacts
**Read `docs/deployment.md` first** -- short version: no DVC remote was
ever configured in Phase 1, so `models/model.pkl` and the processed data
only lived in local `.dvc/cache`, invisible to CI/Docker/Render. `dvc.yaml`
now marks those outputs `cache: false` so they get committed to git
directly instead.

```bash
pip install -r requirements.txt
dvc repro
git add data/raw/data.csv data/raw/.gitignore \
        data/processed/train.csv data/processed/test.csv data/processed/label_map.json data/processed/.gitignore \
        models/model.pkl models/.gitignore \
        dvc.yaml dvc.lock
```

## 3. Commit everything else
```bash
git add .dockerignore .flake8 .pre-commit-config.yaml Dockerfile \
        docker-compose.yml render.yaml pyproject.toml \
        requirements-api.txt requirements-monitoring.txt \
        app/ src/monitor_drift.py src/check_retrain_trigger.py src/compare_metrics.py \
        tests/ docs/deployment.md docs/model_card.md \
        .github/workflows/ci.yml .github/workflows/cd.yml .github/workflows/retrain.yml \
        README.md
git commit -m "Phase 2: FastAPI serving, Docker, CI/CD, monitoring, continuous training"
git push
```

## 4. One-time manual setup (can't be done from a repo alone)
- **Render**: New > Blueprint > connect this repo (uses `render.yaml`).
  Copy the assigned URL into `README.md` and `docs/model_card.md`.
- **GitHub secret**: Render dashboard > your service > Settings > Deploy
  Hook > copy URL > add as repo secret `RENDER_DEPLOY_HOOK_URL` (Settings >
  Secrets and variables > Actions). Without this, `cd.yml` just logs a
  warning and does nothing -- CI still runs fine.

## 5. Verify
```bash
pytest tests/ -v          # 15 tests, no dependency on the real dataset
flake8 app tests && black --check app tests && isort --check-only app tests
docker build -t dropout-api .   # only works after step 2
```

Full context, including the reasoning for every non-obvious decision
(why `cache: false`, why CD is gated instead of using Render's own
auto-deploy, why lint is scoped to app/+tests/), is in `docs/deployment.md`.
