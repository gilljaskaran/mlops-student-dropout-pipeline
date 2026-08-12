# Deployment (Phase 2)

## The model-artifact gap from Phase 1 (read this first)

Phase 1's `dvc.yaml` cached `models/model.pkl` the normal DVC way, and
`.dvc/config` never had a remote configured. That's fine for `dvc repro` on
one machine, but it means the model file only ever existed in whoever's
local `.dvc/cache` -- it was never in git and never push-able anywhere.
Docker builds, GitHub Actions runners, and Render all start from a fresh
`git clone`, so none of them could ever see it.

Fix applied for Phase 2: `models/model.pkl` in the `train` stage is now
`cache: false` (same pattern the repo already used for `metrics.json`), and
`models/.gitignore` no longer excludes it. DVC still tracks its hash in
`dvc.lock` for reproducibility, but the file itself now lives in git like
any other tracked artifact.

**The proper alternative**, if this ever became a real production system,
is a real DVC remote (S3, GCS, Google Drive, or a free option like
DagsHub) plus `dvc push`/`dvc pull` in CI. We didn't set that up because it
needs cloud credentials tied to someone's account -- `cache: false` gets a
working deployment without that dependency, at the cost of a ~32 MB binary
sitting in git history. Worth revisiting if this project needed a real
model registry.

The same problem applies to `data/raw/data.csv` and the
`data/processed/*.csv` splits -- also DVC-cached-only, also invisible to a
fresh clone. `src/monitor_drift.py` (drift detection) and
`.github/workflows/retrain.yml` (scheduled retraining) both need these
files to exist in the repo, so the same `cache: false` treatment was
applied to the `prepare` stage's outputs, and `data/raw/data.csv` is now
committed directly to git alongside its existing `.dvc` pointer file (it's
~520 KB and CC BY 4.0 licensed -- see docs/dataset.md).

**One-time step before any of the below works:** run `dvc repro`, then
commit the result:
```bash
dvc repro
git add data/raw/data.csv data/raw/.gitignore \
        data/processed/train.csv data/processed/test.csv data/processed/label_map.json data/processed/.gitignore \
        models/model.pkl models/.gitignore \
        dvc.yaml dvc.lock
git commit -m "phase2: track data + model in git (no DVC remote configured)"
git push
```

Without this step: the API can't start (no model.pkl), `docker-build` in CI
skips itself with a warning, and `monitor_drift.py` / `retrain.yml` have
nothing to compare against -- all fail gracefully with a clear message
rather than a confusing error, but nothing actually works until this runs.

## Running locally

```bash
pip install -r requirements-api.txt
uvicorn app.main:app --reload --port 8000
# http://localhost:8000/docs for interactive Swagger UI
```

## Running with Docker

```bash
docker build -t dropout-api:latest .
docker run -p 8000:8000 dropout-api:latest
curl http://localhost:8000/health
```

Or with Compose (adds log volume + resource limits, mirrors the Docker lab's
production-readiness section):
```bash
docker compose up -d
docker compose logs -f api
```

## Deploying to Render (free tier)

Render was the platform called out in Week 8 as the easiest free option
with Docker support and auto-deploy on push. This repo includes
`render.yaml` so it can be deployed as a Blueprint:

1. Push the committed `models/model.pkl` from the step above.
## CI-gated deploys (not Render's own auto-deploy)

`render.yaml` sets `autoDeploy: false` on purpose. Render's default
behaviour redeploys on every push to the connected branch regardless of
whether anything actually passed -- fine for a demo, not what Week 6/8
mean by "CI/CD for model deployment". Instead, `.github/workflows/cd.yml`
only calls Render's deploy hook after `ci.yml` (lint + tests) has passed on
`main`.

One-time setup: Render dashboard > service > Settings > Deploy Hook > copy
the URL > add it to this repo as a GitHub secret named
`RENDER_DEPLOY_HOOK_URL` (Settings > Secrets and variables > Actions).

2. In the Render dashboard: **New > Blueprint**, connect
   `gilljaskaran/mlops-student-dropout-pipeline`. Render reads `render.yaml`
   and creates a `student-dropout-api` web service on the free plan.
   - Alternative without the blueprint file: **New > Web Service** > connect
     the repo > Environment: **Docker** > leave the Dockerfile path as
     `./Dockerfile`.
3. First deploy will take a few minutes (installs scikit-learn/pandas).
   Render calls `GET /health` automatically once it's up.
4. Copy the public URL Render assigns (`https://student-dropout-api-xxxx.onrender.com`)
   into the README and `docs/model_card.md` "Review Information" section.

**Free-tier caveat:** the service spins down after 15 minutes of no
traffic and takes ~30-60s to cold-start on the next request -- expected
during a live demo if there's a gap between testing it. Hit `/health` a
minute before presenting to warm it up.

### Alternatives (also free, not set up here)

- **Hugging Face Spaces** -- Docker Spaces support the same `Dockerfile`
  as-is; good fit if the team wants a Spaces-hosted portfolio piece instead.
- **Railway** -- $5/month free credit, connects to GitHub the same way,
  slightly different config format (no `render.yaml` equivalent needed,
  it auto-detects the Dockerfile).

## Verifying a live deployment

```bash
curl https://<your-render-url>/health
curl https://<your-render-url>/model-info
curl -X POST https://<your-render-url>/predict -H "Content-Type: application/json" \
  -d '{"Marital status": 1, "Application mode": 1, ...}'   # see /model-info for the full field list
```
