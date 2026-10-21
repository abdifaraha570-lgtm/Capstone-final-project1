# jjushycsh-noshow-analytics

Predicting and reducing missed referral appointments at Jigjiga University Sheikh Hassan Yabare
Comprehensive Specialized Hospital (JJUSHYCSH). BAN6800 capstone - Nexford University.

| Module | Increment in this repo |
|---|---|
| 1 | Vision and Ethical AI Charter (submitted as a document; the charter is restated as binding rules in `docs/governance/ethical_ai_framework.md`) |
| 2 | `docs/module2/` - Project Overview Report (architecture, RAID log, privacy plan, sprint plan), project board ticket register, `environment.yml`; data dictionary in `docs/data_dictionary.md` |
| **3 (Sprint 1)** | **Data pipeline: DAG, Great Expectations suites, anonymization, bias check, audit log, Docker** |
| 4 | `module4/` - random-forest champion, MLflow tracking + registry, SHAP/DiCE, Fairlearn audit, model card, FastAPI `/predict` |
| 5 | `module5/` - live Streamlit dashboard (overview, results, XAI, what-if, ethics, limitations) and stakeholder deck |
| **Final Project** | **`monitoring/` (drift + fairness monitoring), `retrain.py` (gated retraining), `.github/workflows/` (CI/CD), `docker-compose.yml`, `render.yaml`, `docs/governance/` (ethical AI framework, monitoring plan, incident response, decommissioning, explanation store, decision log)** |

## Final Project submission map (where each requirement lives)
| Requirement (BAN6800 Final Project) | Evidence in this repository |
|---|---|
| Complete repository, Modules 2–5 | `docs/module2/` (M2), `src/`, `dags/`, `tests/`, `Dockerfile` (M3), `module4/` (M4), `module5/` (M5) |
| README with overview, setup, live links | this file |
| CI/CD (GitHub Actions) | `.github/workflows/ci.yml` (lint, 19 unit tests, monitoring smoke test, synthetic pipeline + gated retraining, API image smoke test, GHCR push); `.github/workflows/retrain.yml` (monthly) |
| Live API `/predict` | `module4/api/main.py`, `Dockerfile.api`, `render.yaml` |
| Live dashboard + XAI dashboard (SHAP) | `module5/app.py` (pages *What drives predictions*, *What-if*), link below |
| Monitoring dashboard (drift + fairness) | `monitoring/monitor.py` (PSI/KS drift, Fairlearn-style group metrics, Evidently report when installed), `monitoring/outputs/`, dashboard page *Monitoring (drift & fairness)* |
| Docker Compose (pipeline + API + monitoring) | `docker-compose.yml` (pipeline, mlflow, api, dashboard, monitoring) |
| Model retraining with MLflow | `retrain.py` (gates: PR-AUC, recall ≥ 0.70, TPR gap ≤ 0.10, no excluded inputs) |
| Model card, fairness report, explanation store | `module4/outputs/model_card.md`, `module4/outputs/fairness_report.md`, `docs/governance/explanation_store.md` |
| Ethical AI framework, AIA, compliance, trust statement, governance | `docs/governance/` |
| Data governance and privacy | `docs/data_governance.md`, `docs/anonymization_plan.md`, `src/data_prep/anonymize.py` |
| Environment | `environment.yml` (conda), `requirements*.txt` |

## Data
Development uses the public *Medical Appointment No Shows* dataset (Hoppen, 2016; 110,527 rows, 14 columns).
Download `KaggleV2-May-2016.csv` from Kaggle and place it in `data/external/`. Data files are never
committed (see `.gitignore`); DVC tracks them. A future JJUSHYCSH extract follows the same data contract
(`docs/data_dictionary.md`) after ethics and DPO approval.

## Pipeline stages
`ingest -> validate_raw (GX) -> clean -> anonymize -> build_features -> validate_processed (GX) -> publish_outputs + bias_check -> governance_report`

| Stage | Code |
|---|---|
| Ingestion + immutable snapshot | `src/data_prep/ingest.py` |
| Cleaning + quality report | `src/data_prep/clean.py` |
| Anonymization (PII removal) | `src/data_prep/anonymize.py` |
| Features + patient history | `src/features/build_features.py` |
| Integration, splits, dashboard feed | `src/data_prep/integrate.py` |
| Validation suites | `src/validation/gx_suite.py` (Great Expectations); `src/validation/pandas_suite.py` (same checks, no install) |
| Bias detection | `src/validation/bias_check.py` |
| Access control / retention | `src/governance/access_control.py` |
| Privacy audit log | `src/utils/audit_log.py` -> `logs/privacy_audit.jsonl` |
| Orchestration | `dags/noshow_etl_dag.py` (Apache Airflow) |

## Run it
```bash
# 1. local  (use `python run_pipeline.py --skip-gx` if Great Expectations cannot be installed)
pip install -r requirements-pipeline.txt
export NOSHOW_HASH_SALT="<secret from the data officer>"
pytest -q tests
python run_pipeline.py
# Data Docs: open gx/uncommitted/data_docs/local_site/index.html

# 2. Docker (tests run first, then the pipeline)
docker build -t noshow-pipeline:0.3.0 .
docker run --rm -e NOSHOW_HASH_SALT=$NOSHOW_HASH_SALT \
  -v $(pwd)/data:/opt/pipeline/data -v $(pwd)/reports:/opt/pipeline/reports \
  -v $(pwd)/logs:/opt/pipeline/logs -v $(pwd)/gx:/opt/pipeline/gx noshow-pipeline:0.3.0

# 3. Airflow: copy dags/ into $AIRFLOW_HOME/dags, set PIPELINE_ROOT to this repo, trigger jjushycsh_noshow_etl
```

## Outputs (input to Module 4)
- `data/processed/appointments_model_table.parquet` - one row per appointment, pseudonymized
- `data/processed/train|validation|test.parquet` - patient-grouped 70/15/15 split, seed 42
- `data/processed/dashboard_daily_noshow.parquet` - aggregate-only feed for the Module 5 dashboard
- `reports/` - data-quality report, GX results, bias tables and chart

## Versioning
Git Flow (`main` / `develop` / `feature/*`), SemVer tags (`v0.3.0` pipeline, `v1.0.0` integrated system), DVC for data.

## Live system (Final Project)
| Component | Where |
|---|---|
| Dashboard (Module 5) + explainability page | https://jjushycsh-noshow-dashboard-9nubvdk2zitnkwww33dukn.streamlit.app |
| Monitoring dashboard | same app, sidebar page **Monitoring (drift & fairness)** (reads `monitoring/outputs/monitoring_summary.json`) |
| Prediction API (Module 4) | `https://jjushycsh-noshow-api.onrender.com/docs` - deployed from `Dockerfile.api` via `render.yaml` (Render free tier; cold start ~1 min) |
| CI/CD | `.github/workflows/ci.yml` - ruff, pytest (pipeline + module4), monitoring smoke test, API image build + `/health` and `/predict` smoke test, push to GHCR |
| Scheduled monitoring + retraining | `.github/workflows/retrain.yml` - 5th of every month (and on demand): `monitoring/monitor.py --demo`, `retrain.py --dry-run`; opens an issue when a batch is in ALERT |
| Experiment tracking / registry | MLflow experiment `jjushycsh-noshow-m4` (Module 4 runs; `retrain.py` logs challengers to `jjushycsh_noshow`), registered model `jjushycsh_noshow_risk` (`docker compose up mlflow` -> http://localhost:5000) |

### Run everything with Docker Compose
```bash
export NOSHOW_HASH_SALT="<secret>"
docker compose up --build            # pipeline -> mlflow (5000) -> api (8000) -> dashboard (8501) -> monitoring
docker compose run --rm monitoring   # re-run drift/fairness monitoring only
docker compose run --rm monitoring python retrain.py --dry-run   # gated retraining check (needs data/processed)
```

### Monitoring and retraining (without Docker)
To try the full loop without patient data: `python scripts/make_sample_data.py --rows 20000 --out data/external/KaggleV2-May-2016.csv`,
then `python run_pipeline.py --skip-gx`, then `python retrain.py --dry-run` (CI runs exactly this on every push).

```bash
python monitoring/monitor.py --demo                      # writes monitoring/outputs/monitoring_summary.json + .html
python monitoring/monitor.py --reference module5/data/scored_test.csv --current data/new_batch.csv --out monitoring/outputs
python retrain.py --dry-run                              # gates: PR-AUC, recall >= 0.70, TPR gaps <= 0.10, no excluded inputs
```

### Deploy the API on Render
Connect this repository in Render -> *New Blueprint* (`render.yaml`). The service builds `Dockerfile.api` and exposes `/health`, `/predict`, `/docs`.

## Governance pack
`docs/governance/ethical_ai_framework.md` (charter, Algorithmic Impact Assessment, GDPR/CCPA/HIPAA + Proclamation 1321/2024 compliance report, public trust statement, governance structure), `monitoring_plan.md`, `incident_response.md`, `decommissioning_plan.md`, `explanation_store.md`, `decision_log.md`.

