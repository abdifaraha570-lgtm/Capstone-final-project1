# Module 4 — Predictive Model Development & Validation Pack (Sprint 2)
JJUSHYCSH referral no-show risk model · **Abdi Farah Ahmed** · BAN6800, Nexford University

This folder is the Sprint 2 increment of `jjushycsh-noshow-analytics`. It **consumes the Module 3 outputs
unchanged** — the pseudonymised model table, the patient-grouped 70/15/15 split and the bias flags — and adds
the model, its validation, explainability, fairness audit, model card, registry entry and API.

```
jjushycsh-noshow-analytics/          <- Module 2/3 repository
├── run_pipeline.py, src/, dags/ ... <- Module 3 (Sprint 1) - unchanged
├── data/processed/                  <- Module 3 outputs: train / validation / test  (+ model table)
├── reports/                         <- Module 3 outputs: bias_flags.json, data_quality_report.json, split_summary.json
└── module4/                         <- THIS FOLDER (Sprint 2)
    ├── run_all.py                   one command -> every deliverable
    ├── m4/config.py                 targets, fairness thresholds, feature list (all inherited from Modules 1-3)
    ├── m4/data.py                   loads the Module 3 hand-over; serving-time features = Module 3 definitions
    ├── m4/modeling.py               baselines, necessity review, 3 tuned algorithms, MLflow, registry
    ├── m4/xai_fairness.py           SHAP, counterfactuals (DiCE), Fairlearn audit + mitigation, sensitivity
    ├── m4/docs.py, m4/report.py     Model Card, Fairness Metrics Report, Model Validation Report (.docx)
    ├── api/main.py                  FastAPI POST /predict with tier, outreach action and reason codes
    ├── notebooks/shap_analysis.ipynb, notebooks/colab_run_module4.ipynb
    ├── tests/test_m4.py             7 unit tests (no patient data needed)
    └── outputs/                     written by run_all.py
```

## What builds on what
| Module | Carried into Module 4 as |
|---|---|
| 1 Vision | 70 % high-risk recall target; Low/Medium/High tiers and outreach actions served by the API; "a flag adds support, never reduces care"; ethical risk register (geography-as-penalty -> `area_code` excluded) |
| 2 Project Overview | Equal-opportunity release threshold (TPR gap <= 0.10; DI ratio is a diagnostic only); logistic-regression transparent baseline vs XGBoost (tickets S2-01..S2-05); SHAP for analysts + plain-language reason codes for OPD users; MLflow; Gate 3/Gate 4; RAID R-02/R-04/R-05; necessity review of Alcoholism / Handcap |
| 3 Data Pipeline | `data/processed/train|validation|test` used as-is (zero patient overlap asserted); Module 3 feature names (`lead_days`, `chronic_count`, `prior_visits`, `first_visit` ...); age bands 0-5 ... 70+; bias flag "disability group 2 %" -> separate fairness check; open item "necessity review in Module 4" -> Table 2 of the report |

## Run it (recommended: Google Colab, `notebooks/colab_run_module4.ipynb`)
The Colab notebook uploads the repo + Kaggle file, runs Module 3 then Module 4, opens the MLflow UI and the
API docs for screenshots, executes the SHAP notebook and downloads an outputs zip.

Locally (Python 3.10-3.12; the laptop's Python 3.14 cannot install xgboost/shap):
```bash
cd jjushycsh-noshow-analytics
pip install -r module4/requirements.txt
export NOSHOW_HASH_SALT="<secret>"          # Windows: set NOSHOW_HASH_SALT=<secret>
python run_pipeline.py --skip-gx             # Module 3 -> data/processed + reports   (skip if already done)
cd module4
python run_all.py                            # add --quick for a 3-minute smoke test
mlflow ui --backend-store-uri sqlite:///mlflow.db        # http://127.0.0.1:5000 -> screenshots
uvicorn api.main:app --reload                            # http://127.0.0.1:8000/docs -> POST /predict
python -m api.test_api
pytest -q tests
jupyter notebook notebooks/shap_analysis.ipynb           # Run All, save with outputs
```
Other inputs: `python run_all.py --data ../data/processed/appointments_model_table.csv` (single Module 3 table,
re-split with the Module 3 code and seed) or `--data path/to/KaggleV2-May-2016.csv` (runs the Module 3 pipeline first).

## Deliverables map
| Required | File |
|---|---|
| Model Validation Report (1,800-2,200 words) | `outputs/BAN6800_Module4_Model_Validation_Report_Abdi_Farah_Ahmed.docx` |
| SHAP analysis notebook | `notebooks/shap_analysis.ipynb` (executed) |
| Fairness Metrics Report | `outputs/fairness_report.md` (+ Appendix B of the report) |
| Model Card | `outputs/model_card.md` (+ Appendix A) |
| MLflow tracking + registry | screenshots + `outputs/mlflow_runs_export.csv`; model `jjushycsh_noshow_risk`, alias `champion`, stage Staging |
| Serialized model | `outputs/model/noshow_model.joblib` |
| FastAPI /predict | `api/main.py` |
| Counterfactuals / sensitivity | `outputs/counterfactuals*.csv`, `outputs/sensitivity_analysis.csv` |
| Results (every number in the report) | `outputs/results.json` |

## Finish the report
Everything highlighted **yellow** in the .docx needs you: two short personal-insight paragraphs (your DHIS2 /
Somali Region experience), the three screenshots and the GitHub link. `run_all.py` prints the body word count;
keep the total (with your paragraphs) inside 1,800-2,200 and read every numbers-driven sentence against the tables.

## Design choices (all documented in the report)
- Module 3 split reused: train = tuning (StratifiedGroupKFold by `patient_key`), validation = champion choice and
  thresholds (70 % / 90 % recall), test = reported once.
- Champion chosen on validation PR-AUC, never on test.
- Necessity review: a sensitive flag stays an input only if dropping it costs > 0.005 CV PR-AUC; a dropped flag is
  also removed from `chronic_count` (data minimisation, Proclamation 1321/2024). All flags stay in the fairness audit.
- Excluded: `sms_received` (intervention leakage), `area_code` (geography-as-penalty), `gender` (audit only),
  `appt_month` (calendar proxy), `lead_band` (redundant with `lead_days`).
- Mitigation only if the largest TPR gap > 0.10: group thresholds chosen on validation; Fairlearn ThresholdOptimizer cross-check.
- Forward-in-time check (earliest 80 % vs latest 20 % of dates) as the drift signal for the monitoring plan.
