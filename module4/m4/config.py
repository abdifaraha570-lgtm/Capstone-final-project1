"""Central configuration for the BAN6800 Module 4 increment (JJUSHYCSH referral no-show risk model).

Everything here is inherited from, or decided in continuity with, Modules 1-3:
  * Module 1 Vision Document  -> success criteria (>= 70 % high-risk recall), "a flag can add support, never reduce care"
  * Module 2 Project Overview -> fairness objectives, protected attributes, explainability framework
  * Module 3 Data Pipeline    -> the clean, pseudonymised model table, the patient-grouped 70/15/15 split,
                                 the bias-detection flags and the open item "necessity review of sensitive fields"
Edit the values below only if your Module 1/2 documents set different targets.
"""
import os
from pathlib import Path

AUTHOR = "Abdi Farah Ahmed"
UNIVERSITY = "Nexford University"

M4_ROOT = Path(__file__).resolve().parents[1]                 # .../module4
REPO_ROOT = Path(os.getenv("PIPELINE_ROOT", M4_ROOT.parent))  # the jjushycsh-noshow-analytics repository

# ---- Module 3 hand-over (inputs) ----
PROCESSED_DIR = REPO_ROOT / "data" / "processed"   # appointments_model_table + train/validation/test from Module 3
M3_REPORTS_DIR = REPO_ROOT / "reports"             # bias_flags.json, data_quality_report.json, split_summary.json
SPLIT_FILES = ("train", "validation", "test")

# ---- Module 4 outputs ----
OUT = M4_ROOT / "outputs"
FIG = OUT / "figures"
MODEL_DIR = OUT / "model"

# ---- Experiment tracking ----
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{(M4_ROOT / 'mlflow.db').as_posix()}")
EXPERIMENT = "jjushycsh-noshow-m4"
REGISTERED_MODEL = "jjushycsh_noshow_risk"

# ---- Protocol ----
RANDOM_STATE = 42          # same seed as Module 3
CV_FOLDS = 3               # StratifiedGroupKFold (grouped by patient_key) inside the Module 3 training split
N_ITER = 12                # random-search candidates per algorithm (--quick uses 3)

# ---- Business success criteria (Module 1 Vision Document) ----
TARGET_RECALL = 0.70       # High tier must capture >= 70 % of true no-shows
MEDIUM_RECALL = 0.90       # Medium tier boundary
MIN_GAIN_OVER_RULE = 0.02  # champion must beat the transparent lead-time rule by >= 0.02 ROC-AUC

# ---- Fairness criteria (Module 2 Project Overview, Section 6 and KPI table) ----
EOD_MAX = 0.10             # RELEASE THRESHOLD: absolute TPR gap (max-min) <= 0.10 across monitored groups;
                           # FPR gap reported alongside (equalized odds), Hardt et al. (2016)
DI_MIN = 0.80              # DIAGNOSTIC ONLY (Module 2: selection rates are "not sufficient evidence of fairness"):
                           # four-fifths rule on selection-rate ratio, reported, not a release gate

# ---- Feature necessity review (open item handed over from Module 3, slide 10) ----
SENSITIVE_HEALTH_FLAGS = ["alcoholism", "has_disability"]
CHRONIC_COMPONENTS = ["hypertension", "diabetes", "alcoholism"]   # Module 3: chronic_count = sum of these;
                                                                  # a dropped flag is also removed from the count
NECESSITY_TOLERANCE = 0.005   # keep a sensitive flag only if dropping it costs > 0.005 CV PR-AUC

# ---- Feature set (column names exactly as produced by Module 3 build_features.py) ----
NUMERIC = ["age", "lead_days", "prior_visits", "prior_no_shows", "prior_no_show_rate", "chronic_count"]
BINARY = ["scholarship", "hypertension", "diabetes", "alcoholism", "has_disability", "same_day", "first_visit"]
CATEGORICAL = ["appt_weekday"]
FEATURES = NUMERIC + BINARY + CATEGORICAL
TARGET = "no_show"
GROUP_KEY = "patient_key"
DATE_COL = "appointment_date"

# Audited for fairness (Module 3 BIAS_GROUPS). gender is deliberately NOT a model input.
PROTECTED = ["gender", "age_band", "scholarship", "has_disability"]
AGE_BANDS = ["0-5", "6-17", "18-34", "35-54", "55-69", "70+"]   # Module 3 bands

# Columns in the Module 3 model table that are excluded from the model on purpose:
EXCLUDED = {
    "sms_received": "Intervention leakage: the SMS is the outreach the model is meant to target.",
    "area_code": "Geography-as-penalty risk (Module 1 risk register); k-anonymised area is kept for audit only.",
    "gender": "No outreach rationale; retained only for the fairness audit.",
    "appt_month": "Only ~6 weeks of data: month is a calendar proxy, not a patient characteristic.",
    "lead_band": "Redundant with lead_days (kept for the dashboard, not the model).",
    "patient_key / appointment_key / dates": "Identifiers and raw dates carry no predictive meaning.",
}

# Module 1 prescriptive rule (served by the API)
ACTIONS = {
    "High": "Phone call 48 h before + community health worker follow-up; offer an earlier slot if available",
    "Medium": "SMS reminder + phone call 24-48 h before",
    "Low": "Standard SMS reminder",
}
