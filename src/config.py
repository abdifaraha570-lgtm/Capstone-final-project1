"""Central configuration for the JJUSHYCSH no-show data pipeline (Sprint 1 / Module 3)."""
import os
from pathlib import Path

ROOT = Path(os.getenv("PIPELINE_ROOT", Path(__file__).resolve().parents[1]))

RAW_FILE = ROOT / "data" / "external" / os.getenv("RAW_FILENAME", "KaggleV2-May-2016.csv")
RAW_SNAPSHOT_DIR = ROOT / "data" / "raw"          # immutable, checksummed copy (restricted tier)
INTERIM_DIR = ROOT / "data" / "interim"           # cleaned + pseudonymized (analyst tier)
PROCESSED_DIR = ROOT / "data" / "processed"       # model-ready feature table + splits
REPORTS_DIR = ROOT / "reports"
LOG_DIR = ROOT / "logs"
AUDIT_LOG = LOG_DIR / "privacy_audit.jsonl"

# Salt for pseudonymous keys. Must come from a secret store / env var, never from Git.
HASH_SALT = os.getenv("NOSHOW_HASH_SALT", "dev-only-salt-change-me")

RANDOM_SEED = 42
MIN_ROWS = 5000

RAW_COLUMNS = [
    "PatientId", "AppointmentID", "Gender", "ScheduledDay", "AppointmentDay", "Age",
    "Neighbourhood", "Scholarship", "Hipertension", "Diabetes", "Alcoholism",
    "Handcap", "SMS_received", "No-show",
]

# Direct identifiers that must never leave the restricted tier.
DIRECT_IDENTIFIERS = ["PatientId", "AppointmentID"]
# Fields a future local JJUSHYCSH extract may carry and that must be dropped on ingest.
LOCAL_PII_FIELDS = ["patient_name", "phone_number", "mrn", "kebele", "national_id", "address"]

# Groups monitored for representation bias (Module 1 fairness objectives).
BIAS_GROUPS = ["gender", "age_band", "scholarship", "has_disability"]
MIN_GROUP_SHARE = 0.05       # flag a group below 5% of records
MAX_RATE_GAP = 0.10          # flag label-rate gap above 10 percentage points
