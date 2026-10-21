"""Loading the Module 3 hand-over and turning an API request into the same feature row.

Preferred input (exact continuity with Module 3): the three split files written by
`run_pipeline.py` - data/processed/train, validation, test (.parquet or .csv).
Fallbacks, in order:
  2. a single Module 3 model table  -> re-split with the Module 3 GroupShuffleSplit code and seed
  3. the raw Kaggle CSV             -> the Module 3 pipeline is imported and run (needs the repository),
                                       so the model never trains on data that skipped the Module 3 gates
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from m4 import config as C

M3_MIN_COLUMNS = {"patient_key", "appointment_date", "scheduled_date", "no_show", "lead_days",
                  "prior_visits", "prior_no_shows", "prior_no_show_rate", "first_visit",
                  "chronic_count", "has_disability", "age_band", "same_day", "appt_weekday"}


# ------------------------------------------------------------------ reading
def _read(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    for c in ("scheduled_date", "appointment_date"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


def _find(dir_: Path, stem: str) -> Path | None:
    for ext in (".parquet", ".csv"):
        p = dir_ / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def md5_of(paths) -> str:
    h = hashlib.md5()
    for p in paths:
        h.update(Path(p).read_bytes())
    return h.hexdigest()


def _coerce(df: pd.DataFrame) -> pd.DataFrame:
    """Types exactly as Module 4 expects them (Module 3 already guarantees the values)."""
    df = df.copy()
    for c in C.NUMERIC:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    for c in C.BINARY:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    df[C.TARGET] = pd.to_numeric(df[C.TARGET], errors="coerce").astype(int)
    for c in C.CATEGORICAL + [p for p in C.PROTECTED if p not in C.BINARY]:   # gender, age_band, weekday
        df[c] = df[c].astype(str)
    return df.dropna(subset=C.NUMERIC).reset_index(drop=True)


def load_m3_splits(processed_dir: Path):
    """Route 1: the Module 3 train / validation / test files."""
    files = {s: _find(processed_dir, s) for s in C.SPLIT_FILES}
    if not all(files.values()):
        return None
    parts = {s: _coerce(_read(p)) for s, p in files.items()}
    missing = M3_MIN_COLUMNS - set(parts["train"].columns)
    if missing:
        raise ValueError(f"Module 3 split files are missing columns {sorted(missing)}")
    info = {"route": "module3_splits", "files": {s: str(p) for s, p in files.items()},
            "data_md5": md5_of(files.values())}
    return parts, info


def group_split(df: pd.DataFrame) -> dict:
    """Route 2: identical to Module 3 src/data_prep/integrate.py::split (patient-grouped 70/15/15, seed 42)."""
    gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=C.RANDOM_STATE)
    tr, tmp = next(gss.split(df, groups=df[C.GROUP_KEY]))
    train, temp = df.iloc[tr], df.iloc[tmp]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=C.RANDOM_STATE)
    va, te = next(gss2.split(temp, groups=temp[C.GROUP_KEY]))
    return {"train": train.reset_index(drop=True), "validation": temp.iloc[va].reset_index(drop=True),
            "test": temp.iloc[te].reset_index(drop=True)}


def load_model_table(path: Path):
    df = _read(path)
    if M3_MIN_COLUMNS <= set(df.columns):
        parts = group_split(_coerce(df))
        return parts, {"route": "module3_model_table_resplit", "files": {"model_table": str(path)},
                       "data_md5": md5_of([path])}
    if "No-show" in df.columns or "no-show" in {c.lower() for c in df.columns}:
        return run_module3_pipeline(path)
    raise ValueError(f"{path} is neither a Module 3 model table nor the raw Kaggle file "
                     f"(columns: {list(df.columns)[:8]}...)")


def run_module3_pipeline(raw_csv: Path):
    """Route 3: raw file -> the actual Module 3 code (ingest, clean, anonymise, features, gate, split)."""
    repo = C.REPO_ROOT
    if not (repo / "run_pipeline.py").exists():
        raise FileNotFoundError(
            f"Raw Kaggle file given but the Module 3 repository was not found at {repo}. "
            "Place module4/ inside jjushycsh-noshow-analytics/ (or set PIPELINE_ROOT) so the Module 3 "
            "pipeline can prepare the data, or pass the Module 3 processed directory with --processed-dir.")
    import os, shutil
    os.environ.setdefault("NOSHOW_HASH_SALT", "dev-only-salt-change-me")
    os.environ["PIPELINE_ROOT"] = str(repo)
    ext = repo / "data" / "external"
    ext.mkdir(parents=True, exist_ok=True)
    if Path(raw_csv).resolve() != (ext / "KaggleV2-May-2016.csv").resolve():
        shutil.copy(raw_csv, ext / "KaggleV2-May-2016.csv")
    sys.path.insert(0, str(repo))
    import run_pipeline                                     # Module 3 code
    run_pipeline.main(skip_gx=True)
    parts, info = load_m3_splits(repo / "data" / "processed")
    info["route"] = "raw_csv_via_module3_pipeline"
    return parts, info


def load_inputs(processed_dir: Path | None = None, data_file: Path | None = None):
    """Returns ({'train','validation','test'}, info)."""
    if data_file:
        return load_model_table(Path(data_file))
    processed_dir = Path(processed_dir or C.PROCESSED_DIR)
    got = load_m3_splits(processed_dir)
    if got:
        return got
    mt = _find(processed_dir, "appointments_model_table")
    if mt:
        return load_model_table(mt)
    raise FileNotFoundError(
        f"No Module 3 outputs in {processed_dir}. Run the Module 3 pipeline first "
        "(python run_pipeline.py --skip-gx) or pass --data <file>.")


def load_m3_reports(reports_dir: Path | None = None) -> dict:
    """Module 3 evidence carried forward into the Module 4 report (bias flags, quality report, split summary)."""
    d = Path(reports_dir or C.M3_REPORTS_DIR)
    out = {}
    for name in ("bias_flags", "data_quality_report", "split_summary"):
        p = d / f"{name}.json"
        if p.exists():
            try:
                out[name] = json.loads(p.read_text())
            except json.JSONDecodeError:
                pass
    p = d / "bias_representation.csv"
    if p.exists():
        out["bias_representation"] = pd.read_csv(p).to_dict(orient="records")
    return out


def apply_minimisation(df: pd.DataFrame, dropped) -> pd.DataFrame:
    """Data minimisation after the necessity review: a dropped sensitive flag must not survive inside
    chronic_count either, so the count is rebuilt from the remaining Module 3 components."""
    comps = [c for c in C.CHRONIC_COMPONENTS if c not in dropped]
    if len(comps) < len(C.CHRONIC_COMPONENTS):
        df = df.copy()
        df["chronic_count"] = df[comps].sum(axis=1).astype(float)
    return df


# ------------------------------------------------------------------ serving
def age_band(age: float) -> str:
    """Same bins as Module 3 build_features.AGE_BINS."""
    return str(pd.cut([age], [-1, 5, 17, 34, 54, 69, 200], labels=C.AGE_BANDS)[0])


def features_from_request(req: dict, dropped=()) -> pd.DataFrame:
    """One-row feature frame from an API request, built with the Module 3 feature definitions
    so serving uses exactly the representation the model was trained on (no training/serving skew)."""
    sched = pd.Timestamp(req["scheduled_date"]).tz_localize(None) if pd.Timestamp(req["scheduled_date"]).tzinfo \
        else pd.Timestamp(req["scheduled_date"])
    appt = pd.Timestamp(req["appointment_date"]).tz_localize(None) if pd.Timestamp(req["appointment_date"]).tzinfo \
        else pd.Timestamp(req["appointment_date"])
    lead = int((appt.normalize() - sched.normalize()).days)
    if lead < 0:
        raise ValueError("appointment_date is before scheduled_date")
    prior_visits = int(req.get("prior_visits", 0))
    prior_no_shows = int(req.get("prior_no_shows", 0))
    if prior_no_shows > prior_visits:
        raise ValueError("prior_no_shows cannot exceed prior_visits")
    flags = {k: int(bool(int(req.get(k, 0)))) for k in ["scholarship", "hypertension", "diabetes", "alcoholism"]}
    has_dis = int(int(req.get("has_disability", req.get("handicap", 0))) > 0)
    row = {
        "age": float(min(max(float(req["age"]), 0), 90)),      # Module 3 top-codes age at 90
        "lead_days": float(lead),
        "prior_visits": float(prior_visits), "prior_no_shows": float(prior_no_shows),
        "prior_no_show_rate": (prior_no_shows / prior_visits) if prior_visits else 0.0,
        "chronic_count": float(sum(flags[c] for c in C.CHRONIC_COMPONENTS if c not in dropped)),
        **flags, "has_disability": has_dis,
        "same_day": int(lead == 0), "first_visit": int(prior_visits == 0),
        "appt_weekday": appt.day_name(),
    }
    return pd.DataFrame([row])
