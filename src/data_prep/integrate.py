"""Stage 5 - Integration & output: merge sources, write model table, splits and dashboard feed."""
import json

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from src import config
from src.utils.audit_log import write_event
from src.utils.io import save_table

MODEL_COLUMNS = [
    "appointment_key", "patient_key", "scheduled_date", "appointment_date",
    "gender", "age", "age_band", "area_code", "scholarship", "hypertension", "diabetes",
    "alcoholism", "has_disability", "chronic_count", "sms_received", "lead_days",
    "lead_band", "same_day", "appt_weekday", "appt_month", "prior_visits",
    "prior_no_shows", "prior_no_show_rate", "first_visit", "no_show",
]


def merge_outreach_log(df: pd.DataFrame, outreach: pd.DataFrame | None) -> pd.DataFrame:
    """Future local phase: LEFT JOIN the outreach log on appointment_key so every
    scheduled visit is kept even when no reminder was logged."""
    if outreach is None:
        return df
    before = len(df)
    out = df.merge(outreach, on="appointment_key", how="left", validate="one_to_one")
    assert len(out) == before, "outreach join must not add or drop appointments"
    return out


def split(df: pd.DataFrame) -> dict:
    """Patient-grouped split: one patient never appears in both train and test,
    otherwise their history would leak and inflate Module 4 recall."""
    gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=config.RANDOM_SEED)
    train_idx, temp_idx = next(gss.split(df, groups=df["patient_key"]))
    train, temp = df.iloc[train_idx], df.iloc[temp_idx]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=config.RANDOM_SEED)
    val_idx, test_idx = next(gss2.split(temp, groups=temp["patient_key"]))
    return {"train": train, "validation": temp.iloc[val_idx], "test": temp.iloc[test_idx]}


def dashboard_feed(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate-only view for the Module 5 dashboard / Regional Health Bureau."""
    return (df.groupby(["appointment_date", "age_band", "gender", "lead_band"], observed=True)
              .agg(scheduled=("no_show", "size"), missed=("no_show", "sum"))
              .reset_index()
              .assign(no_show_rate=lambda x: (x["missed"] / x["scheduled"]).round(4)))


def run(df: pd.DataFrame, outreach: pd.DataFrame | None = None) -> dict:
    df = merge_outreach_log(df, outreach)[MODEL_COLUMNS]
    paths = {"model_table": str(save_table(df, config.PROCESSED_DIR / "appointments_model_table"))}
    parts = split(df)
    summary = {}
    for name, part in parts.items():
        paths[name] = str(save_table(part, config.PROCESSED_DIR / f"{name}"))
        summary[name] = {"rows": len(part), "no_show_rate": round(float(part["no_show"].mean()), 4)}
    feed = dashboard_feed(df)
    paths["dashboard_feed"] = str(save_table(feed, config.PROCESSED_DIR / "dashboard_daily_noshow"))
    (config.REPORTS_DIR / "split_summary.json").write_text(json.dumps(summary, indent=2))
    write_event("WRITE", "processed", "model-ready outputs", rows_out=len(df), outputs=list(paths))
    return {"paths": paths, "splits": summary, "frames": parts}
