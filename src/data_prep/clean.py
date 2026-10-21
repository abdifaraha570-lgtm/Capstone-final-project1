"""Stage 2 - Cleaning: types, target, duplicates, invalid values, date logic.

Every rule records how many rows it touched in a data-quality report, so the
choices made here are visible in the slides and reproducible later.
"""
import json

import numpy as np
import pandas as pd

from src import config
from src.utils.audit_log import audited

RENAME = {
    "PatientId": "patient_id", "AppointmentID": "appointment_id", "Gender": "gender",
    "ScheduledDay": "scheduled_ts", "AppointmentDay": "appointment_ts", "Age": "age",
    "Neighbourhood": "neighbourhood", "Scholarship": "scholarship",
    "Hipertension": "hypertension", "Diabetes": "diabetes", "Alcoholism": "alcoholism",
    "Handcap": "handicap_level", "SMS_received": "sms_received", "No-show": "no_show",
}
BINARY = ["scholarship", "hypertension", "diabetes", "alcoholism", "sms_received"]

quality_report: dict = {}


def _log(rule: str, n: int, action: str) -> None:
    quality_report[rule] = {"rows_affected": int(n), "action": action}


@audited("TRANSFORM", "appointments_raw", "standardise names and types")
def standardise(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=RENAME).copy()
    _log("missing_values_any_column", df.isna().any(axis=1).sum(), "counted; rows with a missing target are dropped")
    # IDs are read as text then parsed, so 15-digit PatientIds are not rounded by float conversion
    df["patient_id"] = pd.to_numeric(df["patient_id"], errors="coerce").round().astype("Int64").astype(str)
    df["appointment_id"] = pd.to_numeric(df["appointment_id"], errors="coerce").astype("Int64")
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["handicap_level"] = pd.to_numeric(df["handicap_level"], errors="coerce").fillna(0).astype(int)
    for c in BINARY:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int).clip(0, 1)
    df["gender"] = df["gender"].str.strip().str.upper()
    df["neighbourhood"] = df["neighbourhood"].str.strip().str.upper()
    # Timestamps are UTC ISO strings; scheduled has a time, appointment is date-only
    df["scheduled_ts"] = pd.to_datetime(df["scheduled_ts"], utc=True, errors="coerce")
    df["appointment_ts"] = pd.to_datetime(df["appointment_ts"], utc=True, errors="coerce")
    bad_dates = df["scheduled_ts"].isna() | df["appointment_ts"].isna()
    _log("unparseable_dates", bad_dates.sum(), "dropped")
    df = df[~bad_dates]
    # Target: "Yes" in the source means the patient did NOT attend -> no_show = 1
    df["no_show"] = df["no_show"].str.strip().str.lower().map({"yes": 1, "no": 0})
    missing_target = df["no_show"].isna()
    _log("missing_or_invalid_target", missing_target.sum(), "dropped")
    return df[~missing_target].astype({"no_show": int})


@audited("TRANSFORM", "appointments_raw", "remove duplicates")
def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    dup_id = df.duplicated("appointment_id", keep="first")
    _log("duplicate_appointment_id", dup_id.sum(), "dropped (keep first)")
    df = df[~dup_id]
    # Rows identical in every field except AppointmentID are double entries.
    # (Same patient + same booking second but a DIFFERENT outcome = two real visits, so kept.)
    key = [c for c in df.columns if c != "appointment_id"]
    dup_entry = df.duplicated(key, keep="first")
    _log("identical_rows_except_appointment_id", dup_entry.sum(), "dropped (keep first)")
    same_booking = df[~dup_entry].duplicated(["patient_id", "scheduled_ts", "appointment_ts"], keep=False)
    _log("same_patient_same_booking_time_different_details", same_booking.sum(), "kept - separate visits; flagged for HIS review")
    return df[~dup_entry]


@audited("TRANSFORM", "appointments_raw", "fix invalid values")
def fix_invalid(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    bad_age = (df["age"] < 0) | (df["age"] > 110) | df["age"].isna()
    _log("age_out_of_range_(<0_or_>110)", bad_age.sum(), "dropped - cannot be a valid patient age")
    df = df[~bad_age]
    bad_gender = ~df["gender"].isin(["F", "M"])
    _log("gender_not_F_or_M", bad_gender.sum(), "set to 'U' (unknown), kept")
    df.loc[bad_gender, "gender"] = "U"
    # Lead time is measured on calendar days: appointment_ts carries no clock time
    df["scheduled_date"] = df["scheduled_ts"].dt.tz_convert(None).dt.normalize()
    df["appointment_date"] = df["appointment_ts"].dt.tz_convert(None).dt.normalize()
    negative = df["scheduled_date"] > df["appointment_date"]
    _log("scheduled_after_appointment", negative.sum(), "dropped - impossible sequence")
    df = df[~negative]
    multi = df["handicap_level"] > 1
    _log("handicap_level_above_1", multi.sum(), "kept; binary has_disability flag derived")
    df["has_disability"] = (df["handicap_level"] > 0).astype(int)
    return df.drop(columns=["handicap_level", "scheduled_ts", "appointment_ts"])


def run(df: pd.DataFrame) -> pd.DataFrame:
    quality_report.clear()
    quality_report["rows_received"] = {"rows_affected": len(df), "action": "input"}
    out = fix_invalid(deduplicate(standardise(df)))
    quality_report["rows_after_cleaning"] = {"rows_affected": len(out), "action": "output"}
    quality_report["no_show_rate_after_cleaning"] = {"rows_affected": round(float(out["no_show"].mean()), 4), "action": "share"}
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.REPORTS_DIR / "data_quality_report.json").write_text(json.dumps(quality_report, indent=2))
    return out
