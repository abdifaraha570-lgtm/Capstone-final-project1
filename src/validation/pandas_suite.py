"""Pandas validator that runs the SAME checks as the Great Expectations suites.

Used when Great Expectations cannot be installed (e.g. a locked-down laptop).
Writes reports/gx_<suite>.json in the same format, so the results can be rendered
and compared with a later Great Expectations run.
"""
import json
import re

import pandas as pd

from src import config
from src.utils.audit_log import write_event

BINARY_RAW = ["Scholarship", "Hipertension", "Diabetes", "Alcoholism", "SMS_received"]
BINARY_PROCESSED = ["scholarship", "hypertension", "diabetes", "alcoholism",
                    "sms_received", "has_disability", "same_day", "first_visit", "no_show"]
ISO = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$"


def _row(name, column, bad, n, mostly=1.0, observed=None):
    ok = (1 - bad / n) >= mostly if n else False
    return {"expectation": name, "column": column, "success": bool(ok),
            "unexpected_count": int(bad), "observed_value": observed}


def raw_checks(df):
    n, out = len(df), []
    num = {c: pd.to_numeric(df[c], errors="coerce") for c in ["Age", "Handcap"] + BINARY_RAW}
    out.append(_row("expect_table_row_count_to_be_between", "(table)", 0 if n >= config.MIN_ROWS else n, n, observed=n))
    out.append(_row("expect_table_columns_to_match_ordered_list", "(table)", 0 if list(df.columns) == config.RAW_COLUMNS else 1, 1))
    out.append(_row("expect_column_values_to_be_unique", "AppointmentID", df["AppointmentID"].duplicated().sum(), n))
    out.append(_row("expect_column_values_to_be_in_set", "Gender", (~df["Gender"].isin(["F", "M"])).sum(), n))
    out.append(_row("expect_column_values_to_be_in_set", "No-show", (~df["No-show"].isin(["Yes", "No"])).sum(), n))
    out.append(_row("expect_column_values_to_be_between", "Age", (~num["Age"].between(0, 110)).sum(), n, mostly=0.999))
    out.append(_row("expect_column_values_to_be_between", "Handcap", (~num["Handcap"].between(0, 4)).sum(), n))
    for c in ["ScheduledDay", "AppointmentDay"]:
        out.append(_row("expect_column_values_to_match_regex", c, (~df[c].astype(str).str.match(ISO)).sum(), n))
    for c in config.RAW_COLUMNS:
        out.append(_row("expect_column_values_to_not_be_null", c, df[c].isna().sum(), n))
    for c in BINARY_RAW:
        out.append(_row("expect_column_values_to_be_in_set", c, (~num[c].isin([0, 1])).sum(), n))
    return out


def processed_checks(df):
    from src.data_prep.integrate import MODEL_COLUMNS
    n, out = len(df), []
    out.append(_row("expect_table_row_count_to_be_between", "(table)", 0 if n >= config.MIN_ROWS else n, n, observed=n))
    out.append(_row("expect_column_values_to_be_unique", "appointment_key", df["appointment_key"].duplicated().sum(), n))
    out.append(_row("expect_column_values_to_match_regex", "patient_key",
                    (~df["patient_key"].astype(str).str.match(r"^[0-9a-f]{16}$")).sum(), n))
    out.append(_row("expect_column_values_to_be_between", "age", (~df["age"].between(0, 90)).sum(), n))
    out.append(_row("expect_column_values_to_be_between", "lead_days", (~df["lead_days"].between(0, 365)).sum(), n))
    out.append(_row("expect_column_values_to_be_between", "prior_no_show_rate", (~df["prior_no_show_rate"].between(0, 1)).sum(), n))
    out.append(_row("expect_column_values_to_be_in_set", "gender", (~df["gender"].isin(["F", "M", "U"])).sum(), n))
    mean = float(df["no_show"].mean())
    out.append(_row("expect_column_mean_to_be_between", "no_show", 0 if 0.05 <= mean <= 0.40 else 1, 1, observed=round(mean, 4)))
    out.append(_row("expect_table_columns_to_match_set", "(table)", 0 if set(df.columns) == set(MODEL_COLUMNS) else 1, 1))
    for c in ["patient_key", "appointment_key", "no_show", "lead_days", "age_band"]:
        out.append(_row("expect_column_values_to_not_be_null", c, df[c].isna().sum(), n))
    for c in BINARY_PROCESSED:
        out.append(_row("expect_column_values_to_be_in_set", c, (~df[c].isin([0, 1])).sum(), n))
    return out


def validate(df: pd.DataFrame, suite_name: str) -> dict:
    details = raw_checks(df) if suite_name.startswith("raw") else processed_checks(df)
    stats = {"suite": suite_name, "engine": "pandas", "success": all(d["success"] for d in details),
             "expectations": len(details), "passed": sum(d["success"] for d in details), "details": details}
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.REPORTS_DIR / f"gx_{suite_name}.json").write_text(json.dumps(stats, indent=2, default=str))
    write_event("VALIDATE", suite_name, "quality gate (pandas validator)",
                success=stats["success"], passed=stats["passed"], expectations=stats["expectations"])
    return stats
