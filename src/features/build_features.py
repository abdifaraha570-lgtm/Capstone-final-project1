"""Stage 4 - Transformation / feature engineering (tests the Module 2 RCA hypotheses)."""
import numpy as np
import pandas as pd

from src.utils.audit_log import audited

AGE_BINS = [-1, 5, 17, 34, 54, 69, 200]
AGE_LABELS = ["0-5", "6-17", "18-34", "35-54", "55-69", "70+"]
LEAD_BINS = [-1, 0, 3, 7, 14, 30, 10_000]
LEAD_LABELS = ["same_day", "1-3d", "4-7d", "8-14d", "15-30d", "31d+"]


@audited("TRANSFORM", "appointments_anon", "engineer scheduling and history features")
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Scheduling-lag hypothesis
    df["lead_days"] = (df["appointment_date"] - df["scheduled_date"]).dt.days
    df["lead_band"] = pd.cut(df["lead_days"], LEAD_BINS, labels=LEAD_LABELS).astype(str)
    df["same_day"] = (df["lead_days"] == 0).astype(int)
    df["appt_weekday"] = df["appointment_date"].dt.day_name()
    df["appt_month"] = df["appointment_date"].dt.month
    # Age bands are used for fairness auditing and reporting
    df["age_band"] = pd.cut(df["age"], AGE_BINS, labels=AGE_LABELS).astype(str)
    df["chronic_count"] = df[["hypertension", "diabetes", "alcoholism"]].sum(axis=1)
    return df


@audited("TRANSFORM", "appointments_anon", "integrate patient visit history (leakage-safe)")
def add_history(df: pd.DataFrame) -> pd.DataFrame:
    """Self-join on patient_key: each visit only sees visits booked BEFORE it,
    so the history features never use the outcome being predicted."""
    df = df.sort_values(["patient_key", "appointment_date", "scheduled_date", "appointment_key"]).copy()
    g = df.groupby("patient_key", sort=False)
    df["prior_visits"] = g.cumcount()
    df["prior_no_shows"] = g["no_show"].cumsum() - df["no_show"]
    df["prior_no_show_rate"] = np.where(df["prior_visits"] > 0,
                                        df["prior_no_shows"] / df["prior_visits"].replace(0, np.nan), 0.0)
    df["first_visit"] = (df["prior_visits"] == 0).astype(int)
    return df.reset_index(drop=True)


def run(df: pd.DataFrame) -> pd.DataFrame:
    return add_history(add_features(df))
