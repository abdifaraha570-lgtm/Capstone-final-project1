"""Module 4 unit tests (run from module4/:  pytest -q tests).
They use a small synthetic Module-3-style table, so no patient data is needed."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from m4 import config as C                     # noqa: E402
from m4 import data as D                       # noqa: E402
from m4.modeling import threshold_for_recall, metrics_at, make_preprocessor   # noqa: E402
from m4.xai_fairness import group_table, summarise   # noqa: E402


def synthetic(n=3000, seed=1):
    rng = np.random.default_rng(seed)
    appt = pd.Timestamp("2016-05-01") + pd.to_timedelta(rng.integers(0, 40, n), unit="D")
    lead = rng.integers(0, 30, n)
    df = pd.DataFrame({
        "patient_key": [f"p{i:05d}" for i in rng.integers(0, n // 2, n)],
        "appointment_key": [f"a{i:05d}" for i in range(n)],
        "scheduled_date": appt - pd.to_timedelta(lead, unit="D"), "appointment_date": appt,
        "gender": rng.choice(["F", "M"], n), "age": rng.integers(0, 90, n),
        "scholarship": rng.integers(0, 2, n), "hypertension": rng.integers(0, 2, n),
        "diabetes": rng.integers(0, 2, n), "alcoholism": rng.integers(0, 2, n),
        "has_disability": (rng.random(n) < 0.03).astype(int), "sms_received": rng.integers(0, 2, n),
        "lead_days": lead, "same_day": (lead == 0).astype(int),
        "appt_weekday": appt.day_name(), "appt_month": appt.month,
        "prior_visits": rng.integers(0, 5, n), "prior_no_shows": 0,
        "no_show": (rng.random(n) < 0.2).astype(int),
    })
    df["age_band"] = pd.cut(df["age"], [-1, 5, 17, 34, 54, 69, 200], labels=C.AGE_BANDS).astype(str)
    df["chronic_count"] = df[["hypertension", "diabetes", "alcoholism"]].sum(axis=1)
    df["prior_no_shows"] = (df["prior_visits"] * rng.random(n)).astype(int)
    df["prior_no_show_rate"] = np.where(df["prior_visits"] > 0, df["prior_no_shows"] / df["prior_visits"].replace(0, 1), 0)
    df["first_visit"] = (df["prior_visits"] == 0).astype(int)
    df["lead_band"] = "x"; df["area_code"] = "AREA_001"
    return df


def test_group_split_has_no_patient_overlap():
    parts = D.group_split(D._coerce(synthetic()))
    assert not set(parts["train"]["patient_key"]) & set(parts["test"]["patient_key"])
    assert not set(parts["train"]["patient_key"]) & set(parts["validation"]["patient_key"])
    assert abs(len(parts["train"]) / 3000 - 0.70) < 0.05


def test_request_features_match_module3_row():
    row = D._coerce(synthetic()).iloc[7]
    req = {"age": row.age, "scheduled_date": row.scheduled_date, "appointment_date": row.appointment_date,
           "scholarship": row.scholarship, "hypertension": row.hypertension, "diabetes": row.diabetes,
           "alcoholism": row.alcoholism, "has_disability": row.has_disability,
           "prior_visits": row.prior_visits, "prior_no_shows": row.prior_no_shows}
    X = D.features_from_request(req)
    for c in C.FEATURES:
        assert str(X[c].iloc[0]) == str(row[c]) or float(X[c].iloc[0]) == float(row[c]), c


def test_minimisation_rebuilds_chronic_count():
    df = D._coerce(synthetic())
    out = D.apply_minimisation(df, ["alcoholism"])
    assert (out["chronic_count"] == df["hypertension"] + df["diabetes"]).all()
    X = D.features_from_request({"age": 40, "scheduled_date": "2026-10-01", "appointment_date": "2026-10-05",
                                 "hypertension": 1, "alcoholism": 1}, dropped=("alcoholism",))
    assert X["chronic_count"].iloc[0] == 1.0


def test_request_rejects_impossible_inputs():
    with pytest.raises(ValueError):
        D.features_from_request({"age": 30, "scheduled_date": "2026-10-10", "appointment_date": "2026-10-01"})
    with pytest.raises(ValueError):
        D.features_from_request({"age": 30, "scheduled_date": "2026-10-01", "appointment_date": "2026-10-02",
                                 "prior_visits": 1, "prior_no_shows": 2})


def test_threshold_reaches_target_recall():
    rng = np.random.default_rng(0)
    y = (rng.random(5000) < 0.2).astype(int)
    p = np.clip(0.2 + 0.3 * y + rng.normal(0, 0.2, 5000), 0, 1)
    t = threshold_for_recall(y, p, 0.70)
    assert metrics_at(y, p, t)["recall"] >= 0.70


def test_preprocessor_outputs_expected_columns():
    df = D._coerce(synthetic())
    prep = make_preprocessor(C.FEATURES).fit(df[C.FEATURES])
    names = list(prep.get_feature_names_out())
    assert "age" in names and any(n.startswith("appt_weekday_") for n in names)
    assert "gender" not in names and "sms_received" not in names and "area_code" not in names


def test_fairness_summary_bounds():
    y = np.array([1, 0, 1, 0, 1, 0, 1, 0]); pred = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    g = np.array(["F", "F", "F", "F", "M", "M", "M", "M"])
    s = summarise(group_table(y, pred, g))
    assert 0 <= s["di_ratio"] <= 1 and 0 <= s["tpr_gap"] <= 1
