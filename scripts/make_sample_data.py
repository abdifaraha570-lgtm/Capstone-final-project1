"""Create a small SYNTHETIC file with the exact Kaggle schema, for unit tests and CI.

It is NOT used for analysis: real runs use the Kaggle file (Hoppen, 2016) placed in
data/external/. Keeping a synthetic fixture means CI never needs patient data.
Deliberate defects are injected (bad age, duplicate ID, scheduled-after-appointment)
so the cleaning rules and validation suites have something to catch.
"""
import argparse

import numpy as np
import pandas as pd


def make(n: int = 6000, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    appt = pd.Timestamp("2016-04-29") + pd.to_timedelta(rng.integers(0, 40, n), unit="D")
    lead = rng.choice([0, 1, 2, 5, 10, 20, 40], n, p=[.35, .1, .1, .15, .12, .1, .08])
    sched = appt - pd.to_timedelta(lead, unit="D") + pd.to_timedelta(rng.integers(7, 18, n), unit="h")
    age = rng.integers(0, 100, n)
    sms = (rng.random(n) < .32).astype(int)
    p_ns = .12 + .12 * (lead > 3) + .05 * ((age > 12) & (age < 35))
    df = pd.DataFrame({
        "PatientId": (rng.integers(1_000, 1_000 + n // 2, n) * 1e7).astype(float),
        "AppointmentID": np.arange(5_600_000, 5_600_000 + n),
        "Gender": rng.choice(["F", "M"], n, p=[.65, .35]),
        "ScheduledDay": sched.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "AppointmentDay": appt.strftime("%Y-%m-%dT00:00:00Z"),
        "Age": age,
        "Neighbourhood": rng.choice([f"BAIRRO {i}" for i in range(60)], n),
        "Scholarship": (rng.random(n) < .1).astype(int),
        "Hipertension": (rng.random(n) < .2).astype(int),
        "Diabetes": (rng.random(n) < .07).astype(int),
        "Alcoholism": (rng.random(n) < .03).astype(int),
        "Handcap": rng.choice([0, 1, 2], n, p=[.978, .02, .002]),
        "SMS_received": sms,
        "No-show": np.where(rng.random(n) < p_ns, "Yes", "No"),
    })
    df.loc[0, "Age"] = -1                                            # invalid age
    df.loc[1, "AppointmentID"] = df.loc[2, "AppointmentID"]           # duplicate ID
    df.loc[3, "ScheduledDay"] = "2016-06-30T09:00:00Z"                # after appointment
    df.loc[3, "AppointmentDay"] = "2016-05-02T00:00:00Z"
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=6000)
    ap.add_argument("--out", default="tests/fixtures/sample_appointments.csv")
    a = ap.parse_args()
    make(a.rows).to_csv(a.out, index=False)
    print("wrote", a.out)
