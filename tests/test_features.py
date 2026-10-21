import pandas as pd

from scripts.make_sample_data import make
from src.data_prep import anonymize, clean, integrate
from src.features import build_features


def _features():
    return build_features.run(anonymize.run(clean.run(make(6000).astype(str))))


def test_lead_days_non_negative():
    assert (_features()["lead_days"] >= 0).all()


def test_history_has_no_leakage():
    df = _features()
    # the first visit of every patient must have zero prior history
    first = df[df["prior_visits"] == 0]
    assert (first["prior_no_shows"] == 0).all()
    assert (df["prior_no_shows"] <= df["prior_visits"]).all()


def test_patient_grouped_split_has_no_overlap():
    parts = integrate.split(_features())
    train = set(parts["train"]["patient_key"]); test = set(parts["test"]["patient_key"])
    assert train.isdisjoint(test)
