import pandas as pd

from scripts.make_sample_data import make
from src.data_prep import anonymize, clean


def _anon(extra_pii=False):
    raw = make(6000).astype(str)
    cleaned = clean.run(raw)
    if extra_pii:
        # Simulate a future local JJUSHYCSH extract carrying direct identifiers
        try:
            from faker import Faker
            fake = Faker(); Faker.seed(1)
            cleaned["patient_name"] = [fake.name() for _ in range(len(cleaned))]
            cleaned["phone_number"] = [fake.msisdn() for _ in range(len(cleaned))]
        except ImportError:
            cleaned["patient_name"] = "Test Name"
            cleaned["phone_number"] = "0911000000"
    return anonymize.run(cleaned)


def test_direct_identifiers_removed():
    out = _anon()
    for col in ["patient_id", "appointment_id", "neighbourhood"]:
        assert col not in out.columns


def test_local_pii_fields_dropped():
    out = _anon(extra_pii=True)
    assert "patient_name" not in out.columns and "phone_number" not in out.columns


def test_pseudonym_is_stable_and_salted():
    a = anonymize.pseudonymise("123", salt="s1")
    assert a == anonymize.pseudonymise("123", salt="s1")
    assert a != anonymize.pseudonymise("123", salt="s2")
    assert len(a) == 16


def test_age_top_coded():
    assert _anon()["age"].max() <= 90
