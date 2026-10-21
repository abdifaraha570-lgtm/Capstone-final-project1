"""Stage 3 - Anonymization: remove direct identifiers, generalise quasi-identifiers.

Plan (see docs/anonymization_plan.md):
  * Direct IDs (PatientId, AppointmentID) -> salted SHA-256 keys; originals dropped.
  * Local PII fields (names, phone, MRN, kebele, national ID) -> dropped on sight.
  * Neighbourhood (quasi-identifier) -> coded area; areas with < K patients -> "OTHER".
  * Age kept as integer for modelling but banded for reporting; ages >= 90 top-coded.
  * Dates reduced to calendar day (clock time already removed in cleaning).
"""
import hashlib

import pandas as pd

from src import config
from src.utils.audit_log import audited, write_event

K_ANONYMITY = 20


def pseudonymise(value: str, salt: str = config.HASH_SALT) -> str:
    return hashlib.sha256(f"{salt}|{value}".encode()).hexdigest()[:16]


@audited("ANONYMIZE", "appointments_clean", "remove and pseudonymise identifiers")
def anonymise(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    pii_present = [c for c in config.LOCAL_PII_FIELDS if c in df.columns]
    if pii_present:
        write_event("ANONYMIZE", "appointments_clean", "drop local PII fields", dropped_columns=pii_present)
    df = df.drop(columns=pii_present)

    df["patient_key"] = df["patient_id"].astype(str).map(pseudonymise)
    df["appointment_key"] = df["appointment_id"].astype(str).map(pseudonymise)
    df = df.drop(columns=["patient_id", "appointment_id"])

    # Generalise neighbourhood: small areas are grouped so no area identifies a few people
    patients_per_area = df.groupby("neighbourhood")["patient_key"].nunique()
    small = patients_per_area[patients_per_area < K_ANONYMITY].index
    codes = {n: f"AREA_{i:03d}" for i, n in enumerate(sorted(patients_per_area.index))}
    df["area_code"] = df["neighbourhood"].map(codes)
    df.loc[df["neighbourhood"].isin(small), "area_code"] = "AREA_OTHER"
    df = df.drop(columns=["neighbourhood"])
    write_event("ANONYMIZE", "appointments_clean", "generalise neighbourhood",
                areas=len(codes), areas_merged_small=len(small), k=K_ANONYMITY)

    df["age"] = df["age"].clip(upper=90).astype(int)       # top-code the very old
    return df


def check_no_identifiers(df: pd.DataFrame) -> None:
    forbidden = set(config.DIRECT_IDENTIFIERS) | set(config.LOCAL_PII_FIELDS) | {"patient_id", "appointment_id", "neighbourhood"}
    leaked = forbidden & set(df.columns)
    if leaked:
        raise AssertionError(f"Identifier columns leaked past anonymisation: {sorted(leaked)}")


def run(df: pd.DataFrame) -> pd.DataFrame:
    out = anonymise(df)
    check_no_identifiers(out)
    return out
