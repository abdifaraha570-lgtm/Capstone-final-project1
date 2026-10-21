"""Data governance: role-based access tiers enforced in code (see docs/data_governance.md)."""
from src.utils.audit_log import write_event

# tier -> roles allowed to read it
ACCESS_POLICY = {
    "restricted_raw": {"his_data_officer", "data_engineer"},
    "pseudonymised":  {"his_data_officer", "data_engineer", "analyst"},
    "aggregate":      {"his_data_officer", "data_engineer", "analyst", "opd_coordinator",
                       "department_head", "medical_director", "regional_health_bureau"},
}
DATASET_TIER = {
    "data/raw": "restricted_raw",
    "data/interim": "pseudonymised",
    "data/processed/appointments_model_table": "pseudonymised",
    "data/processed/train": "pseudonymised",
    "data/processed/validation": "pseudonymised",
    "data/processed/test": "pseudonymised",
    "data/processed/dashboard_daily_noshow": "aggregate",
}
RETENTION_DAYS = {"restricted_raw": 365, "pseudonymised": 730, "aggregate": 1825, "audit_log": 2555}


def authorize(role: str, dataset: str) -> bool:
    tier = DATASET_TIER.get(dataset, "restricted_raw")      # unknown data defaults to most restricted
    allowed = role in ACCESS_POLICY[tier]
    write_event("READ" if allowed else "ACCESS_DENIED", dataset, "access check", role=role, tier=tier)
    return allowed
