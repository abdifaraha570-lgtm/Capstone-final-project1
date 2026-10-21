# Explanation Store — how explanations are kept for audit

Every `/predict` call returns, and the service stores, an **explanation record**:

| Field | Content |
|---|---|
| `request_id`, `timestamp` | UUID, UTC time |
| `appointment_key` | Pseudonymous key from the Module 3 anonymiser (no identifiers) |
| `model_version`, `data_md5` | MLflow registered version and training-data hash from the bundle |
| `inputs` | The 12 model inputs as used (after `features_from_request`) |
| `probability`, `threshold_applied`, `tier`, `action` | Score, age-band threshold used, Low/Medium/High, recommended outreach |
| `reason_codes` | Top SHAP contributions in plain language (as returned to the user) |
| `shap_values` | Per-feature SHAP values (vector) for exact reproduction |
| `override` | Filled later by the dashboard when staff change the action, with reason |

**Storage:** append-only Parquet files partitioned by month (`explanations/year=/month=`) in the hospital's protected
storage; the global SHAP summary, counterfactual table (`module4/outputs/counterfactuals.csv`) and sensitivity analysis are
versioned with the model in MLflow. **Access:** model owner, DPO, Governance Committee; read-only. **Retention:** 24 months,
then deleted with the scored lists. **Audit use:** incident root-cause analysis, patient "why was I contacted" requests,
and the quarterly fairness review (explanations are re-aggregated by group to check that the drivers do not differ unfairly).
