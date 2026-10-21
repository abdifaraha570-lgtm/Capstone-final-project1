"""Data validation with Great Expectations (GX Core 1.x).

Two suites act as quality gates:
  raw_appointments_suite       - runs right after ingestion (is the source what we expect?)
  processed_appointments_suite - runs before outputs are published (is the model table safe?)
A failed critical expectation stops the DAG. Results are rendered as Data Docs HTML
(screenshots for the Module 3 slides come from gx/uncommitted/data_docs/local_site/index.html).
"""
import json

import pandas as pd

from src import config
from src.utils.audit_log import write_event

BINARY_RAW = ["Scholarship", "Hipertension", "Diabetes", "Alcoholism", "SMS_received"]
BINARY_PROCESSED = ["scholarship", "hypertension", "diabetes", "alcoholism",
                    "sms_received", "has_disability", "same_day", "first_visit", "no_show"]


def raw_expectations(gx):
    E = gx.expectations
    exps = [
        E.ExpectTableRowCountToBeBetween(min_value=config.MIN_ROWS),
        E.ExpectTableColumnsToMatchOrderedList(column_list=config.RAW_COLUMNS),
        E.ExpectColumnValuesToBeUnique(column="AppointmentID"),
        E.ExpectColumnValuesToBeInSet(column="Gender", value_set=["F", "M"]),
        E.ExpectColumnValuesToBeInSet(column="No-show", value_set=["Yes", "No"]),
        # mostly=0.999: we expect a handful of bad ages; cleaning removes them
        E.ExpectColumnValuesToBeBetween(column="Age", min_value=0, max_value=110, mostly=0.999),
        E.ExpectColumnValuesToBeBetween(column="Handcap", min_value=0, max_value=4),
        E.ExpectColumnValuesToMatchRegex(column="ScheduledDay", regex=r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$"),
        E.ExpectColumnValuesToMatchRegex(column="AppointmentDay", regex=r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$"),
    ]
    exps += [E.ExpectColumnValuesToNotBeNull(column=c) for c in config.RAW_COLUMNS]
    exps += [E.ExpectColumnValuesToBeInSet(column=c, value_set=[0, 1]) for c in BINARY_RAW]
    return exps


def processed_expectations(gx):
    E = gx.expectations
    exps = [
        E.ExpectTableRowCountToBeBetween(min_value=config.MIN_ROWS),
        E.ExpectColumnValuesToBeUnique(column="appointment_key"),
        E.ExpectColumnValuesToMatchRegex(column="patient_key", regex=r"^[0-9a-f]{16}$"),
        E.ExpectColumnValuesToBeBetween(column="age", min_value=0, max_value=90),
        E.ExpectColumnValuesToBeBetween(column="lead_days", min_value=0, max_value=365),
        E.ExpectColumnValuesToBeBetween(column="prior_no_show_rate", min_value=0, max_value=1),
        E.ExpectColumnValuesToBeInSet(column="gender", value_set=["F", "M", "U"]),
        E.ExpectColumnMeanToBeBetween(column="no_show", min_value=0.05, max_value=0.40),
    ]
    # Privacy gate: the processed table must contain exactly the approved columns,
    # so a raw identifier (PatientId, neighbourhood, a phone number...) fails the run.
    from src.data_prep.integrate import MODEL_COLUMNS
    exps.append(E.ExpectTableColumnsToMatchSet(column_set=MODEL_COLUMNS, exact_match=True))
    exps += [E.ExpectColumnValuesToNotBeNull(column=c) for c in ["patient_key", "appointment_key", "no_show", "lead_days", "age_band"]]
    exps += [E.ExpectColumnValuesToBeInSet(column=c, value_set=[0, 1]) for c in BINARY_PROCESSED]
    return exps


def _numeric_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Raw file is read as text; cast numeric columns only for validation."""
    df = df.copy()
    for c in ["Age", "Handcap"] + BINARY_RAW:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def validate(df: pd.DataFrame, suite_name: str) -> dict:
    import great_expectations as gx

    context = gx.get_context(mode="file", project_root_dir=str(config.ROOT))
    source = context.data_sources.add_or_update_pandas(name="noshow_pandas")
    try:
        asset = source.get_asset(suite_name)
    except Exception:
        asset = source.add_dataframe_asset(name=suite_name)
    try:
        batch_def = asset.get_batch_definition("whole_table")
    except Exception:
        batch_def = asset.add_batch_definition_whole_dataframe("whole_table")

    vd_name, cp_name = f"{suite_name}_vd", f"{suite_name}_checkpoint"
    # rebuild on every run so the code is the single source of truth (dependency order)
    for store, name in ((context.checkpoints, cp_name), (context.validation_definitions, vd_name),
                        (context.suites, suite_name)):
        try:
            store.delete(name)
        except Exception:
            pass
    suite = context.suites.add(gx.ExpectationSuite(name=suite_name))

    if suite_name.startswith("raw"):
        df, exps = _numeric_raw(df), raw_expectations(gx)
    else:
        exps = processed_expectations(gx)
    for e in exps:
        suite.add_expectation(e)
    suite.save()

    vd = context.validation_definitions.add(gx.ValidationDefinition(name=vd_name, data=batch_def, suite=suite))
    try:
        from great_expectations.checkpoint import UpdateDataDocsAction
    except ImportError:
        from great_expectations.checkpoint.actions import UpdateDataDocsAction
    checkpoint = context.checkpoints.add(gx.Checkpoint(
        name=cp_name, validation_definitions=[vd],
        actions=[UpdateDataDocsAction(name="update_data_docs")],
        result_format={"result_format": "SUMMARY"},
    ))
    result = checkpoint.run(batch_parameters={"dataframe": df})

    details = _details(result)
    stats = {"suite": suite_name, "success": bool(result.success),
             "expectations": len(suite.expectations),
             "passed": sum(d["success"] for d in details), "details": details}
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.REPORTS_DIR / f"gx_{suite_name}.json").write_text(json.dumps(stats, indent=2))
    write_event("VALIDATE", suite_name, "great expectations quality gate", **stats)
    return stats


def _details(result) -> list:
    """Per-expectation outcome, saved so slides can show exactly what passed/failed."""
    rows = []
    try:
        for vr in result.run_results.values():
            d = vr.to_json_dict() if hasattr(vr, "to_json_dict") else dict(vr)
            for r in d.get("results", []):
                cfg = r.get("expectation_config", {}) or {}
                kw = cfg.get("kwargs", {}) or {}
                res = r.get("result", {}) or {}
                rows.append({
                    "expectation": cfg.get("type") or cfg.get("expectation_type"),
                    "column": kw.get("column", "(table)"),
                    "success": bool(r.get("success")),
                    "unexpected_count": res.get("unexpected_count"),
                    "observed_value": res.get("observed_value"),
                })
    except Exception as exc:        # never let reporting break the pipeline
        rows.append({"expectation": "details_unavailable", "column": str(exc), "success": True})
    return rows
