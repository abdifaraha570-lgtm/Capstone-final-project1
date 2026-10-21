"""Airflow DAG - JJUSHYCSH referral no-show ETL (Sprint 1 / Module 3).

ingest -> validate_raw -> clean -> anonymize -> build_features
       -> validate_processed -> [publish_outputs, bias_check] -> governance_report

Proxy phase: the Kaggle file is static, so the DAG is triggered manually.
Local phase: schedule "0 18 * * *" so scores are ready 24-72 h before visits.
Intermediate data is passed through files (not XCom) so no patient rows sit in
the Airflow metadata database.
"""
import os
import sys
from datetime import datetime, timedelta

from airflow.decorators import dag, task

sys.path.insert(0, os.getenv("PIPELINE_ROOT", "/opt/pipeline"))

DEFAULT_ARGS = {"owner": "his_data_officer", "retries": 2, "retry_delay": timedelta(minutes=10)}


@dag(dag_id="jjushycsh_noshow_etl", start_date=datetime(2026, 9, 1), schedule=None,
     catchup=False, default_args=DEFAULT_ARGS, max_active_runs=1,
     tags=["ban6800", "module3", "referral-no-show"])
def noshow_etl():

    @task
    def ingest() -> str:
        from src import config
        from src.data_prep import ingest as ing
        from src.utils.io import save_table
        df = ing.run()
        return str(save_table(df.astype(str), config.INTERIM_DIR / "stage_raw"))

    @task
    def validate_raw(path: str) -> str:
        import pandas as pd
        from src.validation import gx_suite
        df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path, dtype=str)
        gx_suite.validate(df, "raw_appointments_suite")   # warnings logged; cleaning handles them
        return path

    @task
    def clean_and_anonymize(path: str) -> str:
        import pandas as pd
        from src import config
        from src.data_prep import anonymize, clean
        from src.utils.io import save_table
        df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path, dtype=str)
        anon = anonymize.run(clean.run(df))
        os.remove(path)   # raw staging copy is deleted once pseudonymised
        return str(save_table(anon, config.INTERIM_DIR / "appointments_anon"))

    @task
    def build_features(path: str) -> str:
        from src import config
        from src.features import build_features as bf
        from src.utils.io import load_table, save_table
        from pathlib import Path
        df = bf.run(load_table(Path(path).with_suffix("")))
        return str(save_table(df, config.INTERIM_DIR / "appointments_features"))

    @task
    def validate_processed(path: str) -> str:
        from pathlib import Path
        from src.data_prep.integrate import MODEL_COLUMNS
        from src.utils.io import load_table
        from src.validation import gx_suite
        result = gx_suite.validate(load_table(Path(path).with_suffix(""))[MODEL_COLUMNS],
                                   "processed_appointments_suite")
        if not result["success"]:
            raise ValueError("Great Expectations gate failed - stopping before publish")
        return path

    @task
    def publish_outputs(path: str) -> dict:
        from pathlib import Path
        from src.data_prep import integrate
        from src.utils.io import load_table
        return integrate.run(load_table(Path(path).with_suffix("")))["splits"]

    @task
    def bias_check(path: str) -> int:
        from pathlib import Path
        from src.utils.io import load_table
        from src.validation import bias_check as bc
        return len(bc.run(load_table(Path(path).with_suffix("")))["flags"])

    @task
    def governance_report(splits: dict, n_flags: int) -> None:
        from src.utils.audit_log import write_event
        write_event("VALIDATE", "pipeline_run", "sprint-1 governance summary",
                    splits=splits, bias_flags=n_flags)

    features = build_features(clean_and_anonymize(validate_raw(ingest())))
    checked = validate_processed(features)
    governance_report(publish_outputs(checked), bias_check(checked))


noshow_etl()
