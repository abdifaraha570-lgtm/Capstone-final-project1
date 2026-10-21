"""Run the full pipeline locally (same steps the Airflow DAG orchestrates).

    python run_pipeline.py              # full run incl. Great Expectations
    python run_pipeline.py --skip-gx    # same checks via the pandas validator (no GX install)
"""
import argparse
import json
import logging
import os
import uuid

from src import config
from src.data_prep import anonymize, clean, ingest, integrate
from src.features import build_features
from src.utils.io import save_table
from src.validation import bias_check


def main(skip_gx: bool = False) -> dict:
    os.environ.setdefault("PIPELINE_RUN_ID", uuid.uuid4().hex[:8])
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    raw = ingest.run()
    if skip_gx:
        from src.validation import pandas_suite as gx_suite     # same checks, no GX install needed
    else:
        from src.validation import gx_suite
    gate = gx_suite.validate(raw, "raw_appointments_suite")
    logging.info("Raw validation gate: success=%s", gate["success"])    # raw may fail 'mostly' checks; cleaning fixes them

    cleaned = clean.run(raw)
    anon = anonymize.run(cleaned)
    save_table(anon, config.INTERIM_DIR / "appointments_anon")
    features = build_features.run(anon)

    gate = gx_suite.validate(features[integrate.MODEL_COLUMNS], "processed_appointments_suite")
    if not gate["success"]:
        raise SystemExit("Processed data failed the validation gate - outputs not published.")

    outputs = integrate.run(features)
    splits = {n: features.loc[part.index] for n, part in outputs["frames"].items()}
    bias = bias_check.run(features, splits)
    summary = {"quality": clean.quality_report, "splits": outputs["splits"], "bias_flags": bias["flags"]}
    print(json.dumps(summary, indent=2, default=str))
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-gx", action="store_true")
    main(**vars(ap.parse_args()))
