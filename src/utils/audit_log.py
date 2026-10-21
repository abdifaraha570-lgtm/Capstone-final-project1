"""Privacy audit logging: one JSON line per data access or transformation.

Every pipeline step is wrapped with @audited so the log answers: who touched which
dataset, when, for what purpose, how many rows went in/out, and the file hash.
The log is append-only; it never contains patient-level values.
"""
import functools
import getpass
import hashlib
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from src import config

_logger = logging.getLogger("privacy_audit")


def _actor() -> str:
    return os.getenv("PIPELINE_ACTOR") or getpass.getuser()


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_event(action: str, dataset: str, purpose: str, **details) -> dict:
    event = {
        "ts_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "actor": _actor(),
        "run_id": os.getenv("PIPELINE_RUN_ID", "local"),
        "action": action,          # READ | WRITE | TRANSFORM | VALIDATE | ANONYMIZE | ACCESS_DENIED
        "dataset": dataset,
        "purpose": purpose,
        **details,
    }
    Path(config.AUDIT_LOG).parent.mkdir(parents=True, exist_ok=True)
    with open(config.AUDIT_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, default=str) + "\n")
    _logger.info("%s %s rows_in=%s rows_out=%s", action, dataset,
                 details.get("rows_in"), details.get("rows_out"))
    return event


def audited(action: str, dataset: str, purpose: str = "no-show outreach analytics"):
    """Decorator for DataFrame -> DataFrame steps; logs row counts and duration."""
    def wrap(fn):
        @functools.wraps(fn)
        def inner(df, *args, **kwargs):
            start = time.perf_counter()
            rows_in = len(df) if hasattr(df, "__len__") else None
            out = fn(df, *args, **kwargs)
            rows_out = len(out) if hasattr(out, "__len__") else None
            write_event(action, dataset, purpose, step=fn.__name__,
                        rows_in=rows_in, rows_out=rows_out,
                        columns_out=list(getattr(out, "columns", [])),
                        seconds=round(time.perf_counter() - start, 3))
            return out
        return inner
    return wrap
