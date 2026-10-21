"""Stage 1 - Ingestion: read the source extract, check schema, freeze an immutable snapshot."""
import json
import os
import shutil
import stat
from datetime import datetime, timezone

import pandas as pd

from src import config
from src.utils.audit_log import file_sha256, write_event


class SchemaError(ValueError):
    pass


def read_raw(path=config.RAW_FILE) -> pd.DataFrame:
    # Read everything as string first so type problems are caught by the cleaning
    # step and not silently coerced by pandas at load time.
    df = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])
    write_event("READ", str(path.name), "ingest source extract",
                rows_out=len(df), sha256=file_sha256(path))
    return df


def check_schema(df: pd.DataFrame) -> None:
    missing = [c for c in config.RAW_COLUMNS if c not in df.columns]
    extra = [c for c in df.columns if c not in config.RAW_COLUMNS]
    if missing:
        raise SchemaError(f"Missing expected columns: {missing}")
    if len(df) < config.MIN_ROWS:
        raise SchemaError(f"Only {len(df)} rows; minimum is {config.MIN_ROWS}")
    if extra:
        # A future local extract may carry names/phones; they are dropped, never stored.
        write_event("TRANSFORM", "raw", "drop unexpected columns at ingest",
                    dropped_columns=extra)


def snapshot(path=config.RAW_FILE) -> dict:
    """Copy the source into data/raw with a checksum manifest; make it read-only."""
    config.RAW_SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    digest = file_sha256(path)
    target = config.RAW_SNAPSHOT_DIR / f"appointments_{digest[:12]}.csv"
    if not target.exists():
        shutil.copy2(path, target)
        os.chmod(target, stat.S_IRUSR | stat.S_IWUSR)   # 600: restricted tier
    manifest = {
        "source": str(path.name), "snapshot": target.name, "sha256": digest,
        "ingested_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (config.RAW_SNAPSHOT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    write_event("WRITE", target.name, "immutable raw snapshot", sha256=digest)
    return manifest


def run() -> pd.DataFrame:
    df = read_raw()
    check_schema(df)
    snapshot()
    return df[config.RAW_COLUMNS]
