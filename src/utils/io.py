"""Small IO helpers: Parquet when pyarrow is available, CSV otherwise."""
from pathlib import Path
import pandas as pd


def save_table(df: pd.DataFrame, path_no_ext: Path) -> Path:
    path_no_ext.parent.mkdir(parents=True, exist_ok=True)
    try:
        import pyarrow  # noqa: F401
        path = path_no_ext.with_suffix(".parquet")
        df.to_parquet(path, index=False)
    except ImportError:
        path = path_no_ext.with_suffix(".csv")
        df.to_csv(path, index=False)
    return path


def load_table(path_no_ext: Path) -> pd.DataFrame:
    pq, csv = path_no_ext.with_suffix(".parquet"), path_no_ext.with_suffix(".csv")
    if pq.exists():
        return pd.read_parquet(pq)
    return pd.read_csv(csv, parse_dates=["scheduled_date", "appointment_date"])
