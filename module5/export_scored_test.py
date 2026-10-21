"""Write the small, identifier-free file the dashboard needs for its interactive fairness view.

    python module5/export_scored_test.py        (run from the repository root, after module4/run_all.py)

Output: module5/data/scored_test.csv  - one row per Module 3 TEST appointment with the champion's predicted
probability, the true outcome, the four audited attributes and the main drivers (no keys, no dates).
Also copies results.json, model_card.md, fairness_report.md and the figures into module5/data/ so the
dashboard folder is self-contained for deployment.
"""
import shutil
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
M4 = ROOT / "module4"
sys.path.insert(0, str(M4))
from m4 import config as C          # noqa: E402
from m4 import data as D            # noqa: E402

OUT = ROOT / "module5" / "data"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    bundle = joblib.load(M4 / "outputs" / "model" / "noshow_model.joblib")
    parts, info = D.load_inputs(processed_dir=ROOT / "data" / "processed")
    test = D.apply_minimisation(parts["test"], bundle.get("dropped_flags") or [])
    p = bundle["pipeline"].predict_proba(test[bundle["features"]])[:, 1]
    keep = ["no_show", "gender", "age_band", "scholarship", "has_disability",
            "age", "lead_days", "prior_visits", "prior_no_shows", "prior_no_show_rate", "same_day", "appt_weekday"]
    df = test[keep].copy()
    df.insert(0, "p_no_show", p.round(4))
    df.to_csv(OUT / "scored_test.csv", index=False)
    print(f"wrote {OUT / 'scored_test.csv'} ({len(df):,} rows, {len(df.columns)} columns, no identifiers)")

    for f in ["results.json", "model_card.md", "fairness_report.md", "sensitivity_analysis.csv", "counterfactuals.csv"]:
        src = M4 / "outputs" / f
        if src.exists():
            shutil.copy(src, OUT / f)
    (OUT / "figures").mkdir(exist_ok=True)
    for png in (M4 / "outputs" / "figures").glob("*.png"):
        shutil.copy(png, OUT / "figures" / png.name)
    (OUT / "model").mkdir(exist_ok=True)
    shutil.copy(M4 / "outputs" / "model" / "noshow_model.joblib", OUT / "model" / "noshow_model.joblib")
    print("copied results, figures, reports and model into module5/data/")


if __name__ == "__main__":
    main()
