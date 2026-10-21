"""BAN6800 Module 4 - one command produces every deliverable from the Module 3 hand-over.

    python module4/run_all.py                                  # uses ../data/processed + ../reports from Module 3
    python module4/run_all.py --processed-dir path/to/processed --reports-dir path/to/reports
    python module4/run_all.py --data path/to/appointments_model_table.csv   # single Module 3 table (re-split, seed 42)
    python module4/run_all.py --data path/to/KaggleV2-May-2016.csv          # raw file -> runs the Module 3 pipeline first
Add --quick for a smoke test (3 search candidates per algorithm).
"""
import os
os.environ.setdefault("PYTHONWARNINGS", "ignore::FutureWarning,ignore::UserWarning")
import argparse
import datetime as dt
import json
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import joblib
import numpy as np
import pandas as pd

from m4 import config as C
from m4 import data as D
from m4 import modeling, xai_fairness as xf, docs


def _jsonable(o):
    if isinstance(o, pd.DataFrame):
        return o.to_dict(orient="records")
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, float) and np.isnan(o):
        return None
    return o


def step(name, fn, errors):
    print(f"[{name}] ...", flush=True)
    try:
        return fn()
    except Exception as e:
        traceback.print_exc()
        errors.append(f"{name}: {e}")
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed-dir", default=None, help="Module 3 data/processed (train/validation/test)")
    ap.add_argument("--reports-dir", default=None, help="Module 3 reports/ (bias_flags.json etc.)")
    ap.add_argument("--data", default=None, help="single file instead of --processed-dir (model table or raw Kaggle CSV)")
    ap.add_argument("--quick", action="store_true", help="fewer search iterations (smoke test)")
    ap.add_argument("--no-report", action="store_true")
    a = ap.parse_args()
    if a.quick:
        C.N_ITER = 3
    for d in (C.OUT, C.FIG, C.MODEL_DIR):
        d.mkdir(parents=True, exist_ok=True)
    errors = []

    # ---- 1. Module 3 hand-over
    parts, info = D.load_inputs(a.processed_dir, a.data)
    m3 = D.load_m3_reports(a.reports_dir)
    train, val, test = parts["train"], parts["validation"], parts["test"]
    print(f"Data route: {info['route']} | train {len(train):,} / validation {len(val):,} / test {len(test):,} "
          f"| no-show {train[C.TARGET].mean():.1%} / {val[C.TARGET].mean():.1%} / {test[C.TARGET].mean():.1%}")
    overlap = set(train[C.GROUP_KEY]) & set(test[C.GROUP_KEY])
    assert not overlap, f"{len(overlap)} patients appear in both train and test - split is not patient-grouped"

    # ---- 2. Data bias on the training split (continues the Module 3 bias suite)
    bias = {}
    for g in C.PROTECTED:
        t = train.groupby(train[g].astype(str))[C.TARGET].agg(["size", "mean"])
        bias[g] = [{"group": k, "share": float(v["size"] / len(train)), "no_show_rate": float(v["mean"])}
                   for k, v in t.iterrows()]

    # ---- 3. Necessity review of sensitive health flags (open item from Module 3)
    necessity = step("Necessity review", lambda: modeling.necessity_review(train), errors) or \
        {"table": [], "dropped": [], "final_features": list(C.FEATURES), "tolerance": C.NECESSITY_TOLERANCE}
    features = necessity["final_features"]
    parts = {k: D.apply_minimisation(v, necessity["dropped"]) for k, v in parts.items()}   # chronic_count rebuilt
    train, val, test = parts["train"], parts["validation"], parts["test"]
    print(f"  final features ({len(features)}): {features}; dropped flags: {necessity['dropped']}")

    # ---- 4. Baselines + three tuned algorithms (MLflow)
    results, fitted, champ, registry = modeling.train_all(parts, info, features)
    f = fitted[champ]
    p_test = f["pipe"].predict_proba(test[features])[:, 1]
    temporal = step("Temporal check", lambda: modeling.temporal_check(f["pipe"], parts, features), errors) or {"status": "error"}

    # ---- 5. XAI + fairness + sensitivity
    shap_res = step("SHAP", lambda: xf.shap_analysis(f["pipe"], test[features]), errors) or {}
    cf_res = step("Counterfactuals", lambda: xf.counterfactuals(f["pipe"], test, f["t_high"], features), errors)
    fair = xf.fairness_audit(test[C.TARGET].values, p_test, f["t_high"], test)
    xf.plot_fairness(fair, C.FIG / "fairness_before.png")
    mit = step("Mitigation", lambda: xf.mitigate(f, parts, features, fair), errors) or {"applied": False, "reason": "error"}
    if mit.get("applied"):
        xf.plot_fairness(mit["after"], C.FIG / "fairness_after.png", " (after)")
    sens = step("Sensitivity", lambda: xf.sensitivity(f["pipe"], test, f["t_high"], features), errors)

    # ---- 6. Serialise + export
    bundle = {"pipeline": f["pipe"], "threshold_high": f["t_high"], "threshold_medium": f["t_med"],
              "features": features, "dropped_flags": necessity["dropped"], "algorithm": champ, "data_md5": info["data_md5"], "data_route": info["route"],
              "group_thresholds": mit.get("group_thresholds") if mit.get("applied") else None,
              "group_attribute": mit.get("attribute") if mit.get("applied") else None,
              "author": C.AUTHOR, "created": dt.datetime.now().isoformat(timespec="seconds")}
    joblib.dump(bundle, C.MODEL_DIR / "noshow_model.joblib")
    n_runs = modeling.export_mlflow_runs()

    R = {"run_date": dt.date.today().isoformat(), "author": C.AUTHOR, "data_md5": info["data_md5"],
         "data_route": info["route"], "data_files": info["files"], "m3": m3,
         "n_train": len(train), "n_val": len(val), "n_test": len(test),
         "train_no_show": float(train[C.TARGET].mean()), "val_no_show": float(val[C.TARGET].mean()),
         "test_no_show": float(test[C.TARGET].mean()),
         "date_min": str(train[C.DATE_COL].min().date()), "date_max": str(test[C.DATE_COL].max().date()),
         "features": features, "necessity": necessity, "bias": bias, "results": results, "champion": champ,
         "registry": registry, "temporal_check": temporal, "shap": shap_res,
         "counterfactuals": cf_res[0] if cf_res else None, "dice_status": cf_res[1] if cf_res else "error",
         "fairness_before": fair, "mitigation": mit, "sensitivity": sens, "mlflow_runs": n_runs,
         "errors": errors, "n_iter": C.N_ITER, "cv_folds": C.CV_FOLDS,
         "has": {"mlflow": modeling.HAS_MLFLOW, "shap": xf.HAS_SHAP, "xgboost": modeling.HAS_XGB}}
    R = _jsonable(R)
    (C.OUT / "results.json").write_text(json.dumps(R, indent=2, default=str))
    (C.OUT / "model_card.md").write_text(docs.model_card_md(R), encoding="utf-8")
    (C.OUT / "fairness_report.md").write_text(docs.fairness_report_md(R), encoding="utf-8")

    if not a.no_report:
        try:
            from m4.report import build_report
            print("Report written:", build_report(R))
        except Exception:
            traceback.print_exc(); errors.append("report")
    print("\nDONE." + (f" Non-fatal errors: {errors}" if errors else " No errors."))


if __name__ == "__main__":
    main()
