"""Automated, gated retraining of the JJUSHYCSH no-show risk model (Final Project).

Triggered monthly by .github/workflows/retrain.yml, by the `monitoring` service, or manually.  The flow is:

    1. load the Module 3 splits (data/processed/train|validation|test) — optionally appended with a new labelled batch
    2. retrain the champion algorithm (random forest, Module 4 hyper-parameters) on train
    3. select the operating threshold and the age-band thresholds on validation (same rule as Module 4)
    4. GATES (all must pass, otherwise the candidate is rejected and the champion stays in service):
         a. validation PR-AUC >= champion PR-AUC - 0.01
         b. validation recall at the operating threshold >= 0.70
         c. true-positive-rate gap on every monitored attribute <= 0.10 after mitigation
         d. no deliberately excluded attribute among the inputs (gender, area code, SMS receipt and the flags dropped
            by the Module 4 necessity review); monitored attributes such as welfare enrolment may be inputs but are
            audited for fairness in gate c; data hash recorded
    5. log the run to MLflow (experiment "jjushycsh_noshow", registered model "jjushycsh_noshow_risk") and, if all
       gates pass, save the bundle to module4/outputs/model/noshow_model.joblib with alias "challenger-approved";
       promotion to "champion" is a human decision recorded in docs/governance/decision_log.md.

usage: python retrain.py [--new-batch data/processed/new_batch.csv] [--dry-run]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "module4"))
OUT = ROOT / "module4" / "outputs"
MONITORED = ["gender", "age_band", "scholarship", "has_disability"]          # fairness-audited groups (gate c)
EXCLUDED_INPUTS = ["gender", "area_code", "sms_received", "appt_month"]     # never model inputs (Module 4 config.EXCLUDED)
GATES = {"pr_auc_tolerance": 0.01, "recall_min": 0.70, "tpr_gap_max": 0.10}


def load_split(name: str) -> pd.DataFrame:
    for ext in ("parquet", "csv"):
        p = ROOT / "data" / "processed" / f"{name}.{ext}"
        if p.exists():
            return pd.read_parquet(p) if ext == "parquet" else pd.read_csv(p)
    raise FileNotFoundError(f"data/processed/{name}.parquet|csv not found - run run_pipeline.py first")


def tpr_gap(y, yhat, g):
    tprs = [yhat[(g == k) & (y == 1)].mean() for k in pd.unique(g) if ((g == k) & (y == 1)).any()]
    return float(max(tprs) - min(tprs)) if tprs else 0.0


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--new-batch"); ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.pipeline import Pipeline

    champion = joblib.load(OUT / "model" / "noshow_model.joblib")
    features = champion["features"]; target = "no_show"
    try:
        train, val, test = load_split("train"), load_split("validation"), load_split("test")
    except FileNotFoundError as e:
        print(f"retrain: {e}. Nothing retrained; the champion stays in service.")
        return 3
    if a.new_batch:
        nb = pd.read_csv(a.new_batch); train = pd.concat([train, nb], ignore_index=True)
    data_md5 = hashlib.md5(pd.util.hash_pandas_object(train[features + [target]]).values).hexdigest()

    # excluded attributes must never be inputs (Module 1 charter / Module 4 config.EXCLUDED and necessity review)
    leaked = [c for c in features if c in EXCLUDED_INPUTS + list(champion.get("dropped_flags", []))]

    champ_pipe = champion["pipeline"]
    params = champ_pipe.steps[-1][1].get_params() if hasattr(champ_pipe, "steps") else {}
    rf = RandomForestClassifier(**{k: v for k, v in params.items() if k in RandomForestClassifier().get_params()})
    from sklearn.base import clone
    # clone, never reuse: fitting the champion's own preprocessor in place would silently alter the champion
    pre = clone(champ_pipe.steps[0][1]) if hasattr(champ_pipe, "steps") and len(champ_pipe.steps) > 1 else "passthrough"
    cand = Pipeline([("pre", pre), ("rf", rf)]) if pre != "passthrough" else Pipeline([("rf", rf)])
    cand.fit(train[features], train[target])

    pv_c = cand.predict_proba(val[features])[:, 1]; pv_0 = champ_pipe.predict_proba(val[features])[:, 1]
    pr_c, pr_0 = average_precision_score(val[target], pv_c), average_precision_score(val[target], pv_0)

    # operating threshold: highest threshold giving recall >= 0.70 on validation (Module 4 rule)
    ths = np.linspace(0.2, 0.8, 121); rec = [(val[target][pv_c >= t].sum() / val[target].sum()) for t in ths]
    thr = float(max([t for t, r in zip(ths, rec) if r >= GATES["recall_min"]], default=0.5))
    # age-band thresholds: Module 4 mitigation rule - per band, the highest threshold that still reaches 70 % recall
    group_thr = {}
    for g in sorted(val["age_band"].astype(str).unique()):
        m = (val["age_band"].astype(str) == g).values
        yg = val[target].values[m]
        ok = [t for t in ths if yg.sum() == 0 or (pv_c[m][yg == 1] >= t).mean() >= GATES["recall_min"]]
        group_thr[g] = float(max(ok)) if ok else float(ths[0])
    yhat_g = (pv_c >= val["age_band"].astype(str).map(group_thr).values).astype(int)
    recall_g = yhat_g[val[target] == 1].mean()
    gaps = {attr: tpr_gap(val[target].values, yhat_g, val[attr].astype(str).values) for attr in MONITORED if attr in val.columns}

    gates = {"pr_auc": {"candidate": float(pr_c), "champion": float(pr_0), "pass": bool(pr_c >= pr_0 - GATES["pr_auc_tolerance"])},
             "recall": {"value": float(recall_g), "pass": bool(recall_g >= GATES["recall_min"])},
             "fairness": {"tpr_gaps": gaps, "pass": bool(all(v <= GATES["tpr_gap_max"] for v in gaps.values()))},
             "no_excluded_inputs": {"leaked": leaked, "pass": not leaked}}
    approved = all(g["pass"] for g in gates.values())
    log = {"run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "data_md5": data_md5, "n_train": int(len(train)),
           "threshold": thr, "group_thresholds": group_thr, "gates": gates, "approved": approved,
           "test_roc_auc": float(roc_auc_score(test[target], cand.predict_proba(test[features])[:, 1]))}

    # MLflow tracking (optional)
    try:
        import mlflow
        mlflow.set_experiment("jjushycsh_noshow")
        with mlflow.start_run(run_name=f"retrain-{log['run_at'][:10]}"):
            mlflow.log_params({"algorithm": "random_forest", "data_md5": data_md5, "n_train": len(train), "threshold": thr})
            mlflow.log_metrics({"val_pr_auc": pr_c, "val_recall": recall_g, "test_roc_auc": log["test_roc_auc"],
                                **{f"tpr_gap_{k}": v for k, v in gaps.items()}})
            mlflow.set_tags({"approved": str(approved), "gate_fairness": str(gates["fairness"]["pass"])})
            mlflow.log_dict(log, "retrain_log.json")
            log["mlflow_run_id"] = mlflow.active_run().info.run_id
    except Exception as e:
        log["mlflow"] = f"not logged ({type(e).__name__})"

    (OUT / "retrain_log.json").write_text(json.dumps(log, indent=2, default=float))
    print(json.dumps({k: log[k] for k in ("approved", "threshold", "gates")}, indent=2, default=float))
    if approved and not a.dry_run:
        bundle = dict(champion, pipeline=cand, threshold_high=thr, group_thresholds=group_thr, data_md5=data_md5,
                      created=log["run_at"], alias="challenger-approved")
        joblib.dump(bundle, OUT / "model" / "noshow_model_challenger.joblib")
        print("challenger saved: module4/outputs/model/noshow_model_challenger.joblib (promotion requires governance sign-off)")
    return 0 if approved else 1


if __name__ == "__main__":
    sys.exit(main())
