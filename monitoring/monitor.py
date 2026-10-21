"""Model, data-drift and fairness monitoring for the JJUSHYCSH no-show risk model (Final Project).

Compares a CURRENT batch of scored appointments with a REFERENCE batch (the Module 3 validation split, or the
previous month) and writes:
    outputs/monitoring_summary.json   machine-readable status (read by the Streamlit "Monitoring" page and by CI)
    outputs/monitoring_report.html    human-readable report
    outputs/evidently_drift.html      Evidently AI DataDrift report, when the `evidently` package is installed

Works with plain pandas/numpy/scipy so it runs in CI and on Streamlit Cloud; Evidently and Fairlearn are used
when available (docker-compose `monitoring` service installs them).

usage:
    python monitoring/monitor.py --reference module5/data/scored_test.csv --current <new_batch.csv> --out monitoring/outputs
    python monitoring/monitor.py --demo        # splits the scored test set and adds a simulated lead-time shift
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

# ----------------------------------------------------------------------------- thresholds (from the Module 4 model card)
THRESHOLD_HIGH = 0.556            # champion single operating threshold (recall ~71%)
# Age-band thresholds adopted after the Module 4 fairness mitigation (equal opportunity on age); the deployed
# decision rule. Loaded from the model bundle when present so monitoring always mirrors the served model.
GROUP_ATTR = "age_band"
GROUP_THRESHOLDS = {"0-5": 0.5930, "6-17": 0.6115, "18-34": 0.6288, "35-54": 0.5644, "55-69": 0.4739, "70+": 0.4690}
try:
    import joblib
    _b = joblib.load(ROOT / "module4" / "outputs" / "model" / "noshow_model.joblib")
    GROUP_THRESHOLDS = {str(k): float(v) for k, v in _b.get("group_thresholds", GROUP_THRESHOLDS).items()}
    GROUP_ATTR = _b.get("group_attribute", GROUP_ATTR); THRESHOLD_HIGH = float(_b.get("threshold_high", THRESHOLD_HIGH))
except Exception:
    pass


def decide(df: pd.DataFrame, prob="p_no_show") -> pd.Series:
    """Flag = 1 when the probability reaches the deployed threshold (age-band specific, else the single threshold)."""
    if GROUP_ATTR in df.columns:
        thr = df[GROUP_ATTR].astype(str).map(GROUP_THRESHOLDS).fillna(THRESHOLD_HIGH)
    else:
        thr = pd.Series(THRESHOLD_HIGH, index=df.index)
    return (df[prob].astype(float) >= thr).astype(int)
PROTECTED = ["gender", "age_band", "scholarship", "has_disability"]
NUMERIC = ["age", "lead_days", "prior_visits", "prior_no_shows", "prior_no_show_rate"]
CATEGORICAL = ["same_day", "appt_weekday", "gender", "age_band", "scholarship", "has_disability"]
LIMITS = {
    "psi_warn": 0.10, "psi_alert": 0.20,          # population stability index
    "ks_p_alert": 0.01,                           # KS test p-value
    "recall_min": 0.65, "precision_min": 0.28,    # performance floor (model KPI: recall >= 0.70 at acceptance)
    "tpr_gap_max": 0.10,                          # Module 2 release threshold (equal opportunity)
    "di_min": 0.80,                               # disparate impact diagnostic (four-fifths rule)
    "flag_rate_band": (0.35, 0.55),               # expected share flagged
}


# ----------------------------------------------------------------------------- metrics
def psi(ref: pd.Series, cur: pd.Series, bins: int = 10) -> float:
    """Population stability index on quantile bins built from the reference."""
    ref = ref.dropna().astype(float); cur = cur.dropna().astype(float)
    if ref.nunique() <= 1 or len(cur) == 0:
        return 0.0
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    r = np.histogram(ref, bins=edges)[0] / len(ref)
    c = np.histogram(cur, bins=edges)[0] / len(cur)
    r = np.clip(r, 1e-4, None); c = np.clip(c, 1e-4, None)
    return float(np.sum((c - r) * np.log(c / r)))


def psi_categorical(ref: pd.Series, cur: pd.Series) -> float:
    cats = sorted(set(ref.dropna().astype(str)) | set(cur.dropna().astype(str)))
    r = ref.astype(str).value_counts(normalize=True).reindex(cats).fillna(0).clip(lower=1e-4)
    c = cur.astype(str).value_counts(normalize=True).reindex(cats).fillna(0).clip(lower=1e-4)
    return float(np.sum((c - r) * np.log(c / r)))


def ks(ref: pd.Series, cur: pd.Series):
    try:
        from scipy.stats import ks_2samp
        s = ks_2samp(ref.dropna(), cur.dropna())
        return float(s.statistic), float(s.pvalue)
    except Exception:
        return None, None


def performance(df: pd.DataFrame, prob="p_no_show", label="no_show", thr=THRESHOLD_HIGH) -> dict | None:
    if label not in df.columns or df[label].isna().all():
        return None
    y = df[label].astype(int).values; p = df[prob].astype(float).values; yhat = decide(df, prob).values
    tp = int(((yhat == 1) & (y == 1)).sum()); fp = int(((yhat == 1) & (y == 0)).sum())
    fn = int(((yhat == 0) & (y == 1)).sum()); tn = int(((yhat == 0) & (y == 0)).sum())
    out = {"n": int(len(y)), "base_rate": float(y.mean()), "flag_rate": float(yhat.mean()),
           "recall": tp / max(tp + fn, 1), "precision": tp / max(tp + fp, 1), "tp": tp, "fp": fp, "fn": fn, "tn": tn}
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss
        out["roc_auc"] = float(roc_auc_score(y, p)); out["pr_auc"] = float(average_precision_score(y, p))
        out["brier"] = float(brier_score_loss(y, p))
    except Exception:
        pass
    return out


def fairness(df: pd.DataFrame, attr: str, prob="p_no_show", label="no_show", thr=THRESHOLD_HIGH) -> dict | None:
    if attr not in df.columns:
        return None
    d = df[[attr, prob] + ([label] if label in df.columns else [])].dropna()
    yhat = decide(d, prob)
    rows = []
    for g, sub in d.groupby(attr):
        r = {"group": str(g), "n": int(len(sub)), "selection_rate": float(yhat.loc[sub.index].mean())}
        if label in d.columns:
            y = sub[label].astype(int); yh = yhat.loc[sub.index]
            r["tpr"] = float(yh[y == 1].mean()) if (y == 1).any() else None
            r["fpr"] = float(yh[y == 0].mean()) if (y == 0).any() else None
        rows.append(r)
    sel = [r["selection_rate"] for r in rows]
    out = {"attribute": attr, "groups": rows,
           "dp_difference": float(max(sel) - min(sel)),
           "di_ratio": float(min(sel) / max(sel)) if max(sel) > 0 else None}
    tprs = [r["tpr"] for r in rows if r.get("tpr") is not None]; fprs = [r["fpr"] for r in rows if r.get("fpr") is not None]
    if tprs:
        out["tpr_gap"] = float(max(tprs) - min(tprs)); out["fpr_gap"] = float(max(fprs) - min(fprs)) if fprs else None
        out["eo_difference"] = float(max(out["tpr_gap"], out["fpr_gap"] or 0))
    # Fairlearn cross-check when available
    try:
        from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
        if label in d.columns:
            out["fairlearn_dp_difference"] = float(demographic_parity_difference(d[label], yhat, sensitive_features=d[attr]))
            out["fairlearn_eo_difference"] = float(equalized_odds_difference(d[label], yhat, sensitive_features=d[attr]))
    except Exception:
        pass
    return out


# ----------------------------------------------------------------------------- main
def run(reference: pd.DataFrame, current: pd.DataFrame, out: Path, batch_name="current") -> dict:
    out.mkdir(parents=True, exist_ok=True)
    alerts, warnings_ = [], []

    # 1. feature drift
    drift = []
    for col in NUMERIC:
        if col in reference.columns and col in current.columns:
            v = psi(reference[col], current[col]); k, p = ks(reference[col], current[col])
            status = "ALERT" if v >= LIMITS["psi_alert"] else "WARN" if v >= LIMITS["psi_warn"] else "OK"
            drift.append({"feature": col, "type": "numeric", "psi": round(v, 4), "ks_stat": None if k is None else round(k, 4),
                          "ks_p": None if p is None else round(p, 6), "ref_mean": float(reference[col].mean()),
                          "cur_mean": float(current[col].mean()), "status": status})
    for col in CATEGORICAL:
        if col in reference.columns and col in current.columns:
            v = psi_categorical(reference[col], current[col])
            status = "ALERT" if v >= LIMITS["psi_alert"] else "WARN" if v >= LIMITS["psi_warn"] else "OK"
            drift.append({"feature": col, "type": "categorical", "psi": round(v, 4), "status": status})
    for d in drift:
        if d["status"] == "ALERT": alerts.append(f"Feature drift: {d['feature']} PSI={d['psi']}")
        elif d["status"] == "WARN": warnings_.append(f"Feature drift (watch): {d['feature']} PSI={d['psi']}")

    # 2. prediction drift
    pred = {"psi": round(psi(reference["p_no_show"], current["p_no_show"]), 4),
            "ref_mean_prob": float(reference["p_no_show"].mean()), "cur_mean_prob": float(current["p_no_show"].mean()),
            "ref_flag_rate": float(decide(reference).mean()),
            "cur_flag_rate": float(decide(current).mean())}
    pred["status"] = "ALERT" if pred["psi"] >= LIMITS["psi_alert"] or not (LIMITS["flag_rate_band"][0] <= pred["cur_flag_rate"] <= LIMITS["flag_rate_band"][1]) \
        else "WARN" if pred["psi"] >= LIMITS["psi_warn"] else "OK"
    if pred["status"] == "ALERT": alerts.append(f"Prediction drift: PSI={pred['psi']}, flag rate {pred['cur_flag_rate']:.2f}")
    elif pred["status"] == "WARN": warnings_.append(f"Prediction drift (watch): PSI={pred['psi']}")

    # 3. performance (when outcomes are known — typically 30 days after the appointment)
    perf_ref = performance(reference); perf_cur = performance(current)
    if perf_cur:
        if perf_cur["recall"] < LIMITS["recall_min"]: alerts.append(f"Recall {perf_cur['recall']:.2f} below floor {LIMITS['recall_min']}")
        if perf_cur["precision"] < LIMITS["precision_min"]: warnings_.append(f"Precision {perf_cur['precision']:.2f} below floor {LIMITS['precision_min']}")

    # 4. fairness drift
    fair = {}
    for attr in PROTECTED:
        f = fairness(current, attr)
        if f:
            fair[attr] = f
            if f.get("tpr_gap") is not None and f["tpr_gap"] > LIMITS["tpr_gap_max"]:
                alerts.append(f"Fairness: {attr} TPR gap {f['tpr_gap']:.3f} > {LIMITS['tpr_gap_max']}")
            if f.get("di_ratio") is not None and f["di_ratio"] < LIMITS["di_min"]:
                warnings_.append(f"Fairness (diagnostic): {attr} disparate-impact ratio {f['di_ratio']:.2f} < {LIMITS['di_min']}")

    status = "ALERT" if alerts else "WARN" if warnings_ else "OK"
    summary = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "batch": batch_name,
               "n_reference": int(len(reference)), "n_current": int(len(current)), "threshold_high": THRESHOLD_HIGH,
               "group_thresholds": GROUP_THRESHOLDS,
               "limits": LIMITS, "status": status, "alerts": alerts, "warnings": warnings_,
               "feature_drift": drift, "prediction_drift": pred, "performance_reference": perf_ref,
               "performance_current": perf_cur, "fairness_current": fair,
               "action": {"OK": "No action. Next check at the next monthly batch.",
                          "WARN": "Review at the monthly model-governance meeting; recheck next batch.",
                          "ALERT": "Open an incident (docs/incident_response.md): pause automated use of scores for the affected group/feature, "
                                   "investigate root cause, and trigger retrain.py if drift is confirmed."}[status]}
    (out / "monitoring_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    (out / "monitoring_report.html").write_text(render_html(summary))
    # optional Evidently report
    try:
        from evidently import Report
        from evidently.presets import DataDriftPreset
        rep = Report([DataDriftPreset()])
        snap = rep.run(reference_data=reference[NUMERIC + CATEGORICAL + ["p_no_show"]], current_data=current[NUMERIC + CATEGORICAL + ["p_no_show"]])
        snap.save_html(str(out / "evidently_drift.html")); summary["evidently"] = "evidently_drift.html"
    except Exception as e:  # evidently not installed or API changed: pandas report stands on its own
        summary["evidently"] = f"not generated ({type(e).__name__})"
    (out / "monitoring_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    return summary


def render_html(s: dict) -> str:
    def tbl(rows, cols):
        h = "".join(f"<th>{c}</th>" for c in cols)
        b = "".join("<tr>" + "".join(f"<td>{'' if r.get(c) is None else (round(r[c], 4) if isinstance(r[c], float) else r[c])}</td>" for c in cols) + "</tr>" for r in rows)
        return f"<table border=1 cellpadding=4 style='border-collapse:collapse'><tr>{h}</tr>{b}</table>"
    colour = {"OK": "#2e7d32", "WARN": "#f9a825", "ALERT": "#c62828"}[s["status"]]
    html = [f"<html><body style='font-family:Calibri,Arial'><h1>JJUSHYCSH no-show model — monitoring report</h1>",
            f"<p>Generated {s['generated_at']} · batch <b>{s['batch']}</b> · reference n={s['n_reference']} · current n={s['n_current']}</p>",
            f"<h2 style='color:{colour}'>Status: {s['status']}</h2><p>{s['action']}</p>"]
    if s["alerts"]: html.append("<h3>Alerts</h3><ul>" + "".join(f"<li>{a}</li>" for a in s["alerts"]) + "</ul>")
    if s["warnings"]: html.append("<h3>Warnings</h3><ul>" + "".join(f"<li>{a}</li>" for a in s["warnings"]) + "</ul>")
    html.append("<h3>Feature drift (PSI: warn ≥ 0.10, alert ≥ 0.20)</h3>" + tbl(s["feature_drift"], ["feature", "type", "psi", "ks_stat", "ks_p", "ref_mean", "cur_mean", "status"]))
    html.append("<h3>Prediction drift</h3>" + tbl([s["prediction_drift"]], ["psi", "ref_mean_prob", "cur_mean_prob", "ref_flag_rate", "cur_flag_rate", "status"]))
    if s["performance_current"]:
        html.append("<h3>Performance (current vs reference)</h3>" + tbl([dict(s["performance_reference"] or {}, batch="reference"), dict(s["performance_current"], batch="current")],
                                                                        ["batch", "n", "base_rate", "flag_rate", "recall", "precision", "roc_auc", "pr_auc", "brier"]))
    for attr, f in s["fairness_current"].items():
        html.append(f"<h3>Fairness — {attr} (TPR gap {f.get('tpr_gap')}, DI ratio {f.get('di_ratio')})</h3>" + tbl(f["groups"], ["group", "n", "selection_rate", "tpr", "fpr"]))
    html.append("</body></html>")
    return "\n".join(html)


def demo(out: Path):
    """Demonstration on the Module 4 scored test set: a no-drift batch and a simulated lead-time shift."""
    df = pd.read_csv(ROOT / "module5" / "data" / "scored_test.csv")
    rng = np.random.default_rng(42)
    mask = rng.random(len(df)) < 0.5
    ref, cur = df[mask].copy(), df[~mask].copy()
    s1 = run(ref, cur, out / "demo_no_drift", batch_name="demo: second half of test set (no drift expected)")
    # simulated shift: appointments booked 3 days further ahead; probabilities moved by the Module 4 sensitivity result (+0.154 mean)
    shifted = cur.copy(); shifted["lead_days"] = shifted["lead_days"] + 3; shifted["same_day"] = 0
    shifted["p_no_show"] = (shifted["p_no_show"] + 0.154).clip(0, 1)
    s2 = run(ref, shifted, out / "demo_lead_time_shift", batch_name="demo: simulated +3-day lead-time shift")
    (out / "monitoring_summary.json").write_text(json.dumps({"latest": s1, "stress_test": s2}, indent=2, default=float))
    print("no-drift batch:", s1["status"], s1["alerts"]); print("shifted batch:", s2["status"], s2["alerts"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference"); ap.add_argument("--current"); ap.add_argument("--out", default=str(HERE / "outputs"))
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()
    if a.demo or not (a.reference and a.current):
        demo(Path(a.out))
    else:
        s = run(pd.read_csv(a.reference), pd.read_csv(a.current), Path(a.out), batch_name=Path(a.current).stem)
        print(s["status"], s["alerts"])
