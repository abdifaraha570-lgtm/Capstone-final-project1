"""Explainability (SHAP global + local, counterfactuals), fairness audit + mitigation, sensitivity.
Implements the Module 2 explainability framework on the Module 3 feature definitions."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, recall_score

from m4 import config as C
from m4.modeling import metrics_at, threshold_for_recall

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

PRETTY = {"lead_days": "Lead time (days booked ahead)", "age": "Patient age",
          "prior_no_show_rate": "Prior no-show rate", "prior_visits": "Number of prior visits",
          "prior_no_shows": "Number of prior no-shows", "chronic_count": "Chronic conditions (count)",
          "same_day": "Same-day appointment", "first_visit": "First visit (no history)",
          "scholarship": "Welfare programme enrolment", "hypertension": "Hypertension",
          "diabetes": "Diabetes", "alcoholism": "Alcohol-use disorder", "has_disability": "Disability recorded"}
BINARY_PRETTY = {PRETTY[f] for f in ["same_day", "first_visit", "scholarship", "hypertension", "diabetes",
                                     "alcoholism", "has_disability"]}


def pretty(f):
    if f.startswith("appt_weekday_"):
        return f"Appointment on {f.split('_')[-1]}"
    return PRETTY.get(f, f)


# ------------------------------------------------------------------ SHAP
def shap_analysis(pipe, X_test, n_sample=2000, seed=C.RANDOM_STATE):
    if not HAS_SHAP:
        return {"error": "shap not installed"}
    prep, clf = pipe.named_steps["prep"], pipe.named_steps["clf"]
    Xs = X_test.sample(min(n_sample, len(X_test)), random_state=seed)
    Xt = pd.DataFrame(prep.transform(Xs), columns=prep.get_feature_names_out(), index=Xs.index)
    display = Xt.copy()
    for c in C.NUMERIC:                       # show real units (days, years, counts) instead of z-scores
        if c in Xs.columns:
            display[c] = Xs[c].values
    try:
        if clf.__class__.__name__ == "LogisticRegression":
            explainer = shap.LinearExplainer(clf, Xt)
        else:
            explainer = shap.TreeExplainer(clf)
        sv = explainer(Xt)
    except Exception:
        bg = shap.sample(Xt, 200, random_state=seed)
        explainer = shap.Explainer(lambda a: clf.predict_proba(a)[:, 1], bg)
        sv = explainer(Xt.iloc[:500]); display = display.iloc[:500]; Xs = Xs.iloc[:500]
    vals = np.asarray(sv.values, dtype=float)
    bv = np.asarray(sv.base_values, dtype=float)
    if vals.ndim == 3:                        # random forest returns one slice per class
        vals = vals[:, :, 1]
        bv = bv[..., 1] if bv.ndim >= 1 and bv.shape[-1] == 2 else bv
    n = vals.shape[0]
    bv = np.full(n, float(bv.ravel()[0])) if bv.size == 1 else bv.reshape(n, -1)[:, -1]
    sv = shap.Explanation(values=vals, base_values=bv, data=display.values[:n].astype(float),
                          feature_names=[pretty(f) for f in Xt.columns])
    Xs = Xs.iloc[:n]

    plt.figure(); shap.plots.beeswarm(sv, max_display=12, show=False)
    plt.title("SHAP summary: what drives predicted no-show risk", fontsize=10)
    plt.tight_layout(); plt.savefig(C.FIG / "shap_beeswarm.png", dpi=160, bbox_inches="tight"); plt.close()
    plt.figure(); shap.plots.bar(sv, max_display=12, show=False)
    plt.title("Mean |SHAP| (global feature importance)", fontsize=10)
    plt.tight_layout(); plt.savefig(C.FIG / "shap_bar.png", dpi=160, bbox_inches="tight"); plt.close()

    imp = pd.Series(np.abs(sv.values).mean(0), index=sv.feature_names).sort_values(ascending=False)
    imp.to_csv(C.OUT / "shap_global_importance.csv", header=["mean_abs_shap"])
    direction = {}
    for j, f in enumerate(sv.feature_names):
        v, s = sv.data[:, j].astype(float), sv.values[:, j]
        direction[f] = float(np.corrcoef(v, s)[0, 1]) if np.std(v) > 0 and np.std(s) > 0 else 0.0

    # local explanations: highest-risk, lowest-risk and a borderline appointment
    p = pipe.predict_proba(Xs)[:, 1]
    local = []
    for label, idx in [("high", int(np.argmax(p))), ("low", int(np.argmin(p)))]:
        plt.figure(); shap.plots.waterfall(sv[idx], max_display=10, show=False)
        plt.title(f"Local explanation: {label}-risk appointment (p={p[idx]:.2f})", fontsize=10)
        plt.tight_layout(); plt.savefig(C.FIG / f"shap_local_{label}.png", dpi=160, bbox_inches="tight"); plt.close()
        contrib = pd.Series(sv.values[idx], index=sv.feature_names)
        top = contrib.reindex(contrib.abs().sort_values(ascending=False).index)[:3]
        local.append({"case": label, "prob": float(p[idx]), "top_factors": [(k, float(v)) for k, v in top.items()]})
    return {"importance": imp.to_dict(), "direction": direction, "local": local,
            "explainer": explainer.__class__.__name__, "n_explained": int(len(sv.values))}


# ------------------------------------------------------------------ Counterfactuals
def counterfactuals(pipe, test, t_high, features, n_cases=5):
    """Actionable counterfactuals: only lead time / appointment weekday may change
    (age, health status and history are not things the hospital or patient can alter)."""
    X, y = test[features], test[C.TARGET].values
    p = pipe.predict_proba(X)[:, 1]
    cand = np.where((p >= t_high) & (y == 1) & (X["lead_days"].values >= 4))[0]
    rng = np.random.default_rng(C.RANDOM_STATE)
    picks = rng.choice(cand, size=min(n_cases, len(cand)), replace=False) if len(cand) else []
    rows = []
    for i in picks:
        base = X.iloc[[i]].copy(); orig_lead = int(base["lead_days"].iloc[0])
        found = None
        for lead in [d for d in [14, 10, 7, 5, 3, 2, 1, 0] if d < orig_lead]:
            for wd in [base["appt_weekday"].iloc[0], "Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]:
                x = base.copy(); x["lead_days"] = float(lead); x["same_day"] = int(lead == 0); x["appt_weekday"] = wd
                pn = float(pipe.predict_proba(x)[:, 1][0])
                if pn < t_high:
                    found = (lead, wd, pn); break
            if found:
                break
        rows.append({"test_row": int(i), "age": float(base["age"].iloc[0]), "orig_lead_days": orig_lead,
                     "orig_weekday": base["appt_weekday"].iloc[0], "orig_prob": float(p[i]),
                     "cf_lead_days": found[0] if found else None, "cf_weekday": found[1] if found else None,
                     "cf_prob": found[2] if found else None, "flips_below_threshold": bool(found)})
    cf = pd.DataFrame(rows); cf.to_csv(C.OUT / "counterfactuals.csv", index=False)

    dice_status = "not run"
    try:
        import dice_ml
        num = [f for f in C.NUMERIC if f in features]
        tr = test[features + [C.TARGET]].copy()
        d = dice_ml.Data(dataframe=tr, continuous_features=num, outcome_name=C.TARGET)
        m = dice_ml.Model(model=pipe, backend="sklearn")
        exp = dice_ml.Dice(d, m, method="random")
        q = X.iloc[list(picks[:3])] if len(picks) else X.iloc[:1]
        res = exp.generate_counterfactuals(q, total_CFs=3, desired_class="opposite",
                                           features_to_vary=["lead_days", "appt_weekday"],
                                           permitted_range={"lead_days": [0, 60]})
        pd.concat([c.final_cfs_df.assign(query=k) for k, c in enumerate(res.cf_examples_list)]
                  ).to_csv(C.OUT / "counterfactuals_dice.csv", index=False)
        dice_status = "DiCE (random) counterfactuals saved to outputs/counterfactuals_dice.csv"
    except ImportError:
        dice_status = "dice-ml not installed; custom actionable search used"
    except Exception as e:
        dice_status = f"DiCE failed ({str(e)[:80]}); custom actionable search used"
    return cf, dice_status


# ------------------------------------------------------------------ Fairness
def group_table(y, pred, groups):
    rows = []
    for g in sorted(pd.unique(groups)):
        m = groups == g
        yt, pt = y[m], pred[m]
        rows.append({"group": g, "n": int(m.sum()), "base_rate": float(yt.mean()),
                     "selection_rate": float(pt.mean()),
                     "tpr": float(pt[yt == 1].mean()) if (yt == 1).any() else np.nan,
                     "fpr": float(pt[yt == 0].mean()) if (yt == 0).any() else np.nan,
                     "precision": float(yt[pt == 1].mean()) if (pt == 1).any() else np.nan})
    return pd.DataFrame(rows)


def summarise(tab):
    sr, tpr, fpr = tab["selection_rate"], tab["tpr"], tab["fpr"]
    return {"dp_difference": float(sr.max() - sr.min()),
            "di_ratio": float(sr.min() / sr.max()) if sr.max() > 0 else np.nan,
            "tpr_gap": float(tpr.max() - tpr.min()), "fpr_gap": float(fpr.max() - fpr.min()),
            "eo_difference": float(max(tpr.max() - tpr.min(), fpr.max() - fpr.min()))}


def _flags(d):
    d["passes_di"] = bool(d["di_ratio"] >= C.DI_MIN)          # diagnostic (four-fifths rule)
    d["passes_tpr"] = bool(d["tpr_gap"] <= C.EOD_MAX)          # Module 2 release threshold (equal opportunity)
    d["passes_eo"] = bool(d["eo_difference"] <= C.EOD_MAX)     # stricter: TPR and FPR gaps (equalized odds)
    return d


def fairness_audit(y, p, t, frame, pred=None):
    pred = (p >= t).astype(int) if pred is None else pred
    out = {}
    for a in C.PROTECTED:
        tab = group_table(y, pred, frame[a].astype(str).values)
        out[a] = _flags({"table": tab, **summarise(tab)})
    try:  # cross-check with Fairlearn's reference implementation
        from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
        for a in C.PROTECTED:
            s = frame[a].astype(str).values
            out[a]["fairlearn_dp_diff"] = float(demographic_parity_difference(y, pred, sensitive_features=s))
            out[a]["fairlearn_eo_diff"] = float(equalized_odds_difference(y, pred, sensitive_features=s))
    except ImportError:
        pass
    return out


def mitigate(fitted, parts, features, audit_before):
    """Post-processing for equal opportunity: group-specific thresholds chosen on the VALIDATION split
    (never on the test set) so every group reaches the same recall target."""
    val, test = parts["validation"], parts["test"]
    worst = max(C.PROTECTED, key=lambda a: audit_before[a]["tpr_gap"])
    if audit_before[worst]["tpr_gap"] <= C.EOD_MAX:
        return {"applied": False, "attribute": worst,
                "reason": f"largest TPR gap ({audit_before[worst]['tpr_gap']:.3f}, {worst}) is within the "
                          f"tolerance of {C.EOD_MAX}"}
    yva, pva = val[C.TARGET].values, fitted["p_val"]
    gva, gte = val[worst].astype(str).values, test[worst].astype(str).values
    thr = {g: threshold_for_recall(yva[gva == g], pva[gva == g], C.TARGET_RECALL) for g in np.unique(gva)
           if (yva[gva == g] == 1).sum() >= 20}
    p = fitted["pipe"].predict_proba(test[features])[:, 1]
    t_vec = np.array([thr.get(g, fitted["t_high"]) for g in gte])
    pred = (p >= t_vec).astype(int)
    y = test[C.TARGET].values
    after = fairness_audit(y, p, None, test, pred=pred)
    perf = metrics_at(y, p, 0.0)          # roc/pr from probabilities
    perf.update({k: v for k, v in metrics_at(y, pred.astype(float), 0.5).items()
                 if k in ("accuracy", "precision", "recall", "f1", "flag_rate", "tp", "fp", "fn", "tn")})
    res = {"applied": True, "attribute": worst, "group_thresholds": thr, "after": after,
           "performance_after": perf, "method": "group-specific thresholds chosen on validation (equal opportunity)"}
    try:
        from fairlearn.postprocessing import ThresholdOptimizer
        to = ThresholdOptimizer(estimator=fitted["pipe"], constraints="true_positive_rate_parity",
                                objective="balanced_accuracy_score", prefit=True, predict_method="predict_proba")
        to.fit(val[features], yva, sensitive_features=gva)
        pr = np.asarray(to.predict(test[features], sensitive_features=gte, random_state=C.RANDOM_STATE))
        tab = group_table(y, pr, gte)
        res["fairlearn_threshold_optimizer"] = {**summarise(tab), "recall": float(recall_score(y, pr)),
                                                "flag_rate": float(np.mean(pr))}
    except Exception as e:
        res["fairlearn_threshold_optimizer"] = {"status": f"not run ({str(e)[:60]})"}
    return res


def plot_fairness(audit, path, title_suffix=""):
    n = len(C.PROTECTED)
    fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 3.6))
    for ax, a in zip(np.atleast_1d(axes), C.PROTECTED):
        tab = audit[a]["table"]; x = np.arange(len(tab)); w = 0.27
        ax.bar(x - w, tab["selection_rate"], w, label="Flag rate")
        ax.bar(x, tab["tpr"], w, label="Recall (TPR)")
        ax.bar(x + w, tab["fpr"], w, label="FPR")
        ax.set_xticks(x); ax.set_xticklabels(tab["group"], fontsize=7, rotation=30); ax.set_ylim(0, 1)
        ax.set_title(f"{a}{title_suffix}\nDI={audit[a]['di_ratio']:.2f}, EO diff={audit[a]['eo_difference']:.2f}", fontsize=8.5)
    np.atleast_1d(axes)[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


# ------------------------------------------------------------------ Sensitivity
def sensitivity(pipe, test, t_high, features, n=5000):
    X = test[features].sample(min(n, len(test)), random_state=C.RANDOM_STATE).copy()
    y = test.loc[X.index, C.TARGET].values
    p0 = pipe.predict_proba(X)[:, 1]
    hi0 = p0 >= t_high

    def perturb(fn):
        x = X.copy(); fn(x); x["same_day"] = (x["lead_days"] == 0).astype(int)
        p = pipe.predict_proba(x)[:, 1]
        return {"mean_abs_delta_prob": float(np.mean(np.abs(p - p0))), "mean_delta_prob": float(np.mean(p - p0)),
                "high_tier_flips_pct": float(100 * np.mean((p >= t_high) != hi0)), "roc_auc": float(roc_auc_score(y, p))}

    scen = {
        "lead_days +3": lambda x: x.__setitem__("lead_days", x["lead_days"] + 3),
        "lead_days -3 (floor 0)": lambda x: x.__setitem__("lead_days", (x["lead_days"] - 3).clip(lower=0)),
        "lead_days x1.2": lambda x: x.__setitem__("lead_days", (x["lead_days"] * 1.2).round()),
        "age +5 years": lambda x: x.__setitem__("age", (x["age"] + 5).clip(upper=90)),
        "age -5 years (floor 0)": lambda x: x.__setitem__("age", (x["age"] - 5).clip(lower=0)),
        "prior no-show rate +0.10": lambda x: x.__setitem__("prior_no_show_rate", (x["prior_no_show_rate"] + .1).clip(upper=1)),
        "welfare flag flipped": lambda x: x.__setitem__("scholarship", 1 - x["scholarship"]),
    }
    rng = np.random.default_rng(C.RANDOM_STATE)

    def noise(x):
        for c in ["age", "lead_days"]:
            x[c] = (x[c] + rng.normal(0, 0.10 * x[c].std(), len(x))).clip(lower=0).round()
    scen["Gaussian noise (10% SD) on age and lead time"] = noise
    rows = [{"scenario": "unchanged", "mean_abs_delta_prob": 0.0, "mean_delta_prob": 0.0,
             "high_tier_flips_pct": 0.0, "roc_auc": float(roc_auc_score(y, p0))}]
    rows += [{"scenario": k, **perturb(f)} for k, f in scen.items()]
    sens = pd.DataFrame(rows); sens.to_csv(C.OUT / "sensitivity_analysis.csv", index=False)

    ts = np.linspace(0.05, 0.95, 37)
    rec = [recall_score(y, (p0 >= t).astype(int), zero_division=0) for t in ts]
    fl = [np.mean(p0 >= t) for t in ts]
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.plot(ts, rec, label="Recall"); ax.plot(ts, fl, label="Share of appointments flagged")
    ax.axvline(t_high, color="red", ls=":", label=f"Operating threshold {t_high:.2f}")
    ax.set(xlabel="Decision threshold", ylabel="Rate", title="Threshold sensitivity (test sample)")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(C.FIG / "threshold_sensitivity.png", dpi=160); plt.close(fig)
    return sens
