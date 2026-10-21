"""JJUSHYCSH Referral No-Show Risk - stakeholder dashboard (BAN6800 Module 5, Abdi Farah Ahmed).

Run locally:   streamlit run module5/app.py
Deploy:        Hugging Face Spaces (Streamlit SDK) or Streamlit Community Cloud - see module5/README.md

Everything shown comes from the Module 4 run (module5/data/, written by export_scored_test.py):
results.json, the figures, the scored Module 3 test split (no identifiers) and the serialised champion model.
The dashboard describes and explores results; it does not make recommendations.
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:                    # pragma: no cover
    plt = None

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
FIG = DATA / "figures"
NAVY, TEAL, OCHRE, RED, GREY = "#1F3B57", "#2E6B65", "#C8933D", "#B2432F", "#8A8F98"
AUTHOR = "Abdi Farah Ahmed"
TPR_GAP_MAX, DI_MIN, TARGET_RECALL = 0.10, 0.80, 0.70

st.set_page_config(page_title="JJUSHYCSH No-Show Risk", page_icon="🏥", layout="wide")


# ----------------------------------------------------------------------------- data
@st.cache_data
def load_results():
    return json.loads((DATA / "results.json").read_text())


@st.cache_data
def load_scored():
    df = pd.read_csv(DATA / "scored_test.csv")
    for c in ["gender", "age_band", "scholarship", "has_disability", "appt_weekday"]:
        df[c] = df[c].astype(str)
    return df


@st.cache_resource
def load_model():
    try:
        import joblib
        return joblib.load(DATA / "model" / "noshow_model.joblib"), None
    except Exception as e:           # version mismatch on the host, etc.
        return None, str(e)[:160]


R = load_results()
S = load_scored()
CH = R["champion"]; CR = R["results"][CH]
CH_NAME = CH.replace("_", " ").title()
T_HIGH, T_MED = CR["threshold"], CR.get("threshold_medium", CR["threshold"] * 0.8)
def _secret(name):
    """Optional secret; Streamlit raises if no secrets.toml exists at all, so treat that as 'not set'."""
    try:
        return st.secrets.get(name, "")
    except Exception:
        return ""


API_URL = os.getenv("NOSHOW_API_URL") or _secret("NOSHOW_API_URL")

PRETTY = {"lead_days": "Days booked ahead", "age": "Age", "prior_no_show_rate": "Share of past visits missed",
          "prior_visits": "Past visits", "prior_no_shows": "Past missed visits", "chronic_count": "Chronic conditions",
          "same_day": "Same-day booking", "first_visit": "First visit", "scholarship": "Welfare enrolment",
          "hypertension": "Hypertension", "diabetes": "Diabetes"}
GROUP_LABEL = {"gender": "Gender", "age_band": "Age band", "scholarship": "Welfare enrolment", "has_disability": "Disability recorded"}


def pct(x, d=0):
    return "n/a" if x is None else f"{100 * float(x):.{d}f}%"


def tier(p):
    return "High" if p >= T_HIGH else "Medium" if p >= T_MED else "Low"


ACTIONS = {"High": "Phone call 48 h before + community health worker follow-up; offer an earlier slot if available",
           "Medium": "SMS reminder + phone call 24-48 h before", "Low": "Standard SMS reminder"}


def metrics_at(y, p, t):
    pred = (p >= t).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    return {"recall": tp / max(tp + fn, 1), "precision": tp / max(tp + fp, 1), "flag_rate": pred.mean(),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def group_metrics(df, attr, t):
    """Per-group selection rate, recall (TPR), FPR, precision - Fairlearn if available, pandas otherwise."""
    y, pred, g = df["no_show"].values, (df["p_no_show"].values >= t).astype(int), df[attr].values
    try:
        from fairlearn.metrics import MetricFrame, selection_rate, true_positive_rate, false_positive_rate, count
        from sklearn.metrics import precision_score
        mf = MetricFrame(metrics={"n": count, "flag_rate": selection_rate, "recall": true_positive_rate,
                                  "fpr": false_positive_rate,
                                  "precision": lambda yt, yp: precision_score(yt, yp, zero_division=0)},
                         y_true=y, y_pred=pred, sensitive_features=g)
        tab = mf.by_group.reset_index().rename(columns={attr: "group", "sensitive_feature_0": "group"})
        tab["base_rate"] = pd.Series(y).groupby(g).mean().reindex(tab["group"]).values
        engine = "Fairlearn MetricFrame"
    except Exception:
        rows = []
        for grp in sorted(pd.unique(g)):
            m = g == grp; yt, yp = y[m], pred[m]
            rows.append({"group": grp, "n": int(m.sum()), "base_rate": yt.mean(), "flag_rate": yp.mean(),
                         "recall": yp[yt == 1].mean() if (yt == 1).any() else np.nan,
                         "fpr": yp[yt == 0].mean() if (yt == 0).any() else np.nan,
                         "precision": yt[yp == 1].mean() if (yp == 1).any() else np.nan})
        tab = pd.DataFrame(rows); engine = "pandas"
    summ = {"tpr_gap": float(tab["recall"].max() - tab["recall"].min()),
            "fpr_gap": float(tab["fpr"].max() - tab["fpr"].min()),
            "di_ratio": float(tab["flag_rate"].min() / tab["flag_rate"].max()) if tab["flag_rate"].max() > 0 else np.nan}
    return tab, summ, engine


def bar(ax, labels, values, colors, title, ylim=(0, 1), fmt="{:.0%}"):
    b = ax.bar(labels, values, color=colors)
    for rect, v in zip(b, values):
        ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.01, fmt.format(v), ha="center", fontsize=9)
    ax.set_ylim(*ylim); ax.set_title(title, fontsize=11); ax.spines[["top", "right"]].set_visible(False)


# ----------------------------------------------------------------------------- layout
st.sidebar.title("JJUSHYCSH referral no-show risk")
st.sidebar.caption(f"BAN6800 capstone · {AUTHOR} · Nexford University")
page = st.sidebar.radio("Go to", ["Overview", "Model results", "What drives predictions", "What-if: score an appointment",
                                  "Ethical compliance (fairness)", "Monitoring (drift & fairness)", "Limitations & transparency", "About the data"])
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Champion model:** {CH_NAME}  \n**High-risk threshold:** {T_HIGH:.2f}  \n"
                    f"**Run date:** {R['run_date']}  \n**Data:** Module 3 test split, {R['n_test']:,} appointments")
st.sidebar.info("Advisory tool. A risk flag can only add outreach support; it never cancels, delays or deprioritises care "
                "(Module 1 Ethical AI Charter).")

# ============================================================================ Overview
if page == "Overview":
    st.title("Predicting missed referral appointments at JJUSHYCSH")
    st.markdown("Jigjiga University Sheikh Hassan Yabare Comprehensive Specialized Hospital is the only tertiary referral "
                "hospital for the Somali Region's six to seven million people. Every missed follow-up delays care for one "
                "patient and wastes a specialist slot another patient travelled far for. This dashboard shows what the "
                "Module 4 model found, how it decides, and how fairly it treats different groups.")
    rule = R["results"]["baseline_lead_time_rule"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("No-shows caught (recall)", pct(CR["recall"]), f"target ≥ {TARGET_RECALL:.0%}")
    c2.metric("Appointments flagged", pct(CR["flag_rate"]), "outreach workload")
    c3.metric("Precision of a flag", pct(CR["precision"]), f"vs {pct(R['test_no_show'])} base rate")
    c4.metric("Ranking quality (ROC-AUC)", f"{CR['roc_auc']:.2f}", f"rule baseline {rule['roc_auc']:.2f}")
    st.markdown("#### What a week of 1,000 referral appointments looks like with the model")
    n = 1000; base = R["test_no_show"]
    ns = n * base; caught = ns * CR["recall"]; flagged = n * CR["flag_rate"]
    if plt:
        fig, ax = plt.subplots(figsize=(8, 2.6))
        ax.barh(["Flagged for extra outreach", "No-shows among them (caught)", "No-shows not flagged (missed)"],
                [flagged, caught, ns - caught], color=[OCHRE, TEAL, RED])
        for i, v in enumerate([flagged, caught, ns - caught]):
            ax.text(v + 8, i, f"{v:,.0f}", va="center", fontsize=10)
        ax.set_xlim(0, n); ax.invert_yaxis(); ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlabel("appointments (per 1,000)")
        st.pyplot(fig, use_container_width=True)
    st.caption(f"Based on the champion's test-split performance: no-show rate {pct(R['test_no_show'], 1)}, "
               f"recall {pct(CR['recall'], 1)}, flag rate {pct(CR['flag_rate'], 1)}.")
    st.markdown("#### Plain-language model explanation")
    st.markdown(f"""
The model looks at each scheduled referral or follow-up visit **24–72 hours before the appointment** and estimates the
chance the patient will not come. It learns from past appointments only from things known at booking time: how far ahead
the visit was booked, the patient's own attendance history, age, chronic conditions and welfare enrolment. It then sorts
appointments into **Low, Medium and High** risk so that limited SMS, phone-call and community-health-worker capacity goes
where it is most needed. The champion ({CH_NAME}) was chosen on a validation set and reported once on a test set of
{R['n_test']:,} appointments it had never seen.
""")

# ============================================================================ Model results
elif page == "Model results":
    st.title("Model results")
    res = R["results"]
    order = ["baseline_lead_time_rule", "logistic_regression", "random_forest"] + \
            [k for k in res if k in ("xgboost", "hist_gradient_boosting")]
    names = {"baseline_lead_time_rule": "Lead-time rule (no ML)", "logistic_regression": "Logistic regression",
             "random_forest": "Random forest", "xgboost": "XGBoost", "hist_gradient_boosting": "Gradient boosting"}
    tab = pd.DataFrame([{"Model": names[k] + (" ★ champion" if k == CH else ""),
                         "No-shows caught (recall)": pct(res[k]["recall"], 1), "Precision": pct(res[k]["precision"], 1),
                         "Flagged": pct(res[k]["flag_rate"], 1), "ROC-AUC": f"{res[k]['roc_auc']:.3f}",
                         "PR-AUC": f"{res[k]['pr_auc']:.3f}"} for k in order if k in res])
    st.dataframe(tab, hide_index=True, use_container_width=True)
    st.caption("Every model's threshold was set on the validation split to reach 70 % recall; the table shows what "
               "happened on the untouched test split. Full confusion-matrix counts are in the Q&A appendix of the slides.")
    if plt:
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
        ks = [k for k in order if k in res]
        bar(axes[0], [names[k].split(" (")[0] for k in ks], [res[k]["roc_auc"] for k in ks],
            [NAVY if k == CH else GREY for k in ks], "Ranking quality (ROC-AUC, higher is better)", (0.4, 0.85), "{:.3f}")
        bar(axes[1], [names[k].split(" (")[0] for k in ks], [res[k]["precision"] for k in ks],
            [TEAL if k == CH else GREY for k in ks], "Precision at 70 % recall", (0, 0.5))
        for a in axes: a.tick_params(axis="x", labelsize=8)
        st.pyplot(fig, use_container_width=True)
    c1, c2 = st.columns(2)
    if (FIG / "roc_pr_curves.png").exists():
        c1.image(str(FIG / "roc_pr_curves.png"), caption="ROC and precision–recall curves (test split)")
    if (FIG / "confusion_champion.png").exists():
        c2.image(str(FIG / "confusion_champion.png"), caption="Champion confusion matrix at the operating threshold")
    tc = R.get("temporal_check", {})
    if tc.get("status") == "ok":
        st.markdown(f"**Does it hold up over time?** Refitting on appointments before {tc['cut_date']} and scoring the "
                    f"later {tc['n_late']:,} gave ROC-AUC {tc['roc_auc']:.3f} and recall {pct(tc['recall_at_early_threshold'])} "
                    "at the earlier threshold - the drift signal the monitoring plan tracks.")

# ============================================================================ Drivers
elif page == "What drives predictions":
    st.title("What drives a prediction")
    imp = R.get("shap", {}).get("importance", {})
    dirn = R.get("shap", {}).get("direction", {})
    if imp:
        top = list(imp.items())[:10]
        st.markdown("Average influence of each factor on predicted risk (SHAP values, Lundberg & Lee, 2017). "
                    "Teal bars push risk **up** when the factor is high or present; grey bars push it **down**.")
        if plt:
            fig, ax = plt.subplots(figsize=(8, 4))
            labels = [k for k, _ in top][::-1]; vals = [v for _, v in top][::-1]
            cols = [TEAL if dirn.get(k, 0) > 0 else GREY for k in labels]
            ax.barh(labels, vals, color=cols); ax.set_xlabel("mean |SHAP| (impact on predicted probability)")
            ax.spines[["top", "right"]].set_visible(False)
            st.pyplot(fig, use_container_width=True)
        st.markdown("""
**In business terms**
- *Days booked ahead* - the longer the gap between booking and the visit, the more time for travel plans, money or family
  arrangements to fall through. Same-day bookings are rarely missed.
- *Attendance history* - a patient who has missed before is more likely to miss again; a first visit has no history, so the
  model relies on the booking pattern.
- *Age* - adolescents and young adults miss more often; older patients with chronic follow-up attend more reliably.
- *Chronic conditions, welfare enrolment* - smaller effects; chronic follow-up creates routine, welfare enrolment signals
  cost barriers.
""")
        c1, c2 = st.columns(2)
        if (FIG / "shap_beeswarm.png").exists():
            c1.image(str(FIG / "shap_beeswarm.png"), caption="SHAP summary: each dot is one appointment")
        if (FIG / "shap_local_high.png").exists():
            c2.image(str(FIG / "shap_local_high.png"), caption="One high-risk appointment explained")
    else:
        st.warning("SHAP results are not in results.json for this run.")
    st.markdown("**What the model does *not* use:** gender, neighbourhood/area, whether an SMS was sent, alcohol-use "
                "disorder and disability. The last two were tested and dropped because they added nothing to accuracy "
                "(data minimisation); all are still checked in the fairness audit.")

# ============================================================================ What-if
elif page == "What-if: score an appointment":
    st.title("What-if: score an appointment")
    st.markdown("Change the inputs and see how the risk tier responds. This uses the same serialised model and the same "
                "feature definitions as the Module 4 API.")
    bundle, err = load_model()
    c1, c2, c3 = st.columns(3)
    age = c1.slider("Patient age", 0, 90, 27)
    lead = c1.slider("Days between booking and appointment", 0, 60, 21)
    weekday = c1.selectbox("Appointment weekday", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"], 3)
    prior_visits = c2.slider("Earlier appointments on record", 0, 20, 3)
    prior_ns = c2.slider("...of which missed", 0, 20, 1)
    scholarship = c2.checkbox("Welfare programme enrolment", True)
    hyp = c3.checkbox("Hypertension", False); diab = c3.checkbox("Diabetes", False)
    alc = c3.checkbox("Alcohol-use disorder (audit only)", False); dis = c3.checkbox("Disability recorded (audit only)", False)
    prior_ns = min(prior_ns, prior_visits)
    dropped = set((bundle or {}).get("dropped_flags") or ["alcoholism", "has_disability"])
    flags = {"scholarship": int(scholarship), "hypertension": int(hyp), "diabetes": int(diab), "alcoholism": int(alc)}
    comps = [c for c in ["hypertension", "diabetes", "alcoholism"] if c not in dropped]
    row = {"age": float(age), "lead_days": float(lead), "prior_visits": float(prior_visits),
           "prior_no_shows": float(prior_ns), "prior_no_show_rate": prior_ns / prior_visits if prior_visits else 0.0,
           "chronic_count": float(sum(flags[c] for c in comps)), **flags, "has_disability": int(dis),
           "same_day": int(lead == 0), "first_visit": int(prior_visits == 0), "appt_weekday": weekday}
    p = None; source = ""
    if bundle is not None:
        try:
            X = pd.DataFrame([row])[bundle["features"]]
            p = float(bundle["pipeline"].predict_proba(X)[:, 1][0]); source = "serialised champion model"
        except Exception as e:
            err = str(e)[:160]
    if p is None and API_URL:
        try:
            import requests
            body = {"age": age, "scheduled_date": "2026-10-01T09:00:00",
                    "appointment_date": (pd.Timestamp("2026-10-01") + pd.Timedelta(days=lead)).isoformat(),
                    "scholarship": int(scholarship), "hypertension": int(hyp), "diabetes": int(diab), "alcoholism": int(alc),
                    "has_disability": int(dis), "prior_visits": prior_visits, "prior_no_shows": prior_ns}
            r = requests.post(API_URL.rstrip("/") + "/predict", json=body, timeout=10); r.raise_for_status()
            p = r.json()["no_show_probability"]; source = f"Module 4 API at {API_URL}"
        except Exception as e:
            err = str(e)[:160]
    if p is None:
        # Transparent fallback so the page still demonstrates the logic: nearest neighbours in the scored test split.
        d = S.copy()
        d["dist"] = (abs(d["lead_days"] - lead) / 10 + abs(d["age"] - age) / 20 +
                     abs(d["prior_no_show_rate"] - row["prior_no_show_rate"]) * 2 + abs(d["prior_visits"] - prior_visits) / 5)
        p = float(d.nsmallest(200, "dist")["p_no_show"].mean())
        source = "average of the 200 most similar scored test appointments (model file could not be loaded here)"
        if err:
            st.caption(f"Model/API unavailable on this host: {err}")
    t = tier(p)
    colour = {"High": RED, "Medium": OCHRE, "Low": TEAL}[t]
    st.markdown(f"<div style='padding:14px;border-radius:8px;background:{colour};color:white;font-size:20px'>"
                f"Predicted no-show probability <b>{p:.0%}</b> → <b>{t} risk</b></div>", unsafe_allow_html=True)
    st.markdown(f"**Outreach action for this tier (Module 1 rule):** {ACTIONS[t]}")
    st.caption(f"Thresholds: High ≥ {T_HIGH:.2f}, Medium ≥ {T_MED:.2f}. Source of this score: {source}.")
    if bundle is not None and p is not None and plt and lead > 0:
        st.markdown("#### How the lead time alone changes this appointment's risk")
        try:
            leads = list(range(0, 61, 3)); ps = []
            for L in leads:
                r2 = dict(row); r2["lead_days"] = float(L); r2["same_day"] = int(L == 0)
                ps.append(float(bundle["pipeline"].predict_proba(pd.DataFrame([r2])[bundle["features"]])[:, 1][0]))
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.plot(leads, ps, color=NAVY, marker="o", ms=3); ax.axhline(T_HIGH, color=RED, ls=":", label="High threshold")
            ax.axhline(T_MED, color=OCHRE, ls=":", label="Medium threshold"); ax.axvline(lead, color=GREY, lw=0.8)
            ax.set_xlabel("days booked ahead"); ax.set_ylabel("predicted probability"); ax.set_ylim(0, 1); ax.legend(fontsize=8)
            ax.spines[["top", "right"]].set_visible(False)
            st.pyplot(fig, use_container_width=True)
        except Exception:
            pass
    st.markdown("#### Three example appointments")
    examples = [("A", "27-year-old, booked 21 days ahead, 3 earlier visits of which 1 missed, welfare enrolled",
                 dict(age=27, lead_days=21, prior_visits=3, prior_no_shows=1, scholarship=1)),
                ("B", "64-year-old with hypertension and diabetes, booked 2 days ahead, 6 earlier visits all attended",
                 dict(age=64, lead_days=2, prior_visits=6, prior_no_shows=0, hypertension=1, diabetes=1)),
                ("C", "16-year-old, first visit, booked 35 days ahead",
                 dict(age=16, lead_days=35, prior_visits=0, prior_no_shows=0))]
    rows = []
    for lab, desc, kw in examples:
        rr = {"age": 0.0, "lead_days": 0.0, "prior_visits": 0.0, "prior_no_shows": 0.0, "scholarship": 0, "hypertension": 0,
              "diabetes": 0, "alcoholism": 0, "has_disability": 0, "appt_weekday": "Tuesday"}
        rr.update({k: float(v) if k in ("age", "lead_days", "prior_visits", "prior_no_shows") else v for k, v in kw.items()})
        rr["prior_no_show_rate"] = rr["prior_no_shows"] / rr["prior_visits"] if rr["prior_visits"] else 0.0
        rr["chronic_count"] = float(sum(rr[c] for c in comps)); rr["same_day"] = int(rr["lead_days"] == 0)
        rr["first_visit"] = int(rr["prior_visits"] == 0)
        pe = None
        if bundle is not None:
            try:
                pe = float(bundle["pipeline"].predict_proba(pd.DataFrame([rr])[bundle["features"]])[:, 1][0])
            except Exception:
                pe = None
        rows.append({"Case": lab, "Appointment": desc, "Risk": f"{pe:.0%} → {tier(pe)}" if pe is not None else "see slides"})
    st.table(pd.DataFrame(rows))

# ============================================================================ Fairness
elif page == "Ethical compliance (fairness)":
    st.title("Ethical compliance dashboard")
    st.markdown(f"""
**What fair means here (Module 2 plan).** The benefit of a flag is extra outreach, so the question is whether at-risk
patients in every group are *found* at the same rate. The release threshold is a gap of no more than **{TPR_GAP_MAX:.2f}**
in recall (true-positive rate) between the best- and worst-served group. The flag-rate ratio (four-fifths rule, ≥ {DI_MIN})
and the false-positive-rate gap are shown as diagnostics. Groups are the four the Module 3 bias suite monitors.
""")
    t = st.slider("Decision threshold (the operating point is marked)", 0.05, 0.95, float(round(T_HIGH, 3)), 0.001, format="%.3f")
    overall = metrics_at(S["no_show"].values, S["p_no_show"].values, t)
    c1, c2, c3 = st.columns(3)
    c1.metric("Recall at this threshold", pct(overall["recall"], 1)); c2.metric("Precision", pct(overall["precision"], 1))
    c3.metric("Share flagged", pct(overall["flag_rate"], 1))
    summary_rows = []; tabs = {}
    for attr in ["gender", "age_band", "scholarship", "has_disability"]:
        tab, summ, engine = group_metrics(S, attr, t); tabs[attr] = tab
        summary_rows.append({"Attribute": GROUP_LABEL[attr], "Recall gap": f"{summ['tpr_gap']:.3f}",
                             f"Within {TPR_GAP_MAX:.2f} (release)": "✅ yes" if summ["tpr_gap"] <= TPR_GAP_MAX else "⚠️ no",
                             "False-positive gap": f"{summ['fpr_gap']:.3f}", "Flag-rate ratio": f"{summ['di_ratio']:.2f}",
                             f"≥ {DI_MIN} (diagnostic)": "✅" if summ["di_ratio"] >= DI_MIN else "ℹ️ below"})
    st.dataframe(pd.DataFrame(summary_rows), hide_index=True, use_container_width=True)
    st.caption(f"Computed live from the scored Module 3 test split ({len(S):,} appointments) with {engine}.")
    if plt:
        fig, axes = plt.subplots(1, 4, figsize=(13, 3.2))
        for ax, attr in zip(axes, tabs):
            tab = tabs[attr]
            _lbl = {"F": "Women", "M": "Men", "0": "No", "1": "Yes"}
            ax.bar(tab["group"].astype(str).map(lambda g: _lbl.get(g, g)), tab["recall"], color=TEAL)
            ax.axhline(TARGET_RECALL, color=RED, ls=":", lw=1)
            ax.set_ylim(0, 1); ax.set_title(f"Recall by {GROUP_LABEL[attr].lower()}", fontsize=10)
            ax.tick_params(axis="x", labelsize=8, rotation=30); ax.spines[["top", "right"]].set_visible(False)
        axes[0].set_ylabel("share of true no-shows flagged")
        st.pyplot(fig, use_container_width=True)
    with st.expander("Group tables (n, no-show base rate, flag rate, recall, false-positive rate, precision)"):
        for attr, tab in tabs.items():
            st.markdown(f"**{GROUP_LABEL[attr]}**")
            st.dataframe(tab.round(3), hide_index=True, use_container_width=True)
    st.markdown("#### What was found at the operating threshold, and what was done")
    fb = R["fairness_before"]; mit = R["mitigation"]
    worst = max(fb, key=lambda a: fb[a]["tpr_gap"])
    st.markdown(f"- Largest recall gap in the Module 4 audit: **{GROUP_LABEL.get(worst, worst)}**, gap "
                f"{fb[worst]['tpr_gap']:.3f} ({'within' if fb[worst]['tpr_gap'] <= TPR_GAP_MAX else 'above'} the release threshold).")
    if mit.get("applied"):
        a = mit["after"][mit["attribute"]]; pa = mit["performance_after"]
        st.markdown(f"- **Mitigation applied** on {GROUP_LABEL.get(mit['attribute'], mit['attribute'])}: group-specific thresholds "
                    f"chosen on the validation split so each group reaches {TARGET_RECALL:.0%} recall. Gap after: "
                    f"{a['tpr_gap']:.3f}; overall recall {pct(pa['recall'], 1)}, flag rate {pct(pa['flag_rate'], 1)}. "
                    "Using the attribute at decision time is acceptable only because the decision adds support; it needs "
                    "ethics-committee approval before use.")
        if (FIG / "fairness_after.png").exists():
            st.image(str(FIG / "fairness_after.png"), caption="Group metrics after mitigation")
    else:
        st.markdown(f"- **No mitigation was needed**: {mit.get('reason', '')}. The routine stays in the pipeline and runs at every retraining.")
    st.markdown("- The disability group is only ~2 % of records (flagged in Module 3), so its estimates are less precise "
                "and are watched separately.")

# ============================================================================ Monitoring (Final Project)
elif page == "Monitoring (drift & fairness)":
    import json as _json
    st.title("Monitoring: data drift, prediction drift and fairness drift")
    st.markdown("Reads `monitoring/outputs/monitoring_summary.json`, written by `monitoring/monitor.py` (run monthly by the "
                "`retrain` GitHub Actions workflow or the `monitoring` service in docker-compose). Thresholds come from the "
                "Module 4 model card and the Module 2 fairness release criteria.")
    mon_paths = [HERE.parent / "monitoring" / "outputs" / "monitoring_summary.json", DATA / "monitoring_summary.json"]
    mon = next((p for p in mon_paths if p.exists()), None)
    if mon is None:
        st.warning("No monitoring summary found. Run `python monitoring/monitor.py --demo` from the repository root.")
    else:
        M = _json.loads(mon.read_text())
        batches = {"Latest batch": M.get("latest", M), "Stress test (simulated +3-day lead-time shift)": M.get("stress_test")}
        batches = {k: v for k, v in batches.items() if v}
        choice = st.radio("Batch", list(batches), horizontal=True)
        S = batches[choice]
        colour = {"OK": "green", "WARN": "orange", "ALERT": "red"}[S["status"]]
        st.markdown(f"### Status: :{colour}[{S['status']}]  \n*{S['batch']}* · generated {S['generated_at']} · "
                    f"reference n = {S['n_reference']:,}, current n = {S['n_current']:,}")
        st.info(S["action"])
        if S["alerts"]:
            st.error("**Alerts**\n\n" + "\n".join(f"- {a}" for a in S["alerts"]))
        if S["warnings"]:
            st.warning("**Warnings**\n\n" + "\n".join(f"- {a}" for a in S["warnings"]))
        c1, c2, c3 = st.columns(3)
        pdft = S["prediction_drift"]; pc = S.get("performance_current") or {}; pr = S.get("performance_reference") or {}
        c1.metric("Prediction PSI", f"{pdft['psi']:.3f}", "alert ≥ 0.20")
        c2.metric("Flag rate", pct(pdft["cur_flag_rate"]), f"reference {pct(pdft['ref_flag_rate'])}")
        if pc:
            c3.metric("Recall (current)", pct(pc["recall"]), f"reference {pct(pr.get('recall', 0))}; floor 65%")
        st.markdown("#### Feature drift (population stability index)")
        fd = pd.DataFrame(S["feature_drift"])
        st.dataframe(fd, use_container_width=True, hide_index=True)
        st.markdown("#### Fairness on the current batch (deployed age-band thresholds)")
        rows = []
        for attr, f in S["fairness_current"].items():
            rows.append({"attribute": attr, "TPR gap": f.get("tpr_gap"), "FPR gap": f.get("fpr_gap"),
                         "DP difference": f.get("dp_difference"), "DI ratio": f.get("di_ratio"),
                         "release rule (TPR gap ≤ 0.10)": "pass" if (f.get("tpr_gap") or 0) <= 0.10 else "FAIL"})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        with st.expander("Per-group detail"):
            for attr, f in S["fairness_current"].items():
                st.markdown(f"**{attr}**"); st.dataframe(pd.DataFrame(f["groups"]), hide_index=True)
        st.caption("Limits: PSI warn 0.10 / alert 0.20; recall floor 0.65; TPR gap ≤ 0.10 (release rule); DI ratio ≥ 0.80 "
                   "(diagnostic). An ALERT opens the incident process in docs/incident_response.md.")

# ============================================================================ Limitations
elif page == "Limitations & transparency":
    st.title("Limitations and transparency statement")
    st.markdown(f"""
#### What the model can do
- Rank scheduled referral/follow-up appointments by likelihood of being missed, 24–72 h ahead, and group them into tiers.
- Explain each score with its main drivers (reason codes) so a coordinator knows *why* a call is suggested.
- Catch about {pct(CR['recall'])} of no-shows while flagging about {pct(CR['flag_rate'])} of appointments (test split).

#### What it cannot do
- Tell you *that* a specific patient will miss: most flagged patients would still have attended
  (precision {pct(CR['precision'])}); a flag is a reason to reach out, not a judgement.
- Reflect Jigjiga yet: it was trained on a Brazilian proxy dataset (Hoppen, 2016) because hospital data were not
  available. It is a validated **method**, not a deployable JJUSHYCSH model, until retrained on local records.
- See distance, transport, season or phone coverage - likely decisive here, absent from the proxy data.
- Capture seasonality: the data cover six weeks.
- Be used to cancel, overbook against or deprioritise anyone - the governance policy forbids it and the API says so.

#### Transparency commitments (Module 1 charter, Module 2 privacy plan)
- Every score is advisory; staff keep full discretion and overrides are logged.
- Gender and geography are not inputs; fairness is monitored by gender, age band, welfare status and disability.
- Pseudonymised data remain personal data under Ethiopia's Personal Data Protection Proclamation No. 1321/2024; the
  Module 3 access tiers apply to scores as well.
- Oversight: Hospital Research & Ethics Review Committee and the AI & Data Governance Committee.
- Monitoring after any deployment: monthly recall and per-group recall gap, weekly flag rate vs outreach capacity,
  monthly input drift, quarterly calibration and override review.
""")
    st.markdown("#### Model card")
    mc = DATA / "model_card.md"
    if mc.exists():
        with st.expander("Open the Module 4 Model Card"):
            st.markdown(mc.read_text(encoding="utf-8"))

# ============================================================================ About
elif page == "About the data":
    st.title("About the data and the pipeline")
    q = R.get("m3", {}).get("data_quality_report", {})
    st.markdown(f"""
- **Source:** Medical Appointment No Shows (Hoppen, 2016), 110,527 appointments from Brazil, used as a disclosed proxy.
- **Module 3 pipeline:** ingest → validate → clean → anonymise (salted keys, k = 20 areas) → features → validate → publish.
  {f"{q['rows_received']['rows_affected']:,} records in, {q['rows_after_cleaning']['rows_affected']:,} model-ready rows." if q else ""}
- **Split:** patient-grouped 70/15/15 (no patient in two sets): train {R['n_train']:,}, validation {R['n_val']:,},
  test {R['n_test']:,}; no-show rate {pct(R['test_no_show'], 1)} in the test split.
- **Features used:** {', '.join(PRETTY.get(f, f) for f in R['features'])}.
- **Excluded on purpose:** SMS sent (it is the intervention), area code (geography must not become a penalty), gender
  (audit only), calendar month, and - after the necessity review - alcohol-use disorder and disability.
- **Data hash:** `{R['data_md5'][:12]}…` tagged on every MLflow run.
- **This dashboard** reads `results.json`, the figures and a scored copy of the test split with no identifiers.
""")
    st.markdown("**Sources:** Ahmed, A. F. (2026a-d) BAN6800 Modules 1–4; Hoppen (2016); Hardt, Price & Srebro (2016); "
                "Lundberg & Lee (2017); Mitchell et al. (2019); Weerts et al. (2023); Federal Ministry of Justice, Ethiopia (2024).")
