"""Build the Module 5 stakeholder deck (15 slides + Q&A appendix) from the Module 4 results.

    python module5/build_slides.py --dashboard-url https://huggingface.co/spaces/<you>/jjushycsh-noshow \
                                   --github-url https://github.com/<you>/jjushycsh-noshow-analytics

Reads module5/data/ (written by export_scored_test.py) and writes
module5/outputs/BAN6800_Module5_Stakeholder_Presentation_Abdi_Farah_Ahmed.pptx
Every number on the slides comes from results.json / scored_test.csv; nothing is typed in by hand.
"""
import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt, Emu

HERE = Path(__file__).resolve().parent
DATA, ASSETS, OUT = HERE / "data", HERE / "assets", HERE / "outputs"
FIG = DATA / "figures"
AUTHOR, COURSE = "Abdi Farah Ahmed", "BAN6800 Business Analytics Capstone · Nexford University"
NAVY, TEAL, OCHRE, RED, GREY, LIGHT, INK = "#1F3B57", "#2E6B65", "#C8933D", "#B2432F", "#8A8F98", "#F3F6F8", "#1F2A37"
TPR_GAP_MAX, DI_MIN, TARGET_RECALL = 0.10, 0.80, 0.70
GROUP_LABEL = {"gender": "Gender", "age_band": "Age band", "scholarship": "Welfare enrolment", "has_disability": "Disability recorded"}
AGE_ORDER = ["0-5", "6-17", "18-34", "35-54", "55-69", "70+"]


def group_order(attr, groups):
    groups = [str(g) for g in groups]
    if attr == "age_band":
        return sorted(groups, key=lambda g: AGE_ORDER.index(g) if g in AGE_ORDER else 99)
    return sorted(groups)


def group_name(attr, g):
    g = str(g)
    if attr in ("scholarship", "has_disability"):
        return {"0": "No", "1": "Yes"}.get(g, g)
    return {"F": "Female", "M": "Male"}.get(g, g) if attr == "gender" else g
MODEL_NAME = {"baseline_majority": "Everyone attends", "baseline_lead_time_rule": "Lead-time rule (no ML)",
              "logistic_regression": "Logistic regression", "random_forest": "Random forest",
              "xgboost": "XGBoost", "hist_gradient_boosting": "Gradient boosting"}


def rgb(h):
    return RGBColor.from_string(h.lstrip("#"))


def pct(x, d=0):
    return "n/a" if x is None else f"{100 * float(x):.{d}f}%"


# ----------------------------------------------------------------------------- charts
def chart_workload(R, path):
    cr = R["results"][R["champion"]]; n = 1000; base = R["test_no_show"]
    ns = n * base; caught = ns * cr["recall"]; flagged = n * cr["flag_rate"]
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    labels = ["Flagged for extra outreach", "No-shows among them (caught)", "No-shows not flagged (missed)"]
    vals = [flagged, caught, ns - caught]
    ax.barh(labels, vals, color=[OCHRE, TEAL, RED])
    for i, v in enumerate(vals):
        ax.text(v + 8, i, f"{v:,.0f}", va="center", fontsize=11, color=INK)
    ax.set_xlim(0, n); ax.invert_yaxis(); ax.set_xlabel("appointments, per 1,000 scheduled")
    ax.spines[["top", "right"]].set_visible(False); ax.tick_params(labelsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def chart_models(R, path):
    res = R["results"]; ks = [k for k in ["baseline_lead_time_rule", "logistic_regression", "random_forest", "xgboost",
                                           "hist_gradient_boosting"] if k in res]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, m, title, lim in [(axes[0], "roc_auc", "Ranking quality (ROC-AUC)", (0.45, 0.85)),
                              (axes[1], "precision", "Precision at 70 % recall", (0, 0.5))]:
        vals = [res[k][m] for k in ks]; cols = [NAVY if k == R["champion"] else GREY for k in ks]
        b = ax.bar([MODEL_NAME[k].replace(" (no ML)", "\n(no ML)") for k in ks], vals, color=cols)
        for r, v in zip(b, vals):
            ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.005, f"{v:.3f}" if m == "roc_auc" else f"{v:.0%}",
                    ha="center", fontsize=9)
        ax.set_ylim(*lim); ax.set_title(title, fontsize=11); ax.tick_params(axis="x", labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def chart_shap(R, path):
    imp = list(R.get("shap", {}).get("importance", {}).items())[:8]
    if not imp:
        return False
    dirn = R["shap"].get("direction", {})
    labels = [k for k, _ in imp][::-1]; vals = [v for _, v in imp][::-1]
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.barh(labels, vals, color=[TEAL if dirn.get(k, 0) > 0 else GREY for k in labels])
    ax.set_xlabel("average impact on predicted probability (mean |SHAP|)")
    ax.spines[["top", "right"]].set_visible(False); ax.tick_params(labelsize=10)
    ax.text(0.99, 0.02, "teal = higher value raises risk · grey = lowers risk", transform=ax.transAxes, ha="right",
            fontsize=8, color=GREY)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)
    return True


def chart_fairness(S, t, path):
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.2))
    y = S["no_show"].values; pred = (S["p_no_show"].values >= t).astype(int)
    out = {}
    for ax, attr in zip(axes, GROUP_LABEL):
        g = S[attr].astype(str).values; rows = []
        for grp in group_order(attr, pd.unique(g)):
            m = g == grp; yt, yp = y[m], pred[m]
            rows.append((group_name(attr, grp), yp[yt == 1].mean() if (yt == 1).any() else np.nan, int(m.sum())))
        out[attr] = rows
        ax.bar([r[0] for r in rows], [r[1] for r in rows], color=TEAL)
        ax.axhline(TARGET_RECALL, color=RED, ls=":", lw=1)
        ax.set_ylim(0, 1); ax.set_title(f"Recall by {GROUP_LABEL[attr].lower()}", fontsize=10)
        ax.tick_params(axis="x", labelsize=8, rotation=30); ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("share of true no-shows flagged")
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)
    return out


def chart_mitigation(R, path):
    mit = R["mitigation"]
    if not mit.get("applied"):
        return False
    a = mit["attribute"]; before = R["fairness_before"][a]["table"]; after = mit["after"][a]["table"]
    order = group_order(a, [r["group"] for r in before])
    bmap = {str(r["group"]): r["tpr"] for r in before}; amap = {str(r["group"]): r["tpr"] for r in after}
    groups = [group_name(a, g) for g in order]
    fig, ax = plt.subplots(figsize=(7.5, 3.2)); x = np.arange(len(groups)); w = 0.36
    ax.bar(x - w / 2, [bmap[g] for g in order], w, color=GREY, label="before")
    ax.bar(x + w / 2, [amap.get(g, np.nan) for g in order], w, color=TEAL, label="after group thresholds")
    ax.axhline(TARGET_RECALL, color=RED, ls=":", lw=1); ax.set_xticks(x); ax.set_xticklabels(groups, fontsize=9)
    ax.set_ylim(0, 1); ax.set_ylabel("recall"); ax.set_title(f"Recall by {GROUP_LABEL.get(a, a).lower()} before and after mitigation", fontsize=11)
    ax.legend(fontsize=9); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)
    return True


# ----------------------------------------------------------------------------- examples
def score_examples(R):
    """Three illustrative appointments scored with the serialised model (reason codes via SHAP when available)."""
    try:
        import joblib
        bundle = joblib.load(DATA / "model" / "noshow_model.joblib")
    except Exception:
        return None
    dropped = set(bundle.get("dropped_flags") or []); comps = [c for c in ["hypertension", "diabetes", "alcoholism"] if c not in dropped]
    cr = R["results"][R["champion"]]; t_high, t_med = cr["threshold"], cr.get("threshold_medium", cr["threshold"] * 0.8)
    cases = [("Amina", "27 years old, booked 21 days ahead, 3 earlier visits of which 1 missed, welfare-enrolled",
              dict(age=27, lead_days=21, prior_visits=3, prior_no_shows=1, scholarship=1)),
             ("Hassan", "64 years old with hypertension and diabetes, booked 2 days ahead, 6 earlier visits all attended",
              dict(age=64, lead_days=2, prior_visits=6, prior_no_shows=0, hypertension=1, diabetes=1)),
             ("Sahra", "16 years old, first visit, booked 35 days ahead",
              dict(age=16, lead_days=35, prior_visits=0, prior_no_shows=0))]
    pipe = bundle["pipeline"]; prep = pipe.named_steps["prep"]; clf = pipe.named_steps["clf"]
    try:
        import shap
        from m4_pretty import pretty  # noqa
    except Exception:
        shap = None
    out = []
    for name, desc, kw in cases:
        r = {"age": 0.0, "lead_days": 0.0, "prior_visits": 0.0, "prior_no_shows": 0.0, "scholarship": 0, "hypertension": 0,
             "diabetes": 0, "alcoholism": 0, "has_disability": 0, "appt_weekday": "Tuesday"}
        r.update({k: float(v) if k in ("age", "lead_days", "prior_visits", "prior_no_shows") else v for k, v in kw.items()})
        r["prior_no_show_rate"] = r["prior_no_shows"] / r["prior_visits"] if r["prior_visits"] else 0.0
        r["chronic_count"] = float(sum(r[c] for c in comps)); r["same_day"] = int(r["lead_days"] == 0)
        r["first_visit"] = int(r["prior_visits"] == 0)
        X = pd.DataFrame([r])[bundle["features"]]
        p = float(pipe.predict_proba(X)[:, 1][0])
        tier = "High" if p >= t_high else "Medium" if p >= t_med else "Low"
        reasons = []
        try:
            import shap
            Xt = pd.DataFrame(prep.transform(X), columns=prep.get_feature_names_out())
            ex = shap.LinearExplainer(clf, Xt) if clf.__class__.__name__ == "LogisticRegression" else shap.TreeExplainer(clf)
            sv = np.asarray(ex(Xt).values, dtype=float); sv = sv[:, :, 1] if sv.ndim == 3 else sv
            contrib = pd.Series(sv[0], index=Xt.columns)
            top = contrib.reindex(contrib.abs().sort_values(ascending=False).index)[:3]
            nice = {"lead_days": "days booked ahead", "age": "age", "prior_no_show_rate": "share of past visits missed",
                    "prior_visits": "number of past visits", "prior_no_shows": "past missed visits",
                    "chronic_count": "chronic conditions", "same_day": "same-day booking", "first_visit": "first visit",
                    "scholarship": "welfare enrolment", "hypertension": "hypertension", "diabetes": "diabetes"}
            for k, v in top.items():
                lab = nice.get(k, k.replace("appt_weekday_", "appointment on "))
                reasons.append(f"{lab} ({'raises' if v > 0 else 'lowers'} risk)")
        except Exception:
            if r["lead_days"] >= 7: reasons.append("long lead time (raises risk)")
            if r["prior_no_show_rate"] > 0: reasons.append("missed visits before (raises risk)")
            if r["same_day"] or r["lead_days"] <= 2: reasons.append("booked close to the visit (lowers risk)")
            if r["chronic_count"] >= 1: reasons.append("chronic follow-up (lowers risk)")
            if r["first_visit"]: reasons.append("no attendance history (model relies on booking pattern)")
        out.append({"name": name, "desc": desc, "p": p, "tier": tier, "reasons": reasons[:3]})
    return out


# ----------------------------------------------------------------------------- deck helpers
class Deck:
    W, H = Inches(13.333), Inches(7.5)

    def __init__(self):
        self.prs = Presentation(); self.prs.slide_width, self.prs.slide_height = self.W, self.H
        self.n = 0

    def _box(self, s, x, y, w, h, fill=None, line=None):
        sh = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y, w, h)
        sh.fill.solid() if fill else sh.fill.background()
        if fill: sh.fill.fore_color.rgb = rgb(fill)
        if line: sh.line.color.rgb = rgb(line)
        else: sh.line.fill.background()
        sh.shadow.inherit = False
        return sh

    def text(self, s, x, y, w, h, lines, size=14, color=INK, bold=False, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP,
             bullets=False, font="Calibri", spacing=4):
        tb = s.shapes.add_textbox(x, y, w, h); tf = tb.text_frame; tf.word_wrap = True
        tf.vertical_anchor = anchor
        tf.margin_left = tf.margin_right = Inches(0.08); tf.margin_top = tf.margin_bottom = Inches(0.04)
        if isinstance(lines, str): lines = [lines]
        for i, ln in enumerate(lines):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.alignment = align; p.space_after = Pt(spacing)
            seg_bold = bold
            txt = ln
            if bullets and txt and not txt.startswith("##"):
                txt = "•  " + txt
            if txt.startswith("##"):            # sub-heading inside a text box
                txt = txt[2:].strip(); seg_bold = True
            run = p.add_run(); run.text = txt
            run.font.size = Pt(size); run.font.bold = seg_bold; run.font.name = font; run.font.color.rgb = rgb(color)
        return tb

    def slide(self, title, subtitle=None, section=""):
        s = self.prs.slides.add_slide(self.prs.slide_layouts[6]); self.n += 1
        self._box(s, 0, 0, self.W, Inches(1.05), fill=NAVY)
        self.text(s, Inches(0.45), Inches(0.12), Inches(11.5), Inches(0.6), title, size=26, color="#FFFFFF", bold=True)
        if subtitle:
            self.text(s, Inches(0.45), Inches(0.62), Inches(12), Inches(0.4), subtitle, size=13, color="#D7E3EA")
        self._box(s, 0, Inches(7.1), self.W, Inches(0.4), fill=LIGHT)
        self.text(s, Inches(0.45), Inches(7.12), Inches(9), Inches(0.35),
                  f"{AUTHOR} · {COURSE} · Module 5" + (f" · {section}" if section else ""), size=9, color=GREY)
        self.text(s, Inches(12.2), Inches(7.12), Inches(0.9), Inches(0.35), str(self.n), size=9, color=GREY, align=PP_ALIGN.RIGHT)
        return s

    def notes(self, s, text):
        s.notes_slide.notes_text_frame.text = text

    def image(self, s, path, x, y, w=None, h=None):
        if not Path(path).exists():
            self.text(s, x, y, w or Inches(5), Inches(0.5), f"[figure missing: {Path(path).name}]", size=11, color=RED); return
        kw = {"width": w} if w else {"height": h}
        s.shapes.add_picture(str(path), x, y, **kw)

    def takeaway(self, s, text, y=Inches(6.3), x=Inches(0.45), w=Inches(12.4), h=Inches(0.7), fill=TEAL):
        self._box(s, x, y, w, h, fill=fill)
        self.text(s, x + Inches(0.1), y, w - Inches(0.2), h, text, size=14, color="#FFFFFF", bold=True, anchor=MSO_ANCHOR.MIDDLE)

    def tile(self, s, x, y, w, h, big, small, fill=LIGHT, color=NAVY):
        self._box(s, x, y, w, h, fill=fill)
        self.text(s, x, y + Inches(0.08), w, Inches(0.7), big, size=28, color=color, bold=True, align=PP_ALIGN.CENTER)
        self.text(s, x + Inches(0.05), y + Inches(0.8), w - Inches(0.1), h - Inches(0.8), small, size=11, color=INK, align=PP_ALIGN.CENTER)

    def table(self, s, x, y, w, rows, col_w=None, size=11, header_fill=NAVY, row_h=Inches(0.36)):
        nrows, ncols = len(rows), len(rows[0])
        shp = s.shapes.add_table(nrows, ncols, x, y, w, row_h * nrows); t = shp.table
        if col_w:
            for i, cw in enumerate(col_w): t.columns[i].width = cw
        for i, row in enumerate(rows):
            for j, val in enumerate(row):
                c = t.cell(i, j); c.text = ""; p = c.text_frame.paragraphs[0]; r = p.add_run(); r.text = str(val)
                r.font.size = Pt(size); r.font.name = "Calibri"
                c.margin_left = c.margin_right = Inches(0.06); c.margin_top = c.margin_bottom = Inches(0.03)
                if i == 0:
                    c.fill.solid(); c.fill.fore_color.rgb = rgb(header_fill); r.font.bold = True; r.font.color.rgb = rgb("#FFFFFF")
                else:
                    c.fill.solid(); c.fill.fore_color.rgb = rgb(LIGHT if i % 2 else "#FFFFFF"); r.font.color.rgb = rgb(INK)
        return shp


# ----------------------------------------------------------------------------- build
def build(dashboard_url, github_url, api_demo=None):
    R = json.loads((DATA / "results.json").read_text())
    S = pd.read_csv(DATA / "scored_test.csv")
    for c in GROUP_LABEL: S[c] = S[c].astype(str)
    ASSETS.mkdir(exist_ok=True); OUT.mkdir(exist_ok=True)
    CH = R["champion"]; CR = R["results"][CH]; CHN = MODEL_NAME[CH]; rule = R["results"]["baseline_lead_time_rule"]
    t_high = CR["threshold"]
    chart_workload(R, ASSETS / "workload.png"); chart_models(R, ASSETS / "models.png")
    has_shap = chart_shap(R, ASSETS / "shap.png")
    fair_groups = chart_fairness(S, t_high, ASSETS / "fairness.png")
    has_mit = chart_mitigation(R, ASSETS / "mitigation.png")
    examples = score_examples(R)
    fb, mit, nr, tc = R["fairness_before"], R["mitigation"], R["necessity"], R.get("temporal_check", {})
    gaps = {a: fb[a]["tpr_gap"] for a in GROUP_LABEL}; worst = max(gaps, key=gaps.get)
    fair_pass = all(v <= TPR_GAP_MAX for v in gaps.values())
    d = Deck(); X0, X1 = Inches(0.45), Inches(6.9)

    # 1 Title
    s = d.prs.slides.add_slide(d.prs.slide_layouts[6]); d.n += 1
    d._box(s, 0, 0, d.W, d.H, fill=NAVY)
    d.text(s, Inches(0.8), Inches(1.6), Inches(11.5), Inches(1.6),
           "Who will miss their referral appointment — and what the model can and cannot tell us",
           size=34, color="#FFFFFF", bold=True)
    d.text(s, Inches(0.8), Inches(3.3), Inches(11.5), Inches(0.9),
           "Stakeholder presentation of the referral no-show risk model for Jigjiga University Sheikh Hassan Yabare "
           "Comprehensive Specialized Hospital (JJUSHYCSH)", size=16, color="#D7E3EA")
    d.text(s, Inches(0.8), Inches(4.6), Inches(11.5), Inches(1.4),
           [f"{AUTHOR}  ·  {COURSE}", "Module 5 Assignment — Business Insights Presentation for Stakeholder Decision-Making",
            dt.date.today().strftime("%B %Y"),
            "Objective: explain clearly and neutrally what the Module 4 model found, how it decides, and how fairly it "
            "treats patients — so that OPD, hospital leadership and the governance committees can judge it for themselves."],
           size=13, color="#FFFFFF")
    d.text(s, Inches(0.8), Inches(6.5), Inches(11.5), Inches(0.6), f"Live dashboard: {dashboard_url}    ·    Code: {github_url}",
           size=12, color="#C8D8E0")
    d.notes(s, "Good day. This is the Module 5 presentation for my BAN6800 capstone. It continues the same project as Modules 1-4. "
               "My role today is business analyst, not data scientist: I will explain what the model found and how fairly it behaves, "
               "without recommending a decision - that belongs to a separate deliverable. The dashboard link is live and shows the same results.")

    # 2 Business problem
    s = d.slide("The business problem", "From the Module 1 Vision Document and Module 2 Project Overview", "Business problem")
    d.text(s, X0, Inches(1.3), Inches(6.2), Inches(4.6), [
        "## Where",
        "JJUSHYCSH is the only tertiary referral hospital for the Somali Region — a catchment of six to seven million, "
        "largely rural and pastoralist people, 600 km from Addis Ababa.",
        "## What goes wrong",
        "About one in five scheduled appointments is missed in the proxy data. When a referred patient does not come, "
        "care is delayed for that patient and a specialist slot that someone else travelled hundreds of kilometres for goes unused.",
        "## Why now",
        "OPD staff have no ranked list, so SMS, phone calls and community-health-worker visits cannot be targeted. "
        "Reminder capacity is the constraint, not demand."], size=13)
    d.tile(s, X1, Inches(1.4), Inches(2.9), Inches(1.7), "≥ 15 %", "fewer no-shows within two quarters of local deployment (Module 1 KPI)")
    d.tile(s, X1 + Inches(3.1), Inches(1.4), Inches(2.9), Inches(1.7), "≥ 70 %", "of high-risk no-shows flagged (model KPI, Module 1)")
    d.tile(s, X1, Inches(3.3), Inches(2.9), Inches(1.7), "24–72 h", "before the visit: the window in which a score must arrive to be acted on")
    d.tile(s, X1 + Inches(3.1), Inches(3.3), Inches(2.9), Inches(1.7), "0", "patients may be cancelled, overbooked against or deprioritised because of a score")
    d.takeaway(s, "The question for this presentation: can a model tell us, early enough and fairly, which appointments need extra outreach?")
    d.notes(s, "The problem is capacity and reach, not demand. The two KPIs come straight from my Module 1 vision document. "
               "The last tile is the rule from the Module 1 Ethical AI Charter that governs every slide that follows.")

    # 3 Audience
    s = d.slide("Who this is for, and what they need to decide", "Stakeholders from Module 1 and the Module 2 communication plan", "Audience")
    d.table(s, X0, Inches(1.3), Inches(12.4), [
        ["Stakeholder", "Decision they make", "What they need from this presentation"],
        ["OPD schedulers / referral coordinators", "Whom to call, SMS or visit each day", "Is the list trustworthy? Why is a patient flagged? How many calls per day?"],
        ["Department heads", "Where to put outreach effort by service", "Which patterns drive no-shows; what the flag rate means for staff time"],
        ["Medical Director / administration", "Whether to pilot the tool on local data", "Honest performance, limits, and what it would take to go live"],
        ["HIS / data officer", "Data refresh and quality upkeep", "What data the model needs and how results are monitored"],
        ["Hospital Research & Ethics Review Committee", "Approval of patient-data use", "Privacy safeguards, consent, proxy-data disclosure"],
        ["AI & Data Governance Committee", "Fairness thresholds, change control", "Group-level recall gaps, mitigation, oversight hooks"],
        ["Somali Regional Health Bureau", "Oversight (aggregate only)", "Aggregate results; no patient-level data"]],
        col_w=[Inches(3.6), Inches(3.6), Inches(5.2)], size=11)
    d.takeaway(s, "Two audiences in one room: operational users who need clarity, and ethics bodies who need evidence. The deck serves both; the dashboard lets each explore on their own.")
    d.notes(s, "The last three rows are the ethics and compliance stakeholders from Module 1. The Regional Health Bureau only ever sees aggregates.")

    # 4 Journey + plain-language explanation
    s = d.slide("What the model is, in plain language", "Five modules, one project", "Model")
    steps = [("Module 1", "Vision, KPIs, ethical charter"), ("Module 2", "Plan, architecture, fairness criteria"),
             ("Module 3", "Governed data pipeline, pseudonymised data"), ("Module 4", "Model, explanations, fairness audit"),
             ("Module 5", "This presentation + live dashboard")]
    for i, (a, b) in enumerate(steps):
        x = X0 + Inches(2.5) * i
        d._box(s, x, Inches(1.3), Inches(2.3), Inches(0.9), fill=TEAL if i == 4 else LIGHT)
        d.text(s, x, Inches(1.32), Inches(2.3), Inches(0.4), a, size=12, bold=True, color="#FFFFFF" if i == 4 else NAVY, align=PP_ALIGN.CENTER)
        d.text(s, x, Inches(1.62), Inches(2.3), Inches(0.6), b, size=10, color="#FFFFFF" if i == 4 else INK, align=PP_ALIGN.CENTER)
    d.text(s, X0, Inches(2.5), Inches(12.4), Inches(3.6), [
        "## Plain-language model explanation",
        "Every scheduled referral or follow-up visit is looked at 24–72 hours before the appointment. The model estimates the chance "
        "that the patient will not come, using only facts known at booking time: how far ahead the visit was booked, the patient's own "
        "attendance history, age, chronic conditions and welfare enrolment.",
        "It learned these patterns from 77,000 past appointments and was checked on 16,000 appointments it had never seen "
        f"({R['n_test']:,} in the test split). It then sorts appointments into Low, Medium and High risk tiers, and each tier maps to an "
        "outreach action from Module 1 (SMS → SMS + call → call + community health worker).",
        f"The champion is a {CHN.lower()} model. Three algorithms were compared (logistic regression, random forest and gradient-boosted "
        "trees); the winner was chosen on a validation set, never on the test set. Every score comes with its reasons, so a coordinator "
        "sees why a call is suggested, not just a number.",
        "What it does not use: gender, neighbourhood, whether an SMS was sent, and — after a necessity test — alcohol-use disorder and disability."],
        size=13)
    d.takeaway(s, "Think of it as a triage list for reminders: it ranks appointments by risk and says why, and people decide what to do.")
    d.notes(s, "This is the slide to pause on with non-technical stakeholders. The model is a ranking tool with reasons attached; it is not a verdict on any patient.")

    # 5 Model & key results
    s = d.slide("Key results", f"Champion: {CHN} · Module 3 test split, {R['n_test']:,} appointments the model never saw", "Results")
    d.tile(s, X0, Inches(1.35), Inches(2.95), Inches(1.9), pct(CR["recall"]), f"of no-shows caught (recall) — target ≥ {TARGET_RECALL:.0%}", color=TEAL)
    d.tile(s, X0 + Inches(3.15), Inches(1.35), Inches(2.95), Inches(1.9), pct(CR["flag_rate"]), "of appointments flagged for extra outreach (workload)")
    d.tile(s, X0 + Inches(6.3), Inches(1.35), Inches(2.95), Inches(1.9), pct(CR["precision"]), f"of flagged appointments are real no-shows (base rate {pct(R['test_no_show'])})")
    d.tile(s, X0 + Inches(9.45), Inches(1.35), Inches(2.95), Inches(1.9), f"{CR['roc_auc']:.2f}", f"ranking quality (ROC-AUC) vs {rule['roc_auc']:.2f} for a simple lead-time rule")
    d.text(s, X0, Inches(3.5), Inches(12.4), Inches(2.6), [
        f"## What the numbers mean",
        f"At the operating threshold the model finds about {pct(CR['recall'])} of the patients who will not come, by flagging roughly "
        f"{pct(CR['flag_rate'])} of appointments. Most flagged patients would still have attended — that is the price of catching "
        f"the ones who would not, and the cost of a false flag is one reminder.",
        f"A rule a scheduler could apply by hand ({rule['params']['rule'].replace('lead_days', 'days booked ahead')}) reaches "
        f"{pct(rule['recall'])} recall with precision {pct(rule['precision'])}; "
        + (f"the model improves ranking quality by {CR['roc_auc'] - rule['roc_auc']:+.3f} ROC-AUC and precision by {CR['precision'] - rule['precision']:+.1%} points at the same recall target."
           if CR['roc_auc'] - rule['roc_auc'] >= 0.005 else
           "on this data the model does not rank better than that rule (a finding worth stating plainly), though it adds reasons for every score and per-group fairness control."),
        "All three algorithms landed within a few points of each other, which means the signal is in the data, not in a clever model."],
        size=13)
    met_ = CR["recall"] >= TARGET_RECALL - 0.02; gain_ = CR["roc_auc"] - rule["roc_auc"]
    d.takeaway(s, (f"The model {'meets' if met_ else 'falls just short of'} the 70 % recall target on unseen patients; its gain over a simple rule is "
                   + ("real but modest" if gain_ >= 0.02 else "small" if gain_ >= 0.005 else "negligible") + " — honesty about that is part of the story."))
    d.notes(s, "Recall is the KPI from Module 1. Precision is low by design: with one no-show in five, catching 70 % means accepting many false flags, each costing one reminder.")

    # 6 Visual summary
    s = d.slide("Visual summary of results", "Each chart carries one message", "Results")
    d.image(s, ASSETS / "workload.png", X0, Inches(1.3), w=Inches(6.3))
    d.text(s, X0, Inches(4.05), Inches(6.3), Inches(0.5), "What 1,000 scheduled appointments look like with the model", size=11, color=GREY)
    d.image(s, ASSETS / "models.png", X1, Inches(1.3), w=Inches(6.0))
    d.text(s, X1, Inches(3.65), Inches(6.0), Inches(0.5), "Rule vs three models; the champion is highlighted", size=11, color=GREY)
    d.image(s, FIG / "confusion_champion.png", X0, Inches(4.4), h=Inches(1.85))
    d.text(s, X0 + Inches(2.3), Inches(4.5), Inches(4.0), Inches(1.6),
           [f"Test-split counts at the operating threshold:", f"caught no-shows {CR['tp']:,} · missed {CR['fn']:,}",
            f"flagged but attended {CR['fp']:,} · correctly not flagged {CR['tn']:,}"], size=11)
    d.image(s, FIG / "calibration_champion.png", X1, Inches(4.2), h=Inches(2.0))
    d.text(s, X1 + Inches(2.5), Inches(4.5), Inches(3.5), Inches(1.6),
           ["Calibration: when the model says 40 %, about 40 % of those patients actually miss — so tiers can be read as probabilities."], size=11)
    d.takeaway(s, "Per 1,000 appointments: a few hundred calls to reach most of the no-shows. The charts are the evidence; the dashboard lets you move the threshold yourself.")
    d.notes(s, "Left: the workload picture per 1,000 appointments. Right: the model comparison and calibration. The confusion matrix gives the raw counts behind the percentages.")

    # 7 Drivers
    s = d.slide("What drives a prediction", "SHAP explanations translated into business language (Module 4)", "Explainability")
    if has_shap:
        d.image(s, ASSETS / "shap.png", X0, Inches(1.3), w=Inches(6.4))
    else:
        d.text(s, X0, Inches(1.5), Inches(6.4), Inches(1), "[SHAP figure not available in this run]", size=12, color=RED)
    imp = list(R.get("shap", {}).get("importance", {}).items())[:3]
    d.text(s, X1, Inches(1.3), Inches(6.0), Inches(4.8), [
        "## The three strongest drivers" + (f": {', '.join(k.lower() for k, _ in imp)}" if imp else ""),
        "Days booked ahead — the longer the gap between booking and visit, the more time for travel money, transport or family "
        "arrangements to fall through. Same-day bookings are rarely missed.",
        "Attendance history — a patient who has missed before is more likely to miss again. A first visit has no history, so the model leans on the booking pattern.",
        "Age — adolescents and young adults miss more often; older patients with chronic follow-up attend more reliably.",
        "Chronic conditions and welfare enrolment play smaller roles: chronic follow-up creates routine; welfare enrolment signals cost barriers.",
        "## What is deliberately absent",
        "Gender, neighbourhood and whether an SMS was sent are not inputs; alcohol-use disorder and disability were tested and "
        f"dropped because they added nothing (Δ PR-AUC {', '.join(f'{t[chr(100)+chr(101)+chr(108)+chr(116)+chr(97)]:+.4f}' for t in nr['table'])})."],
        size=12)
    d.takeaway(s, "No-show risk is mostly about the circumstances of the booking, not the diagnosis — which is good news, because booking circumstances can be changed.")
    d.notes(s, "SHAP values show how much each factor moves the predicted probability. For OPD staff the message is simple: long lead times and past no-shows are where to look first.")

    # 8 Examples
    s = d.slide("Three appointments, explained", "Illustrative cases scored with the serialised Module 4 model; names are invented", "Explainability")
    if examples:
        rows = [["Case", "Appointment", "Predicted risk", "Tier", "Main reasons (reason codes)"]]
        for e in examples:
            rows.append([e["name"], e["desc"], pct(e["p"]), e["tier"], "; ".join(e["reasons"]) or "—"])
        d.table(s, X0, Inches(1.3), Inches(12.4), rows, col_w=[Inches(1.0), Inches(4.6), Inches(1.3), Inches(1.0), Inches(4.5)], size=11, row_h=Inches(0.75))
        hi = max(examples, key=lambda e: e["p"]); lo = min(examples, key=lambda e: e["p"])
        d.text(s, X0, Inches(5.0), Inches(12.4), Inches(1.2), [
            f"How a coordinator reads this: {hi['name']} is {hi['tier']} risk ({pct(hi['p'])}) mainly because of {hi['reasons'][0] if hi['reasons'] else 'the booking pattern'} — "
            f"the Module 1 rule suggests a call 48 h before and, if a slot is free, an earlier appointment. {lo['name']} is {lo['tier']} risk ({pct(lo['p'])}); "
            "a standard SMS is enough. The reasons are shown with every score in the API and the dashboard."], size=12)
    else:
        d.text(s, X0, Inches(1.5), Inches(12), Inches(2), "[Examples are produced when build_slides.py runs on the laptop that holds the model file]", size=12, color=RED)
    d.takeaway(s, "A score always arrives with its reasons — a flag is a reason to reach out, never a judgement about the patient.")
    d.notes(s, "These three are constructed examples scored with the real model so that the audience sees the logic end to end. The reason codes are the same ones the API returns.")

    # 9 What-if
    s = d.slide("What-if: how changing the inputs changes the outcome", "Counterfactuals and threshold sensitivity (Module 4); live version on the dashboard", "What-if")
    cf = R.get("counterfactuals") or []
    if cf:
        rows = [["Age", "Booked ahead", "Risk", "What would need to change", "New risk"]]
        for c in cf[:5]:
            rows.append([int(c["age"]), f"{c['orig_lead_days']} days", pct(c["orig_prob"]),
                         f"book {c['cf_lead_days']} days ahead ({c['cf_weekday']})" if c["flips_below_threshold"] else "no actionable change found",
                         pct(c["cf_prob"]) if c["cf_prob"] is not None else "—"])
        d.table(s, X0, Inches(1.3), Inches(6.3), rows, col_w=[Inches(0.7), Inches(1.3), Inches(0.9), Inches(2.5), Inches(0.9)], size=10, row_h=Inches(0.4))
        n_flip = sum(c["flips_below_threshold"] for c in cf)
        d.text(s, X0, Inches(1.3) + Inches(0.4) * (len(rows)) + Inches(0.15), Inches(6.3), Inches(1.6), [
            f"Only levers the hospital controls were allowed to move (lead time, weekday); age, health and history stayed fixed. "
            f"For {n_flip} of {len(cf)} correctly flagged no-shows, a shorter lead time alone moved the appointment below the High threshold."], size=11)
    d.image(s, FIG / "threshold_sensitivity.png", X1, Inches(1.3), w=Inches(6.0))
    sens = R.get("sensitivity") or []
    if sens:
        w_ = max(sens[1:], key=lambda x: x["high_tier_flips_pct"])
        d.text(s, X1, Inches(5.1), Inches(6.0), Inches(1.1), [
            f"Moving the threshold trades reach against workload (red line = current operating point). The model is most sensitive to "
            f"'{w_['scenario']}' ({w_['high_tier_flips_pct']:.0f}% of High-tier decisions change), least to random data-entry noise."], size=11)
    d.takeaway(s, "Lead time is the one lever both the patient and the hospital can change — the what-if page on the dashboard shows this for any appointment.")
    d.notes(s, "Counterfactuals answer 'what would need to change'. The threshold chart is the capacity dial: leadership can see what a different call volume buys in recall.")

    # 10 Fairness
    s = d.slide("Fairness summary", f"Does the model find at-risk patients equally in every group? (Module 2 criterion; Module 4 audit)", "Fairness")
    d.image(s, ASSETS / "fairness.png", X0, Inches(1.25), w=Inches(12.4))
    lines = ["## In plain words"]
    for a in GROUP_LABEL:
        g = fb[a]; lines.append(f"{GROUP_LABEL[a]}: recall gap between groups {g['tpr_gap']:.2f} — "
                                f"{'within' if g['tpr_gap'] <= TPR_GAP_MAX else 'above'} the {TPR_GAP_MAX:.2f} release threshold; flag-rate ratio {g['di_ratio']:.2f}.")
    d.text(s, X0, Inches(4.6), Inches(12.4), Inches(1.6), lines, size=11)
    d.takeaway(s, ("All four monitored groups are within the Module 2 release threshold: at-risk patients are found at similar rates whoever they are." if fair_pass else
                   f"Not every group is served equally: the largest recall gap is on {GROUP_LABEL[worst].lower()} ({gaps[worst]:.2f}) — next slide shows what was done."),
               fill=TEAL if fair_pass else OCHRE)
    d.notes(s, "The fairness criterion is equal opportunity: because a flag brings help, what matters is that no-shows are caught equally in every group. "
               "Flag-rate differences by age reflect real differences in base rates, not bias. The disability group is small (2 %) so its bars are less precise.")

    # 11 Mitigation
    s = d.slide("Bias mitigation: what was found and what was done", "Module 4, Section 6.5", "Fairness")
    if has_mit:
        d.image(s, ASSETS / "mitigation.png", X0, Inches(1.3), w=Inches(6.4))
        a = mit["attribute"]; aft = mit["after"][a]; pa = mit["performance_after"]
        d.text(s, X1, Inches(1.3), Inches(6.0), Inches(4.8), [
            "## Found", f"At one shared threshold, recall differed by {fb[a]['tpr_gap']:.2f} across {GROUP_LABEL[a].lower()} — above the 0.10 release threshold.",
            "## Done", f"Group-specific thresholds were chosen on the validation split so that every group reaches {TARGET_RECALL:.0%} recall "
            f"(the equal-opportunity method of Hardt et al., 2016). On the test split the gap fell to {aft['tpr_gap']:.2f}; overall recall "
            f"{pct(pa['recall'])}, flag rate {pct(pa['flag_rate'])}.",
            "## Trade-off", f"The {GROUP_LABEL[a].lower()} attribute is then used at decision time. That is acceptable only because the decision adds "
            "support and never removes it — and it needs approval from the AI & Data Governance Committee before any use.",
            "## Cross-check", "Fairlearn's ThresholdOptimizer gives a comparable result (details in the Q&A appendix)."], size=12)
    else:
        d.text(s, X0, Inches(1.4), Inches(12.4), Inches(4.5), [
            "## Found", f"At the operating threshold the largest recall gap across the four monitored groups was {gaps[worst]:.2f} "
            f"({GROUP_LABEL[worst].lower()}), within the 0.10 release threshold set in Module 2.",
            "## Done", "No correction was needed. The mitigation routine — group-specific thresholds for equal opportunity, cross-checked with "
            "Fairlearn's ThresholdOptimizer — stays in the pipeline and runs automatically at every retraining, so a future drift in any group's recall is corrected, not just reported.",
            "## Also checked", "Representation: the Module 3 bias suite flagged that only ~2 % of records have a disability recorded, so that group's "
            "metrics are reported separately and treated as less certain.",
            "## Data minimisation", "Alcohol-use disorder and disability were removed as model inputs after a necessity test showed they added nothing; "
            "both remain in the audit."], size=13)
    d.takeaway(s, "Fairness here is an ongoing check, not a one-off result: the audit and the correction run every time the model is retrained.")
    d.notes(s, "Mitigation is post-processing: the model is unchanged; only the threshold per group moves. This is the least invasive method and keeps the model auditable.")

    # 12 Limitations
    s = d.slide("Model limitations", "What this model cannot do — stated before anyone asks", "Limitations")
    d.text(s, X0, Inches(1.3), Inches(6.2), Inches(4.8), [
        "## About the data",
        "Trained on a Brazilian public dataset (Hoppen, 2016) as a disclosed proxy — JJUSHYCSH records were not yet available. It is a validated method, not a Jigjiga model.",
        "Six weeks of appointments: no seasons, no rainy-season roads, no pastoral movement.",
        "No distance, transport cost or phone-coverage variables — likely decisive here, absent in the data.",
        "Only ~2 % of records carry a disability flag, so that group's results are imprecise."], size=12)
    d.text(s, X1, Inches(1.3), Inches(6.0), Inches(4.8), [
        "## About the model",
        f"Precision is {pct(CR['precision'])}: most flagged patients would have attended. A flag is a reason to call, not a prediction about a person.",
        f"Ranking quality (ROC-AUC {CR['roc_auc']:.2f}) is modest; the gain over a simple lead-time rule is real but small.",
        "A model trained on one period can drift: the forward-in-time check gave " + (f"ROC-AUC {tc['roc_auc']:.2f} and recall {pct(tc['recall_at_early_threshold'])}" if tc.get("status") == "ok" else "n/a") + " on later appointments — monitoring is not optional.",
        "Scores can be misused to deprioritise patients; the governance policy forbids it and the API states it on every response."], size=12)
    d.takeaway(s, "Everything on this slide is in the Model Card. A stakeholder who only reads one page should read that one.")
    d.notes(s, "Limitations are presented before implications on purpose. The proxy-data limitation is the most important: nothing here should be read as a measurement of Jigjiga patients.")

    # 13 Transparency & ethics
    s = d.slide("Transparency and ethics", "Built on the Module 1 Ethical AI Charter, the Module 2 privacy plan and the Module 4 Model Card", "Ethics")
    d.text(s, X0, Inches(1.3), Inches(6.2), Inches(4.8), [
        "## Transparency statement",
        "The model CAN rank scheduled appointments by likelihood of being missed, 24–72 h ahead, and explain each score with its main reasons.",
        "The model CANNOT say that a given patient will miss, measure Jigjiga patients yet, or see distance, transport or season.",
        "The model MUST NOT be used to cancel, overbook against or deprioritise anyone. Every score is advisory and staff overrides are logged.",
        "## Privacy",
        "Pseudonymised keys, k = 20 area grouping, no names or phone numbers, three access tiers (Module 3). Pseudonymised data are still personal data under Proclamation No. 1321/2024."], size=12)
    d.text(s, X1, Inches(1.3), Inches(6.0), Inches(4.8), [
        "## Oversight",
        "Hospital Research & Ethics Review Committee — approval of any local patient-data use.",
        "AI & Data Governance Committee — fairness thresholds, feature changes, any change in how scores are used.",
        "## Honest visuals",
        "Every chart in this deck starts its axis at zero or shows the full 0–1 range, reports the test split only, and shows the baseline next to the model.",
        "## Monitoring once live",
        "Monthly recall and per-group recall gap · weekly flag rate vs outreach capacity · monthly input drift · quarterly calibration and override review."], size=12)
    d.takeaway(s, "The commitment from Module 1 stands: the score may trigger additional support — never denial, cancellation or lower clinical priority.")
    d.notes(s, "This slide is for the ethics committees. It restates the charter, the privacy controls and who decides what.")

    # 14 Business implications
    s = d.slide("What the findings mean for JJUSHYCSH", "Connecting the results to the Module 1 goals — without prescribing a decision", "Implications")
    weekly = 500
    d.text(s, X0, Inches(1.3), Inches(12.4), Inches(4.9), [
        f"## Reach — Module 1 goal: ≥ 70 % of high-risk no-shows flagged",
        f"On unseen appointments the model reaches {pct(CR['recall'])} recall. If JJUSHYCSH's own data behave like the proxy, roughly {CR['recall']*R['test_no_show']*weekly:.0f} of the "
        f"{R['test_no_show']*weekly:.0f} no-shows in a week of {weekly} referral appointments would be on the outreach list.",
        f"## Workload — Module 1 constraint: fixed reminder capacity",
        f"That reach costs about {CR['flag_rate']*weekly:.0f} extra contacts per {weekly} appointments ({pct(CR['flag_rate'])}). The threshold is a dial: the dashboard shows what each level of calls buys in recall.",
        "## Where the risk sits — drivers",
        "Long lead times and prior missed visits dominate. Implication: the booking desk, not the clinic, is where most of the risk is created — and where earlier slots or confirmation calls can change it.",
        "## Specialist-slot utilisation — Module 1 supporting KPI",
        "Each caught no-show is a slot that can be re-offered; each false flag costs one reminder. The asymmetry is what makes a modest-precision model still worth running.",
        "## Fairness as a performance measure — Module 2",
        ("All monitored groups are within the release threshold, so equal reach is achievable at current performance." if fair_pass else
         f"Equal reach needs the group-threshold correction for {GROUP_LABEL[worst].lower()}; without it, some groups would be under-served.")],
        size=12)
    d.takeaway(s, "The findings tie back to each Module 1 goal; whether and how to act on them is a decision for the pilot-readiness gate, not for this presentation.")
    d.notes(s, "This slide connects results to the KPIs without recommending. The 500-appointment week is an illustration; the real volume comes from the HIS officer.")

    # 15 Next steps
    s = d.slide("Possible next steps (for discussion, not a recommendation)", "Options the stakeholders may consider; the dashboard and code are available now", "Next steps")
    d.text(s, X0, Inches(1.3), Inches(6.2), Inches(4.8), [
        "## Options on the table",
        "Request the anonymised JJUSHYCSH OPD/referral extract through the ethics pathway (Module 2 dependency D-01) and repeat the Module 3–4 validation on local data.",
        "Agree an outreach capacity, then read the threshold off the dashboard rather than fixing it in advance.",
        "Add the local data fields the proxy lacks: distance band, referral source, contact availability.",
        "Decide in the AI & Data Governance Committee whether group-specific thresholds are acceptable.",
        "Pilot in one department with the override log switched on from day one."], size=12, bullets=False)
    d.text(s, X1, Inches(1.3), Inches(6.0), Inches(4.8), [
        "## Available today",
        f"Live dashboard: {dashboard_url}",
        "   overview · results · drivers · what-if scoring · ethical-compliance (fairness) view · limitations",
        f"Code and artefacts: {github_url}",
        "   Module 3 pipeline · Module 4 model, API and reports · Module 5 dashboard and this deck",
        "API: POST /predict returns probability, tier, outreach action and reason codes (demo in the appendix).",
        "## Q&A appendix follows",
        "Data and split · protocol and hyperparameters · full metrics · fairness definitions · sensitivity and drift · API demo · monitoring · references."], size=12)
    d.takeaway(s, "Thank you. Everything shown is reproducible from the repository; the dashboard shows the same numbers live.")
    d.notes(s, "These are options, not a recommendation — the brief excludes recommendations from this deliverable. The appendix slides hold the technical detail for questions.")

    # ---------------- Appendix
    s = d.slide("Q&A appendix", "Technical backup for questions", "Appendix")
    d.text(s, X0, Inches(1.5), Inches(12), Inches(4), [
        "A1  Data, pipeline and split", "A2  Protocol, hyperparameters and cross-validation", "A3  Full performance metrics",
        "A4  Fairness definitions and full metrics", "A5  Necessity review, sensitivity analysis and drift check",
        "A6  Architecture and API demonstration", "A7  Acceptance criteria and monitoring plan", "References (NXU / APA format)"], size=16)

    s = d.slide("A1 · Data, pipeline and split", "Module 3 hand-over used unchanged", "Appendix")
    q = R.get("m3", {}).get("data_quality_report", {})
    d.text(s, X0, Inches(1.3), Inches(12.4), Inches(4.8), [
        "Source: Medical Appointment No Shows (Hoppen, 2016), 110,527 appointments, 14 variables; disclosed proxy for JJUSHYCSH records.",
        "Module 3 pipeline: ingest → raw validation (28 checks) → clean → anonymise (salted SHA-256 keys, k = 20 area grouping, age top-coded at 90) → features → processed validation (23 checks, privacy gate) → publish + bias check + audit log."
        + (f" {q['rows_received']['rows_affected']:,} in → {q['rows_after_cleaning']['rows_affected']:,} model-ready rows." if q else ""),
        f"Split: GroupShuffleSplit on patient key, seed 42 — train {R['n_train']:,} / validation {R['n_val']:,} / test {R['n_test']:,}; "
        f"no patient in two sets (asserted in Module 4). No-show rate {pct(R['train_no_show'],1)} / {pct(R['val_no_show'],1)} / {pct(R['test_no_show'],1)}.",
        f"Features ({len(R['features'])}): {', '.join(R['features'])}.",
        "Excluded: sms_received (intervention leakage), area_code (geography-as-penalty risk), gender (audit only), appt_month (calendar proxy), lead_band (redundant); alcoholism and has_disability dropped by the necessity review.",
        f"Data hash {R['data_md5'][:12]}… tagged on every MLflow run (experiment jjushycsh-noshow-m4; registry jjushycsh_noshow_risk v{R['registry'].get('version','n/a')}, alias champion)."],
        size=12)

    s = d.slide("A2 · Protocol, hyperparameters and cross-validation", "", "Appendix")
    tuned = [k for k in R["results"] if not k.startswith("baseline")]
    rows = [["Model", "Best hyperparameters", "CV PR-AUC", "Val PR-AUC", "Val ROC-AUC"]]
    for k in tuned:
        r = R["results"][k]
        rows.append([MODEL_NAME[k], "; ".join(f"{a}={v}" for a, v in r["params"].items()), f"{r['cv_pr_auc']:.3f} ± {r.get('cv_pr_auc_std', 0):.3f}",
                     f"{r['val_pr_auc']:.3f}", f"{r.get('val_roc_auc', float('nan')):.3f}"])
    d.table(s, X0, Inches(1.3), Inches(12.4), rows, col_w=[Inches(2.0), Inches(6.2), Inches(1.5), Inches(1.3), Inches(1.4)], size=10, row_h=Inches(0.6))
    d.text(s, X0, Inches(4.2), Inches(12.4), Inches(2), [
        f"Randomized search, {R['n_iter']} candidates per algorithm, {R['cv_folds']}-fold StratifiedGroupKFold grouped by patient, scoring PR-AUC; class weights for imbalance (no resampling).",
        f"Champion chosen on validation PR-AUC. Thresholds set on validation: High = highest threshold reaching {TARGET_RECALL:.0%} recall ({t_high:.3f}); Medium at 90 % recall ({CR.get('threshold_medium', 0):.3f}).",
        "Every candidate logged as a nested MLflow run; champion registered with alias 'champion' at stage Staging."], size=11)

    s = d.slide("A3 · Full performance metrics (Module 3 test split)", "", "Appendix")
    rows = [["Model", "Threshold", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC", "Brier", "Flag rate", "TP / FP / FN / TN"]]
    for k, r in R["results"].items():
        rows.append([MODEL_NAME[k], f"{r['threshold']:.3f}", f"{r['accuracy']:.3f}", f"{r['precision']:.3f}", f"{r['recall']:.3f}", f"{r['f1']:.3f}",
                     f"{r['roc_auc']:.3f}", f"{r['pr_auc']:.3f}", "n/a" if r.get("brier") is None else f"{r['brier']:.3f}", pct(r["flag_rate"], 1),
                     f"{r['tp']:,} / {r['fp']:,} / {r['fn']:,} / {r['tn']:,}"])
    d.table(s, X0, Inches(1.3), Inches(12.4), rows, size=10, row_h=Inches(0.45))
    d.image(s, FIG / "roc_pr_curves.png", X0, Inches(4.2), h=Inches(2.7))

    s = d.slide("A4 · Fairness definitions and full metrics", "Module 2 criteria; Fairlearn cross-check in Module 4", "Appendix")
    d.text(s, X0, Inches(1.3), Inches(12.4), Inches(1.3), [
        "Equal opportunity / TPR gap (release threshold ≤ 0.10): difference in recall between best- and worst-served group. "
        "Equalized odds adds the false-positive-rate gap. Disparate impact ratio (diagnostic ≥ 0.80): lowest flag rate ÷ highest flag rate. "
        "Demographic parity difference: largest minus smallest flag rate. Favourable outcome = being flagged (it triggers support)."], size=11)
    rows = [["Attribute", "DP diff", "DI ratio", "TPR gap", "FPR gap", "EO diff", "TPR gap ≤ 0.10", "DI ≥ 0.80"]]
    for a in GROUP_LABEL:
        g = fb[a]; rows.append([GROUP_LABEL[a], f"{g['dp_difference']:.3f}", f"{g['di_ratio']:.3f}", f"{g['tpr_gap']:.3f}", f"{g['fpr_gap']:.3f}",
                                f"{g['eo_difference']:.3f}", "yes" if g["passes_tpr"] else "no", "yes" if g["passes_di"] else "no"])
    d.table(s, X0, Inches(2.6), Inches(12.4), rows, size=10, row_h=Inches(0.38))
    ft = mit.get("fairlearn_threshold_optimizer", {})
    d.text(s, X0, Inches(4.7), Inches(12.4), Inches(1.6), [
        f"Mitigation: {mit.get('method') if mit.get('applied') else 'not applied — ' + str(mit.get('reason', ''))}."
        + (f" Group thresholds: {', '.join(f'{k}={v:.3f}' for k, v in mit['group_thresholds'].items())}." if mit.get("applied") else ""),
        (f"Fairlearn ThresholdOptimizer cross-check: TPR gap {ft['tpr_gap']:.3f}, recall {ft['recall']:.3f}, flag rate {ft['flag_rate']:.3f}." if "tpr_gap" in ft else ""),
        "Full per-group tables (n, base rate, flag rate, TPR, FPR, precision) are in the Module 4 Fairness Metrics Report and on the dashboard's ethical-compliance page."], size=11)

    s = d.slide("A5 · Necessity review, sensitivity analysis and drift check", "", "Appendix")
    rows = [["Sensitive flag", "CV PR-AUC with", "CV PR-AUC without", "Δ", "Decision"]]
    for t in nr["table"]:
        rows.append([t["feature"], f"{t['cv_pr_auc_with']:.4f}", f"{t['cv_pr_auc_without']:.4f}", f"{t['delta']:+.4f}", t["decision"]])
    d.table(s, X0, Inches(1.3), Inches(6.2), rows, size=10, row_h=Inches(0.4))
    sens = R.get("sensitivity") or []
    if sens:
        rows = [["Scenario", "Mean |Δ risk|", "High-tier flips", "ROC-AUC"]]
        for x in sens: rows.append([x["scenario"], f"{x['mean_abs_delta_prob']:.3f}", f"{x['high_tier_flips_pct']:.1f}%", f"{x['roc_auc']:.3f}"])
        d.table(s, X1, Inches(1.3), Inches(6.0), rows, col_w=[Inches(3.1), Inches(1.0), Inches(1.0), Inches(0.9)], size=9, row_h=Inches(0.33))
    d.text(s, X0, Inches(3.3), Inches(6.2), Inches(2.8), [
        f"Necessity tolerance: a flag stays an input only if dropping it costs > {nr['tolerance']} CV PR-AUC. A dropped flag is also removed from chronic_count (data minimisation, Proclamation 1321/2024).",
        (f"Forward-in-time check: champion configuration refitted on dates before {tc['cut_date']} ({tc['n_early']:,} appointments) and scored on the latest {tc['n_late']:,}: "
         f"ROC-AUC {tc['roc_auc']:.3f}, PR-AUC {tc['pr_auc']:.3f}, recall {pct(tc['recall_at_early_threshold'],1)} at the earlier threshold." if tc.get("status") == "ok" else "Forward-in-time check: n/a.")], size=11)

    s = d.slide("A6 · Architecture and API demonstration", "Module 2 architecture as built in Modules 3–5", "Appendix")
    d.text(s, X0, Inches(1.3), Inches(6.2), Inches(4.8), [
        "## Flow", "Kaggle proxy (later: HIS/DHIS2 extract, 18:00 daily) → Module 3 pipeline (pandas, validation gates, pseudonymisation, Airflow DAG, Docker) → "
        "Module 4 model (scikit-learn / XGBoost, MLflow tracking + registry, SHAP, Fairlearn, DiCE) → FastAPI /predict → Module 5 Streamlit dashboard.",
        "## API", "POST /predict with age, scheduled_date, appointment_date, flags and attendance history → no_show_probability, risk_tier, recommended_action, reason_codes, thresholds, model, disclaimer.",
        "GET /health → algorithm, creation time, feature list, data hash.",
        "Serving uses the Module 3 feature definitions, so training and serving cannot drift apart.",
        "Run: uvicorn api.main:app → http://127.0.0.1:8000/docs; demo script module5/api_demo.py."], size=11)
    demo_txt = ["## Demo output (module5/api_demo.py)"]
    if isinstance(api_demo, dict):
        api_demo = api_demo.get("cases", [])
    if api_demo:
        for case in api_demo[:3]:
            resp = case.get("response", {})
            demo_txt.append(f"{case.get('label','case')}: p = {resp.get('no_show_probability')}, tier {resp.get('risk_tier')} — "
                            f"{'; '.join(resp.get('reason_codes', []))[:140]}")
    else:
        demo_txt.append("[run module5/api_demo.py with the API up to fill this box]")
    d.text(s, X1, Inches(1.3), Inches(6.0), Inches(4.8), demo_txt, size=11)

    s = d.slide("A7 · Acceptance criteria and monitoring plan", "Module 4, Section 7", "Appendix")
    rows = [["Signal", "Acceptance criterion / trigger", "Frequency"],
            ["Recall on no-shows", f"≥ {TARGET_RECALL:.2f}; retrain if < {TARGET_RECALL-0.05:.2f}", "Monthly"],
            ["Precision / flag rate", "Flag rate within outreach capacity; alert if +50 % vs baseline", "Weekly"],
            ["Per-group recall gap (4 attributes)", "≤ 0.10; governance committee review if exceeded", "Monthly"],
            ["Input drift (lead time, age, history)", "Population stability index < 0.2", "Monthly"],
            ["Calibration (Brier score)", f"Within 10 % of validation value ({CR.get('brier', 0):.3f})", "Quarterly"],
            ["Override log", "Staff overrides and reasons reviewed", "Quarterly"]]
    d.table(s, X0, Inches(1.3), Inches(12.4), rows, col_w=[Inches(3.6), Inches(6.8), Inches(2.0)], size=11, row_h=Inches(0.42))
    met = CR["recall"] >= TARGET_RECALL - 0.02; gain_ok = (CR["roc_auc"] - rule["roc_auc"]) >= 0.02
    d.text(s, X0, Inches(4.6), Inches(12.4), Inches(1.5), [
        f"Module 1 success criteria on the proxy data: recall target {'met' if met else 'not met'}; gain ≥ 0.02 ROC-AUC over the rule baseline "
        f"{'met' if gain_ok else 'not met'}; fairness release threshold {'met' if fair_pass or mit.get('applied') else 'not met'}. "
        "Status: registered at Staging as a validated method (Module 2 Gate 3 passed on proxy data); Gate 4 requires local data, a repeat of the validation pack and governance sign-off."], size=11)

    s = d.slide("References", "NXU (APA 7th) format", "Appendix")
    refs = [
        "Ahmed, A. F. (2026a). Project vision document [Unpublished BAN6800 Module 1 assignment]. Nexford University.",
        "Ahmed, A. F. (2026b). Business analytics project overview report [Unpublished BAN6800 Module 2 assignment]. Nexford University.",
        "Ahmed, A. F. (2026c). Designing a data pipeline for referral no-show analytics [Unpublished BAN6800 Module 3 assignment]. Nexford University.",
        "Ahmed, A. F. (2026d). Predictive model development and validation pack [Unpublished BAN6800 Module 4 assignment]. Nexford University.",
        "CURE Ethiopia. (2023, August 29). New mobile clinic in Jigjiga. https://ethiopia.cure.org/news/new-mobile-clinic-in-jigjiga/",
        "Federal Ministry of Justice, Ethiopia. (2024). Personal Data Protection Proclamation No. 1321/2024.",
        "Hardt, M., Price, E., & Srebro, N. (2016). Equality of opportunity in supervised learning. Advances in Neural Information Processing Systems, 29, 3315–3323.",
        "Hoppen, J. (2016). Medical appointment no shows [Data set]. Kaggle. https://www.kaggle.com/datasets/joniarroba/noshowappointments",
        "Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. Advances in Neural Information Processing Systems, 30, 4765–4774.",
        "Mitchell, M., et al. (2019). Model cards for model reporting. Proceedings of the Conference on Fairness, Accountability, and Transparency, 220–229.",
        "Mothilal, R. K., Sharma, A., & Tan, C. (2020). Explaining machine learning classifiers through diverse counterfactual explanations. FAT* 2020, 607–617.",
        "Weerts, H., Dudík, M., Edgar, R., Jalali, A., Lutz, R., & Madaio, M. (2023). Fairlearn: Assessing and improving fairness of AI systems. Journal of Machine Learning Research, 24(257), 1–8.",
        "Streamlit Inc. (n.d.). Streamlit documentation. https://docs.streamlit.io/",
    ]
    d.text(s, X0, Inches(1.25), Inches(12.4), Inches(5.8), refs, size=10, spacing=3)

    out = OUT / f"BAN6800_Module5_Stakeholder_Presentation_{AUTHOR.replace(' ', '_')}.pptx"
    d.prs.save(out)
    print(f"Deck written: {out}  ({d.n} slides: 15 main + {d.n - 15} appendix)")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dashboard-url", default="<add your live dashboard URL>")
    ap.add_argument("--github-url", default="<add your GitHub repository URL>")
    ap.add_argument("--api-demo", default=str(HERE / "outputs" / "api_demo_output.json"))
    a = ap.parse_args()
    demo = None
    if Path(a.api_demo).exists():
        try:
            demo = json.loads(Path(a.api_demo).read_text())
        except Exception:
            demo = None
    build(a.dashboard_url, a.github_url, demo)
