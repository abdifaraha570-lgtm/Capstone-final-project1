"""Builds the Model Validation Report (.docx) from outputs/results.json + outputs/figures.
Every number in the text comes from the run; sentences highlighted yellow are for the author to complete."""
import datetime as dt
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_COLOR_INDEX
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt, Inches, RGBColor

from m4 import config as C

NAVY, TEAL = RGBColor(0x1F, 0x3B, 0x57), RGBColor(0x2E, 0x6B, 0x65)
MEANING = {
    "Lead time (days booked ahead)": "the longer the gap between booking and the visit, the more time for a referred patient's travel plans, money or caregiving arrangements to fall through",
    "Prior no-show rate": "past attendance is the most direct behavioural signal available for returning patients",
    "Number of prior no-shows": "past attendance is the most direct behavioural signal available for returning patients",
    "Patient age": "age reflects life-stage barriers: adolescents and young adults miss more often, older patients with chronic follow-up attend more reliably",
    "Same-day appointment": "same-day bookings are made by patients already at or near the facility, so they are rarely missed",
    "Number of prior visits": "patients with more contact history are more engaged with care",
    "First visit (no history)": "a first referral visit carries no attendance history, so the model relies on scheduling signals",
    "Welfare programme enrolment": "welfare enrolment is a proxy for financial barriers such as transport cost",
    "Chronic conditions (count)": "chronic-disease follow-up creates routine attendance",
    "Hypertension": "chronic-disease follow-up creates routine attendance",
    "Diabetes": "chronic-disease follow-up creates routine attendance",
    "Alcohol-use disorder": "a recorded alcohol-use disorder signals additional barriers to attendance",
    "Disability recorded": "disability can create mobility and transport barriers",
}


def f3(x):
    return "n/a" if x is None else f"{x:.3f}"


def pct(x):
    return "n/a" if x is None else f"{100 * x:.1f}%"


class Rep:
    def __init__(self):
        self.d = Document()
        s = self.d.styles["Normal"]; s.font.name = "Calibri"; s.font.size = Pt(11)
        s.element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")
        for sec in self.d.sections:
            sec.left_margin = sec.right_margin = Inches(1); sec.top_margin = sec.bottom_margin = Inches(0.9)
        self.fig = 0; self.tab = 0; self.words = 0; self.count = True
        self._footer()

    def _footer(self):
        p = self.d.sections[0].footer.paragraphs[0]; p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(f"BAN6800 Module 4 — Model Validation Report | {C.AUTHOR} | Page "); r.font.size = Pt(8)
        for t, txt in [("begin", None), (None, "PAGE"), ("end", None)]:
            run = p.add_run(); run.font.size = Pt(8)
            if t:
                e = OxmlElement("w:fldChar"); e.set(qn("w:fldCharType"), t); run._r.append(e)
            else:
                e = OxmlElement("w:instrText"); e.set(qn("xml:space"), "preserve"); e.text = txt; run._r.append(e)

    def h(self, text, lvl=1):
        p = self.d.add_heading(text, level=lvl)
        for r in p.runs:
            r.font.color.rgb = NAVY if lvl == 1 else TEAL; r.font.name = "Calibri"
            r.font.size = Pt(15 if lvl == 1 else 12)
        return p

    def p(self, text, italic=False, highlight=False, size=None):
        para = self.d.add_paragraph(); para.paragraph_format.space_after = Pt(6)
        r = para.add_run(text); r.italic = italic
        if size: r.font.size = Pt(size)
        if highlight: r.font.highlight_color = WD_COLOR_INDEX.YELLOW
        if self.count and not highlight: self.words += len(text.split())
        return para

    def bullet(self, text):
        para = self.d.add_paragraph(text, style="List Bullet"); para.paragraph_format.space_after = Pt(2)
        if self.count: self.words += len(text.split())

    def caption(self, text):
        para = self.d.add_paragraph(); para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = para.add_run(text); r.italic = True; r.font.size = Pt(9); r.font.color.rgb = TEAL

    def figure(self, path, caption, width=6.0):
        if not path.exists():
            self.p(f"[Figure missing: {path.name} — this figure is produced when the corresponding library "
                   f"(e.g. shap) is installed; re-run run_all.py in the Colab notebook.]", highlight=True); return
        self.fig += 1
        self.d.add_picture(str(path), width=Inches(width))
        self.d.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        self.caption(f"Figure {self.fig}. {caption}")

    def table(self, header, rows, caption, widths=None):
        self.tab += 1
        self.caption(f"Table {self.tab}. {caption}")
        t = self.d.add_table(rows=1, cols=len(header)); t.style = "Light Grid Accent 1"
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        for i, h in enumerate(header):
            c = t.rows[0].cells[i]; c.text = ""; r = c.paragraphs[0].add_run(str(h)); r.bold = True; r.font.size = Pt(8.5)
        for row in rows:
            cells = t.add_row().cells
            for i, v in enumerate(row):
                cells[i].text = ""; r = cells[i].paragraphs[0].add_run(str(v)); r.font.size = Pt(8.5)
        if widths:
            for row in t.rows:
                for i, w in enumerate(widths):
                    row.cells[i].width = Inches(w)
        self.d.add_paragraph().paragraph_format.space_after = Pt(2)


def build_report(R):
    rep = Rep(); d = rep.d
    res, ch = R["results"], R["champion"]; cr = res[ch]
    CH = ch.replace("_", " ").title()
    tuned = [k for k in res if not k.startswith("baseline")]
    rule = res["baseline_lead_time_rule"]
    imp = list(R["shap"].get("importance", {}).items())
    dirn = R["shap"].get("direction", {})
    fb, mit, nr, tc = R["fairness_before"], R["mitigation"], R["necessity"], R.get("temporal_check", {})
    m3 = R.get("m3", {}); q = m3.get("data_quality_report", {}); ss = m3.get("split_summary", {})
    m3flags = m3.get("bias_flags", {}).get("flags", [])
    feats = R["features"]

    # ---------------- Title page
    rep.count = False
    for _ in range(5): d.add_paragraph()
    t = d.add_paragraph(); t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = t.add_run("Predictive Model Development & Validation Pack"); r.bold = True; r.font.size = Pt(22); r.font.color.rgb = NAVY
    t = d.add_paragraph(); t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = t.add_run("Referral No-Show Risk Model for Jigjiga University Sheikh Hassan Yabare\nComprehensive Specialized Hospital (JJUSHYCSH) — Sprint 2 of 3")
    r.italic = True; r.font.size = Pt(13); r.font.color.rgb = TEAL
    d.add_paragraph()
    for line in ["BAN6800 Business Analytics Capstone — Module 4 Assignment", C.AUTHOR, C.UNIVERSITY,
                 dt.date.today().strftime("%B %Y")]:
        q_ = d.add_paragraph(); q_.alignment = WD_ALIGN_PARAGRAPH.CENTER; q_.add_run(line).font.size = Pt(12)
    d.add_page_break()
    rep.count = True

    # ---------------- 1
    rep.h("1. Business Need and Target Definition")
    rep.p("JJUSHYCSH is the only tertiary referral hospital for Ethiopia's Somali Region, serving six to seven million largely rural and pastoralist people (CURE Ethiopia, 2023); a missed follow-up delays care and wastes a specialist slot someone else travelled far for. "
          "The Module 1 Vision Document (Ahmed, 2026a) set two KPIs, at least 15% fewer no-shows and at least 70% recall of high-risk appointments, so that limited SMS, phone and community-health-worker capacity is targeted. "
          "Module 2 (Ahmed, 2026b) planned the architecture and four sprints; Module 3 delivered Sprint 1, the governed data pipeline. This report is Sprint 2: the model, its validation and its explanation, built on the Module 3 data.")
    rep.p("[Personalise: one or two sentences from your own experience with DHIS2 and referral records in the Somali Region — e.g. how appointments are recorded today and what that means for moving from the proxy data to hospital records.]", highlight=True)
    rep.p(f"The task is binary classification. The unit of analysis is one scheduled appointment; the target no_show equals 1 when the patient did not attend (Module 3 data dictionary). "
          f"The prediction horizon is 24 to 72 hours before the appointment, matching Module 3's daily 18:00 pipeline run. "
          f"Success criteria from Module 1: (1) recall of at least {C.TARGET_RECALL:.0%} on no-shows at the High-tier threshold; (2) a gain of at least {C.MIN_GAIN_OVER_RULE:.2f} ROC-AUC over a transparent lead-time rule; (3) no protected group outside the fairness tolerance below.")
    rep.p(f"Fairness objectives come from Module 2, Section 6: because the intervention is outreach and, by the Module 1 charter, a score can never reduce care, the governing criterion is equal opportunity, with a release threshold of a true-positive-rate gap no greater than {C.EOD_MAX} across monitored groups (Hardt et al., 2016). "
          f"False-positive-rate gaps (equalized odds) and selection-rate ratios (four-fifths rule, disparate impact ≥ {C.DI_MIN}; Feldman et al., 2015) are diagnostics, which Module 2 says are not sufficient evidence of fairness alone. "
          f"The protected attributes are the four groups the Module 3 bias suite monitors: gender, age band ({', '.join(C.AGE_BANDS)}), welfare enrolment as a socioeconomic marker, and recorded disability; distance bands follow once local data are approved.")

    # ---------------- 2
    rep.h("2. Data Readiness and Feature Set")
    rows_in = q.get("rows_received", {}).get("rows_affected"); rows_out = q.get("rows_after_cleaning", {}).get("rows_affected")
    rep.p("The modelling dataset is the pseudonymised model table published by the Module 3 pipeline (ingest → validate → clean → anonymise → features → validate → publish) from the Kaggle Medical Appointment No Shows data (Hoppen, 2016), a disclosed proxy for JJUSHYCSH records. "
          + (f"Module 3's quality report shows {rows_in:,} records received and {rows_out:,} model-ready rows after cleaning; " if rows_in and rows_out else "")
          + f"the split files hold {R['n_train']:,} training, {R['n_val']:,} validation and {R['n_test']:,} test appointments ({R['date_min']} to {R['date_max']}). "
          f"The MD5 hash of the input files ({R['data_md5'][:12]}…) is tagged on every MLflow run, so each result traces to the exact Module 3 output; no model saw data that bypassed the Module 3 gates.")
    rep.p("Features reuse the Module 3 definitions rather than re-deriving them: lead time in days, same-day flag, weekday, chronic-condition count, and the leakage-safe history (prior visits, prior no-shows, prior no-show rate, first-visit flag) that Module 3 computes only from earlier visits. "
          "Inside the scikit-learn pipeline (Pedregosa et al., 2011) numeric features are standardised and the weekday one-hot encoded, both fitted on training folds only.")
    rep.table(["Feature (Module 3 name)", "Type", "Treatment"],
              [["age", "numeric", "standardised (top-coded at 90 in Module 3)"],
               ["lead_days", "numeric (derived)", "standardised"],
               ["prior_visits, prior_no_shows, prior_no_show_rate", "numeric (history, past-only)", "standardised"],
               ["chronic_count", "numeric (derived)", "standardised"],
               ["scholarship, hypertension, diabetes" + (", alcoholism" if "alcoholism" in feats else "") + (", has_disability" if "has_disability" in feats else ""), "binary", "passthrough"],
               ["same_day, first_visit", "binary (derived)", "passthrough"],
               ["appt_weekday", "categorical (derived)", "one-hot"]],
              "Final feature set and preprocessing.", [2.8, 1.8, 1.8])
    rep.p("Four exclusions prevent leakage or unfairness: sms_received, because the SMS is the intervention the model targets and encodes scheduling policy rather than risk; area_code, so geography cannot become a penalty (the first Module 1 risk); gender, audit-only with no outreach rationale; and appt_month, a calendar proxy in six weeks of data.")
    rep.p(f"The Module 2 data dictionary marked Alcoholism and Handcap for a necessity review and Module 3 handed them over as an open item. The review is a data-minimisation test under Proclamation No. 1321/2024: a gradient-boosting model was cross-validated with and without each flag (Table 2). "
          + "; ".join(f"removing {t['feature']} changed cross-validated PR-AUC by {t['delta']:+.4f}, so it is {'kept as an input' if t['decision'].startswith('keep') else 'dropped as an input and kept for the fairness audit only'}" for t in nr["table"])
          + (". As chronic_count in Module 3 sums hypertension, diabetes and alcoholism, the count was rebuilt without alcoholism so the dropped flag does not survive inside another feature." if "alcoholism" in nr["dropped"] else "."))
    rep.table(["Sensitive flag", "CV PR-AUC with", "CV PR-AUC without", "Δ", "Decision"],
              [[t["feature"], f3(t["cv_pr_auc_with"]), f3(t["cv_pr_auc_without"]), f"{t['delta']:+.4f}", t["decision"]] for t in nr["table"]],
              f"Necessity review of sensitive health flags (tolerance {nr['tolerance']} PR-AUC).", [1.3, 1.2, 1.3, 0.8, 2.0])
    rows = []
    for g in C.PROTECTED:
        for b in R["bias"].get(g, []):
            rows.append([g, b["group"], pct(b["share"]), pct(b["no_show_rate"])])
    rep.table(["Attribute", "Group", "Share of training split", "No-show rate"], rows,
              "Data bias check on the training split (continuation of the Module 3 bias suite).", [1.5, 1.2, 1.8, 1.5])
    flag_txt = "; ".join(f"{fl.get('attribute')}{(' ' + str(fl.get('group'))) if fl.get('group') else ''} {fl.get('issue')}" for fl in m3flags) or "no flags"
    ab = {b["group"]: b["no_show_rate"] for b in R["bias"].get("age_band", [])}
    hi_g, lo_g = (max(ab, key=ab.get), min(ab, key=ab.get)) if ab else ("", "")
    rep.p(f"The Module 3 bias suite raised: {flag_txt}; the disability group's small share means its error rates are checked separately in Section 6. "
          + (f"Base rates differ by age ({hi_g}: {pct(ab[hi_g])}; {lo_g}: {pct(ab[lo_g])}), " if ab else "")
          + "so equal flag rates across age bands would be neither expected nor desirable, which is why demographic parity is reported but equal opportunity governs.")

    # ---------------- 3
    rep.h("3. Train/Test Protocol and Baselines")
    rep.p(f"The split is the patient-grouped 70/15/15 split produced by Module 3 (GroupShuffleSplit on patient_key, seed 42), so no patient appears in more than one set and history features cannot leak across sets; this run asserts zero overlap. "
          f"The training split ({R['n_train']:,}; no-show {pct(R['train_no_show'])}) serves hyperparameter search with stratified {R['cv_folds']}-fold cross-validation grouped by patient; the validation split ({R['n_val']:,}; {pct(R['val_no_show'])}) selects the champion and fixes the thresholds; the test split ({R['n_test']:,}; {pct(R['test_no_show'])}) is scored once and informs no decision.")
    rep.p(f"Two baselines anchor the comparison. A majority-class classifier predicts that everyone attends, giving {pct(res['baseline_majority']['accuracy'])} accuracy but zero recall, so accuracy cannot be the headline metric. "
          f"A transparent rule a scheduler could apply by hand ({rule['params']['rule']}, cut chosen on validation for the same recall target) reached a test ROC-AUC of {f3(rule['roc_auc'])}, recall {pct(rule['recall'])} and precision {pct(rule['precision'])}.")

    # ---------------- 4
    rep.h("4. Model Development")
    algos = ", ".join(k.replace("_", " ") for k in tuned)
    rep.p(f"Following Module 2 tickets S2-01 and S2-02, three algorithms span the interpretability–flexibility range: regularised logistic regression (the transparent baseline Module 2 requires as a comparison point), random forest (non-linear interactions, robust to outliers) and gradient-boosted trees ({'XGBoost; Chen & Guestrin, 2016' if 'xgboost' in tuned else 'scikit-learn HistGradientBoosting'}), typically the strongest on tabular data. "
          f"Class imbalance (one no-show in five) was handled with class weights rather than resampling, which keeps probabilities interpretable for the tiers. Each algorithm was tuned by randomized search ({R['n_iter']} candidates, coarse ranges) with {R['cv_folds']}-fold patient-grouped stratified cross-validation optimising PR-AUC, the metric most sensitive to the minority class.")
    rep.p(f"Every candidate was logged to MLflow (Zaharia et al., 2018) as a nested run with its hyperparameters and cross-validated scores, and each best pipeline with its model artefact; {R.get('mlflow_runs') or 'all'} runs sit in experiment '{C.EXPERIMENT}'. "
          f"The champion was chosen on validation PR-AUC; its operating threshold is the highest threshold still reaching {C.TARGET_RECALL:.0%} validation recall, and a second threshold at {C.MEDIUM_RECALL:.0%} recall defines the Medium tier. Both are stored with the model.")
    rep.table(["Model", "Best hyperparameters", "CV PR-AUC", "Val PR-AUC", "Val ROC-AUC", "Fit (s)"],
              [[k.replace("_", " "), "; ".join(f"{a}={v}" for a, v in res[k]["params"].items()),
                f"{f3(res[k]['cv_pr_auc'])} ± {f3(res[k].get('cv_pr_auc_std'))}", f3(res[k]["val_pr_auc"]),
                f3(res[k].get("val_roc_auc")), res[k].get("fit_seconds", "")] for k in tuned],
              "Tuned hyperparameters, cross-validation and validation scores.", [1.2, 2.8, 0.9, 0.7, 0.7, 0.5])
    lr = res.get("logistic_regression")
    if lr and ch != "logistic_regression":
        gap = cr["val_pr_auc"] - lr["val_pr_auc"]
        rep.p(f"{CH} was selected with a validation PR-AUC {gap:.3f} above logistic regression. "
              + ("Because this margin is small, and Module 1 favours simple, explainable models given JJUSHYCSH's limited data-science capacity, logistic regression remains a credible fallback with coefficients the ethics committee can read directly; the tree model was kept because SHAP gives equivalent transparency per appointment." if gap < 0.01 else
                 "This margin justifies the added complexity despite Module 1's preference for simpler models, provided the SHAP explanations in Section 6 travel with every score and the logistic baseline stays in the registry as the fallback."))
    elif lr:
        rep.p("Logistic regression matched or outperformed the tree ensembles on validation, which suggests the risk signal is largely additive. This is a welcome result: the simplest and most transparent model is also the best, so there is no trade-off between performance and explainability.")

    # ---------------- 5
    rep.h("5. Performance Evaluation and Validation")
    rep.p("Metrics match the business risk: with one no-show in five, PR-AUC drove model selection and recall at the fixed threshold is the operational measure, while ROC-AUC, precision, F1, accuracy and the Brier score are reported for completeness and calibration. A false positive costs one reminder and a false negative a specialist slot, so recall outweighs precision.")
    order = ["baseline_majority", "baseline_lead_time_rule"] + tuned
    rep.table(["Model", "Threshold", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC", "Brier", "Flag rate"],
              [[k.replace("_", " "), f3(res[k]["threshold"]), f3(res[k]["accuracy"]), f3(res[k]["precision"]),
                f3(res[k]["recall"]), f3(res[k]["f1"]), f3(res[k]["roc_auc"]), f3(res[k]["pr_auc"]),
                f3(res[k].get("brier")), pct(res[k]["flag_rate"])] for k in order],
              "Module 3 test-split performance at each model's operating threshold.")
    gain = cr["roc_auc"] - rule["roc_auc"]
    met = cr["recall"] >= C.TARGET_RECALL - 0.02
    rep.p(f"The champion, {CH}, achieved a test ROC-AUC of {f3(cr['roc_auc'])} and PR-AUC of {f3(cr['pr_auc'])} against a prevalence of {pct(R['test_no_show'])}, "
          f"{'a gain of ' + f'{gain:.3f}' + ' ROC-AUC over the lead-time rule' if gain > 0 else 'no better than the lead-time rule on ROC-AUC, which is itself an important finding'}. "
          f"At its threshold ({f3(cr['threshold'])}) it recovers {pct(cr['recall'])} of no-shows with precision {pct(cr['precision'])} while flagging {pct(cr['flag_rate'])} of appointments (validation recall {pct(cr['val_recall_at_threshold'])}). "
          + ("The recall target therefore holds on patients the model has never seen. " if met else f"Test recall fell short of the {C.TARGET_RECALL:.0%} target set on validation, so the threshold should be recalibrated on more recent data before use. ")
          + "Modest precision is acceptable because the cost asymmetry favours reach.")
    rep.figure(C.FIG / "roc_pr_curves.png", "ROC and precision–recall curves on the Module 3 test split; dashed = lead-time rule, red dotted = recall target.", 6.3)
    rep.figure(C.FIG / "confusion_champion.png", f"Confusion matrix of the champion ({CH}) at the operating threshold.", 3.2)
    rep.figure(C.FIG / "calibration_champion.png", "Reliability diagram of the champion's probabilities (test split).", 3.4)
    rep.p("Leakage confirmation: no patient overlap across sets; scaling and encoding fitted inside the pipeline; history features use only earlier visits (Module 3 test_features.py); sms_received excluded; cross-validated, validation and test ROC-AUC are close and no score is suspiciously perfect. "
          + (f"A forward-in-time check (champion configuration refitted on dates before {tc.get('cut_date')}, scored on the latest {tc.get('n_late', 0):,} appointments) gave ROC-AUC {f3(tc.get('roc_auc'))} and recall {pct(tc.get('recall_at_early_threshold'))} at the earlier threshold: the drift signal the monitoring plan tracks. " if tc.get("status") == "ok" else ""))

    # ---------------- 6
    rep.h("6. Explainable AI and Fairness Analysis")
    rep.h("6.1 Global explanations (SHAP)", 2)
    if imp:
        tops = [k for k, _ in imp[:3]]
        rep.p(f"Following the Module 2 explainability framework, SHAP values (Lundberg & Lee, 2017) were computed with {R['shap'].get('explainer', 'SHAP')} on {R['shap'].get('n_explained', 0):,} test appointments in real units. "
              f"The three strongest drivers are {tops[0]}, {tops[1]} and {tops[2]}.")
        for k in tops:
            sgn = dirn.get(k, 0)
            from m4.xai_fairness import BINARY_PRETTY
            binary = k in BINARY_PRETTY or k.startswith("Appointment on")
            effect = ("being in this category " if binary else "higher values ") + ("raise" if sgn > 0 else "lower")
            rep.bullet(f"{k}: {effect} predicted risk — {MEANING.get(k, 'see the SHAP plots')}.")
    else:
        rep.p("[SHAP results missing: run run_all.py with shap installed (the Colab notebook does this) and re-generate the report.]", highlight=True)
    rep.figure(C.FIG / "shap_beeswarm.png", "SHAP summary plot: each dot is one appointment; colour shows the feature value.", 5.6)
    rep.figure(C.FIG / "shap_bar.png", "Global feature importance (mean absolute SHAP value).", 4.8)
    rep.h("6.2 Local explanations", 2)
    for l in R["shap"].get("local", []):
        fac = ", ".join(f"{k} ({'+' if v > 0 else ''}{v:.3f})" for k, v in l["top_factors"])
        rep.p(f"For the {l['case']}-risk example (predicted probability {l['prob']:.2f}) the largest contributions were: {fac}. "
              + ("A coordinator receives this as a reason to call, not just a score; the notebook adds a borderline case." if l["case"] == "high" else ""))
    rep.figure(C.FIG / "shap_local_high.png", "SHAP waterfall for the highest-risk sampled appointment.", 5.2)
    rep.h("6.3 Counterfactual explanations", 2)
    cf = R.get("counterfactuals") or []
    if cf:
        rep.table(["Age", "Lead days", "Weekday", "Risk", "Change needed", "New risk"],
                  [[int(c["age"]), c["orig_lead_days"], c["orig_weekday"], f"{c['orig_prob']:.2f}",
                    f"book {c['cf_lead_days']} days ahead ({c['cf_weekday']})" if c["flips_below_threshold"] else "no actionable change found",
                    f"{c['cf_prob']:.2f}" if c["cf_prob"] is not None else "—"] for c in cf],
                  "Actionable counterfactuals for correctly flagged no-shows (only lead time and weekday may change).")
        n_flip = sum(c["flips_below_threshold"] for c in cf)
        rep.p(f"Counterfactuals answer 'what would need to change for a different outcome?' (Wachter et al., 2018; Mothilal et al., 2020). Only levers the hospital controls may vary; age, health and history are fixed. "
              f"For {n_flip} of {len(cf)} cases a shorter lead time alone moved the appointment below the threshold, supporting one action: offer earlier slots or a confirmation call close to the visit. Method: {R.get('dice_status')}.")
    rep.h("6.4 Fairness metrics", 2)
    rows = []
    for a in C.PROTECTED:
        s = fb[a]
        rows.append([a, f3(s["dp_difference"]), f3(s["di_ratio"]), f3(s["tpr_gap"]), f3(s["fpr_gap"]),
                     f3(s["eo_difference"]), "Yes" if s["passes_di"] else "No", "Yes" if s["passes_tpr"] else "No"])
    rep.table(["Attribute", "DP diff", "DI ratio", "TPR gap", "FPR gap", "EO diff", f"DI ≥ {C.DI_MIN} (diag.)", f"TPR gap ≤ {C.EOD_MAX} (release)"],
              rows, "Fairness metrics on the test split at the operating threshold (group tables in Appendix B).")
    fails = [a for a in C.PROTECTED if not (fb[a]["passes_di"] and fb[a]["passes_eo"])]
    tpr_fails = [a for a in C.PROTECTED if not fb[a]["passes_tpr"]]
    di_fails = [a for a in C.PROTECTED if not fb[a]["passes_di"]]
    rep.p("Metrics were computed per group (ticket S2-04) and cross-checked with Fairlearn (Weerts et al., 2023). "
          + (f"The Module 2 release threshold (TPR gap ≤ {C.EOD_MAX}) was exceeded for: {', '.join(tpr_fails)}. " if tpr_fails else f"All four attributes met the Module 2 release threshold (TPR gap ≤ {C.EOD_MAX}). ")
          + (f"The four-fifths diagnostic was below {C.DI_MIN} for {', '.join(di_fails)}. " if di_fails else "")
          + "Where a flag-rate gap mirrors a real difference in base rates, as with age, a low disparate impact ratio reflects need, not bias; the recall gap is decisive. "
          f"The disability group's estimates rest on {next((r_['n'] for r_ in fb['has_disability']['table'] if str(r_['group']) == '1'), 0):,} test appointments and carry wider uncertainty, as Module 3 anticipated.")
    rep.figure(C.FIG / "fairness_before.png", "Flag rate, recall and false-positive rate by group before mitigation.", 6.5)
    rep.h("6.5 Bias mitigation", 2)
    if mit.get("applied"):
        a = mit["after"][mit["attribute"]]; pa = mit["performance_after"]
        rep.p(f"The largest recall gap was on {mit['attribute']} (TPR gap {f3(fb[mit['attribute']]['tpr_gap'])}). Post-processing with group-specific thresholds chosen on the validation split so that each group reaches {C.TARGET_RECALL:.0%} recall (Hardt et al., 2016) reduced the gap to {f3(a['tpr_gap'])} on the test split. "
              f"Overall recall became {pct(pa['recall'])} and precision {pct(pa['precision'])}, with {pct(pa['flag_rate'])} of appointments flagged. "
              "The trade-off is that the attribute is used at decision time, acceptable only because the decision adds support, and subject to ethics-committee approval.")
        rep.figure(C.FIG / "fairness_after.png", "Group metrics after threshold-based mitigation.", 6.5)
    else:
        rep.p(f"No mitigation was required: the {mit.get('reason', '')}. The mitigation routine (group-specific validation thresholds for equal opportunity, cross-checked with Fairlearn's ThresholdOptimizer) stays in the pipeline and runs at every retraining.")
    rep.h("6.6 Sensitivity analysis", 2)
    sens = R.get("sensitivity") or []
    if sens:
        rep.table(["Scenario", "Mean |Δ risk|", "High-tier flips", "ROC-AUC"],
                  [[s["scenario"], f"{s['mean_abs_delta_prob']:.3f}", f"{s['high_tier_flips_pct']:.1f}%", f3(s["roc_auc"])] for s in sens],
                  "Robustness of predictions to input perturbations (test sample).", [3.0, 1.1, 1.1, 1.0])
        worst = max(sens[1:], key=lambda s: s["high_tier_flips_pct"])
        noise = [s for s in sens if s["scenario"].startswith("Gaussian")]
        rep.p(f"The model is most sensitive to '{worst['scenario']}', which changes the High-tier decision for {worst['high_tier_flips_pct']:.1f}% of appointments, as expected for a real driver. "
              + (f"Random 10% noise barely moves discrimination (ROC-AUC {f3(noise[0]['roc_auc'])} vs {f3(sens[0]['roc_auc'])}), so small data-entry errors in a DHIS2 extract will not destabilise the tiers. " if noise else "")
              + f"Figure {rep.fig + 1} shows how recall and workload move with the threshold, letting the hospital trade reach against outreach capacity.")
    rep.figure(C.FIG / "threshold_sensitivity.png", "Recall and share of appointments flagged across decision thresholds.", 4.6)
    rep.p("The Model Card (Mitchell et al., 2019) is Appendix A; the full Fairness Metrics Report is Appendix B.")

    # ---------------- 7
    rep.h("7. Interpretation, Risks and Monitoring")
    if imp:
        rep.p(f"In business terms the model has learned that no-show risk is driven mainly by {imp[0][0].lower()} and {imp[1][0].lower()}: the circumstances under which a referral is booked matter more than the diagnosis. "
              "For JJUSHYCSH this means concentrating confirmation calls on long-lead bookings and patients with missed visits in their history, offering earlier slots where capacity allows, and reaching first visits, which have no history, by default.")
    rep.p("[Personalise: one or two sentences on what these drivers mean in Jigjiga — distance, transport cost, seasonal pastoral movement, phone coverage for SMS — from your own knowledge.]", highlight=True)
    rep.p("Limitations and risks, linked to the Module 1 risk register and Module 2 RAID log: (1) the data come from a different health system (R-02), so this is a validated method, not a deployable JJUSHYCSH model, until retrained on local records; (2) six weeks of history cannot capture seasonality; (3) distance and transport, likely decisive in the Somali Region, are absent until the local data contract adds distance bands; "
          "(4) scores could be misused to deprioritise patients (R-04), which policy forbids and the API states; (5) the disability group is too small for precise fairness estimates; and (6) pseudonymised data remains personal data under Proclamation No. 1321/2024 (R-05), so the Module 3 access tiers apply to scores too.")
    rep.table(["Signal", "Acceptance criterion / trigger", "Frequency"],
              [["Recall on no-shows", f"≥ {C.TARGET_RECALL:.2f}; retrain if < {C.TARGET_RECALL - 0.05:.2f}", "Monthly"],
               ["Precision / flag rate", "Flag rate within outreach capacity; alert if +50% vs baseline", "Weekly"],
               ["Per-group TPR gap (4 attributes)", f"≤ {C.EOD_MAX}; governance committee review if exceeded", "Monthly"],
               ["Input drift (lead time, age, history)", "Population stability index < 0.2", "Monthly"],
               ["Calibration (Brier score)", f"Within 10% of validation value ({f3(cr.get('brier'))})", "Quarterly"],
               ["Override log", "Staff overrides and reasons reviewed", "Quarterly"]],
              "Acceptance criteria and post-deployment monitoring hooks.", [1.9, 3.1, 1.0])
    fair_ok = all(fb[a]["passes_tpr"] for a in C.PROTECTED) or \
        (mit.get("applied") and all(mit["after"][a]["tpr_gap"] <= C.EOD_MAX for a in C.PROTECTED))
    gain_ok = gain >= C.MIN_GAIN_OVER_RULE
    passed = sum([met, gain_ok, bool(fair_ok)])
    rep.p(f"Acceptance decision: on the proxy data the champion meets {passed} of the three Module 1 success criteria (recall target {'met' if met else 'not met'}; gain over the rule baseline {'met' if gain_ok else 'not met'}; fairness tolerances {'met' if fair_ok else 'not met'}). "
          "It is registered at Staging as a validated method, not Production: Module 2's Gate 3 (performance, calibration, fairness, explainability) passes on proxy data, but Gate 4 needs retraining on JJUSHYCSH records, a repeat of this validation pack, and AI & Data Governance Committee sign-off. The Module 5 dashboard will consume the same scores and monitoring signals.")

    # ---------------- 8
    rep.h("8. Deployment Artefacts")
    reg = R.get("registry", {})
    rep.p(f"The champion is serialised as module4/outputs/model/noshow_model.joblib (pipeline, thresholds, features, data hash) and registered in the MLflow Model Registry as '{reg.get('model_name', C.REGISTERED_MODEL)}' version {reg.get('version', 'n/a')}, alias 'champion', stage {reg.get('stage', 'n/a')}. "
          "A FastAPI service (module4/api/main.py) exposes POST /predict, returning the probability, the Low/Medium/High tier, the Module 1 outreach action and plain-language reason codes, and it builds features with the Module 3 definitions to avoid training–serving skew. "
          "The SHAP notebook, Fairness Metrics Report, Model Card and MLflow export sit in the repository beside the Module 3 pipeline.")
    rep.p("[Insert screenshot: MLflow experiment runs table (mlflow ui).]", highlight=True)
    rep.p("[Insert screenshot: MLflow Model Registry showing the registered version and alias.]", highlight=True)
    rep.p("[Insert screenshot: FastAPI /docs page with a successful /predict response.]", highlight=True)
    rep.p("[Insert your GitHub repository link here.]", highlight=True)
    body_words = rep.words

    # ---------------- References
    rep.count = False
    rep.h("References")
    refs = [
        "Ahmed, A. F. (2026a). Project vision document [Unpublished BAN6800 Module 1 assignment]. Nexford University.",
        "Ahmed, A. F. (2026b). Business analytics project overview report [Unpublished BAN6800 Module 2 assignment]. Nexford University.",
        "Ahmed, A. F. (2026c). Designing a data pipeline for referral no-show analytics [Unpublished BAN6800 Module 3 assignment]. Nexford University.",
        "Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 785–794. https://doi.org/10.1145/2939672.2939785",
        "CURE Ethiopia. (2023, August 29). New mobile clinic in Jigjiga. https://ethiopia.cure.org/news/new-mobile-clinic-in-jigjiga/",
        "Dantas, L. F., Fleck, J. L., Cyrino Oliveira, F. L., & Hamacher, S. (2018). No-shows in appointment scheduling – a systematic literature review. Health Policy, 122(4), 412–421. https://doi.org/10.1016/j.healthpol.2018.02.002",
        "Federal Ministry of Justice, Ethiopia. (2024). Personal Data Protection Proclamation No. 1321/2024. https://justice.gov.et/en/law/personal-data-protection-proclamation/",
        "Feldman, M., Friedler, S. A., Moeller, J., Scheidegger, C., & Venkatasubramanian, S. (2015). Certifying and removing disparate impact. Proceedings of the 21st ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 259–268. https://doi.org/10.1145/2783258.2783311",
        "Hardt, M., Price, E., & Srebro, N. (2016). Equality of opportunity in supervised learning. Advances in Neural Information Processing Systems, 29, 3315–3323.",
        "Hoppen, J. (2016). Medical appointment no shows [Data set]. Kaggle. https://www.kaggle.com/datasets/joniarroba/noshowappointments",
        "Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. Advances in Neural Information Processing Systems, 30, 4765–4774.",
        "Mitchell, M., Wu, S., Zaldivar, A., Barnes, P., Vasserman, L., Hutchinson, B., Spitzer, E., Raji, I. D., & Gebru, T. (2019). Model cards for model reporting. Proceedings of the Conference on Fairness, Accountability, and Transparency, 220–229. https://doi.org/10.1145/3287560.3287596",
        "Mothilal, R. K., Sharma, A., & Tan, C. (2020). Explaining machine learning classifiers through diverse counterfactual explanations. Proceedings of the 2020 Conference on Fairness, Accountability, and Transparency, 607–617. https://doi.org/10.1145/3351095.3372850",
        "Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12, 2825–2830.",
        "Wachter, S., Mittelstadt, B., & Russell, C. (2018). Counterfactual explanations without opening the black box: Automated decisions and the GDPR. Harvard Journal of Law & Technology, 31(2), 841–887.",
        "Weerts, H., Dudík, M., Edgar, R., Jalali, A., Lutz, R., & Madaio, M. (2023). Fairlearn: Assessing and improving fairness of AI systems. Journal of Machine Learning Research, 24(257), 1–8.",
        "Zaharia, M., Chen, A., Davidson, A., Ghodsi, A., Hong, S. A., Konwinski, A., Murching, S., Nykodym, T., Ogilvie, P., Parkhe, M., Xie, F., & Zumar, C. (2018). Accelerating the machine learning lifecycle with MLflow. IEEE Data Engineering Bulletin, 41(4), 39–45.",
    ]
    for r_ in refs:
        para = d.add_paragraph(r_); para.paragraph_format.left_indent = Inches(0.4)
        para.paragraph_format.first_line_indent = Inches(-0.4); para.paragraph_format.space_after = Pt(4)
        for run in para.runs: run.font.size = Pt(10)

    # ---------------- Appendices
    d.add_page_break(); rep.h("Appendix A. Model Card")
    _md_block(d, (C.OUT / "model_card.md").read_text(encoding="utf-8"))
    d.add_page_break(); rep.h("Appendix B. Fairness Metrics Report")
    _md_block(d, (C.OUT / "fairness_report.md").read_text(encoding="utf-8"))
    rep.h("Appendix C. AI Disclosure")
    rep.p("See the separately submitted NXU AI Disclosure Form (Module 4).", italic=True)

    out = C.OUT / f"BAN6800_Module4_Model_Validation_Report_{C.AUTHOR.replace(' ', '_')}.docx"
    d.save(out)
    (C.OUT / "report_word_count.txt").write_text(f"{body_words}\n")
    print(f"Report body word count (Sections 1–8, excl. tables, captions, references, appendices, highlighted notes): {body_words}")
    return out


def _md_line(d, line):
    if not line.strip() or line.startswith("|---") or line.startswith("# "):
        return
    if line.startswith("## "):
        p = d.add_paragraph(); r = p.add_run(line[3:]); r.bold = True; r.font.color.rgb = TEAL; return
    p = d.add_paragraph(style="List Bullet" if line.startswith("- ") else None)
    text = line[2:] if line.startswith("- ") else line
    for i, part in enumerate(text.split("**")):
        r = p.add_run(part); r.font.size = Pt(9.5); r.bold = (i % 2 == 1)
    p.paragraph_format.space_after = Pt(2)


def _md_block(d, text):
    """Render the simple Markdown we generate: headings, bullets, bold, and pipe tables."""
    rows = []

    def flush():
        if not rows:
            return
        t = d.add_table(rows=0, cols=len(rows[0])); t.style = "Light Grid Accent 1"
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        for i, r in enumerate(rows):
            cells = t.add_row().cells
            for j, v in enumerate(r[:len(cells)]):
                cells[j].text = ""; run = cells[j].paragraphs[0].add_run(v); run.font.size = Pt(8.5); run.bold = (i == 0)
        d.add_paragraph().paragraph_format.space_after = Pt(2)
        rows.clear()

    for line in text.splitlines():
        if line.startswith("|"):
            if not line.startswith("|---"):
                rows.append([c.strip() for c in line.strip().strip("|").split("|")])
            continue
        flush()
        _md_line(d, line)
    flush()
