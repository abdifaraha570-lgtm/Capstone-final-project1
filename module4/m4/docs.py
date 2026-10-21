"""Writes the Model Card and the Fairness Metrics Report (Markdown) from actual run results."""
from m4 import config as C


def _f(x, d=3):
    try:
        return f"{float(x):.{d}f}"
    except (TypeError, ValueError):
        return str(x)


def fairness_report_md(R):
    L = [f"# Fairness Metrics Report — JJUSHYCSH Referral No-Show Risk Model", "",
         f"Generated {R['run_date']} by {C.AUTHOR} from the Module 3 test split ({R['n_test']:,} appointments). "
         f"Favourable outcome = being flagged High risk (it triggers extra outreach support, never reduced care).", "",
         f"Criteria (Module 2 plan): disparate impact ratio >= {C.DI_MIN}; equalized-odds difference <= {C.EOD_MAX}. "
         "Primary criterion: equal opportunity (recall parity), because the aim is to reach at-risk "
         "patients equally in every group. Protected attributes are the Module 3 bias-audit groups.", ""]
    for a in C.PROTECTED:
        s = R["fairness_before"][a]
        L += [f"## {a}", "", "| Group | n | No-show base rate | Flag rate | Recall (TPR) | FPR | Precision |",
              "|---|---|---|---|---|---|---|"]
        for r in s["table"]:
            L.append(f"| {r['group']} | {r['n']:,} | {_f(r['base_rate'])} | {_f(r['selection_rate'])} | "
                     f"{_f(r['tpr'])} | {_f(r['fpr'])} | {_f(r['precision'])} |")
        L += ["", f"- Demographic parity difference: {_f(s['dp_difference'])}",
              f"- Disparate impact ratio: {_f(s['di_ratio'])} ({'meets' if s['passes_di'] else 'below'} {C.DI_MIN})",
              f"- Equalized-odds difference: {_f(s['eo_difference'])} "
              f"({'meets' if s['passes_eo'] else 'exceeds'} {C.EOD_MAX}); TPR gap {_f(s['tpr_gap'])}, FPR gap {_f(s['fpr_gap'])}"]
        if "fairlearn_eo_diff" in s:
            L.append(f"- Fairlearn cross-check: DP diff {_f(s['fairlearn_dp_diff'])}, EO diff {_f(s['fairlearn_eo_diff'])}")
        L.append("")
    m = R["mitigation"]
    L += ["## Mitigation", ""]
    if m["applied"]:
        a = m["attribute"]; after = m["after"][a]; pa = m["performance_after"]
        L += [f"Applied on **{a}**: {m['method']}.",
              f"Group thresholds: {', '.join(f'{g}={_f(t)}' for g, t in m['group_thresholds'].items())}.",
              f"TPR gap {_f(R['fairness_before'][a]['tpr_gap'])} -> {_f(after['tpr_gap'])}; "
              f"DI ratio {_f(R['fairness_before'][a]['di_ratio'])} -> {_f(after['di_ratio'])}.",
              f"Overall after mitigation: recall {_f(pa['recall'])}, precision {_f(pa['precision'])}, flag rate {_f(pa['flag_rate'])}."]
        ft = m.get("fairlearn_threshold_optimizer", {})
        if "tpr_gap" in ft:
            L.append(f"Fairlearn ThresholdOptimizer comparison: TPR gap {_f(ft['tpr_gap'])}, recall {_f(ft['recall'])}, flag rate {_f(ft['flag_rate'])}.")
    else:
        L.append(f"No mitigation applied: {m['reason']}. The routine stays in the pipeline and runs at every retraining.")
    L += ["", "## Data bias carried forward from Module 3", ""]
    for fl in R.get("m3", {}).get("bias_flags", {}).get("flags", []):
        L.append(f"- Module 3 flag: {fl.get('attribute')} {fl.get('group', '')} — {fl.get('issue')} "
                 f"({', '.join(f'{k}={v}' for k, v in fl.items() if k not in ('attribute', 'group', 'issue'))})")
    if not R.get("m3", {}).get("bias_flags", {}).get("flags"):
        L.append("- No Module 3 bias flags were found in reports/bias_flags.json.")
    return "\n".join(L) + "\n"


def model_card_md(R):
    ch = R["champion"]; r = R["results"][ch]
    imp = list(R["shap"].get("importance", {}).items())[:5]
    fb = R["fairness_before"]; nr = R["necessity"]
    tc = R.get("temporal_check", {})
    return f"""# Model Card — JJUSHYCSH Referral No-Show Risk Model

**Version:** {R['registry'].get('version', 'n/a')} ({R['registry'].get('stage', 'n/a')}) · **Algorithm:** {ch.replace('_', ' ')} · **Date:** {R['run_date']} · **Owner:** {C.AUTHOR} (BAN6800, {C.UNIVERSITY})

## Intended use
Advisory decision support for JJUSHYCSH outpatient/referral coordinators: score each scheduled follow-up 24–72 h ahead and prioritise reminder outreach (SMS → phone call → community health worker). **Out of scope:** cancelling, overbooking against, or deprioritising any patient; any clinical decision. Rule from the Module 1 charter: a flag can add support, never reduce care.

## Training data
Module 3 pseudonymised model table built from the public Kaggle *Medical Appointment No Shows* data (Hoppen, 2016; Brazil), used as a disclosed proxy for JJUSHYCSH records. Patient-grouped 70/15/15 split (Module 3, seed 42): train {R['n_train']:,} / validation {R['n_val']:,} / test {R['n_test']:,}; test no-show rate {R['test_no_show']:.1%}. Data hash {R['data_md5'][:12]}… (route: {R['data_route']}).

## Features
{', '.join(R['features'])}.
Excluded on purpose: {', '.join(C.EXCLUDED.keys())}.
Necessity review of sensitive health flags (Module 3 hand-over): {'; '.join(f"{t['feature']}: Δ CV PR-AUC {t['delta']:+.4f} → {t['decision']}" for t in nr['table'])}.

## Performance (Module 3 test split, threshold {r['threshold']:.3f})
| ROC-AUC | PR-AUC | Recall | Precision | F1 | Accuracy | Brier | Flag rate |
|---|---|---|---|---|---|---|---|
| {_f(r['roc_auc'])} | {_f(r['pr_auc'])} | {_f(r['recall'])} | {_f(r['precision'])} | {_f(r['f1'])} | {_f(r['accuracy'])} | {_f(r['brier'])} | {_f(r['flag_rate'])} |

Forward-in-time check (refit on earliest 80 % of dates, scored on latest 20 %): {('ROC-AUC ' + _f(tc.get('roc_auc')) + ', recall ' + _f(tc.get('recall_at_early_threshold'))) if tc.get('status') == 'ok' else tc.get('status', 'n/a')}.

## Top drivers (mean |SHAP|)
{chr(10).join(f'- {k}: {v:.4f}' for k, v in imp) if imp else '- SHAP not available in this run'}

## Fairness (Module 3 test split)
{chr(10).join(f"- {a}: DI ratio {_f(fb[a]['di_ratio'])}, EO difference {_f(fb[a]['eo_difference'])}, TPR gap {_f(fb[a]['tpr_gap'])}" for a in C.PROTECTED)}
- Mitigation: {R['mitigation'].get('method', 'none') if R['mitigation']['applied'] else 'not required — ' + R['mitigation']['reason']}

## Limitations
- Proxy data from a different health system; must be re-validated on JJUSHYCSH/DHIS2 records before any real use.
- Only ~6 weeks of history; no seasonality, distance-to-facility, transport or rainy-season signals.
- The disability group is 2 % of records (Module 3 flag), so its fairness estimates are less precise.
- Modest discrimination: many flagged patients would have attended (a false positive costs only a reminder).

## Ethical considerations
Gender and geography are not model inputs; fairness is monitored by gender, age band, welfare status and disability. Pseudonymised data remains personal data under Ethiopia's Personal Data Protection Proclamation No. 1321/2024. Oversight: the Hospital Research & Ethics Review Committee and AI & Data Governance Committee proposed in Module 1.

## Monitoring
Monthly: recall and precision vs. acceptance criteria, per-group TPR gap, input drift (lead time, age), flag rate vs. outreach capacity. Retrain if recall < {C.TARGET_RECALL - 0.05:.2f} or any TPR gap > {C.EOD_MAX}.
"""
