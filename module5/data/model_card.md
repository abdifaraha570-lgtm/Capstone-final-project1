# Model Card — JJUSHYCSH Referral No-Show Risk Model

**Version:** 1 (Staging) · **Algorithm:** random forest · **Date:** 2026-10-06 · **Owner:** Abdi Farah Ahmed (BAN6800, Nexford University)

## Intended use
Advisory decision support for JJUSHYCSH outpatient/referral coordinators: score each scheduled follow-up 24–72 h ahead and prioritise reminder outreach (SMS → phone call → community health worker). **Out of scope:** cancelling, overbooking against, or deprioritising any patient; any clinical decision. Rule from the Module 1 charter: a flag can add support, never reduce care.

## Training data
Module 3 pseudonymised model table built from the public Kaggle *Medical Appointment No Shows* data (Hoppen, 2016; Brazil), used as a disclosed proxy for JJUSHYCSH records. Patient-grouped 70/15/15 split (Module 3, seed 42): train 76,706 / validation 16,673 / test 16,520; test no-show rate 20.3%. Data hash 65596f8d2f9e… (route: module3_splits).

## Features
age, lead_days, prior_visits, prior_no_shows, prior_no_show_rate, chronic_count, scholarship, hypertension, diabetes, same_day, first_visit, appt_weekday.
Excluded on purpose: sms_received, area_code, gender, appt_month, lead_band, patient_key / appointment_key / dates.
Necessity review of sensitive health flags (Module 3 hand-over): alcoholism: Δ CV PR-AUC +0.0014 → drop as input (audit only); has_disability: Δ CV PR-AUC +0.0001 → drop as input (audit only).

## Performance (Module 3 test split, threshold 0.556)
| ROC-AUC | PR-AUC | Recall | Precision | F1 | Accuracy | Brier | Flag rate |
|---|---|---|---|---|---|---|---|
| 0.741 | 0.387 | 0.715 | 0.334 | 0.455 | 0.653 | 0.208 | 0.435 |

Forward-in-time check (refit on earliest 80 % of dates, scored on latest 20 %): ROC-AUC 0.737, recall 0.682.

## Top drivers (mean |SHAP|)
- Lead time (days booked ahead): 0.0992
- Same-day appointment: 0.0909
- Patient age: 0.0421
- Prior no-show rate: 0.0225
- Number of prior no-shows: 0.0091

## Fairness (Module 3 test split)
- gender: DI ratio 0.960, EO difference 0.017, TPR gap 0.012
- age_band: DI ratio 0.158, EO difference 0.728, TPR gap 0.728
- scholarship: DI ratio 0.760, EO difference 0.163, TPR gap 0.163
- has_disability: DI ratio 0.680, EO difference 0.201, TPR gap 0.201
- Mitigation: group-specific thresholds chosen on validation (equal opportunity)

## Limitations
- Proxy data from a different health system; must be re-validated on JJUSHYCSH/DHIS2 records before any real use.
- Only ~6 weeks of history; no seasonality, distance-to-facility, transport or rainy-season signals.
- The disability group is 2 % of records (Module 3 flag), so its fairness estimates are less precise.
- Modest discrimination: many flagged patients would have attended (a false positive costs only a reminder).

## Ethical considerations
Gender and geography are not model inputs; fairness is monitored by gender, age band, welfare status and disability. Pseudonymised data remains personal data under Ethiopia's Personal Data Protection Proclamation No. 1321/2024. Oversight: the Hospital Research & Ethics Review Committee and AI & Data Governance Committee proposed in Module 1.

## Monitoring
Monthly: recall and precision vs. acceptance criteria, per-group TPR gap, input drift (lead time, age), flag rate vs. outreach capacity. Retrain if recall < 0.65 or any TPR gap > 0.1.
