# Implementation, Monitoring and Continuous-Fairness Plan

## 1. Deployment plan (phased)
| Phase | What | Exit criterion |
|---|---|---|
| 0. Local validation (month 1–2) | Approved anonymised JJUSHYCSH OPD/referral extract through the Module 3 pipeline; recalibrate; re-run the fairness audit with geography and language added as monitored attributes; establish the local no-show baseline | Recall ≥ 0.70, TPR gap ≤ 0.10 on all attributes, Governance Committee sign-off |
| 1. Shadow mode (month 3) | API scores every scheduled visit; scores visible only to the model owner; outreach unchanged | Flag rate 35–55 %, no ALERT in monitoring for 4 weeks |
| 2. Pilot in one clinic (months 4–6) | Ranked list 24–72 h ahead with tiered outreach (SMS → call → CHW); human review of every list | ≥ 15 % relative reduction in no-shows vs baseline; override rate < 20 %; no unresolved incident |
| 3. Scale-up (months 7–12) | Other follow-up clinics; monthly governance review | KPIs sustained for two consecutive quarters |

Infrastructure: `docker-compose.yml` (pipeline, MLflow, API, dashboard, monitoring); CI in `.github/workflows/ci.yml`;
monthly monitoring + gated retraining in `.github/workflows/retrain.yml`; API deployable with `render.yaml`.

## 2. KPIs
| KPI | Target | Source | Frequency |
|---|---|---|---|
| Referral no-show rate | −15 % relative vs baseline within two quarters | HMIS / OPD register | Monthly |
| Specialist-slot utilisation | ↑ measurable | OPD schedule | Monthly |
| High-risk patients reached within window | ≥ 80 % | Outreach log | Weekly |
| Recall at operating threshold | ≥ 0.70 (floor 0.65) | monitor.py (labels 30 days later) | Monthly |
| Precision | ≥ 0.30 (floor 0.28) | monitor.py | Monthly |
| Flag rate | 35–55 % of appointments | monitor.py | Weekly |
| TPR gap per attribute | ≤ 0.10 | monitor.py / Fairlearn | Monthly |
| Disparate-impact ratio | ≥ 0.80 (diagnostic) | monitor.py | Monthly |
| Override rate and reasons | < 20 %; reviewed | Dashboard log | Monthly |
| API availability / latency | ≥ 99 %; p95 < 500 ms | Render / compose health checks | Continuous |

## 3. Model drift and bias drift detection
* **Input drift:** PSI per feature (warn ≥ 0.10, alert ≥ 0.20) and KS test; Evidently DataDriftPreset when installed.
* **Prediction drift:** PSI of the score distribution; flag rate outside 35–55 %.
* **Performance drift:** recall, precision, ROC-AUC and Brier score recomputed when outcomes arrive (30-day lag).
* **Bias drift:** selection rate, TPR, FPR per group; TPR gap, equalised-odds difference, DI ratio; Fairlearn cross-check.
* **Calibration:** quarterly reliability curve; recalibrate thresholds if Brier worsens by > 0.02.
* The deployed decision rule (age-band thresholds) is read from the model bundle so monitoring always mirrors production.

## 4. Continuous fairness monitoring
Monthly batch → `monitoring/monitor.py` → `monitoring_summary.json` → dashboard "Monitoring" page and GitHub issue on ALERT.
Any TPR gap > 0.10 triggers (a) pause of automated use of scores for the affected group (scores still visible, flagged
"under review"), (b) root-cause analysis within 10 working days, (c) mitigation (threshold re-selection on validation,
reweighting, or feature review) and (d) Governance Committee decision. Small groups (< 200 positives) are reported with
confidence intervals and judged over three months, because single-month gaps are unstable (seen in the demo batch for
disability and welfare enrolment).

## 5. Retraining policy
`retrain.py` runs monthly or on ALERT. A challenger is approved only if validation PR-AUC ≥ champion − 0.01, recall ≥ 0.70,
every TPR gap ≤ 0.10, and no protected attribute is an input. Promotion to champion is a human decision recorded in the
decision log; the previous champion is kept in the MLflow registry for rollback.
