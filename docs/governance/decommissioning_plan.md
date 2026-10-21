# Decommissioning Plan — retiring the model ethically

**When:** the Governance Committee decides to retire the model if (a) local validation fails, (b) an incident cannot be
remediated, (c) the no-show problem is addressed upstream and the KPI gain disappears, (d) the data contract or legal basis
changes, or (e) a successor model is promoted.

| Step | Action |
|---|---|
| 1. Decision | Recorded in `decision_log.md` with reason, date and signatories |
| 2. Stop scoring | API returns HTTP 410 with a "retired" message; dashboard what-if page disabled; compose services `api` and `monitoring` stopped; Render service suspended |
| 3. Communicate | Staff notice (OPD), patient/community notice via the Public Trust Statement, SRHB informed |
| 4. Data | Scored lists and explanation-store records retained for the audit period set by the DPO (default 24 months), then deleted; pseudonymised training data deleted per the Module 3 retention schedule; audit log of deletions kept |
| 5. Artefacts | Model bundle, model card, fairness reports and this plan archived read-only in the MLflow registry (stage "Archived") and the repository tag `retired-<date>` |
| 6. Successor | If a successor exists, run both in shadow for one month and apply the same release gates before cut-over |
| 7. Lessons | Final review report to the Governance and Ethics Committees |
