# Incident Response Plan for Ethical Failures

**Trigger examples:** fairness ALERT (TPR gap > 0.10), evidence that a score influenced a denial or delay of care, a privacy
breach (identifier exposure), systematic errors for a group, a complaint from a patient or staff member, or a drift ALERT that
changes the flag rate by > 10 points.

| Step | Time | Owner | Action |
|---|---|---|---|
| 1. Detect & log | Immediately | Model owner / anyone | Open a GitHub issue labelled `incident` (automated for monitoring ALERTs) and the incident register |
| 2. Contain | ≤ 24 h | Model owner | Switch scores to "advisory — under review" or pause the API (`docker compose stop api`); keep manual outreach running |
| 3. Assess | ≤ 3 working days | Model owner + DPO | Severity (patient harm? privacy? fairness?), affected groups and period, root cause using the explanation store and monitoring history |
| 4. Notify | ≤ 3 working days | Committee chair | Governance Committee, Ethics Committee; patients/community if care or privacy was affected; SRHB and regulator for a privacy breach as required by Proclamation 1321/2024 |
| 5. Remediate | ≤ 10 working days | Model owner | Mitigation (thresholds, retrain, feature removal, data fix); re-run full fairness audit; document |
| 6. Review & close | ≤ 20 working days | Governance Committee | Approve return to service or decommissioning; lessons recorded in `decision_log.md`; monitoring limits adjusted if needed |

Post-incident: a one-page public summary is added to the Public Trust Statement page of the dashboard.
