# Ethical AI Framework — JJUSHYCSH Referral No-Show Risk Model
*BAN6800 Final Project · Abdi Farah Ahmed · October 2026 · consolidates the ethics work of Modules 1–5*

## 1. Purpose and scope
This framework governs the development, deployment, monitoring and retirement of the referral no-show risk model built for
Jigjiga University Sheikh Hassan Yabare Comprehensive Specialized Hospital (JJUSHYCSH). It applies to every component in this
repository: the Module 3 pipeline, the Module 4 model and `/predict` API, the Module 5 dashboard, and the Final Project
monitoring, retraining and CI/CD components.

## 2. Ethical AI Charter (Module 1, unchanged)
1. **Decision support, never decision making.** A risk score may only *add* outreach (SMS, call, community health worker visit);
   it must never be used to cancel, delay, deprioritise or deny an appointment.
2. **Transparency.** Every score is accompanied by plain-language reason codes; the model card, fairness report and
   limitations statement are public within the hospital.
3. **Fairness.** No patient is disadvantaged because of gender, age, welfare status, disability, geography, language or
   pastoralist origin. Release rule: true-positive-rate gap ≤ 0.10 on every monitored attribute; disparate-impact ratio ≥ 0.80
   as a diagnostic.
4. **Privacy by design.** No direct identifiers leave the Module 3 anonymisation step; pseudonymised data remain personal data
   under Ethiopia's Personal Data Protection Proclamation No. 1321/2024.
5. **Human accountability.** Named owners (Section 8) answer for the model; staff may override any flag and overrides are logged.
6. **Local validation before use.** No operational claim is made from the Brazilian proxy data; the model is recalibrated and
   re-audited on an approved JJUSHYCSH extract before it influences any outreach decision.

## 3. Algorithmic Impact Assessment (AIA)
| Dimension | Assessment |
|---|---|
| Decision supported | Which referred/follow-up patients receive reminder or outreach resources 24–72 h before the visit |
| Affected people | ~325,000 women of reproductive age and ~150,000 under-fives in the catchment; all OPD referral patients |
| Severity of a wrong prediction | **False negative:** patient receives no extra reminder (status quo). **False positive:** an unnecessary reminder (minor cost, mild intrusion). No decision reduces access to care, by design. |
| Autonomy / reversibility | Advisory score; staff decide; overrides logged; model can be switched off instantly (Section 7) |
| Vulnerable groups | Older patients (age-band gap found and mitigated), welfare-enrolled patients (gap 0.12, under review), patients with disability (small group, unstable estimates), rural/pastoralist patients (geography deliberately excluded from inputs; to be audited on local data) |
| Data sensitivity | Health appointment data; chronic-condition flags; welfare enrolment. Pseudonymised, access-tiered, retention-limited |
| Explainability | Global SHAP, local SHAP reason codes, DiCE counterfactuals; explanation store for audit |
| Residual risk rating | **Medium-low** while advisory and locally validated; **high** if ever used to ration slots — prohibited |
| Impact level (per Canada's AIA scale) | Level II — moderate, reversible impacts; peer review, notice, human-in-the-loop and monitoring required |

## 4. Regulatory compliance report
| Regime | Applicability | Position |
|---|---|---|
| **Ethiopia Personal Data Protection Proclamation No. 1321/2024** | Primary law for JJUSHYCSH data | Lawful basis: public-interest health service delivery; data minimisation (12 inputs, protected attributes excluded); pseudonymisation with salted hashes; purpose limitation (outreach targeting only); DPO approval required before any local extract; data-subject rights handled through the hospital's records office |
| **GDPR (EU 2016/679)** | Not directly applicable (no EU data subjects); used as the design benchmark | Art. 5 principles met (minimisation, accuracy, storage limitation, integrity); Art. 22 — no solely automated decision with legal/similar effect because the output is advisory and human-reviewed; Art. 35 DPIA equivalent = this AIA |
| **CCPA/CPRA (California)** | Not applicable (no California residents) | Benchmark only: notice, purpose limitation, no sale of data |
| **HIPAA (USA)** | Not applicable (no US covered entity) | Benchmark used for the de-identification standard: the Module 3 anonymiser removes the 18 HIPAA identifier classes; k ≥ 20 area generalisation |
| **Ethiopian MoH HMIS / Information Revolution policies** | Applicable to HMIS data use | Data stays within the regional health information governance; HMIS data-use agreement to be signed by JJUSHYCSH and SRHB |
| **Kaggle dataset licence (proxy data)** | Applicable to development data | Public dataset used for development only; not redistributed in the repository (.gitignore) |

## 5. Public trust statement
*To patients, families and the communities served by JJUSHYCSH:* this tool helps the hospital remember to reach out to patients
who may find it hard to come back for a follow-up visit. It never decides who gets care. It uses only appointment history,
booking lead time, age, and chronic-condition flags — never your name, address, clan, language or phone number. Its scores
are checked every month for fairness between women and men, age groups, welfare-enrolled patients and patients with
disability, and the results are shared with the hospital ethics committee. You can ask hospital staff why you were contacted,
and you can ask not to be contacted. If the tool is found to be unfair or inaccurate, it is switched off until it is fixed.

## 6. Ethical governance structure (ongoing oversight)
| Body / role | Members | Responsibilities | Cadence |
|---|---|---|---|
| **AI & Data Governance Committee** (chair: Medical Director) | Medical Director, OPD head, HMIS/IT lead, DPO, nurse representative, community/patient representative, SRHB MCH Directorate | Approves release, threshold changes, retraining promotions, incident closure and decommissioning; reviews fairness reports | Monthly (30 min standing item); ad hoc for incidents |
| **Hospital Research & Ethics Review Committee** | Per hospital charter | Approves the local data extract and any change of purpose | On request |
| **Model owner** (analytics lead) | Abdi Farah Ahmed | Maintains model, monitoring, documentation; runs the monthly review; first responder to incidents | Continuous |
| **Data Protection Officer** | Hospital DPO | Access tiers, retention, privacy audit log, data-subject requests | Continuous |
| **OPD coordinators** | OPD staff | Use scores, record overrides and reasons, raise concerns | Daily |

Decision log: `docs/governance/decision_log.md`. Explanation store: `docs/governance/explanation_store.md`.
