# Data Governance Framework (Module 3)

## Access tiers (enforced in `src/governance/access_control.py`)
| Tier | Data | Roles with access | Controls |
|---|---|---|---|
| Restricted | `data/raw` snapshot, hash salt | HIS/data officer, data engineer | File mode 600, encrypted disk, named accounts |
| Pseudonymized | interim, model table, splits | + analyst | No direct IDs; GX privacy gate must pass |
| Aggregate | dashboard feed, reports | + OPD coordinators, department heads, Medical Director, Regional Health Bureau | Counts and rates only |

Unknown datasets default to the Restricted tier. Every allowed and denied access is written to the audit log.

## Retention
| Asset | Keep for | Then |
|---|---|---|
| Raw snapshot (local extract) | 12 months | Secure delete; manifest hash kept |
| Pseudonymized tables | 24 months (retraining + audit) | Delete or re-approve |
| Aggregate outputs | 5 years | Archive |
| Privacy audit log | 7 years | Archive, read-only |

## Roles
- Data owner: JJUSHYCSH Medical Director
- Data steward: HIS/data officer (definitions, quality, salt custody)
- Data engineer: pipeline code, GX suites, containers
- AI & Data Governance Committee: reviews bias flags, sensitive-feature use, any change of purpose
- DPO/IT: privacy incidents, encryption, backups

## Usage rules
1. Purpose limitation: outreach targeting for missed referral visits only.
2. A risk flag may trigger extra support; it may never reduce, cancel, or deprioritize care.
3. New use of the data = new ethics review.
4. Bias flags from `bias_check.py` go to the committee; data is not silently reweighted or dropped.
