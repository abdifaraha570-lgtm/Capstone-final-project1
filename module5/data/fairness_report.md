# Fairness Metrics Report — JJUSHYCSH Referral No-Show Risk Model

Generated 2026-10-06 by Abdi Farah Ahmed from the Module 3 test split (16,520 appointments). Favourable outcome = being flagged High risk (it triggers extra outreach support, never reduced care).

Criteria (Module 2 plan): disparate impact ratio >= 0.8; equalized-odds difference <= 0.1. Primary criterion: equal opportunity (recall parity), because the aim is to reach at-risk patients equally in every group. Protected attributes are the Module 3 bias-audit groups.

## gender

| Group | n | No-show base rate | Flag rate | Recall (TPR) | FPR | Precision |
|---|---|---|---|---|---|---|
| F | 10,768 | 0.204 | 0.441 | 0.719 | 0.369 | 0.333 |
| M | 5,752 | 0.200 | 0.423 | 0.707 | 0.352 | 0.335 |

- Demographic parity difference: 0.017
- Disparate impact ratio: 0.960 (meets 0.8)
- Equalized-odds difference: 0.017 (meets 0.1); TPR gap 0.012, FPR gap 0.017
- Fairlearn cross-check: DP diff 0.017, EO diff 0.017

## age_band

| Group | n | No-show base rate | Flag rate | Recall (TPR) | FPR | Precision |
|---|---|---|---|---|---|---|
| 0-5 | 1,780 | 0.184 | 0.504 | 0.893 | 0.416 | 0.327 |
| 18-34 | 3,652 | 0.248 | 0.634 | 0.910 | 0.544 | 0.356 |
| 35-54 | 4,563 | 0.196 | 0.449 | 0.719 | 0.383 | 0.313 |
| 55-69 | 2,808 | 0.166 | 0.116 | 0.232 | 0.093 | 0.332 |
| 6-17 | 2,337 | 0.237 | 0.622 | 0.884 | 0.540 | 0.337 |
| 70+ | 1,380 | 0.148 | 0.100 | 0.181 | 0.086 | 0.268 |

- Demographic parity difference: 0.534
- Disparate impact ratio: 0.158 (below 0.8)
- Equalized-odds difference: 0.728 (exceeds 0.1); TPR gap 0.728, FPR gap 0.458
- Fairlearn cross-check: DP diff 0.534, EO diff 0.728

## scholarship

| Group | n | No-show base rate | Flag rate | Recall (TPR) | FPR | Precision |
|---|---|---|---|---|---|---|
| 0 | 14,954 | 0.198 | 0.422 | 0.696 | 0.354 | 0.327 |
| 1 | 1,566 | 0.244 | 0.555 | 0.859 | 0.457 | 0.377 |

- Demographic parity difference: 0.133
- Disparate impact ratio: 0.760 (below 0.8)
- Equalized-odds difference: 0.163 (exceeds 0.1); TPR gap 0.163, FPR gap 0.103
- Fairlearn cross-check: DP diff 0.133, EO diff 0.163

## has_disability

| Group | n | No-show base rate | Flag rate | Recall (TPR) | FPR | Precision |
|---|---|---|---|---|---|---|
| 0 | 16,194 | 0.203 | 0.437 | 0.718 | 0.366 | 0.334 |
| 1 | 326 | 0.178 | 0.298 | 0.517 | 0.250 | 0.309 |

- Demographic parity difference: 0.140
- Disparate impact ratio: 0.680 (below 0.8)
- Equalized-odds difference: 0.201 (exceeds 0.1); TPR gap 0.201, FPR gap 0.116
- Fairlearn cross-check: DP diff 0.140, EO diff 0.201

## Mitigation

Applied on **age_band**: group-specific thresholds chosen on validation (equal opportunity).
Group thresholds: 0-5=0.593, 18-34=0.629, 35-54=0.564, 55-69=0.474, 6-17=0.611, 70+=0.469.
TPR gap 0.728 -> 0.051; DI ratio 0.158 -> 0.802.
Overall after mitigation: recall 0.694, precision 0.321, flag rate 0.438.
Fairlearn ThresholdOptimizer comparison: TPR gap 0.059, recall 0.903, flag rate 0.620.

## Data bias carried forward from Module 3

- Module 3 flag: has_disability 1 — under-represented (share=0.0201)
