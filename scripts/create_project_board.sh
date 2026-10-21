#!/usr/bin/env bash
# Creates the GitHub Project board for the BAN6800 capstone from the Module 2 ticket register
# (Appendix B of the Module 2 Project Overview Report) plus the Final Project tickets.
# Status reflects what was actually delivered: tickets with no evidence of completion stay open.
#
# Requirements: GitHub CLI (https://cli.github.com), logged in:  gh auth login
#               then add the project scope once:                 gh auth refresh -s project
# Run (Git Bash on Windows, or any terminal on macOS/Linux):     bash scripts/create_project_board.sh
set -euo pipefail

OWNER="abdifaraha570-lgtm"
REPO="$OWNER/capstoneMolude3"
TITLE="JJUSHYCSH Referral No-Show Capstone (BAN6800)"

echo "Creating project..."
PNUM=$(gh project create --owner "$OWNER" --title "$TITLE" --format json --jq .number)
PID=$(gh project view "$PNUM" --owner "$OWNER" --format json --jq .id)
gh project link "$PNUM" --owner "$OWNER" --repo "$REPO" >/dev/null 2>&1 || true

# Sprint field (single select)
gh project field-create "$PNUM" --owner "$OWNER" --name "Sprint" --data-type SINGLE_SELECT \
  --single-select-options "Sprint 0 · Module 2,Sprint 1 · Module 3,Sprint 2 · Module 4,Sprint 3 · Module 5,Final Project" >/dev/null

STATUS_FID=$(gh project field-list "$PNUM" --owner "$OWNER" --format json --jq '.fields[] | select(.name=="Status") | .id')
SPRINT_FID=$(gh project field-list "$PNUM" --owner "$OWNER" --format json --jq '.fields[] | select(.name=="Sprint") | .id')
opt() { gh project field-list "$PNUM" --owner "$OWNER" --format json --jq ".fields[] | select(.name==\"$1\") | .options[] | select(.name==\"$2\") | .id"; }

DONE=$(opt Status "Done"); INPROG=$(opt Status "In Progress"); TODO=$(opt Status "Todo")
S0=$(opt Sprint "Sprint 0 · Module 2"); S1=$(opt Sprint "Sprint 1 · Module 3"); S2=$(opt Sprint "Sprint 2 · Module 4")
S3=$(opt Sprint "Sprint 3 · Module 5"); SF=$(opt Sprint "Final Project")

add() {  # id | sprint-option | status-option | title | acceptance criterion | evidence
  local id="$1" sp="$2" st="$3" title="$4" acc="$5" ev="$6"
  local url item
  url=$(gh issue create -R "$REPO" --title "$id $title" \
        --body "**Acceptance criterion:** $acc"$'\n\n'"**Evidence:** $ev"$'\n\n'"Source: Module 2 ticket register (Appendix B) / Final Project plan.")
  item=$(gh project item-add "$PNUM" --owner "$OWNER" --url "$url" --format json --jq .id)
  gh project item-edit --id "$item" --project-id "$PID" --field-id "$SPRINT_FID" --single-select-option-id "$sp" >/dev/null
  gh project item-edit --id "$item" --project-id "$PID" --field-id "$STATUS_FID" --single-select-option-id "$st" >/dev/null
  if [ "$st" = "$DONE" ]; then gh issue close "$url" --reason completed >/dev/null; fi
  echo "  $id  $title"
}

echo "Adding tickets..."
add S0-01 "$S0" "$DONE" "Initialize repository and Git Flow"          "Repo contains agreed structure and README"            "Module 2 repository; README.md"
add S0-02 "$S0" "$DONE" "Create architecture diagram"                 "All components and data flows documented"             "Module 2 report, Figure 2"
add S0-03 "$S0" "$DONE" "Approve privacy and data-minimization plan"  "Required fields, roles, security, retention defined"  "docs/data_governance.md, docs/anonymization_plan.md"
add S0-04 "$S0" "$DONE" "Create RAID register"                        "Owner, trigger, mitigation, status recorded"          "Module 2 RAID log"
add S0-05 "$S0" "$DONE" "Create preliminary data dictionary"          "Proxy and future local fields documented"             "docs/data_dictionary.md"
add S1-01 "$S1" "$DONE" "Ingest Kaggle proxy data"                    "Raw file immutable; schema validation passes"         "src/data_prep/ingest.py (SHA-256 snapshot); raw suite 28/28"
add S1-02 "$S1" "$DONE" "Profile data quality"                        "Missingness, invalid ages/dates, duplicates documented" "Validation_Results_and_Evidence/data_quality_report.json"
add S1-03 "$S1" "$DONE" "Engineer scheduling features"                "Lead time, day-of-week, age bands reproducible"       "src/features/build_features.py"
add S1-04 "$S1" "$DONE" "Build testable preprocessing pipeline"       "Unit tests and versioned processed data"              "Airflow DAG, tests/, Dockerfile; processed suite 23/23"
add S2-01 "$S2" "$DONE" "Train transparent baseline"                  "Logistic regression metrics logged in MLflow"         "MLflow experiment jjushycsh-noshow-m4"
add S2-02 "$S2" "$DONE" "Train candidate tree model"                  "XGBoost comparison documented"                        "module4/outputs/results.json; Final report Table 4"
add S2-03 "$S2" "$DONE" "Select high-risk threshold"                  "Recall >=70% or limitation explicitly documented"     "Threshold 0.556 (validation recall 70%)"
add S2-04 "$S2" "$DONE" "Run fairness assessment"                     "TPR/FPR/selection metrics by group reported"          "module4/outputs/fairness_report.md"
add S2-05 "$S2" "$DONE" "Generate SHAP analysis and model card"       "Global/local explanations and limitations recorded"   "module4/notebooks/shap_analysis.ipynb; model_card.md"
add S3-01 "$S3" "$DONE" "Implement outreach rules"                    "Risk tier maps to approved support action only"       "Dashboard what-if page; API outreach action"
add S3-02 "$S3" "$DONE" "Build dashboard prototype"                   "Role-based views and KPI monitoring available"        "Live Streamlit dashboard"
add S3-03 "$S3" "$INPROG" "Conduct UAT and workflow review"           "User feedback logged and critical defects resolved"   "Not yet done with OPD users: planned for the shadow-mode phase"
add S3-04 "$S3" "$TODO" "Pilot-readiness governance gate"             "Privacy/fairness/operations sign-off recorded"        "Open: welfare-enrolment TPR gap 0.12 awaits Governance Committee ruling"
add F-01  "$SF" "$DONE" "Drift and fairness monitoring"               "PSI/KS drift, per-group TPR gaps, ALERT status"       "monitoring/monitor.py; monitoring_summary.json"
add F-02  "$SF" "$DONE" "Gated retraining"                            "Challenger registered only if all gates pass"         "retrain.py"
add F-03  "$SF" "$DONE" "CI/CD with GitHub Actions"                   "Tests, monitoring smoke test, image build on push"    ".github/workflows/ci.yml, retrain.yml"
add F-04  "$SF" "$DONE" "Docker Compose multi-container setup"        "Pipeline, MLflow, API, dashboard, monitoring"         "docker-compose.yml"
add F-05  "$SF" "$DONE" "Ethical AI governance pack"                  "Framework, AIA, compliance, trust statement, plans"   "docs/governance/"
add F-06  "$SF" "$TODO" "Deploy live API on Render"                   "Public /predict endpoint reachable"                   "render.yaml (move to Done once the URL responds)"

echo
echo "Done. Open: https://github.com/users/$OWNER/projects/$PNUM"
echo "Switch the view to Board (columns = Status) and take the Figure 3 screenshot."
echo "Tip: View options -> Group by 'Sprint' shows sprint completion per module."
