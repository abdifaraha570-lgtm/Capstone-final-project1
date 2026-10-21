# Module 5 — Business Insights Presentation & Live Dashboard (Sprint 3)
JJUSHYCSH referral no-show risk · **Abdi Farah Ahmed** · BAN6800, Nexford University

Everything in this folder is generated from the Module 4 run — no number is typed by hand.

```
module5/
├── export_scored_test.py   step 1: copies results/figures/model from module4/outputs and writes data/scored_test.csv (no identifiers)
├── app.py                  the Streamlit dashboard (7 pages incl. the Ethical Compliance view with Fairlearn)
├── build_slides.py         step 2: the 15-slide deck + 9 Q&A appendix slides (.pptx) with speaker notes
├── api_demo.py             API demonstration script (prints and saves /health + 3 /predict calls)
├── requirements.txt        dashboard dependencies, pinned to the training laptop
├── HF_SPACE_README.md      front-matter README for the Hugging Face Space (rename to README.md in the Space)
├── .streamlit/config.toml  theme
├── data/                   written by step 1 (results.json, figures, model, scored_test.csv, reports)
├── assets/                 charts drawn by step 2
└── outputs/                the .pptx and api_demo_output.json
```

## Run (from the repository root, after `module4/run_all.py` has completed)
```bash
pip install -r module5/requirements.txt python-pptx
python module5/export_scored_test.py                               # 1. data for the dashboard
streamlit run module5/app.py                                       # 2. dashboard locally -> http://localhost:8501
cd module4 && python -m uvicorn api.main:app &  cd ..              # 3. API (or use --inprocess below)
python module5/api_demo.py            # or: python module5/api_demo.py --inprocess
python module5/build_slides.py --dashboard-url <live URL> --github-url <repo URL>   # 4. the deck
```

## Deploy the dashboard (graded as a live artefact)
### Option A — Hugging Face Spaces (no git needed, ~10 min)
1. huggingface.co → Sign up/in → **New Space** → name `jjushycsh-noshow`, SDK **Streamlit**, hardware CPU basic (free), Public.
2. In the Space → **Files** → *Add file → Upload files*: upload `app.py`, `requirements.txt`, the whole `data/` folder
   (drag the folder) and `.streamlit/config.toml`. Then upload `HF_SPACE_README.md` **renamed to `README.md`**
   (it sets `python_version: "3.12"` and `app_file: app.py`).
3. Commit. The Space builds in 2–4 minutes; the URL is `https://huggingface.co/spaces/<your-username>/jjushycsh-noshow`.
4. Open it, click through all seven pages, and paste the URL into `build_slides.py --dashboard-url` and the submission.

### Option B — Streamlit Community Cloud
Push the repository to GitHub, go to share.streamlit.io → *New app* → repository, branch `main`,
**Main file path `module5/app.py`**, advanced settings → Python 3.12. If the build fails on the root
`requirements.txt` (it lists Airflow etc.), copy `module5/requirements.txt` over the root one in a `dashboard` branch
and deploy that branch.

### If the model file does not load on the host
The what-if page then falls back to the API (set `NOSHOW_API_URL` as a Space secret if you host the API) or to the
nearest scored appointments — the page says which source it used. All other pages read `results.json` only.

## What is where in the deck (Module 5 brief → slide)
| Brief section | Slides |
|---|---|
| Title & objective; business problem; audience & decision needs; ethics stakeholders | 1, 2, 3 |
| Model & key insights, plain-language explanation | 4, 5 |
| Visual summary of results | 6 |
| What drives predictions (SHAP in business language) | 7 |
| 2–3 example explanations; what-if analysis | 8, 9 |
| Fairness summary; bias mitigation | 10, 11 |
| Model limitations; transparency & ethics | 12, 13 |
| Business implications; next steps (no recommendation) | 14, 15 |
| Q&A backup deck | A1–A7 + References |

## Privacy note
`data/scored_test.csv` contains proxy (Kaggle) appointments with predicted probabilities and the four audited
attributes; it has no keys, dates or identifiers. Do not publish a local JJUSHYCSH equivalent without DPO approval.
