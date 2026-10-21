# AI Disclosure - BAN6800 Module 3
Student: Abdi Farah Ahmed | Nexford University | September 2026
(Items marked [CONFIRM] must be checked and completed by me before submission.)

## 1. AI tool used
Claude (Anthropic), used through claude.ai, 23-24 September 2026.

## 2. What I used it for
- Drafting the pipeline code from my Module 1 vision and Module 2 plan: ingestion, cleaning,
  anonymization, feature engineering, integration/split, Great Expectations suites, bias check,
  audit logging, access-control module, Airflow DAG, Dockerfile and pytest tests.
- Drafting the 12-slide deck and speaker notes in the style of my earlier reports.
- Troubleshooting my Windows setup, and creating a Colab notebook to run everything.

## 3. Prompts and iteration
1. I uploaded my Module 1 and Module 2 documents and the Module 3 brief, and asked for the Module 3
   deliverables continuing the same project and language.
2. The first output was tested only on a synthetic sample with the Kaggle schema, because the AI's
   environment had no internet. I asked for the real Kaggle file; the AI explained it could not
   provide it and I downloaded it myself.
3. On my laptop, pip.exe was blocked by an Application Control policy. Using `python -m pip` got
   past that, but pandas 2.2.3 has no installer for my Python 3.14 and building it was also blocked.
   I asked the AI to finish everything; it relaxed the version pins and produced a Colab notebook
   that runs the tests and pipeline, captures the Great Expectations screenshots and inserts the
   real results into the deck.

## 4. Critical evaluation of the AI output
Strengths:
- The design follows my Module 2 architecture and governance controls closely.
- Good technical decisions I checked and agree with: patient-grouped split (prevents leakage),
  history features using only earlier visits, file paths instead of XCom in Airflow, and a
  processed-data column check that works as a privacy gate.

Weaknesses and errors found:
- The first duplicate rule removed every row with the same patient and the same booking second
  (1,334 rows). When I ran it on the real data I found 29 of those groups had different outcomes,
  so they were separate visits, not double entries. The rule was changed to remove only rows
  identical except the appointment ID (618 rows) and to keep and flag the other 1,432.
- The first draft said PatientId is stored in scientific notation; in the real file it is not,
  so the slide was corrected.
- The first requirements file pinned versions that do not install on Python 3.14.
- Great Expectations and Airflow code could not be executed in the AI's environment; the
  first deck had empty screenshot boxes and an estimated figure ("about 2%") until the real run showed 2.0%.
- The audit-log example on slide 11 shows the format, not values from a run.
- The Airflow DAG was not executed in a real Airflow instance [CONFIRM if you ran it].
- [CONFIRM: any other error you saw in Colab and how it was fixed]

What I changed or decided myself:
- [CONFIRM: e.g., edited speaker notes to add my DHIS2 field experience in the Somali Region]
- [CONFIRM: any threshold you changed - k=20, 5% share, 10-point gap, retention periods]

## 5. How I verified it
- Ran the pipeline on the real 110,527-row dataset: 109,899 model-ready rows, no-show rate 20.1%,
  disability group flagged at 2.0% of records. Ran `pytest` (12 tests) and the Great Expectations
  suites in Google Colab [CONFIRM: result].
- Compared slide numbers with reports/data_quality_report.json and reports/bias_representation.csv.
- Checked the Great Expectations Data Docs pages produced by my run.
- Verification sources: Great Expectations documentation (docs.greatexpectations.io); Apache
  Airflow TaskFlow documentation (airflow.apache.org); Fairlearn user guide (fairlearn.org);
  Personal Data Protection Proclamation No. 1321/2024; Hoppen (2016) Kaggle dataset page.

## 6. Reflection
[CONFIRM - write 3-4 sentences in your own words, e.g. what you learned about validating data
before modeling, and how this pipeline would work with real JJUSHYCSH/DHIS2 data.]
