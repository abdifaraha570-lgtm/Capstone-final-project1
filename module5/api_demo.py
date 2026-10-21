"""API demonstration for the Module 4 FastAPI service (Module 5 technical requirement).

    python module5/api_demo.py                      # calls http://127.0.0.1:8000 (start it with: cd module4 && python -m uvicorn api.main:app)
    python module5/api_demo.py --url http://host:port
    python module5/api_demo.py --inprocess          # no server needed: runs the app in-process with FastAPI's TestClient

Prints /health and three /predict calls with plain-language reason codes, and writes
module5/outputs/api_demo_output.json (used by build_slides.py for appendix slide A6).
Equivalent curl:  curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d @module4/api/sample_request.json
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

CASES = [
    ("Amina - 27, booked 21 days ahead, 1 of 3 earlier visits missed, welfare-enrolled",
     {"age": 27, "scheduled_date": "2026-10-01T09:30:00", "appointment_date": "2026-10-22T00:00:00", "scholarship": 1,
      "hypertension": 0, "diabetes": 0, "alcoholism": 0, "has_disability": 0, "prior_visits": 3, "prior_no_shows": 1, "gender": "F"}),
    ("Hassan - 64, hypertension + diabetes, booked 2 days ahead, 6 earlier visits all attended",
     {"age": 64, "scheduled_date": "2026-10-01T09:30:00", "appointment_date": "2026-10-03T00:00:00", "scholarship": 0,
      "hypertension": 1, "diabetes": 1, "alcoholism": 0, "has_disability": 0, "prior_visits": 6, "prior_no_shows": 0, "gender": "M"}),
    ("Sahra - 16, first visit, booked 35 days ahead",
     {"age": 16, "scheduled_date": "2026-10-01T09:30:00", "appointment_date": "2026-11-05T00:00:00", "scholarship": 0,
      "hypertension": 0, "diabetes": 0, "alcoholism": 0, "has_disability": 0, "prior_visits": 0, "prior_no_shows": 0, "gender": "F"}),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--inprocess", action="store_true")
    a = ap.parse_args()

    if a.inprocess:
        sys.path.insert(0, str(ROOT / "module4"))
        from fastapi.testclient import TestClient
        from api.main import app
        client = TestClient(app); get = client.get; post = client.post; base = ""
    else:
        import requests
        get = requests.get; post = requests.post; base = a.url.rstrip("/")

    out = {"base_url": a.url if not a.inprocess else "in-process TestClient", "health": None, "cases": []}
    h = get(base + "/health"); out["health"] = h.json()
    print("GET /health ->", json.dumps(out["health"], indent=2))
    for label, body in CASES:
        r = post(base + "/predict", json=body)
        resp = r.json()
        out["cases"].append({"label": label, "request": body, "status": r.status_code, "response": resp})
        print(f"\nPOST /predict  [{label}]  status {r.status_code}")
        print(f"  probability {resp.get('no_show_probability')}  tier {resp.get('risk_tier')}")
        print(f"  action: {resp.get('recommended_action')}")
        for rc in resp.get("reason_codes", []):
            print(f"  reason: {rc}")
    # a deliberately invalid request to show validation
    bad = dict(CASES[0][1]); bad["prior_no_shows"] = 9
    r = post(base + "/predict", json=bad)
    out["validation_example"] = {"request": bad, "status": r.status_code, "response": r.json()}
    print(f"\nPOST /predict with prior_no_shows > prior_visits -> status {r.status_code}: {r.json()}")
    (HERE / "outputs").mkdir(exist_ok=True)
    (HERE / "outputs" / "api_demo_output.json").write_text(json.dumps(out, indent=2))
    print("\nwrote module5/outputs/api_demo_output.json")


if __name__ == "__main__":
    main()
