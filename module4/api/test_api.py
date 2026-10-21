"""Quick API test:  python -m api.test_api   (run from the module4 folder, after run_all.py)"""
import json
from pathlib import Path
from fastapi.testclient import TestClient
from api.main import app

c = TestClient(app)
print(c.get("/health").json())
body = json.loads((Path(__file__).parent / "sample_request.json").read_text())
r = c.post("/predict", json=body)
print(r.status_code, json.dumps(r.json(), indent=2))
