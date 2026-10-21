"""FastAPI service for the JJUSHYCSH referral no-show risk model (BAN6800 Module 4).
Run from the module4 folder:   uvicorn api.main:app --reload
Then open http://127.0.0.1:8000/docs and try POST /predict.

Module 2, Section 6: OPD/referral users receive short plain-language reason codes with each score,
and the score may trigger additional support, never denial, cancellation or lower clinical priority."""
import sys
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from m4 import config as C                                    # noqa: E402
from m4.data import features_from_request                     # noqa: E402  (Module 3 feature definitions)
from m4.xai_fairness import pretty                            # noqa: E402

BUNDLE = joblib.load(ROOT / "outputs" / "model" / "noshow_model.joblib")
PIPE, T_HIGH, T_MED, FEATURES = (BUNDLE["pipeline"], BUNDLE["threshold_high"],
                                 BUNDLE["threshold_medium"], BUNDLE["features"])

# Optional SHAP explainer for reason codes (the API still works without shap installed).
_EXPLAINER = None
try:
    import shap
    _clf = PIPE.named_steps["clf"]
    _EXPLAINER = shap.TreeExplainer(_clf) if _clf.__class__.__name__ != "LogisticRegression" else None
except Exception:
    _EXPLAINER = None

app = FastAPI(title="JJUSHYCSH Referral No-Show Risk API", version=str(BUNDLE.get("created", "1")),
              description="Advisory decision support only (BAN6800 capstone, " + C.AUTHOR + "). "
                          "Scores must never be used to cancel, overbook against, or deprioritise a patient.")


class Appointment(BaseModel):
    age: float = Field(..., ge=0, le=110, examples=[34])
    scheduled_date: datetime = Field(..., description="When the booking was made", examples=["2026-10-01T09:30:00"])
    appointment_date: datetime = Field(..., examples=["2026-10-20T00:00:00"])
    scholarship: int = Field(0, ge=0, le=1, description="Welfare programme enrolment (1 = yes)")
    hypertension: int = Field(0, ge=0, le=1)
    diabetes: int = Field(0, ge=0, le=1)
    alcoholism: int = Field(0, ge=0, le=1)
    has_disability: int = Field(0, ge=0, le=1)
    prior_visits: int = Field(0, ge=0, description="Earlier appointments on record for this patient")
    prior_no_shows: int = Field(0, ge=0, description="How many of those were missed")
    gender: Optional[Literal["F", "M"]] = Field(None, description="Not used by the model; accepted for fairness-monitoring logs only")


class Prediction(BaseModel):
    no_show_probability: float
    risk_tier: Literal["Low", "Medium", "High"]
    recommended_action: str
    reason_codes: list[str]
    thresholds: dict
    model: str
    disclaimer: str = "Advisory only: a flag adds outreach support and never reduces care."


def _reason_codes(X: pd.DataFrame, p: float) -> list[str]:
    """Plain-language reasons: SHAP contributions when available, else the strongest known drivers."""
    prep = PIPE.named_steps["prep"]
    try:
        Xt = pd.DataFrame(prep.transform(X), columns=prep.get_feature_names_out())
        if _EXPLAINER is not None:
            sv = np.asarray(_EXPLAINER(Xt).values, dtype=float)
            sv = sv[:, :, 1] if sv.ndim == 3 else sv
            contrib = pd.Series(sv[0], index=[pretty(f) for f in Xt.columns])
        else:  # logistic regression: coefficient x standardised value
            clf = PIPE.named_steps["clf"]
            contrib = pd.Series(clf.coef_[0] * Xt.values[0], index=[pretty(f) for f in Xt.columns])
        top = contrib.reindex(contrib.abs().sort_values(ascending=False).index)[:3]
        return [f"{name}: {'raises' if v > 0 else 'lowers'} risk" for name, v in top.items() if abs(v) > 1e-6]
    except Exception:
        r = X.iloc[0]
        codes = []
        if r["lead_days"] >= 7: codes.append("Long lead time between booking and visit: raises risk")
        if r["prior_no_show_rate"] > 0: codes.append("Missed appointments in the past: raises risk")
        if r["same_day"] == 1: codes.append("Same-day booking: lowers risk")
        return codes or ["No dominant factor identified"]


@app.get("/health")
def health():
    return {"status": "ok", "algorithm": BUNDLE["algorithm"], "created": BUNDLE.get("created"),
            "features": FEATURES, "data_md5": BUNDLE.get("data_md5")}


@app.post("/predict", response_model=Prediction)
def predict(a: Appointment):
    try:
        X = features_from_request(a.model_dump() if hasattr(a, "model_dump") else a.dict(),
                                  dropped=BUNDLE.get("dropped_flags") or ())
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    X = X[FEATURES]
    p = float(PIPE.predict_proba(X)[:, 1][0])
    tier = "High" if p >= T_HIGH else "Medium" if p >= T_MED else "Low"
    return Prediction(no_show_probability=round(p, 4), risk_tier=tier, recommended_action=C.ACTIONS[tier],
                      reason_codes=_reason_codes(X, p),
                      thresholds={"high": round(T_HIGH, 4), "medium": round(T_MED, 4)}, model=BUNDLE["algorithm"])
