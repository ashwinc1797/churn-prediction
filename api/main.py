"""FastAPI model service for the churn pipeline.

Run from the project root:
    export CHURN_API_KEY="change-me"            # Windows: set CHURN_API_KEY=change-me
    uvicorn api.main:app --host 0.0.0.0 --port 8000
Docs (interactive): http://localhost:8000/docs
"""
import os
import sys
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))                    # so `schema` and `utils` import
from schema import FEATURES, POSITIVE_CLASS  # noqa: E402
from utils.model_io import load_trusted_model  # noqa: E402

model, meta, _ = load_trusted_model(ROOT / "models" / "churn_pipeline.joblib",
                                    ROOT / "models" / "churn_metadata.json")
POS_IDX = meta["classes"].index(POSITIVE_CLASS)
app = FastAPI(title="Churn model API", version=meta["sha256"][:8])


class Customer(BaseModel):
    """Pydantic validates types and ranges before the model ever sees the data."""
    tenure_months: int = Field(ge=0, le=72)
    monthly_charges: float = Field(ge=18, le=120)
    total_charges: float = Field(ge=0, le=9000)
    support_calls: int = Field(ge=0, le=10)
    contract: Literal["Month-to-month", "One year", "Two year"]
    internet_service: Literal["Fiber optic", "DSL", "No"]
    payment_method: Literal["Electronic check", "Mailed check", "Bank transfer",
                            "Credit card"]
    paperless_billing: Literal["Yes", "No"]
    senior_citizen: Literal["Yes", "No"]


class Prediction(BaseModel):
    churn_probability: float
    label: str
    model_version: str


def require_api_key(x_api_key: str = Header(default="")) -> None:
    expected = os.environ.get("CHURN_API_KEY")
    if not expected or x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": meta["model_type"]}


@app.post("/predict", response_model=list[Prediction],
          dependencies=[Depends(require_api_key)])
def predict(customers: list[Customer]) -> list[Prediction]:
    if not 1 <= len(customers) <= 1000:
        raise HTTPException(status_code=422, detail="Send between 1 and 1000 records")
    X = pd.DataFrame([c.model_dump() for c in customers])[FEATURES]
    probs = model.predict_proba(X)[:, POS_IDX]
    return [Prediction(churn_probability=round(float(p), 4),
                       label="churn" if p >= 0.5 else "stay",
                       model_version=app.version) for p in probs]
