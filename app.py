"""FastAPI service. Start: uvicorn app:app --host 127.0.0.1 --port 8000"""
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from src.inference import load_model_info, predict_record
from src.validation import ValidationError

app = FastAPI(title="Musanze Harvest Intelligence Service", version="1.0.0",
              description="Classroom decision-support prototype on synthetic data.")


class FarmRecord(BaseModel):
    """One farm record. Ranges match the training data."""
    model_config = ConfigDict(extra="forbid")

    rainfall_mm: float = Field(..., ge=600, le=1400)
    avg_temp_c: float = Field(..., ge=12, le=26)
    soil_moisture_pct: float = Field(..., ge=20, le=80)
    fertilizer_kg: float = Field(..., ge=0, le=250)
    field_size_ha: float = Field(..., ge=0.2, le=4.0)
    season: Literal["A", "B"]
    cooperative_support: Literal[0, 1]


def _models_or_503():
    try:
        return load_model_info()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@app.get("/health")
def health():
    """Liveness check; reports whether saved models are available."""
    try:
        info = load_model_info()
        return {"status": "ok", "models_loaded": True,
                "verification_code": info["verification_code"]}
    except FileNotFoundError:
        return {"status": "degraded", "models_loaded": False,
                "detail": "Run run_all.py to create models."}


@app.get("/model-info")
def model_info():
    """Model metadata, metrics and verification code."""
    return _models_or_503()


@app.post("/predict")
def predict(record: FarmRecord):
    """Return yield estimate, risk class and probability, cluster and verification code."""
    _models_or_503()
    try:
        return predict_record(record.model_dump())
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
