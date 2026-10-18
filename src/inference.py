"""Load saved artifacts, validate one farm record and predict."""
import json
import math
from functools import lru_cache

import joblib
import pandas as pd

from src.config import FEATURES, MODELS_DIR, RANGES, SEASONS
from src.validation import ValidationError

NUMERIC_FIELDS = ["rainfall_mm", "avg_temp_c", "soil_moisture_pct",
                  "fertilizer_kg", "field_size_ha"]
DISCLAIMER = ("Classroom decision-support prototype on synthetic data. Not a replacement "
              "for an agronomist. Human review required; not for finance, insurance "
              "or public-service decisions.")


@lru_cache(maxsize=1)
def load_models():
    """Load the three saved pipelines and model_info.json."""
    needed = ["regression_model.joblib", "classification_model.joblib",
              "cluster_model.joblib", "model_info.json"]
    missing = [n for n in needed if not (MODELS_DIR / n).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing saved artifacts {missing}. Run: python run_all.py --student-id <id> --seed <LLLL>")
    info = json.loads((MODELS_DIR / "model_info.json").read_text())
    return (joblib.load(MODELS_DIR / "regression_model.joblib"),
            joblib.load(MODELS_DIR / "classification_model.joblib"),
            joblib.load(MODELS_DIR / "cluster_model.joblib"), info)


def load_model_info() -> dict:
    return load_models()[3]


def validate_record(record) -> dict:
    """Check fields, types and ranges. Raises ValidationError with a helpful message."""
    if not isinstance(record, dict):
        raise ValidationError("Input must be a JSON object with the farm fields.")
    missing = [f for f in FEATURES if f not in record]
    if missing:
        raise ValidationError(f"Missing required field(s): {missing}")
    extra = [k for k in record if k not in FEATURES]
    if extra:
        raise ValidationError(f"Unexpected field(s): {extra}")
    clean = {}
    for f in NUMERIC_FIELDS:
        v = record[f]
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValidationError(f"Field '{f}' must be a number, got {type(v).__name__}.")
        if not math.isfinite(v):
            raise ValidationError(f"Field '{f}' must be finite.")
        lo, hi = RANGES[f]
        if not lo <= v <= hi:
            raise ValidationError(f"Field '{f}'={v} is outside the allowed range {lo} to {hi}.")
        clean[f] = float(v)
    if record["season"] not in SEASONS:
        raise ValidationError(f"Field 'season' must be one of {SEASONS}.")
    clean["season"] = record["season"]
    cs = record["cooperative_support"]
    if isinstance(cs, bool) or not isinstance(cs, int) or cs not in (0, 1):
        raise ValidationError("Field 'cooperative_support' must be the integer 0 or 1.")
    clean["cooperative_support"] = cs
    return clean


def predict_record(record) -> dict:
    """Validate and predict yield, risk, probability and cluster."""
    clean = validate_record(record)
    reg, clf, clu, info = load_models()
    row = pd.DataFrame([clean])[FEATURES]
    proba = float(clf.predict_proba(row)[0, 1])
    cluster = int(clu.predict(row)[0])
    return {
        "yield_estimate_tons": round(float(reg.predict(row)[0]), 4),
        "high_risk": int(clf.predict(row)[0]),
        "risk_probability": round(proba, 4),
        "cluster": cluster,
        "cluster_profile": info["clustering"]["profiles"][str(cluster)]["label"],
        "verification_code": info["verification_code"],
        "disclaimer": DISCLAIMER,
    }
