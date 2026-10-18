import hashlib

import pandas as pd
import pytest

from data.generate_data import generate_dataframe
from src.config import FEATURES
from src.inference import predict_record, validate_record
from src.preprocessing import clean, split
from src.regression import train_regression
from src.validation import (ValidationError, check_leakage, seed_from_student_id,
                            validate_seed)

SEED = 7935


def test_valid_inference(valid_record):
    out = predict_record(valid_record)
    assert out["verification_code"] == "AI-A2-7935"
    assert out["yield_estimate_tons"] > 0
    assert out["high_risk"] in (0, 1)
    assert 0.0 <= out["risk_probability"] <= 1.0
    assert isinstance(out["cluster"], int)


def test_missing_field_rejected(valid_record):
    del valid_record["rainfall_mm"]
    with pytest.raises(ValidationError, match="Missing required"):
        predict_record(valid_record)


def test_wrong_type_rejected(valid_record):
    valid_record["fertilizer_kg"] = "lots"
    with pytest.raises(ValidationError, match="must be a number"):
        predict_record(valid_record)


def test_range_error_rejected(valid_record):
    valid_record["rainfall_mm"] = 5000
    with pytest.raises(ValidationError, match="outside the allowed range"):
        predict_record(valid_record)


def test_bad_category_rejected(valid_record):
    valid_record["season"] = "C"
    with pytest.raises(ValidationError, match="season"):
        validate_record(valid_record)


def test_dataset_reproducible():
    a = generate_dataframe(SEED).to_csv(index=False, float_format="%.4f").encode()
    b = generate_dataframe(SEED).to_csv(index=False, float_format="%.4f").encode()
    assert hashlib.sha256(a).hexdigest() == hashlib.sha256(b).hexdigest()
    df = generate_dataframe(SEED)
    assert len(df) == 240 + (SEED % 41) + 3
    assert df["soil_moisture_pct"].isna().sum() == 13
    assert df["fertilizer_kg"].isna().sum() == 13
    assert df.duplicated().sum() == 3


def test_metrics_reproducible():
    train, test = split(clean(generate_dataframe(SEED)), SEED)
    r1 = train_regression(train, test, SEED)[0]["metrics"]
    r2 = train_regression(train, test, SEED)[0]["metrics"]
    assert r1 == r2


def test_wrong_seed_rejected():
    assert seed_from_student_id("ug2527935@ines.ac.rw") == 7935
    assert validate_seed("ug2527935@ines.ac.rw", 7935) == "AI-A2-7935"
    with pytest.raises(ValidationError):
        validate_seed("ug2527935@ines.ac.rw", 1234)


def test_leakage_guard():
    check_leakage(FEATURES)
    with pytest.raises(ValidationError, match="leakage"):
        check_leakage(FEATURES + ["high_risk"])


def test_split_has_no_overlap():
    train, test = split(clean(generate_dataframe(SEED)), SEED)
    assert set(train.index).isdisjoint(test.index)
    assert len(train) + len(test) == 262
