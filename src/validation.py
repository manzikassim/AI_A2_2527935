"""Identity, schema, range, duplicate and leakage checks."""
import re

import pandas as pd

from src.config import FEATURES, LEAKAGE_COLUMNS, RANGES, REQUIRED_COLUMNS, SEASONS


class ValidationError(ValueError):
    """Raised when identity or data checks fail."""


def seed_from_student_id(student_id: str) -> int:
    """Return LLLL (last four digits) from an ID such as ug2527935@ines.ac.rw."""
    digits = re.sub(r"\D", "", student_id.split("@")[0])
    if len(digits) < 4:
        raise ValidationError(f"Student ID '{student_id}' has fewer than 4 digits.")
    return int(digits[-4:])


def validate_seed(student_id: str, seed: int) -> str:
    """Reject a seed that is not LLLL; return the verification code."""
    expected = seed_from_student_id(student_id)
    if seed != expected:
        raise ValidationError(
            f"Seed {seed} does not match the last four digits ({expected}) of '{student_id}'.")
    return f"AI-A2-{expected:04d}"


def validate_schema(df: pd.DataFrame) -> None:
    """Check required columns exist."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValidationError(f"Missing required columns: {missing}")


def check_ranges(df: pd.DataFrame) -> dict:
    """Return count of out-of-range values per column (NaN is not a range error)."""
    bad = {}
    for col, (lo, hi) in RANGES.items():
        s = df[col].dropna()
        bad[col] = int(((s < lo) | (s > hi)).sum())
    bad["season"] = int((~df["season"].isin(SEASONS)).sum())
    bad["yield_tons"] = int((df["yield_tons"] < 0.5).sum())
    return bad


def check_leakage(feature_columns: list) -> None:
    """Fail if any target-derived column is used as an input feature."""
    leaked = [c for c in feature_columns if c in LEAKAGE_COLUMNS]
    if leaked:
        raise ValidationError(f"Target leakage: {leaked} cannot be input features.")


def validate_dataset(df: pd.DataFrame) -> dict:
    """Run all checks; raise on hard failures; return an evidence report."""
    validate_schema(df)
    check_leakage(FEATURES)
    ranges = check_ranges(df)
    if any(ranges.values()):
        raise ValidationError(f"Out-of-range values found: { {k: v for k, v in ranges.items() if v} }")
    return {
        "rows": int(len(df)),
        "missing_values": {k: int(v) for k, v in df.isna().sum().items() if v},
        "duplicate_rows": int(df.duplicated().sum()),
        "out_of_range": ranges,
        "leakage_check": "passed",
    }
