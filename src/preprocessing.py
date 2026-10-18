"""Cleaning, train/test split and the shared preprocessing pipeline."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (BINARY_FEATURES, CATEGORICAL_FEATURES, FEATURES,
                        NUMERIC_FEATURES, TEST_SIZE)
from src.validation import check_leakage


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicate rows. Missing values are imputed later, train-only."""
    out = df.drop_duplicates(keep="first").copy()
    out.index.name = "row_id"
    return out


def split(df: pd.DataFrame, seed: int):
    """Fixed split (random_state=seed), stratified on high_risk, shared by all tasks."""
    check_leakage(FEATURES)
    return train_test_split(df, test_size=TEST_SIZE, random_state=seed,
                            stratify=df["high_risk"])


def build_preprocessor() -> ColumnTransformer:
    """Median imputation + scaling for numerics, one-hot for season, binary passthrough."""
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler())])
    return ColumnTransformer([
        ("num", numeric, NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="error"), CATEGORICAL_FEATURES),
        ("bin", "passthrough", BINARY_FEATURES),
    ])
