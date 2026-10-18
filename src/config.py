"""Shared constants for the Musanze Harvest Intelligence Service."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
ARTIFACTS_DIR = ROOT / "artifacts"
PLOTS_DIR = ARTIFACTS_DIR / "plots"

NUMERIC_FEATURES = ["rainfall_mm", "avg_temp_c", "soil_moisture_pct",
                    "fertilizer_kg", "field_size_ha"]
CATEGORICAL_FEATURES = ["season"]
BINARY_FEATURES = ["cooperative_support"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES
TARGETS = ["yield_tons", "high_risk"]
LEAKAGE_COLUMNS = ["yield_tons", "yield_per_ha", "high_risk"]
REQUIRED_COLUMNS = FEATURES + TARGETS

# Inclusive valid ranges (generator ranges). Used for data and API checks.
RANGES = {
    "rainfall_mm": (600, 1400),
    "avg_temp_c": (12, 26),
    "soil_moisture_pct": (20, 80),
    "fertilizer_kg": (0, 250),
    "field_size_ha": (0.2, 4.0),
    "cooperative_support": (0, 1),
}
SEASONS = ["A", "B"]
TEST_SIZE = 0.2
