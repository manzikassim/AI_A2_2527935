"""Generate the individual Musanze potato-cooperative dataset (synthetic).

Usage:  python data/generate_data.py --seed 7935
The seed must equal LLLL (last four digits of the registration number).
"""
import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
VERIFICATION_PREFIX = "AI-A2-"
REG_DIGITS = "2527935"


def generate_dataframe(seed: int) -> pd.DataFrame:
    """Create the raw dataset with deliberate missing values and duplicates."""
    rng = np.random.default_rng(seed)
    n = 240 + (seed % 41)

    # Draw order is fixed so the same seed always gives the same data.
    rainfall = rng.uniform(600, 1400, n)
    temp = rng.uniform(12, 26, n)
    soil = rng.uniform(20, 80, n)
    fert = rng.uniform(0, 250, n)
    size = rng.uniform(0.2, 4.0, n)
    season = rng.choice(["A", "B"], size=n)
    support = rng.integers(0, 2, size=n)
    noise = rng.normal(0, 0.80, n)

    season_effect = np.where(season == "A", 0.50, -0.20)
    yield_tons = (0.004 * rainfall + 0.06 * soil + 0.02 * fert + 1.30 * size
                  - 0.25 * np.abs(temp - 18) + 0.70 * support
                  + season_effect + noise)
    yield_tons = np.clip(yield_tons, 0.50, None)

    per_ha = yield_tons / size
    high_risk = (per_ha < np.median(per_ha)).astype(int)

    df = pd.DataFrame({
        "rainfall_mm": rainfall, "avg_temp_c": temp,
        "soil_moisture_pct": soil, "fertilizer_kg": fert,
        "field_size_ha": size, "season": season,
        "cooperative_support": support, "yield_tons": yield_tons,
        "high_risk": high_risk,
    })

    m = round(0.05 * n)
    soil_idx = rng.choice(n, size=m, replace=False)
    fert_idx = rng.choice(n, size=m, replace=False)
    df.loc[soil_idx, "soil_moisture_pct"] = np.nan
    df.loc[fert_idx, "fertilizer_kg"] = np.nan

    dup_idx = rng.choice(n, size=3, replace=False)
    return pd.concat([df, df.iloc[dup_idx]], ignore_index=True)


def write_outputs(seed: int, root: Path = ROOT, reg_digits: str = REG_DIGITS) -> dict:
    """Write the CSV and fingerprint file; return a summary."""
    df = generate_dataframe(seed)
    csv_path = root / "data" / f"AI_A2_{reg_digits}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False, float_format="%.4f", lineterminator="\n")
    digest = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    fp = root / "artifacts" / "data_fingerprint.txt"
    fp.parent.mkdir(parents=True, exist_ok=True)
    fp.write_text(f"verification_code: {VERIFICATION_PREFIX}{seed:04d}\n"
                  f"file: {csv_path.name}\nrows: {len(df)}\n"
                  f"sha256: {digest}\n")
    return {"rows": len(df), "sha256": digest, "path": csv_path,
            "missing": df.isna().sum().to_dict(),
            "duplicates": int(df.duplicated().sum())}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    args = ap.parse_args()
    info = write_outputs(args.seed)
    print(f"Verification code: {VERIFICATION_PREFIX}{args.seed:04d}")
    for k, v in info.items():
        print(f"{k}: {v}")