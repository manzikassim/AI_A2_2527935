import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

STUDENT_ID = "ug2527935@ines.ac.rw"
SEED = 7935


@pytest.fixture(scope="session", autouse=True)
def trained_artifacts():
    """Make sure saved models exist; run the pipeline once if they do not."""
    if not (ROOT / "models" / "model_info.json").exists():
        subprocess.run([sys.executable, str(ROOT / "run_all.py"),
                        "--student-id", STUDENT_ID, "--seed", str(SEED)],
                       check=True, cwd=ROOT)


@pytest.fixture
def valid_record():
    return {"rainfall_mm": 1050.0, "avg_temp_c": 18.5, "soil_moisture_pct": 55.0,
            "fertilizer_kg": 150.0, "field_size_ha": 1.8, "season": "A",
            "cooperative_support": 1}
