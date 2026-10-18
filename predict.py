"""Predict from one JSON farm record using saved artifacts.

Usage: python predict.py --json samples/valid.json
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.inference import predict_record  # noqa: E402
from src.validation import ValidationError  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", required=True, help="Path to a JSON file with one farm record")
    args = ap.parse_args()
    try:
        record = json.loads(Path(args.json).read_text())
    except FileNotFoundError:
        print(f"ERROR: file not found: {args.json}", file=sys.stderr)
        return 2
    except json.JSONDecodeError as exc:
        print(f"ERROR: invalid JSON: {exc}", file=sys.stderr)
        return 2
    try:
        print(json.dumps(predict_record(record), indent=2))
    except (ValidationError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
