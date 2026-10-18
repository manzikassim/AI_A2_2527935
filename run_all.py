"""One-command pipeline: data -> validation -> cleaning -> 3 ML tasks -> artifacts.

Usage: python run_all.py --student-id ug2527935@ines.ac.rw --seed 7935
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from data.generate_data import write_outputs  # noqa: E402
from src import config  # noqa: E402
from src.classification import train_classification  # noqa: E402
from src.clustering import train_clustering  # noqa: E402
from src.plots import plot_clusters, plot_confusion, plot_residuals  # noqa: E402
from src.preprocessing import build_preprocessor, clean, split  # noqa: E402
from src.regression import train_regression  # noqa: E402
from src.validation import (ValidationError, seed_from_student_id,  # noqa: E402
                            validate_dataset, validate_seed)


def _dump(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))


def run(student_id: str, seed: int) -> dict:
    """Run the whole system and return a short summary."""
    code = validate_seed(student_id, seed)
    reg_digits = "".join(c for c in student_id.split("@")[0] if c.isdigit())
    print(f"Verification code: {code}\nStudent ID: {student_id}   Seed: {seed}")

    info = write_outputs(seed, ROOT, reg_digits)
    print(f"Dataset: {info['path'].name}  rows={info['rows']}  SHA-256={info['sha256']}")

    raw = pd.read_csv(info["path"])
    report = validate_dataset(raw)
    report["verification_code"] = code
    _dump(report, config.ARTIFACTS_DIR / "validation_report.json")
    print(f"Validation: missing={report['missing_values']} duplicates={report['duplicate_rows']}")

    df = clean(raw)
    train, test = split(df, seed)
    print(f"After cleaning: {len(df)} rows -> train {len(train)} / test {len(test)}")

    reg_report, reg_model, reg_pred = train_regression(train, test, seed)
    clf_report, clf_model, clf_pred, clf_proba = train_classification(train, test, seed)
    clu_report, clu_model, test_clusters, _ = train_clustering(train, test, seed)
    for r in (reg_report, clf_report, clu_report):
        r["verification_code"] = code
    sel_m = clf_report["metrics"][clf_report["selected_model"]]["test"]
    reg_report["dataset_sha256"] = info["sha256"]

    _dump(reg_report, config.ARTIFACTS_DIR / "regression_metrics.json")
    _dump(clf_report, config.ARTIFACTS_DIR / "classification_metrics.json")
    _dump(clu_report, config.ARTIFACTS_DIR / "clustering_report.json")

    out = pd.DataFrame({
        "row_id": test.index,
        "actual_yield_tons": test["yield_tons"].round(4).values,
        "predicted_yield_tons": pd.Series(reg_pred).round(4).values,
        "yield_error": (pd.Series(reg_pred) - test["yield_tons"].values).round(4).values,
        "abs_yield_error": (pd.Series(reg_pred) - test["yield_tons"].values).abs().round(4).values,
        "actual_high_risk": test["high_risk"].values,
        "predicted_high_risk": clf_pred,
        "risk_probability": pd.Series(clf_proba).round(4).values,
        "cluster": test_clusters,
    })
    out.to_csv(config.ARTIFACTS_DIR / "test_predictions.csv", index=False)

    plot_residuals(test["yield_tons"].values, reg_pred, code)
    plot_confusion(sel_m["confusion_matrix"]["matrix"], code, clf_report["selected_model"])
    plot_clusters(train, test, clu_model, code)

    config.MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(reg_model, config.MODELS_DIR / "regression_model.joblib")
    joblib.dump(clf_model, config.MODELS_DIR / "classification_model.joblib")
    joblib.dump(clu_model, config.MODELS_DIR / "cluster_model.joblib")
    joblib.dump(build_preprocessor().fit(train[config.FEATURES]),
                config.MODELS_DIR / "preprocessor.joblib")
    _dump({
        "service": "Musanze Harvest Intelligence Service",
        "student_id": student_id, "seed": seed, "verification_code": code,
        "dataset_sha256": info["sha256"], "dataset_rows": info["rows"],
        "rows_after_cleaning": int(len(df)), "train_rows": int(len(train)), "test_rows": int(len(test)),
        "random_state": seed, "split": f"test_size={config.TEST_SIZE}, stratified on high_risk",
        "regression_model": reg_report["selected_model"],
        "classification_model": clf_report["selected_model"],
        "regression_test_metrics": reg_report["metrics"][reg_report["selected_model"]]["test"],
        "classification_test_metrics": {k: sel_m[k] for k in ("precision", "recall", "f1", "roc_auc")},
        "clustering": {"selected_k": clu_report["selected_k"], "profiles": clu_report["profiles"]},
        "synthetic_data": True,
        "intended_use": "Classroom decision-support prototype; human review required.",
    }, config.MODELS_DIR / "model_info.json")

    from src.inference import load_models
    load_models.cache_clear()

    print(f"Regression : {reg_report['selected_model']}  {reg_report['metrics'][reg_report['selected_model']]['test']}")
    print(f"Classifier : {clf_report['selected_model']}  F1={sel_m['f1']} AUC={sel_m['roc_auc']}")
    print(f"Clustering : K={clu_report['selected_k']}  silhouettes={clu_report['silhouette_by_k']}")
    print("Artifacts written:")
    for p in sorted(config.ARTIFACTS_DIR.rglob("*")):
        if p.is_file():
            print("  ", p.relative_to(ROOT))
    for p in sorted(config.MODELS_DIR.glob("*")):
        print("  ", p.relative_to(ROOT))
    print(f"Done. {code}")
    return {"verification_code": code, "sha256": info["sha256"]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--student-id", required=True)
    ap.add_argument("--seed", type=int, required=True)
    args = ap.parse_args()
    try:
        run(args.student_id, args.seed)
    except ValidationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
