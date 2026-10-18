"""Classification task: majority baseline vs LogisticRegression vs RandomForest."""
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from src.config import FEATURES
from src.preprocessing import build_preprocessor


def _pipe(model) -> Pipeline:
    return Pipeline([("prep", build_preprocessor()), ("model", model)])


def classification_scores(y_true, y_pred, proba) -> dict:
    """Precision, recall, F1, confusion matrix and ROC-AUC (None if one class only)."""
    auc = (round(float(roc_auc_score(y_true, proba)), 4)
           if len(set(y_true)) == 2 else None)
    return {
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": auc,
        "confusion_matrix": {"labels": [0, 1],
                             "matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()},
    }


def train_classification(train: pd.DataFrame, test: pd.DataFrame, seed: int):
    """Fit candidates, select by 5-fold CV F1 on TRAIN only, report TEST metrics."""
    candidates = {
        "majority_baseline": _pipe(DummyClassifier(strategy="most_frequent")),
        "logistic_regression": _pipe(LogisticRegression(max_iter=1000, random_state=seed)),
        "random_forest": _pipe(RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2, random_state=seed, n_jobs=1)),
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    Xtr, ytr = train[FEATURES], train["high_risk"]
    Xte, yte = test[FEATURES], test["high_risk"]

    results, cv_f1 = {}, {}
    for name, pipe in candidates.items():
        cv_f1[name] = round(float(cross_val_score(pipe, Xtr, ytr, cv=cv, scoring="f1").mean()), 4)
        pipe.fit(Xtr, ytr)
        pred = pipe.predict(Xte)
        proba = pipe.predict_proba(Xte)[:, 1]
        results[name] = {"test": classification_scores(yte, pred, proba),
                         "cv_f1_train": cv_f1[name]}

    real = {k: v for k, v in cv_f1.items() if k != "majority_baseline"}
    best = max(real, key=real.get)
    pred = candidates[best].predict(Xte)
    proba = candidates[best].predict_proba(Xte)[:, 1]
    wrong = test.loc[pred != yte.values, FEATURES].copy()
    wrong["actual"] = yte[pred != yte.values]
    wrong["predicted"] = pred[pred != yte.values]
    wrong["risk_probability"] = proba[pred != yte.values]
    wrong = wrong.assign(yield_per_ha=(test["yield_tons"] / test["field_size_ha"])[wrong.index])
    report = {
        "random_state": seed,
        "metrics": results,
        "selected_model": best,
        "selection_reason": (
            f"{best} had the highest 5-fold cross-validated F1 on the training set "
            f"({real[best]}); the test set was used only for the final report."),
        "misclassified_test_rows": [
            {"row_id": int(i), **{k: (v if isinstance(v, str) else round(float(v), 4))
                                  for k, v in r.items()}}
            for i, r in wrong.iterrows()],
        "note": "high_risk means yield per hectare below the median; classes are balanced by construction.",
    }
    return report, candidates[best], pred, proba
