"""Regression task: mean baseline vs LinearRegression vs RandomForest."""
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline

from src.config import FEATURES
from src.preprocessing import build_preprocessor


def _pipe(model) -> Pipeline:
    return Pipeline([("prep", build_preprocessor()), ("model", model)])


def regression_scores(y_true, y_pred) -> dict:
    """MAE, RMSE and R2 rounded to 4 decimals."""
    return {"MAE": round(float(mean_absolute_error(y_true, y_pred)), 4),
            "RMSE": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
            "R2": round(float(r2_score(y_true, y_pred)), 4)}


def train_regression(train: pd.DataFrame, test: pd.DataFrame, seed: int):
    """Fit candidates, select by 5-fold CV RMSE on TRAIN only, report TEST metrics."""
    candidates = {
        "mean_baseline": _pipe(DummyRegressor(strategy="mean")),
        "linear_regression": _pipe(LinearRegression()),
        "random_forest": _pipe(RandomForestRegressor(
            n_estimators=300, min_samples_leaf=2, random_state=seed, n_jobs=1)),
    }
    cv = KFold(n_splits=5, shuffle=True, random_state=seed)
    Xtr, ytr = train[FEATURES], train["yield_tons"]
    Xte, yte = test[FEATURES], test["yield_tons"]

    results, cv_rmse = {}, {}
    for name, pipe in candidates.items():
        cv_rmse[name] = round(float(-cross_val_score(
            pipe, Xtr, ytr, cv=cv, scoring="neg_root_mean_squared_error").mean()), 4)
        pipe.fit(Xtr, ytr)
        results[name] = {"test": regression_scores(yte, pipe.predict(Xte)),
                         "cv_rmse_train": cv_rmse[name]}

    real = {k: v for k, v in cv_rmse.items() if k != "mean_baseline"}
    best = min(real, key=real.get)
    pred = candidates[best].predict(Xte)
    err = pd.Series(pred - yte.values, index=yte.index)
    worst_id = err.abs().idxmax()
    report = {
        "random_state": seed,
        "metrics": results,
        "selected_model": best,
        "selection_reason": (
            f"{best} had the lowest 5-fold cross-validated RMSE on the training set "
            f"({real[best]}); the test set was used only for the final report."),
        "largest_error_row": {
            "row_id": int(worst_id),
            "actual": round(float(yte.loc[worst_id]), 4),
            "predicted": round(float(pred[list(yte.index).index(worst_id)]), 4),
            "error": round(float(err.loc[worst_id]), 4),
            "features": {k: (v if isinstance(v, str) else round(float(v), 4))
                         for k, v in test.loc[worst_id, FEATURES].items()},
        },
    }
    return report, candidates[best], pred
