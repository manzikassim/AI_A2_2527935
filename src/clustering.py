"""Clustering task: K-means on scaled features, K=2..6, silhouette-based choice."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline

from src.config import FEATURES, NUMERIC_FEATURES
from src.preprocessing import build_preprocessor

FRIENDLY = {"rainfall_mm": "rainfall", "avg_temp_c": "temperature",
            "soil_moisture_pct": "soil moisture", "fertilizer_kg": "fertilizer use",
            "field_size_ha": "field size"}


def _describe(z: dict, season_a: float, support: float) -> str:
    high = [f"higher {FRIENDLY[f]}" for f, v in z.items() if v >= 0.5]
    low = [f"lower {FRIENDLY[f]}" for f, v in z.items() if v <= -0.5]
    traits = high + low or ["conditions close to the average"]
    return (f"{', '.join(traits)}; {season_a:.0%} season A; "
            f"{support:.0%} receive cooperative support")


def train_clustering(train: pd.DataFrame, test: pd.DataFrame, seed: int):
    """Choose K by silhouette on TRAIN, profile clusters, assign test rows. No targets used."""
    Xtr = train[FEATURES]
    scaled = build_preprocessor().fit_transform(Xtr)

    scores = {}
    for k in range(2, 7):
        labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(scaled)
        scores[k] = round(float(silhouette_score(scaled, labels)), 4)
    best_k = max(scores, key=scores.get)

    model = Pipeline([("prep", build_preprocessor()),
                      ("kmeans", KMeans(n_clusters=best_k, n_init=10, random_state=seed))])
    model.fit(Xtr)
    train_labels = model.predict(Xtr)
    test_labels = model.predict(test[FEATURES])

    mean, std = Xtr[NUMERIC_FEATURES].mean(), Xtr[NUMERIC_FEATURES].std()
    profiles = {}
    for c in range(best_k):
        rows = train[train_labels == c]
        z = ((rows[NUMERIC_FEATURES].mean() - mean) / std).round(3)
        season_a = float((rows["season"] == "A").mean())
        support = float(rows["cooperative_support"].mean())
        text = _describe(z.to_dict(), season_a, support)
        profiles[str(c)] = {
            "label": f"Cluster {c}: {text}",
            "size_train": int(len(rows)),
            "center_original_units": {k: round(float(v), 4)
                                      for k, v in rows[NUMERIC_FEATURES].mean().items()},
            "z_scores_vs_training_mean": z.to_dict(),
            "share_season_A": round(season_a, 4),
            "share_cooperative_support": round(support, 4),
            "post_hoc_mean_yield_tons": round(float(rows["yield_tons"].mean()), 4),
            "post_hoc_note": "yield shown only to interpret clusters; it was NOT used to form them",
        }
    report = {
        "random_state": seed,
        "silhouette_by_k": {str(k): v for k, v in scores.items()},
        "selected_k": int(best_k),
        "selection_reason": (f"K={best_k} has the highest silhouette score ({scores[best_k]}). "
                             "Scores below about 0.25 mean clusters overlap and are weak."),
        "profiles": profiles,
        "interpretation": (
            "Features were generated independently and uniformly, so there is no strong natural "
            "grouping; clusters mainly split farms by the dominant condition differences above."),
    }
    return report, model, test_labels, train_labels
