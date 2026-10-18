"""Matplotlib figures saved under artifacts/plots/."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from src.config import FEATURES, PLOTS_DIR
from src.preprocessing import build_preprocessor


def plot_residuals(actual, predicted, code: str) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    resid = np.asarray(predicted) - np.asarray(actual)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].scatter(predicted, resid, alpha=0.7)
    ax[0].axhline(0, color="red", lw=1)
    ax[0].set(xlabel="Predicted yield (tons)", ylabel="Residual (pred - actual)",
              title="Residuals vs predicted")
    ax[1].hist(resid, bins=15, edgecolor="black")
    ax[1].set(xlabel="Residual (tons)", ylabel="Count", title="Residual distribution")
    fig.suptitle(f"Regression residuals - {code}")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "residuals.png", dpi=150)
    plt.close(fig)


def plot_confusion(matrix, code: str, model_name: str) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    m = np.asarray(matrix)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(m, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(m[i, j]), ha="center", va="center", fontsize=14)
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["low risk", "high risk"],
           yticklabels=["low risk", "high risk"], xlabel="Predicted", ylabel="Actual",
           title=f"{model_name}\n{code}")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)


def plot_clusters(train, test, cluster_model, code: str) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    prep = cluster_model.named_steps["prep"]
    pca = PCA(n_components=2, random_state=0).fit(prep.transform(train[FEATURES]))
    fig, ax = plt.subplots(figsize=(6, 5))
    for data, marker, tag in ((train, "o", "train"), (test, "^", "test")):
        pts = pca.transform(prep.transform(data[FEATURES]))
        sc = ax.scatter(pts[:, 0], pts[:, 1], c=cluster_model.predict(data[FEATURES]),
                        cmap="viridis", marker=marker, alpha=0.7, vmin=0,
                        vmax=cluster_model.named_steps["kmeans"].n_clusters - 1, label=tag)
    centers = pca.transform(cluster_model.named_steps["kmeans"].cluster_centers_)
    ax.scatter(centers[:, 0], centers[:, 1], c="red", marker="X", s=150, label="centers")
    ax.set(xlabel="PC1", ylabel="PC2", title=f"Farm clusters (PCA view) - {code}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "clusters.png", dpi=150)
    plt.close(fig)
