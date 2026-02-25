"""
Pitcher Clustering — KMeans Only
=================================
Simple KMeans partition of pitchers by arsenal features. Used for roster grouping.
Predictive power (which features matter for performance) lives in arsenal_predictor.py.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

FIGURES_DIR = Path(__file__).resolve().parents[2] / "figures"
PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"


# ---------------------------------------------------------------------------
# KMeans
# ---------------------------------------------------------------------------

def find_optimal_k(X: np.ndarray, k_range: range = range(3, 12)) -> dict:
    """Evaluate KMeans for a range of k using silhouette score."""
    results = {"k": [], "silhouette": []}
    for k in k_range:
        labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(X)
        results["k"].append(k)
        results["silhouette"].append(silhouette_score(X, labels))
    return results


def run_kmeans(X: np.ndarray, n_clusters: int = 5) -> np.ndarray:
    """Fit KMeans and return cluster labels."""
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    return km.fit_predict(X)


def compute_pca_embedding(X: np.ndarray, n_components: int = 2) -> np.ndarray:
    """PCA for 2D or 3D visualization."""
    pca = PCA(n_components=n_components, random_state=42)
    return pca.fit_transform(X)


def plot_clusters_2d(
    embedding: np.ndarray,
    labels: np.ndarray,
    profiles: pd.DataFrame,
    method_name: str = "PCA",
    save: bool = True,
) -> plt.Figure:
    """Scatter plot of 2D embedding colored by cluster."""
    fig, ax = plt.subplots(figsize=(10, 8))
    unique_labels = sorted(set(labels))
    palette = sns.color_palette("husl", n_colors=max(len(unique_labels), 1))
    color_map = {lab: palette[i] for i, lab in enumerate(unique_labels)}

    for lab in unique_labels:
        mask = labels == lab
        ax.scatter(
            embedding[mask, 0], embedding[mask, 1],
            c=[color_map[lab]], label=f"Cluster {lab}", alpha=0.6, s=20,
        )
    ax.set_xlabel(f"{method_name} 1")
    ax.set_ylabel(f"{method_name} 2")
    ax.set_title("Pitcher Clusters (KMeans)")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURES_DIR / "clusters_kmeans.png", dpi=150, bbox_inches="tight")
    return fig


def plot_elbow(results: dict, save: bool = True) -> plt.Figure:
    """Plot silhouette scores vs k."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(results["k"], results["silhouette"], "o-", color="green")
    ax.set_xlabel("k")
    ax.set_ylabel("Silhouette Score")
    ax.set_title("KMeans: choose k")
    fig.tight_layout()
    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURES_DIR / "elbow_silhouette.png", dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Similarity
# ---------------------------------------------------------------------------

def find_similar_pitchers(
    profiles: pd.DataFrame,
    X_scaled: np.ndarray,
    pitcher_name: str,
    n: int = 5,
    feature_names: list | None = None,
) -> pd.DataFrame:
    """
    Find n most similar pitchers by Euclidean distance on scaled features.
    When feature_names provided, uses pitch-usage-weighted distance from preprocess_data.
    """
    from preprocess_data import compute_weighted_distances_from_row, get_similarity_features

    mask = profiles["player_name"] == pitcher_name
    if not mask.any():
        raise ValueError(f"Pitcher '{pitcher_name}' not found")
    i = int(np.where(mask.values)[0][0])

    if feature_names is not None:
        dists = compute_weighted_distances_from_row(profiles, X_scaled, feature_names, i)
    else:
        from sklearn.metrics.pairwise import euclidean_distances
        dists = euclidean_distances(X_scaled[i : i + 1], X_scaled).flatten()

    order = np.argsort(dists)
    neighbors = order[1 : n + 1]
    result = profiles.iloc[neighbors][["player_name", "pitcher"]].copy()
    result["distance"] = dists[neighbors]
    return result.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_clustering_pipeline(
    profiles: pd.DataFrame | None = None,
    n_clusters: int = 5,
    save_figures: bool = True,
) -> dict:
    """
    Build feature matrix, run KMeans, save profiles + cluster labels.
    Uses full per-pitch arsenal (pct_*, velo_*, spin_*, movement, etc.).
    """
    from preprocess_data import prepare_clustering_matrix, get_similarity_features

    if profiles is None:
        profiles = pd.read_parquet(PROCESSED_DIR / "pitcher_profiles.parquet")

    feature_names = get_similarity_features(profiles)
    X_scaled, feature_names, profiles = prepare_clustering_matrix(profiles, feature_names)

    print("Running KMeans...")
    labels = run_kmeans(X_scaled, n_clusters=n_clusters)
    profiles["cluster_kmeans"] = labels

    pca_embed = compute_pca_embedding(X_scaled, n_components=3)
    profiles["pca_0"] = pca_embed[:, 0]
    profiles["pca_1"] = pca_embed[:, 1]
    profiles["pca_2"] = pca_embed[:, 2]

    if save_figures:
        plot_clusters_2d(pca_embed, labels, profiles, method_name="PCA", save=True)
        elbow = find_optimal_k(X_scaled)
        plot_elbow(elbow, save=True)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    profiles.to_parquet(PROCESSED_DIR / "pitcher_profiles_clustered.parquet", index=False)
    print(f"Saved {len(profiles)} pitchers, k={n_clusters} clusters")

    return {
        "profiles": profiles,
        "X_scaled": X_scaled,
        "feature_names": feature_names,
        "kmeans_labels": labels,
    }


if __name__ == "__main__":
    results = run_clustering_pipeline(n_clusters=5)
    print("\nCluster sizes:", pd.Series(results["kmeans_labels"]).value_counts().sort_index())
