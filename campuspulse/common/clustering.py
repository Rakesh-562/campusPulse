"""KMeans / Agglomerative / DBSCAN / OPTICS + metrics (compare >= 2 methods)."""
import numpy as np, pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, OPTICS
from sklearn.metrics import silhouette_score, davies_bouldin_score


def run_all(X, k=4, eps=0.8, min_samples=10):
    models = {"kmeans": KMeans(k, n_init=10, random_state=0), "agglo": AgglomerativeClustering(k),
              "dbscan": DBSCAN(eps=eps, min_samples=min_samples), "optics": OPTICS(min_samples=min_samples)}
    rows, labels = [], {}
    for name, m in models.items():
        lab = m.fit_predict(X); labels[name] = lab
        mask = lab != -1
        ok = len(set(lab[mask])) > 1
        rows.append(dict(method=name, n_clusters=len(set(lab[mask])), noise_share=1 - mask.mean(),
                         silhouette=silhouette_score(X[mask], lab[mask]) if ok else np.nan,
                         davies_bouldin=davies_bouldin_score(X[mask], lab[mask]) if ok else np.nan))
    return pd.DataFrame(rows), labels


def profile(df: pd.DataFrame, labels) -> pd.DataFrame:
    """Mean of every numeric column per cluster: use this to NAME the clusters."""
    return df.assign(cluster=labels).groupby("cluster").mean(numeric_only=True).round(2)
