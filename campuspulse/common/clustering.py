"""K-Means / DBSCAN / OPTICS + silhouette (compare the methods).

Two rules every pulse gets for free here:
1. Features are standardised first. Otherwise the column with the biggest units
   (watts, clicks, codes) decides every distance and the clusters mean nothing.
2. Subsamples are RANDOM (`sample_rows`), never `df.iloc[:n]`: most of our tables
   are sorted (by course, by date), so the first n rows are not representative.
"""
import numpy as np, pandas as pd
from sklearn.cluster import KMeans, DBSCAN, OPTICS
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def sample_rows(df, n: int, seed: int = 0):
    """Random subsample of at most n rows (keeps the index so labels can be joined back)."""
    return df if len(df) <= n else df.sample(n, random_state=seed)


def choose_eps(Xs, min_samples: int = 10, quantile: float = 0.90) -> float:
    """k-distance heuristic for DBSCAN: eps = a high quantile of each point's distance
    to its min_samples-th neighbour (the 'knee' of the k-distance plot)."""
    d, _ = NearestNeighbors(n_neighbors=min_samples).fit(Xs).kneighbors(Xs)
    eps = float(np.quantile(d[:, -1], quantile))
    if eps <= 0:   # 0/1 survey data: many identical rows, so use the smallest non-zero distance instead
        positive = d[d > 0]
        eps = float(positive.min()) if positive.size else 0.5
    return eps


def run_all(X, k=4, eps=None, min_samples=10, scale=True):
    X = np.asarray(X, dtype=float)
    Xs = StandardScaler().fit_transform(X) if scale else X
    Xs = np.nan_to_num(Xs)
    eps = choose_eps(Xs, min_samples) if eps is None else eps
    models = {"kmeans": KMeans(k, n_init=10, random_state=0),
              "dbscan": DBSCAN(eps=eps, min_samples=min_samples), "optics": OPTICS(min_samples=min_samples)}
    rows, labels = [], {}
    for name, m in models.items():
        lab = m.fit_predict(Xs); labels[name] = lab
        mask = lab != -1
        n_clu = len(set(lab[mask]))
        ok = n_clu > 1 and mask.sum() > n_clu
        rows.append(dict(method=name, n_clusters=n_clu, noise_share=1 - mask.mean(),
                         silhouette=silhouette_score(Xs[mask], lab[mask]) if ok else np.nan,
                         davies_bouldin=davies_bouldin_score(Xs[mask], lab[mask]) if ok else np.nan,
                         eps=eps if name == "dbscan" else np.nan))
    return pd.DataFrame(rows), labels


def run_kmeans(X, k=4, silhouette_sample_size=1200):
    """Fit K-Means on all rows and estimate its silhouette on a manageable random subset."""
    X = np.asarray(X, dtype=float)
    Xs = np.nan_to_num(StandardScaler().fit_transform(X))
    labels = KMeans(k, n_init=10, random_state=0).fit_predict(Xs)
    sample_size = min(silhouette_sample_size, len(Xs))
    silhouette = silhouette_score(Xs, labels, sample_size=sample_size, random_state=0)
    metrics = pd.DataFrame([dict(method="kmeans", n_clusters=len(np.unique(labels)), noise_share=0.0,
                                 silhouette=silhouette, silhouette_sample_size=sample_size,
                                 davies_bouldin=np.nan, eps=np.nan)])
    return metrics, labels


def profile(df: pd.DataFrame, labels) -> pd.DataFrame:
    """Mean of every numeric column per cluster: use this to NAME the clusters."""
    return df.assign(cluster=labels).groupby("cluster").mean(numeric_only=True).round(2)
