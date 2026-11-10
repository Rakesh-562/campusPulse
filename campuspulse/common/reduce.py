"""PCA / SVD / t-SNE / (optional) autoencoder."""
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


def pca(X, n=2):
    Xs = StandardScaler().fit_transform(X)
    m = PCA(n_components=n, random_state=0).fit(Xs)
    return m.transform(Xs), m.explained_variance_ratio_, m


def svd(X_sparse, n=10):
    m = TruncatedSVD(n_components=n, random_state=0)
    return m.fit_transform(X_sparse), m.explained_variance_ratio_, m


def tsne(X, perplexity=30, seed=0):
    return TSNE(2, perplexity=perplexity, init="pca", random_state=seed).fit_transform(StandardScaler().fit_transform(X))


def autoencoder_scores(X, bottleneck=4, epochs=60, seed=0):
    """Reconstruction error per row = anomaly score. Needs torch."""
    import torch, torch.nn as nn
    torch.manual_seed(seed)
    Xs = torch.tensor(StandardScaler().fit_transform(X), dtype=torch.float32)
    d = Xs.shape[1]
    net = nn.Sequential(nn.Linear(d, 16), nn.ReLU(), nn.Linear(16, bottleneck), nn.ReLU(),
                        nn.Linear(bottleneck, 16), nn.ReLU(), nn.Linear(16, d))
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    for _ in range(epochs):
        opt.zero_grad(); ((net(Xs) - Xs) ** 2).mean().backward(); opt.step()
    with torch.no_grad():
        return ((net(Xs) - Xs) ** 2).mean(1).numpy()
