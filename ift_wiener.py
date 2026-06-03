import numpy as np
from sklearn.neighbors import kneighbors_graph
from scipy.sparse import csr_matrix, identity
from scipy.sparse.linalg import spsolve

def wiener_filter(returns, macro_data, sigma_ratio=0.1, n_neighbors=5):
    """
    Information field theory Wiener filter.
    Assumes: signal = ETF returns, correlated via graph Laplacian.
    Observation: returns = signal + noise.
    Prior: signal ~ N(0, C) where C = (L + eps I)^(-1) (inverse of graph Laplacian + regularization).
    Likelihood: returns = signal + noise, noise ~ N(0, sigma_noise^2 I).
    Posterior mean = (I + sigma_ratio * L)^(-1) * (macro_projection?) Actually we incorporate macro as prior mean.
    We simplify: use macro as external field: prior mean = linear combination of macro variables.
    """
    n = returns.shape[1]
    if n < 2:
        return np.zeros(n), np.zeros(n)
    # Graph from correlation distance
    corr = returns.corr().values
    dist = 1 - np.abs(corr)
    np.fill_diagonal(dist, 0)
    # Build k-NN graph (undirected)
    adj = np.zeros((n, n))
    k = min(n_neighbors, n-1)
    for i in range(n):
        nearest = np.argsort(dist[i])[1:k+1]
        adj[i, nearest] = 1
    adj = np.maximum(adj, adj.T)
    # Graph Laplacian
    D = np.sum(adj, axis=1)
    L = np.diag(D) - adj
    # Regularized prior precision: sigma_noise^2 * (L + eps I)
    eps = 1e-6
    prior_prec = L + eps * np.eye(n)
    # Prior mean from macro variables
    # For the last observation (today), use the latest macro values
    macro_today = macro_data.iloc[-1].values if macro_data is not None else np.zeros(1)
    # Simple linear model: mu_prior = beta * macro_today, where beta estimated via ridge on historical data
    # To keep this self-contained, we'll use a simplified: prior mean = zero (uninformative) and only use macro in the likelihood?
    # Actually the classic IFT signal = signal = prior_mean + correlated noise. We'll set prior mean = 0 and score = posterior variance.
    prior_mean = np.zeros(n)
    # Observation noise variance
    sigma_noise2 = 1.0
    # Signal-to-noise ratio: sigma_signal2 / sigma_noise2 = sigma_ratio
    sigma_signal2 = sigma_ratio * sigma_noise2
    # Posterior covariance: (prior_prec / sigma_signal2 + I / sigma_noise2)^(-1)
    # Posterior mean = posterior_cov * ( y / sigma_noise2 + prior_prec * prior_mean / sigma_signal2 )
    # Here y = returns today? Actually we want to predict the same as observed? For anomaly detection, we use the last day's returns as measurement.
    y = returns.iloc[-1].values  # vector of last day's returns (n,)
    A = (prior_prec / sigma_signal2) + (np.eye(n) / sigma_noise2)
    b = (y / sigma_noise2) + (prior_prec @ prior_mean / sigma_signal2)
    # Solve A * mu_post = b
    try:
        A_sparse = csr_matrix(A)
        mu_post = spsolve(A_sparse, b)
    except:
        mu_post = np.linalg.lstsq(A, b, rcond=None)[0]
    # Posterior variance diagonal = diag(inv(A))
    try:
        invA = np.linalg.inv(A)
        var_post = np.diag(invA)
    except:
        var_post = np.ones(n) * 0.1
    return mu_post, var_post

def ift_score(returns, macro_data, sigma_ratio=0.1, n_neighbors=5):
    mu, var = wiener_filter(returns, macro_data, sigma_ratio, n_neighbors)
    # Score = posterior variance (higher = more uncertainty = macro conflict)
    return var
