"""
DTW Weights Module — TUNED
Changes vs original:
  - DTW computed on daily pct_change() RETURNS not raw price levels
    (price scale differences of 8x no longer dominate distances)
  - Sampling: contiguous window from end of training data (not stride)
    (preserves temporal structure, avoids aliasing)
  - Self-similarity (diagonal) excluded from row normalisation
    per Eq.(22) of the paper — a source ETF should not vote for itself
  - Fallback to correlation matrix if fastdtw unavailable
"""

import numpy as np

try:
    from fastdtw import fastdtw
    from scipy.spatial.distance import euclidean
    FASTDTW_AVAILABLE = True
except ImportError:
    FASTDTW_AVAILABLE = False


def compute_dtw_matrix(price_df, max_samples=500):
    """
    Compute pairwise DTW similarity on DAILY RETURNS (not price levels).

    Parameters
    ----------
    price_df : pd.DataFrame
        Training-window price data, ETFs as columns.
    max_samples : int
        Max rows to use. Takes the most recent contiguous block.

    Returns
    -------
    np.ndarray (n_etfs × n_etfs)
        Row-normalised similarity weights; diagonal = 0.
    """
    if not FASTDTW_AVAILABLE:
        print("  fastdtw not available — using correlation fallback")
        return _correlation_fallback(price_df)

    # Convert to daily returns — removes price-level scale differences
    returns_df = price_df.pct_change().dropna()

    # Contiguous tail window (preserves temporal structure)
    if len(returns_df) > max_samples:
        returns_df = returns_df.iloc[-max_samples:]

    assets      = returns_df.columns.tolist()
    n           = len(assets)
    ret_array   = returns_df.values.astype(np.float64)

    dtw_distances = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            x = ret_array[:, i]
            y = ret_array[:, j]
            mask    = ~(np.isnan(x) | np.isnan(y))
            x_clean = x[mask]
            y_clean = y[mask]

            if len(x_clean) < 10:
                dtw_distances[i, j] = np.inf
            else:
                try:
                    dist, _ = fastdtw(x_clean, y_clean,
                                      dist=euclidean, radius=10)
                    dtw_distances[i, j] = dist
                except Exception as e:
                    print(f"    DTW error {assets[i]} vs {assets[j]}: {e}")
                    dtw_distances[i, j] = np.inf

            dtw_distances[j, i] = dtw_distances[i, j]

    return _distances_to_weights(dtw_distances)


def _distances_to_weights(dtw_distances):
    """
    Convert distance matrix → row-normalised similarity weights.
    Diagonal set to 0 (self excluded per Eq.22 — ETF does not transfer to itself).
    """
    n = dtw_distances.shape[0]

    # Replace inf with large finite value
    finite_max = dtw_distances[np.isfinite(dtw_distances) & (dtw_distances > 0)]
    cap        = finite_max.max() * 10 if len(finite_max) > 0 else 1e6
    dtw_distances = np.where(np.isinf(dtw_distances), cap, dtw_distances)

    # Similarity = inverse distance (with smoothing)
    min_nonzero = dtw_distances[dtw_distances > 0].min() if (dtw_distances > 0).any() else 1.0
    similarity  = 1.0 / (1.0 + dtw_distances / (min_nonzero + 1e-10))

    # Zero out diagonal (self excluded)
    np.fill_diagonal(similarity, 0.0)

    # Row-normalise over off-diagonal entries only
    row_sums = similarity.sum(axis=1, keepdims=True)
    similarity = np.where(row_sums > 1e-10, similarity / row_sums, 0.0)

    print(f"  DTW similarity matrix: {similarity.shape}  "
          f"mean_off_diag={similarity[similarity > 0].mean():.4f}")
    return similarity


def _correlation_fallback(price_df):
    """Correlation-based fallback when fastdtw unavailable."""
    returns = price_df.pct_change().dropna()
    corr    = returns.corr().values.astype(np.float64)
    sim     = (corr + 1.0) / 2.0          # map [-1,1] → [0,1]
    np.fill_diagonal(sim, 0.0)
    row_sums = sim.sum(axis=1, keepdims=True)
    sim = np.where(row_sums > 1e-10, sim / row_sums, 0.0)
    return sim
