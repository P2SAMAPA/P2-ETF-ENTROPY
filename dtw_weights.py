"""
DTW Weights Module
Computes pairwise DTW distances between ETF price series and converts
them to transfer-learning weights following the paper's Eq.(21):

    w_i = 1 / d_i   (for source ETFs i ≠ target)

Key fix vs original:
  - Diagonal (self-distance = 0) is EXCLUDED from weight computation.
    Including it caused every ETF to assign ~14% weight to itself in
    the transfer aggregation, which is wrong — the target ETF is the
    *recipient* of knowledge, not a source.
  - DTW is computed on RETURNS (pct_change), not raw price levels,
    making the distance scale-invariant across ETFs.
  - Returns a (n, n) weight matrix where weight[i, j] = weight that
    source ETF j gets when predicting target ETF i.
    Row i sums to 1.0 over j ≠ i (off-diagonal only).
"""

import numpy as np
import pandas as pd
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean


def compute_dtw_matrix(price_df: pd.DataFrame, max_samples: int = 500) -> np.ndarray:
    """
    Compute DTW distance matrix between all ETF return series,
    then convert to a normalised transfer-weight matrix.

    Parameters
    ----------
    price_df : pd.DataFrame
        Price data — columns are ETFs, index is dates.
        Should contain ONLY the training window (2008–2021) to
        avoid look-ahead bias in weight computation.
    max_samples : int
        Cap on series length fed to DTW (O(n²) complexity).

    Returns
    -------
    weight_matrix : np.ndarray, shape (n_etfs, n_etfs)
        weight_matrix[i, j] = normalised transfer weight that
        source j contributes when predicting target i.
        Diagonal is 0 (no self-transfer).
        Each row sums to 1.0 over off-diagonal entries.
    """
    assets = price_df.columns.tolist()
    n = len(assets)

    # ── Use daily returns (scale-invariant) ──────────────────────────
    returns_df = price_df.pct_change().dropna()

    # Subsample if series is very long (DTW is O(N²) per pair)
    if len(returns_df) > max_samples:
        step = max(1, len(returns_df) // max_samples)
        returns_df = returns_df.iloc[::step][:max_samples]
        print(f"  DTW: subsampled to {len(returns_df)} points (step={step})")

    ret_array = returns_df.values  # shape (T, n)

    # ── Compute pairwise DTW distances ───────────────────────────────
    dtw_distances = np.full((n, n), np.inf)
    np.fill_diagonal(dtw_distances, 0.0)

    for i in range(n):
        for j in range(i + 1, n):
            x = ret_array[:, i].astype(np.float64)
            y = ret_array[:, j].astype(np.float64)

            # Remove rows where either series has NaN
            mask = ~(np.isnan(x) | np.isnan(y))
            x_clean, y_clean = x[mask], y[mask]

            if len(x_clean) < 10:
                print(f"  DTW: insufficient data for {assets[i]} vs {assets[j]}, using inf")
                dtw_distances[i, j] = np.inf
            else:
                try:
                    dist, _ = fastdtw(x_clean, y_clean, dist=euclidean, radius=10)
                    dtw_distances[i, j] = dist
                except Exception as e:
                    print(f"  DTW error {assets[i]} vs {assets[j]}: {e}")
                    dtw_distances[i, j] = np.inf

            dtw_distances[j, i] = dtw_distances[i, j]   # symmetric

    # ── Convert distances → weights (Eq. 21: w_i = 1/d_i) ───────────
    # Work only on off-diagonal entries (source ≠ target)
    weight_matrix = np.zeros((n, n))

    for i in range(n):
        row_distances = dtw_distances[i].copy()
        row_distances[i] = np.inf   # exclude self

        # Replace any remaining inf with large finite value so weight → 0
        finite_mask = np.isfinite(row_distances)
        if finite_mask.sum() == 0:
            # Fallback: equal weights across all sources
            weight_matrix[i] = 1.0 / (n - 1)
            weight_matrix[i, i] = 0.0
            continue

        # w_j = 1 / d_ij  for finite distances, 0 otherwise
        raw_weights = np.zeros(n)
        raw_weights[finite_mask] = 1.0 / row_distances[finite_mask]
        raw_weights[i] = 0.0   # ensure self = 0

        # Normalise so off-diagonal row sums to 1
        total = raw_weights.sum()
        weight_matrix[i] = raw_weights / (total + 1e-12)

    print(f"  DTW weight matrix: {weight_matrix.shape}  "
          f"(row sums: min={weight_matrix.sum(axis=1).min():.3f} "
          f"max={weight_matrix.sum(axis=1).max():.3f})")

    return weight_matrix


def compute_dtw_weights_for_target(
    weight_matrix: np.ndarray,
    target_etf: str,
    etf_list: list
) -> dict:
    """
    Extract the source weights for a specific target ETF.

    Parameters
    ----------
    weight_matrix : np.ndarray
        Output of compute_dtw_matrix.
    target_etf : str
    etf_list : list

    Returns
    -------
    dict  {source_etf: weight}  — excludes target_etf itself
    """
    target_idx = etf_list.index(target_etf)
    weights = {}
    for j, etf in enumerate(etf_list):
        if etf != target_etf:
            weights[etf] = weight_matrix[target_idx, j]
    return weights


def compute_simple_correlation_matrix(price_df: pd.DataFrame) -> np.ndarray:
    """
    Fallback: correlation-based weights (much faster than DTW).
    Same off-diagonal-only normalisation as compute_dtw_matrix.
    """
    returns_df = price_df.pct_change().dropna()
    corr = returns_df.corr().values
    n = corr.shape[0]

    # Convert correlation [-1, 1] → distance [0, 2], then invert
    dist = 1.0 - corr
    np.fill_diagonal(dist, np.inf)   # exclude self

    weight_matrix = np.zeros((n, n))
    for i in range(n):
        row = dist[i].copy()
        finite = np.isfinite(row)
        if finite.sum() == 0:
            weight_matrix[i] = 1.0 / (n - 1)
            weight_matrix[i, i] = 0.0
            continue
        raw = np.zeros(n)
        raw[finite] = 1.0 / (row[finite] + 1e-12)
        raw[i] = 0.0
        weight_matrix[i] = raw / (raw.sum() + 1e-12)

    return weight_matrix
