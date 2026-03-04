"""
DTW Weights Module - OPTIMIZED
Computes pairwise DTW similarity between ETFs
Uses sampling for speed on large datasets
"""

import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean


def compute_dtw_matrix(price_df, max_samples=500):
    """
    Compute DTW distance matrix between all ETF price series
    Uses sampling for large datasets to improve speed
    
    Parameters:
    -----------
    price_df : pd.DataFrame
        Price data with ETFs as columns
    max_samples : int
        Maximum number of samples to use for DTW (for speed)
    
    Returns:
    --------
    np.ndarray
        Similarity matrix (inverse of normalized DTW distance)
    """
    assets = price_df.columns.tolist()
    n = len(assets)
    
    # Sample data if too large (DTW is O(n^2))
    if len(price_df) > max_samples:
        print(f"  Sampling {max_samples} points for DTW (from {len(price_df)})")
        price_df = price_df.iloc[::len(price_df)//max_samples][:max_samples]
    
    # Convert to numpy for faster access
    price_array = price_df.values
    
    # Compute pairwise DTW
    dtw_distances = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i == j:
                dtw_distances[i, j] = 0
            elif j > i:
                # Get 1-D arrays
                x = price_array[:, i].flatten()
                y = price_array[:, j].flatten()
                
                # Ensure 1-D
                x = np.asarray(x, dtype=np.float64).ravel()
                y = np.asarray(y, dtype=np.float64).ravel()
                
                # Remove NaN
                mask = ~(np.isnan(x) | np.isnan(y))
                x_clean = x[mask]
                y_clean = y[mask]
                
                if len(x_clean) < 10:
                    dtw_distances[i, j] = np.inf
                else:
                    try:
                        distance, _ = fastdtw(x_clean, y_clean, dist=euclidean, radius=1)
                        dtw_distances[i, j] = distance
                    except Exception as e:
                        print(f"    DTW error for {assets[i]} vs {assets[j]}: {e}")
                        dtw_distances[i, j] = np.inf
                
                # Symmetric
                dtw_distances[j, i] = dtw_distances[i, j]
    
    # Convert distances to similarities (inverse with smoothing)
    # Add small constant to avoid division by zero
    min_nonzero = np.min(dtw_distances[dtw_distances > 0]) if np.any(dtw_distances > 0) else 1
    similarity = 1.0 / (1.0 + dtw_distances / min_nonzero)
    
    # Normalize rows to sum to 1
    row_sums = similarity.sum(axis=1, keepdims=True)
    similarity = similarity / (row_sums + 1e-10)
    
    print(f"  DTW matrix computed: {similarity.shape}")
    return similarity


def compute_simple_correlation_matrix(price_df):
    """
    Fallback: Use correlation instead of DTW (much faster)
    """
    corr = price_df.corr().values
    # Convert correlation to similarity (0 to 1)
    similarity = (corr + 1) / 2
    # Normalize rows
    row_sums = similarity.sum(axis=1, keepdims=True)
    similarity = similarity / (row_sums + 1e-10)
    return similarity
