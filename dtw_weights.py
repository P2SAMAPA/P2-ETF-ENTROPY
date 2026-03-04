import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

def compute_dtw_matrix(price_df):

    assets = price_df.columns
    n = len(assets)
    matrix = np.zeros((n,n))

    for i in range(n):
        for j in range(n):
            if i == j:
                matrix[i,j] = 1
            else:
                distance, _ = fastdtw(price_df.iloc[:,i], price_df.iloc[:,j], dist=euclidean)
                matrix[i,j] = 1 / (distance + 1e-6)

    return matrix
