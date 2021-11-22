"""

Implementation of DI-SIM co-clustering method from Yu et al. (2016)

"""

import numpy as np
import pandas as pd
from sklearn import cluster


def compute_row_norms(mat):
    row_norms = (mat ** 2).sum(axis=1) ** 0.5

    if np.any(row_norms > 0):
        smallest_non_zero = row_norms[row_norms > 0].min()
    else:
        raise ValueError('all rows of matrix are 0')

    row_norms = np.maximum(row_norms, smallest_non_zero)

    return row_norms


def get_clustering(lead_lag_matrix, num_clusters, cluster_edge_direction):

    # convert lead-lag matrix to the adjacency matrix for the directed graph
    A = np.where(lead_lag_matrix > 0, lead_lag_matrix, 0)

    # computing Laplacian
    tau = A.sum(axis=1).mean()
    P = A.sum(axis=0) + tau
    O = A.sum(axis=1) + tau
    L = A / np.sqrt(O[:, None] * P[None, :])

    # computing the top num_cluster left and right SVs
    U, S, VT = np.linalg.svd(L)
    V = VT.T

    left = U[:, :num_clusters]
    right = V[:, :num_clusters]

    row_norms = compute_row_norms(left)
    left /= row_norms[:, None]
    row_norms = compute_row_norms(right)
    right /= row_norms[:, None]

    left_clustering = cluster.k_means(left, n_clusters=num_clusters, n_init=200, max_iter=6000)
    left_clusters = pd.Series(left_clustering[1], index=lead_lag_matrix.index)

    right_clustering = cluster.k_means(right, n_clusters=num_clusters, n_init=200, max_iter=6000)
    right_clusters = pd.Series(right_clustering[1], index=lead_lag_matrix.index)

    if cluster_edge_direction == 'left':
        return left_clusters
    elif cluster_edge_direction == 'right':
        return right_clusters
    else:
        raise ValueError('cluster_edge_direction is not left or right')
