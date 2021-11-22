"""

Implementation of 2 symmetrisation methods + undirected graph spectral clustering (normalised Laplacian) on the
symmetrised adj matrices.

Symmetrisation methods:
naive: Drop direction of edges
degree_disc_bib: degree-discounted bibliometric

Satuluri "Symmetrizations for clustering directed graphs"

"""

import numpy as np
import pandas as pd
from sklearn import cluster


def naive(lead_lag_matrix, num_clusters):

    A = np.abs(lead_lag_matrix)

    clustering = cluster.SpectralClustering(n_clusters=num_clusters, n_init=200, affinity='precomputed',
                                            assign_labels='kmeans')
    clustering = clustering.fit_predict(A)
    clusters = pd.Series(clustering, index=lead_lag_matrix.index)

    return clusters


def degree_disc_bib(lead_lag_matrix, num_clusters):

    # convert lead-lag matrix to the adjacency matrix for the directed graph
    A = np.where(lead_lag_matrix > 0, lead_lag_matrix, 0)

    Do = lead_lag_matrix.sum(1)
    Di = lead_lag_matrix.sum(0)

    # filling in zero values with smallest non-zero value
    Do = np.maximum(Do, Do[Do > 0].min())
    Di = np.maximum(Di, Di[Di > 0].min())

    # computing mat^-0.5 for Do and Di
    Do = np.diag(1/Do ** 0.5)
    Di = np.diag(1/Di ** 0.5)

    U = Do @ A @ Di @ A.T @ Do + Di @ A.T @ Do @ A @ Di

    clustering = cluster.SpectralClustering(n_clusters=num_clusters, n_init=200, affinity='precomputed',
                                            assign_labels='kmeans')
    clustering = clustering.fit_predict(U)
    clusters = pd.Series(clustering, index=lead_lag_matrix.index)

    return clusters
