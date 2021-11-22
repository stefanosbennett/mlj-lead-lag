#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implementation of "Hermitian matrices for clustering directed graphs:
insights and applications" (Cucuringu et al. 2019) clustering algo

"""

import os
import numpy as np
import pandas as pd
from sklearn import cluster
import itertools


# normalisation of adjacency matrix
def normalise_adj(A, method='rw'):
    D = np.abs(A).sum(1)

    # replacing zero values with min in D to avoid 0/0 errors
    D = D.where(D > 0, D[D > 0].min())

    if method == 'rw':
        return np.dot(np.diag(1/D), A)
    elif method == 'sym':
        return np.diag(1/D) ** (0.5) @ A @ np.diag(1/D) ** (0.5)
    else:
        raise NotImplementedError


# clustering algo
def spectral_cluster(A, num_clusters):
    """A is a Hermitian adj matrix"""
    
    # number of eigenvectors used by the algorithm
    l = num_clusters - num_clusters % 2
    
    eigen_vals, eigen_vectors = np.linalg.eigh(A)
    
    # indices for top l eigenvectors
    index = np.argsort(np.abs(eigen_vals))[-l:]
    
    col_subset = eigen_vectors[:, index]
    
    projection = np.dot(col_subset, np.conjugate(col_subset.T))
    
    return np.real(projection)


def get_clustering(lead_lag_matrix, num_clusters, normalise_adj_method):

    A = lead_lag_matrix * np.complex(0, 1)

    if normalise_adj_method is None:
        pass
    elif normalise_adj_method == 'rw':
        A = normalise_adj(A, method='rw')
    else:
        raise NotImplementedError('normalise_adj_method not implemented')

    projection = spectral_cluster(A, num_clusters)
    clustering = cluster.k_means(projection, n_clusters=num_clusters, n_init=200, max_iter=6000)
    clusters = pd.Series(clustering[1], index=lead_lag_matrix.index)

    return clusters


def get_ordered_clustering(lead_lag_matrix, num_clusters):
    """
    Returns a clustering of time series where the node label corresponds to the clusters (where 0 is the
    most leading and (num_clusters - 1) is the least leading cluster.
    RW-normalised Hermitian clustering is performed.
    """
    clustering = get_clustering(lead_lag_matrix, num_clusters, 'rw')

    lead_lag_by_stock = lead_lag_matrix.sum(1)
    lead_lag_by_cluster = lead_lag_by_stock.groupby(clustering).mean()
    map_cluster_to_order = lead_lag_by_cluster.rank(ascending=False).map(lambda num: int(num) - 1)
    clustering = clustering.map(map_cluster_to_order)

    return clustering

