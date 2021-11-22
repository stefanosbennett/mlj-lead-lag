"""
Util functions for real data scripts

"""


import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from tqdm import tqdm
from models.lead_lag_measure import get_lead_lag_matrix
from models.herm_matrix_algo import get_ordered_clustering


def symmetrise_lead_lag_matrix(lead_lag_matrix):
    """
    compute the symmetrised lead-lag matrix to facilitate flow calculations
    entry e_ij gives the flow going from node i to j
    positive => i leads j, negative => j leads i
    """

    lead_lag_matrix_sym = np.triu(lead_lag_matrix) - np.tril(lead_lag_matrix)
    lead_lag_matrix_sym = pd.DataFrame(lead_lag_matrix_sym, index=lead_lag_matrix.index,
                                       columns=lead_lag_matrix.columns)

    return lead_lag_matrix_sym


def get_ordered_matrix(lead_lag_matrix, method, **method_args):
    """
    Re-orders the lead-lag matrix according to given method so that patterns can be seen
    """

    if method == 'row_sums':
        # ordering from most to least leading
        ordering = lead_lag_matrix.mean(1).sort_values(ascending=False).index
        lead_lag_matrix_ordered = lead_lag_matrix.reindex(index=ordering, columns=ordering)

    elif method == 'herm_rw_clustering':
        clustering = get_ordered_clustering(lead_lag_matrix, num_clusters=method_args['num_clusters'])
        ordering = pd.concat([clustering, lead_lag_matrix.mean(1)], axis=1, keys=['cluster_order', 'time_series_order'])
        # double-ordering by cluster and by time series (from most to least leading).
        # within a cluster, the time series are ordered from most to least leading.
        ordering = ordering.sort_values(['cluster_order', 'time_series_order'], ascending=[True, False]).index

        lead_lag_matrix_ordered = lead_lag_matrix.reindex(index=ordering, columns=ordering)

    else:
        raise NotImplementedError

    lead_lag_matrix_ordered_upper = lead_lag_matrix_ordered.values
    #lead_lag_matrix_ordered_upper[np.tril_indices_from(lead_lag_matrix_ordered)] = np.nan

    return lead_lag_matrix_ordered_upper


def compute_jaccard(cross_tab):
    num_sectors = cross_tab.shape[0]
    num_clusters = cross_tab.shape[1]
    # normalisation is the cardinality of the union of sector and cluster set
    normalisation = np.repeat(cross_tab.sum(1).values.reshape(-1, 1), num_clusters, axis=1)
    normalisation += np.repeat(cross_tab.sum(0).values.reshape(1, -1), num_sectors, axis=0)
    normalisation -= cross_tab

    return cross_tab / normalisation


def compute_flow_graph(lead_lag_matrix, clustering, normalise=False):
    """
    :param lead_lag_matrix: is the real skew-symmetric lead-lag matrix
    :param clustering: is a pd.Series with the cluster assignment for each instrument
    :return: cluster meta-graph flow matrix F. Entry i, j of F gives the amount by which cluster i leads cluster j.
                F is skew-symmetric
    """

    cluster_names = clustering.unique().astype('int')
    cluster_names.sort()
    num_clusters = len(cluster_names)
    F = pd.DataFrame(np.zeros((num_clusters, num_clusters)), index=cluster_names, columns=cluster_names)

    for label_1, label_2 in itertools.combinations(cluster_names, 2):
        F.loc[label_1, label_2] = lead_lag_matrix.loc[clustering.index[clustering == label_1],
                                                      clustering.index[clustering == label_2]].sum().sum()
        F.loc[label_2, label_1] = -1 * F.loc[label_1, label_2]

    if normalise:
        cluster_size = clustering.value_counts().reindex(F.index)
        cluster_size_index = cluster_size.index
        cluster_size = cluster_size.values.reshape(-1, 1)
        norm_factor = pd.DataFrame(cluster_size @ cluster_size.T, index=cluster_size_index,
                                   columns=cluster_size_index)

        F = F / norm_factor

    return F


def plot_flow_heatmap(F, clustering, cmap='seismic'):

    F_upper = F.values
    F_upper[np.tril_indices_from(F)] = np.nan
    f = sns.heatmap(pd.DataFrame(F_upper, index=F.index, columns=F.columns), cmap=cmap)

    return f


def plot_flow_network(F, figsize=None):

    edge_list = F.stack()
    edge_list = edge_list[edge_list > 0]
    edge_list.index.names = ['source', 'target']
    edge_list = edge_list.reset_index().rename(columns={0: 'weight'})

    G = nx.from_pandas_edgelist(edge_list, source='source', target='target', edge_attr=True, create_using=nx.DiGraph)
    G.edges(data=True)

    positions = nx.circular_layout(G) # kamada_kawai_layout, circular_layout
    edges, weights = zip(*nx.get_edge_attributes(G, 'weight').items())
    weights = np.array([w for w in weights])
    weights = 5 * weights/weights.max()
    plt.figure(figsize=figsize)
    f = nx.draw_networkx(G, pos=positions, with_labels=True, width=weights, style='solid', alpha=0.75, edge_color=weights,
                         edge_cmap=plt.get_cmap('autumn_r'), node_color='g', edge_vmin=min(weights), edge_vmax=max(weights))

    return f


def compute_largest_e_values_mean(lead_lag_matrix, top_n=1):
    A = lead_lag_matrix * np.complex(0, 1)
    eigen_vals, eigen_vectors = np.linalg.eigh(A)

    return eigen_vals[-top_n:].mean()


def permutation_mc_lead_lag(seed, log_ret):

    # setting the random seed
    np.random.seed(seed)

    log_ret_values = log_ret.values.copy()
    row_permutation = np.random.permutation(range(log_ret.shape[0]))
    log_ret_perm = log_ret_values[row_permutation, :]
    log_ret_perm = pd.DataFrame(log_ret_perm, index=log_ret.index, columns=log_ret.columns)
    lead_lag_matrix_perm = get_lead_lag_matrix(log_ret_perm, method='ccf_auc', max_lag=5, correlation_method='distance')

    return lead_lag_matrix_perm


def compute_test_statistic(test_statistic_fn, feature, clusters):
    feature_mean_by_cluster = feature.groupby(clusters).mean()
    test_statistic = test_statistic_fn(feature_mean_by_cluster)

    return test_statistic


def permute_feature_cluster(seed, test_statistic_fn, feature, clusters):
    """
    Permute the feature values across equities and compute mean feature value by cluster
    """

    np.random.seed(seed)

    permutation = np.random.permutation(len(feature))
    feature = pd.Series(feature.values[permutation], index=feature.index)

    test_statistic = compute_test_statistic(test_statistic_fn, feature, clusters)

    return test_statistic
