"""

Functions for evaluating pairwise lead-lag measures

"""


import numpy as np
import pandas as pd


def rank_eval(lead_lag_matrix, lags):
    row_sums = lead_lag_matrix.mean(axis=1)
    num_clusters = len(np.unique(lags))
    row_sum_clusters = pd.qcut(row_sums, num_clusters, labels=False)

    return row_sum_clusters.corr(-lags, method='pearson').round(2)


def lag_vector_to_lead_lag_matrix(lags):

    lead_lag_matrix = lags.values.reshape(-1, 1) <= lags.values.reshape(1, -1)
    lead_lag_matrix = 2 * lead_lag_matrix - 1
    lead_lag_matrix -= (lags.values.reshape(-1, 1) == lags.values.reshape(1, -1))

    lead_lag_matrix = pd.DataFrame(lead_lag_matrix, index=lags.index, columns=lags.index)

    return lead_lag_matrix


def prop_correctly_classified_pairwise_relations(lead_lag_matrix, true_lead_lag_matrix):
    '''
    Compute proportion of the true lead-lag pairs that were correctly identified by the lead-lag matrix estimate
    :param true_lead_lag_matrix: a matrix indicating the true lead lag relationship
    '''

    lead_lag_matrix = lead_lag_matrix.values.reshape(-1)
    true_lead_lag_matrix = true_lead_lag_matrix.values.reshape(-1)

    subset_ind = (true_lead_lag_matrix != 0)
    lead_lag_matrix = lead_lag_matrix[subset_ind]
    true_lead_lag_matrix = true_lead_lag_matrix[subset_ind]

    correctly_classified = (lead_lag_matrix * true_lead_lag_matrix) > 0
    prop_correctly_classified = correctly_classified.mean()

    return prop_correctly_classified


def prop_correctly_classified_from_lags_vector(lead_lag_matrix, lags):

    true_lead_lag_matrix = lag_vector_to_lead_lag_matrix(lags)
    return prop_correctly_classified_pairwise_relations(lead_lag_matrix, true_lead_lag_matrix)