"""
Functions used for rolling lead-lag and clustering computation

"""

import multiprocessing as mp
import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from functools import partial
from p_tqdm import p_map
from sklearn import metrics
from tqdm import tqdm
from models.lead_lag_measure import get_lead_lag_matrix
from models.herm_matrix_algo import get_ordered_clustering
from real_data.utils import compute_flow_graph


def subset_data_compute_lead_lag(date, log_ret, lookback, **method_kwargs):
    data_subset = log_ret.loc[:date].iloc[-lookback:]

    return get_lead_lag_matrix(data_subset, **method_kwargs)


def get_rolling_lead_lag_matrix(log_ret, lookback, update_freq, num_cpus=7, **method_kwargs):
    """
    returns a dictionary of lead-lag matrices throughout time
    """

    lead_lag_matrix_dict = dict()
    update_date = pd.Timestamp(year=1900, month=1, day=1)
    date_list = []

    # compiling the list of dates at which we compute the lead-lag matrix
    for date in log_ret.index[lookback:]:
        if date > update_date:
            date_list.append(date)
            update_date = date + pd.tseries.offsets.BDay(update_freq)

    p_compute_lead_lag = partial(subset_data_compute_lead_lag, log_ret=log_ret, lookback=lookback, **method_kwargs)
    lead_lag_matrix_rolling = p_map(p_compute_lead_lag, date_list, num_cpus=num_cpus)

    lead_lag_matrix_dict = {date: matrix for date, matrix in zip(date_list, lead_lag_matrix_rolling)}
    lead_lag_matrix_rolling = pd.Series(lead_lag_matrix_dict)
    lead_lag_matrix_rolling.index = pd.DatetimeIndex(lead_lag_matrix_rolling.index)

    return lead_lag_matrix_rolling


def get_cluster_rolling_metric(clusters_rolling, method='adjusted_rand_index'):

    if method == 'adjusted_rand_index':
        metric = pd.DataFrame(np.nan, index=clusters_rolling.index, columns=clusters_rolling.index)

        for date_1, date_2 in itertools.combinations(clusters_rolling.index, 2):
            metric.loc[date_1, date_2] = metrics.adjusted_rand_score(clusters_rolling.loc[date_1], clusters_rolling.loc[date_2])
            metric.loc[date_2, date_1] = metric.loc[date_1, date_2]

        np.fill_diagonal(metric.values, 1)
    else:
        raise NotImplementedError

    return metric


def get_flow_graph_rolling(lead_lag_matrix_rolling, clusters_rolling, normalise=False):
    flow_graph = dict()
    for date in lead_lag_matrix_rolling.index:
        flow_graph[date] = compute_flow_graph(lead_lag_matrix_rolling.loc[date], clusters_rolling.loc[date], normalise)

    flow_graph = pd.Series(flow_graph)
    return flow_graph
