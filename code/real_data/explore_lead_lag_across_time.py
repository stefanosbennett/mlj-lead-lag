#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

explore the lead-lag matrix and resulting hermitian clustering across time

"""

from importlib import reload
import itertools
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from models.lead_lag_measure import get_lead_lag_matrix
from models.herm_matrix_algo import get_ordered_clustering
import real_data.utils
from real_data.utils import get_ordered_matrix, compute_jaccard, compute_flow_graph, plot_flow_heatmap, \
    plot_flow_network
import real_data.rolling_utils
from real_data.rolling_utils import get_rolling_lead_lag_matrix, get_cluster_rolling_metric, get_flow_graph_rolling

log_ret_type = ''  # _residuals
log_ret = pd.read_pickle('./data/real_data/log_returns' + log_ret_type + '.pkl')
sic = pd.read_pickle('./data/real_data/sic.pkl')
sic_names = pd.read_pickle('./data/real_data/sic_names.pkl')
ticker_indicator_type = '_full'  #
ticker_indicator = pd.read_pickle('./data/real_data/ticker_indicator' + ticker_indicator_type + '.pkl')
etf = pd.read_pickle('./data/real_data/etf.pkl')
lookback = 252
update_freq = 252
correlation_method = 'distance'  # ''
lead_lag_matrix_rolling = pd.read_pickle('./data/pickled_data/lead_lag_matrix_rolling' + '_lookback_' + str(lookback) +
                                         '_update_freq_' + str(update_freq) + log_ret_type + '_' + correlation_method + '.pkl')
lead_lag_matrix_rolling = lead_lag_matrix_rolling.iloc[1:]
# getting SPY
if log_ret_type == '_residuals':
    ticker_indicator = ticker_indicator.drop(columns=['SPY'])
    log_ret_full = pd.read_pickle('./data/real_data/log_returns.pkl')
    log_ret_SPY = log_ret_full.SPY
else:
    log_ret_SPY = log_ret.SPY

# the stock subset used for lead_lag_matrix_rolling.pkl
stock_subset = ticker_indicator.iloc[-1]
stock_subset = stock_subset.index[stock_subset]
log_ret = log_ret.loc[:, stock_subset]

# resampling lead_lag_matrix_rolling to yearly frequency
# lead_lag_matrix_rolling = lead_lag_matrix_rolling.resample('BY').last()

#%% COMPUTE ROLLING LEAD-LAG MATRIX

# lead_lag_matrix_rolling = get_rolling_lead_lag_matrix(log_ret, lookback=lookback,
#                                                       update_freq=update_freq,
#                                                       method='ccf_auc', max_lag=5, correlation_method=correlation_method)
# lead_lag_matrix_rolling.to_pickle('./data/pickled_data/lead_lag_matrix_rolling' + '_lookback_' + str(lookback) +
#                                   '_update_freq_' + str(update_freq) + log_ret_type + '_' + correlation_method + '.pkl')

#%% ADJUSTED RAND INDEX BETWEEN YEARLY CLUSTERS
# applying clustering
clusters_rolling = lead_lag_matrix_rolling.apply(get_ordered_clustering, num_clusters=10)

metric = get_cluster_rolling_metric(clusters_rolling)

rename_lambda = lambda name: name.year
sns.heatmap(metric.rename(index=rename_lambda, columns=rename_lambda))
plt.savefig('./figures/real_data/yearly_ari_heatmap.pdf')
plt.show()

#%% COMPUTING TOTAL NET FLOW TO/FROM EACH ORDERED CLUSTER FOR EACH YEAR
# computing ordered clusters
clusters_rolling = lead_lag_matrix_rolling.apply(get_ordered_clustering, num_clusters=10)

flow_graph = get_flow_graph_rolling(lead_lag_matrix_rolling, clusters_rolling, normalise=True)
flow_graph.apply(lambda df: df.sum(1)).plot()
# plt.savefig('./figures/ordered_cluster_total_flow_normalised.png')
plt.show()
