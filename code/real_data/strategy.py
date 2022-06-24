#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
create statistical arbitrage application using clustering pre-computed lead-lag matrix

"""

#%%
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
from importlib import reload
from functools import partial
import time
import numpy as np
import pandas as pd
import multiprocessing as mp
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from models.herm_matrix_algo import get_ordered_clustering
import real_data.utils
from real_data.utils import get_ordered_matrix, compute_jaccard, compute_flow_graph, plot_flow_heatmap, \
    plot_flow_network
import real_data.rolling_utils
from real_data.rolling_utils import get_cluster_rolling_metric, get_flow_graph_rolling, get_rolling_lead_lag_matrix
from real_data.sharpe_ratio_test import sharpe_ratio_test
from p_tqdm import p_map
import real_data.strategy_utils
from real_data.strategy_utils import LinearARModel, weighted_flow_signal, run_strat, \
    sound_notification, perm_test, VAR_signal

log_ret_type = ''  # _residuals
log_ret = pd.read_pickle('./data/real_data/log_returns' + log_ret_type + '.pkl')
sic = pd.read_pickle('./data/real_data/sic.pkl')
sic_names = pd.read_pickle('./data/real_data/sic_names.pkl')
ticker_indicator_type = '_full'  #
ticker_indicator = pd.read_pickle('./data/real_data/ticker_indicator' + ticker_indicator_type + '.pkl')
etf = pd.read_pickle('./data/real_data/etf.pkl')
lookback = 252
update_freq = 63
correlation_method = 'distance'  # 'distance'
lead_lag_matrix_rolling = pd.read_pickle('./data/pickled_data/lead_lag_matrix_rolling' + '_lookback_' + str(lookback) +
                                         '_update_freq_' + str(update_freq) + log_ret_type + '_' + correlation_method + '.pkl')
clusters_rolling = pd.read_pickle('./data/pickled_data/clusters_rolling_lookback_' + str(lookback) + '_update_freq_' +
                                  str(update_freq) + log_ret_type + '_' + correlation_method + '.pkl')
flow_graph_rolling = pd.read_pickle('./data/pickled_data/flow_graph_rolling_lookback_' + str(lookback) + '_update_freq_' +
                                    str(update_freq) + log_ret_type + '_' + correlation_method + '.pkl')

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

#%% COMPUTE LEAD-LAG ROLLING MATRIX
# correlation_method = 'kendall'
# lead_lag_matrix_rolling = get_rolling_lead_lag_matrix(log_ret, lookback=lookback,
#                                                       update_freq=update_freq,
#                                                       method='ccf_auc', max_lag=5, correlation_method=correlation_method)
# lead_lag_matrix_rolling.to_pickle('./data/pickled_data/lead_lag_matrix_rolling' + '_lookback_' + str(lookback) +
#                                   '_update_freq_' + str(update_freq) + log_ret_type + '_' + correlation_method + '.pkl')
#
#%% COMPUTE ROLLING CLUSTERING AND FLOW GRAPH

# clusters_rolling = lead_lag_matrix_rolling.apply(get_ordered_clustering, num_clusters=10)
# clusters_rolling.to_pickle('./data/pickled_data/clusters_rolling_lookback_' + str(lookback) + '_update_freq_' +
#                            str(update_freq) + log_ret_type + '_' + correlation_method + '.pkl')
# flow_graph_rolling = get_flow_graph_rolling(lead_lag_matrix_rolling, clusters_rolling, normalise=True)
# flow_graph_rolling.to_pickle('./data/pickled_data/flow_graph_rolling_lookback_' + str(lookback) + '_update_freq_' +
#                              str(update_freq) + log_ret_type + '_' + correlation_method + '.pkl')

#%%
position, pnl = run_strat(log_ret, clusters_rolling, flow_graph_rolling,
                          reversion_days=1,
                          cut_off_quantile=0.9,
                          signal_fn=weighted_flow_signal,
                          flow_graph_sign=True,
                          signal_sign=True,
                          smooth_days=4)
sound_notification(1, 250)
pnl.sum(1).cumsum().plot(); plt.show()
position_normalised = position.apply(lambda col: col/pnl.sum(1).rolling(21, min_periods=5).std().shift(1).ffill().bfill())
pnl_normalised = position_normalised * log_ret
pnl_normalised.loc['2002':].sum(1).cumsum().plot(); plt.show()

print(sharpe_ratio_test(pnl.sum(1).loc['2002':]))
print(sharpe_ratio_test(pnl_normalised.sum(1).loc['2002':]))

print(((pnl_normalised.sum(1)/position_normalised.abs().sum(1)).loc['2002':].mean() * 1e4).round(1), ' bps')

# mean bps return
(pnl_normalised.sum(1)/position_normalised.abs().sum(1)).mean() * 1e4

pnl_linear = position_normalised * (np.exp(log_ret) - 1)
# scaling for 10% annualised volatility
pnl_linear *= 0.1 / (pnl_linear.sum(1).std() * np.sqrt(252))
print(sharpe_ratio_test(pnl_linear.sum(1).loc['2002':]))
(((pnl_linear.sum(1) + 1).loc['2002':].cumprod() - 1) * 100).plot()
plt.ylabel('Cumulative return (%)')
# plt.savefig('./figures/real_data/pnl_curve.pdf')
plt.show()

# saving the strategy linear pnl
pnl_linear.to_pickle('./data/pickled_data/pnl_linear_cluster_strategy.pkl')

# mean bps return
print(pnl_linear.sum(1).mean() * 1e4)

pnl_linear.sum(1).loc['2002':].corr(log_ret_SPY.loc['2002':])

#%% Lasso VAR model

position, pnl = run_strat(log_ret, clusters_rolling, flow_graph_rolling,
                          signal_fn=VAR_signal, reversion_days=5)

pnl.sum(1).cumsum().plot(); plt.show()
position_normalised = position.apply(lambda col: col/pnl.sum(1).rolling(21, min_periods=5).std().shift(1).ffill().bfill())
pnl_normalised = position_normalised * log_ret
pnl_normalised.loc['2002':].sum(1).cumsum().plot(); plt.show()

print(sharpe_ratio_test(pnl.sum(1).loc['2002':]))
print(sharpe_ratio_test(pnl_normalised.sum(1).loc['2002':]))

print(((pnl_normalised.sum(1)/position_normalised.abs().sum(1)).loc['2002':].mean() * 1e4).round(1), ' bps')

# mean bps return
(pnl_normalised.sum(1)/position_normalised.abs().sum(1)).mean() * 1e4

pnl_linear = (position_normalised * (np.exp(log_ret) - 1))
# scaling for 10% annualised volatility
pnl_linear *= 0.1 / (pnl_linear.sum(1).std() * np.sqrt(252))
print(sharpe_ratio_test(pnl_linear.sum(1).loc['2002':]))
(((pnl_linear.sum(1) + 1).loc['2002':].cumprod() - 1) * 100).plot()
plt.ylabel('Cumulative return (%)')
# plt.savefig('./figures/real_data/pnl_curve.pdf')
plt.show()

# saving the Lasso VAR model linear pnl
pnl_linear.to_pickle('./data/pickled_data/pnl_linear_lasso_var.pkl')

# computing the correlation between the Lasso VAR and strategy pnls
pnl_linear_lasso_var = pd.read_pickle('./data/pickled_data/pnl_linear_lasso_var.pkl')
pnl_linear_cluster_strategy = pd.read_pickle('./data/pickled_data/pnl_linear_cluster_strategy.pkl')

pnl_linear_lasso_var.sum(1).loc['2002':].corr(pnl_linear_cluster_strategy.sum(1).loc['2002':])

#%% market buy and hold

log_ret.sum(1).cumsum().plot()
plt.show()
log_ret_SPY.cumsum().plot()

sharpe_ratio_test(log_ret_SPY)
sharpe_ratio_test(np.exp(log_ret_SPY.loc['2002':]) - 1)
sharpe_ratio_test(log_ret.sum(1))
# SPY mean bps return
(np.exp(log_ret_SPY.loc['2002':]) - 1).mean() * 1e4

#%% ablation study: effect of permuting clusters

sharpe_perm = []
n_perm = 200  # 200
seeds = range(n_perm)

perm_test = partial(perm_test, log_ret=log_ret, lead_lag_matrix_rolling=lead_lag_matrix_rolling,
                    clusters_rolling=clusters_rolling)

# sharpe_perm = list(map(perm_test, seeds))
sharpe_perm = p_map(perm_test, seeds, num_cpus=7)
sharpe_perm = pd.Series(sharpe_perm)
sharpe_perm.to_pickle('./data/pickled_data/sharpe_perm.pkl')

#%% MC permutation test p-value
sharpe_perm = pd.read_pickle('./data/pickled_data/sharpe_perm.pkl')

real_sharpe = 0.62
sharpe_perm.append(pd.Series([real_sharpe]))
sharpe_perm.hist(); plt.show()
print((sharpe_perm >= real_sharpe).mean())

# reject test hypothesis at p < 0.05 significance
