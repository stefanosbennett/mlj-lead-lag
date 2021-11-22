#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
explore the lead-lag matrix and resulting hermitian clustering

"""

import pickle
import itertools
import time
import networkx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from functools import partial
from tqdm import tqdm
from p_tqdm import p_map
from importlib import reload
from models.lead_lag_measure import get_lead_lag_matrix
from models.herm_matrix_algo import get_ordered_clustering
import real_data.utils
from real_data.utils import get_ordered_matrix, compute_jaccard, compute_flow_graph, plot_flow_heatmap, \
    plot_flow_network, compute_largest_e_values_mean, permutation_mc_lead_lag, compute_test_statistic, \
    permute_feature_cluster

log_ret_type = '' # '_residuals'
correlation_method = '_distance'
restricted = '' # _restricted
log_ret = pd.read_pickle('./data/real_data/log_returns' + log_ret_type + '.pkl')
sic = pd.read_pickle('./data/real_data/sic.pkl')
sic_names = pd.read_pickle('./data/real_data/sic_names.pkl')
ticker_indicator_type = '_full'  #
ticker_indicator = pd.read_pickle('./data/real_data/ticker_indicator' + ticker_indicator_type + '.pkl')
etf = pd.read_pickle('./data/real_data/etf.pkl')
market_beta = pd.read_pickle('./data/real_data/market_beta.pkl')
mean_dollar_volume = pd.read_pickle('./data/real_data/mean_dollar_volume.pkl')
market_cap = pd.read_pickle('./data/real_data/market_cap.pkl')
lead_lag_matrix = pd.read_pickle('./data/pickled_data/lead_lag_matrix' + log_ret_type + correlation_method + restricted
                                 + '.pkl')
gics_sector = pd.read_pickle('./data/real_data/gics_sector.pkl')
gics_name = pd.read_pickle('./data/real_data/gics_name.pkl')

if log_ret_type == '_residuals':
    ticker_indicator = ticker_indicator.drop(columns=['SPY'])

# the stock subset used for lead_lag_matrix.pkl
stock_subset = ticker_indicator.iloc[-1]
stock_subset = stock_subset.index[stock_subset]
log_ret = log_ret.loc[:, stock_subset]

sic = sic.reindex(stock_subset)
gics_sector = gics_sector.reindex(stock_subset)

# computing average vol
volatility = log_ret.std()

# computing hermitian rw clustering
hermitian_clustering = get_ordered_clustering(lead_lag_matrix, num_clusters=10)

#%% COMPUTING LEAD-LAG MATRIX
# restricted = True
#
# start = time.time()
# correlation_method = 'distance'
# if restricted:
#     log_ret = log_ret.loc[:'2006']
#
# lead_lag_matrix = get_lead_lag_matrix(log_ret, method='ccf_auc', max_lag=5, correlation_method=correlation_method)
# end = time.time()
# print(np.round(end - start), ' seconds')
# sns.heatmap(lead_lag_matrix); plt.show()
#
# if restricted:
#     file_name = './data/pickled_data/lead_lag_matrix' + log_ret_type + '_' + correlation_method + '_restricted' + '.pkl'
# else:
#     file_name = './data/pickled_data/lead_lag_matrix' + log_ret_type + '_' + correlation_method + '.pkl'
#
# lead_lag_matrix.to_pickle(file_name)

#%% HEATMAP OF ORDERED LEAD-LAG MATRIX
ordering_method = ['row_sums', 'herm_rw_clustering'][1]
_ = get_ordered_matrix(lead_lag_matrix, method=ordering_method, num_clusters=10)
_ = pd.DataFrame(_)
fig, ax = plt.subplots(figsize=(4, 3))
sns.heatmap(_, cmap='seismic', xticklabels=False, yticklabels=False, ax=ax)
plt.tight_layout()
# plt.savefig('./figures/real_data/heatmap_herm_rw_clustering_ordering.pdf')
plt.show()


#%% SIMILARITY BETWEEN CLUSTERING AND SIC SECTOR MEMBERSHIP


# looking at sic
common_stocks = hermitian_clustering.index.intersection(sic.index)
sic_subset = sic.reindex(index=common_stocks)
tab = pd.crosstab(sic_subset.map(sic_names), hermitian_clustering, rownames=['sector'], colnames=['cluster'])
# computing Jaccard index
fig, ax = plt.subplots(figsize=(5, 3))
sns.heatmap(compute_jaccard(tab), cmap='viridis', ax=ax)
plt.tight_layout()
# plt.savefig('./figures/real_data/jaccard_index_sectors_clusters.pdf')
plt.show()

# Counting the number of equities per sic sector
sic_count = sic.value_counts()
sic_count.index = sic_count.index.map(sic_names)
# sic_count.to_csv('./data/real_data/sic_count.csv')
sic_count.plot.bar(); plt.show()

# Counting the number of equities per data-driven sector
hermitian_count = hermitian_clustering.value_counts()
hermitian_count = hermitian_count.sort_index()
hermitian_count.to_csv('./data/real_data/hermitian_count.csv')
hermitian_count.plot.bar(); plt.show()

# looking at gics sector clustering
gics_sector = gics_sector.dropna()
common_stocks = hermitian_clustering.index.intersection(gics_sector.index)
tab = pd.crosstab(gics_sector.map(gics_name), hermitian_clustering.reindex(index=common_stocks), rownames=['sector'],
                  colnames=['cluster'])
sns.heatmap(compute_jaccard(tab), cmap='viridis')
plt.tight_layout()
plt.show()

# relationship between gics and sic secotrs
tab = pd.crosstab(gics_sector.map(gics_name), sic.reindex(index=common_stocks).map(sic_names), rownames=['gics'],
                  colnames=['sic'])
sns.heatmap(compute_jaccard(tab), cmap='viridis')
plt.tight_layout()
plt.show()


#%% NETWORK OF FLOW BETWEEN CLUSTERS

flow_graph = compute_flow_graph(lead_lag_matrix, hermitian_clustering, normalise=True)
plot_flow_network(flow_graph, figsize=(4, 4))
plt.tight_layout()
plt.savefig('./figures/real_data/flow_graph_network.pdf')
plt.show()
plt.close()


# visualising flow graph when we cluster by sic code
common_stocks = lead_lag_matrix.index.intersection(sic.index)
sic_subset = sic.reindex(index=common_stocks)
clustering = pd.Series(sic_subset)
lead_lag_subset = lead_lag_matrix.reindex(index=common_stocks, columns=common_stocks)

flow_graph = compute_flow_graph(lead_lag_subset, clustering, normalise=True)
flow_graph.rename(index=sic_names, columns=sic_names, inplace=True)
# plot_flow_heatmap(F, clustering)
plot_flow_network(flow_graph)
# plt.savefig('./figures/real_data/sector_flow_graph_network.pdf')
plt.show()


#%% COMPARISON OF HERMITIAN RW AND SIC CLUSTERING FLOW STRENGTH

# hermitian rw clustering
flow_graph_hermitian = compute_flow_graph(lead_lag_matrix, hermitian_clustering, normalise=True)
# sic clustering
clustering_sic = sic
flow_graph_sic = compute_flow_graph(lead_lag_matrix, clustering_sic, normalise=True)

flow_graph_hermitian.stack()[flow_graph_hermitian.stack() > 0].hist(label='Hermitian RW', color='r', alpha=0.8)
flow_graph_sic.stack()[flow_graph_sic.stack() > 0].hist(label='SIC', color='b', alpha=0.8)
plt.xlabel('Net flow between pairs of clusters')
plt.legend()
plt.savefig('./figures/real_data/hermitian_rw_sic_histogram.pdf')
plt.show()


#%% PERMUTATION TEST OF FLOW MATRIX -- RUN

# n_samples = 100
# lead_lag_perm_results = p_map(permutation_mc_lead_lag, [log_ret] * n_samples, num_cpus=7)
# with open('./data/pickled_data/lead_lag_perm_results.pkl', 'wb') as f:
#     pickle.dump(lead_lag_perm_results, f)

#%% PERMUTATION TEST OF FLOW MATRIX -- ANALYSIS

with open('./data/pickled_data/lead_lag_perm_results.pkl', 'rb') as f:
    lead_lag_perm_results = pickle.load(f)

# compute e_val_results from lead_lag_matrix_list
top_n = 1
e_val_results = [compute_largest_e_values_mean(lead_lag_mat, top_n) for lead_lag_mat in lead_lag_perm_results]
# p-value is computed under the null hypothesis so the real data is added to e_val_results
real_e_val_mean = compute_largest_e_values_mean(lead_lag_matrix.values, top_n)
e_val_results.append(real_e_val_mean)
e_val_results = pd.Series(e_val_results)
print((e_val_results >= real_e_val_mean).mean())
e_val_results.hist(); plt.show()
# highly significant p-value

#%% testing association of lead-lag value/clusterings on equity characteristics

# matrix scatter plot
df_features = pd.concat({
    'adj_row_sum': lead_lag_matrix.mean(1),
    'market_beta': market_beta,
    'volume': np.log(mean_dollar_volume),
    'market_cap': np.log(market_cap),
    'volatility': volatility
}, axis=1)
df_features.dropna(axis=0, inplace=True)

pd.plotting.scatter_matrix(df_features); plt.tight_layout(); plt.show()

# np.log(market_cap), np.log(mean_dollar_volume)
feature_name = ['Market capitalisation', 'Average daily volume'][0]
feature = {'Average daily volume': mean_dollar_volume, 'Market capitalisation': market_cap}[feature_name]
feature = feature.reindex(index=lead_lag_matrix.index)

# correlation of feature with lead-lag matrix rowsums
rowsums = lead_lag_matrix.where(lead_lag_matrix > 0, 0).mean(1)
print(rowsums.corr(feature, method='spearman'))
plt.scatter(feature, rowsums); plt.show()

feature.groupby(hermitian_clustering).mean().plot.bar()
plt.ylabel(feature_name)
plt.xlabel('Ordered clusters')
plt.savefig('./figures/real_data/bar_plot_' + feature_name.lower().replace(' ', '_') + '.pdf')
plt.show()

# test_statistic_fn = np.std
# test_statistic_fn = lambda val: val.quantile(0.75) - val.quantile(0.25)
test_statistic_fn = lambda val: val.iloc[:3].mean() - val.iloc[-3:].mean()

permute_feature_cluster_partial = partial(permute_feature_cluster, test_statistic_fn=test_statistic_fn, feature=feature,
                                          clusters=hermitian_clustering)

n_samples = 1000
seeds = range(n_samples)
test_results = list(map(permute_feature_cluster_partial, seeds))
real_test_result = compute_test_statistic(test_statistic_fn, feature, hermitian_clustering)
test_results.append(real_test_result)
test_results = pd.Series(test_results)
test_results.hist(); plt.show()
print((test_results >= real_test_result).mean())


# jaccard similarity between hermitian clustering and quantile clustering
# ordered clusters (0: largest feature value)
feature_clustering = 9 - pd.qcut(feature, 10, labels=False)
tab = pd.crosstab(feature_clustering, hermitian_clustering, rownames=['feature qt'], colnames=['cluster'])
# computing Jaccard index
sns.heatmap(compute_jaccard(tab), cmap='viridis')
plt.tight_layout()
plt.show()

# examining feature association on a more granular basis
lead_lag_matrix_rolling = pd.read_pickle('./data/pickled_data/lead_lag_matrix_rolling_lookback_252_update_freq_252_distance.pkl')
lead_lag_matrix_rolling = lead_lag_matrix_rolling.iloc[1:]
lead_lag_matrix_rolling.index = lead_lag_matrix_rolling.index.year
rowsums_rolling = lead_lag_matrix_rolling.apply(lambda mat: mat.mean(1))
corr_rolling = rowsums_rolling.apply(lambda rank: rank.corr(feature, method='spearman'), axis=1)
corr_rolling.plot.bar()
plt.ylabel('Spearman correlation')
plt.tight_layout()
plt.savefig('./figures/real_data/rolling_rowsum_' + feature_name.lower().replace(' ', '_') + '_corr.pdf')
plt.show()
