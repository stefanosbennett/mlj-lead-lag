"""
Script to run parallel lead-lag permutation computation

"""
import os
import pickle
import numpy as np
import pandas as pd
from functools import partial
from p_tqdm import p_map
from real_data.utils import permutation_mc_lead_lag

log_ret_type = '' # '_residuals'
log_ret = pd.read_pickle('./data/real_data/log_returns' + log_ret_type + '.pkl')
ticker_indicator_type = '_full'  #
ticker_indicator = pd.read_pickle('./data/real_data/ticker_indicator' + ticker_indicator_type + '.pkl')

if log_ret_type == '_residuals':
    ticker_indicator = ticker_indicator.drop(columns=['SPY'])

# the stock subset used for lead_lag_matrix.pkl
stock_subset = ticker_indicator.iloc[-1]
stock_subset = stock_subset.index[stock_subset]
log_ret = log_ret.loc[:, stock_subset]

# checking if SLURM multiprocessing
try:
    ncpu = os.environ["SLURM_JOB_CPUS_PER_NODE"]
    ncpu = sum(int(num) for num in ncpu.split(','))
    print('number of SLURM CPUS: ', ncpu)
except KeyError:
    ncpu = 1

n_samples = 200
seeds = range(n_samples)

permutation_mc_lead_lag = partial(permutation_mc_lead_lag, log_ret=log_ret)
lead_lag_perm_results = p_map(permutation_mc_lead_lag, seeds, num_cpus=ncpu)

with open('./data/pickled_data/lead_lag_perm_results.pkl', 'wb') as f:
    pickle.dump(lead_lag_perm_results, f)
