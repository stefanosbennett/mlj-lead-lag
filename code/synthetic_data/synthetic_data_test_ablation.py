"""

Ablation test for number of cluster hyperparameters on lead-lag detection and clustering pipeline

"""

import os
import numpy as np
import pandas as pd
import multiprocessing as mp
import matplotlib.pyplot as plt
from functools import partial
from sklearn.model_selection import ParameterGrid
from p_tqdm import p_map
from synthetic_data.synthetic_data_test_utils import run_grid_experiment

# checking if SLURM multiprocessing
try:
    ncpu = os.environ["SLURM_JOB_CPUS_PER_NODE"]
    ncpu = sum(int(num) for num in ncpu.split(','))
    print('number of SLURM CPUS: ', ncpu)
except KeyError:
    # ncpu = 1
    ncpu = 6

#%% grid evaluation experiment

T_range = [250]  # 2p, 5p
p_range = [100]  # 20
num_clusters_range = [10]  # 2, 5, 10

sigma_eps_range = np.concatenate((np.linspace(0, 1, 6), np.linspace(2, 4, 3)))

data_param_grid_dict = [
    dict(dependence=['linear'], T=T_range, p=p_range, num_clusters=num_clusters_range,
         sigma_z=[1], sigma_eps=sigma_eps_range),
    dict(dependence=['cosine'], T=T_range, p=p_range, num_clusters=num_clusters_range,
         sigma_eps=sigma_eps_range),
    dict(dependence=['legendre'], T=T_range, p=p_range, num_clusters=num_clusters_range,
         sigma_eps=sigma_eps_range),
    dict(dependence=['hermite'], T=T_range, p=p_range, num_clusters=num_clusters_range,
         sigma_eps=sigma_eps_range),
    dict(dependence=['heterogeneous'], T=T_range, p=p_range, num_clusters=num_clusters_range,
         sigma_eps=sigma_eps_range, n_factors=[2])
]

correlation_method_range = ['distance']

method_param_grid_dict = [
    dict(method=['ccf_auc'], max_lag=[5], correlation_method=correlation_method_range)
    # dict(method=['ccf_at_max_lag'], max_lag=[5], correlation_method=correlation_method_range)
]

cluster_param_grid_dict = dict(
    cluster_method=['hermitian_rw'],
    num_clusters_hyperparam=[5, 9, 10, 11, 15]
)

data_param_grid = list(ParameterGrid(data_param_grid_dict))
method_param_grid = list(ParameterGrid(method_param_grid_dict))
cluster_param_grid = list(ParameterGrid(cluster_param_grid_dict))

results_frame_list = []

num_reps = ncpu  # 48
seeds = range(num_reps)
# testing code
# subset = 2
# data_param_grid = data_param_grid[:subset]

run_grid_experiment_partial = partial(run_grid_experiment, data_param_grid=data_param_grid,
                                      method_param_grid=method_param_grid, cluster_param_grid=cluster_param_grid)

results_frame_list = p_map(run_grid_experiment_partial, seeds, num_cpus=ncpu)

results_frame = pd.concat(results_frame_list, axis=1, keys=range(num_reps), names=['rep', 'scores'])

# saving results
experiment_name = 'experiment_ablation'
# filename if running on desktop
# filename = '../../data/synthetic_data/lead_lag_results/' + experiment_name + '.pkl'
# filename if running on SLURM
filename = './data/synthetic_data/lead_lag_results/' + experiment_name + '.pkl'

os.makedirs(os.path.dirname(filename), exist_ok=True)
results_frame.to_pickle(filename)

