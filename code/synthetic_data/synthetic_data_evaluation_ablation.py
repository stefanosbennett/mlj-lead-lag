"""

Evaluate the results of the synthetic data grid experiment

"""

import numpy as np
import pickle5 as pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# load in results here
with open('./data/synthetic_data/lead_lag_results/experiment_ablation.pkl', 'rb') as fh:
    results = pickle.load(fh, encoding='unicode_escape')

results = results.stack('rep')

dependence = ['linear', 'cosine', 'legendre', 'hermite', 'heterogeneous'][0]
T = 250
num_clusters = 10
n_factors = 2  # 2

results = results.xs(100, level='p').xs(dependence, level='dependence') \
    .xs(T, level='T') \
    .xs(num_clusters, level='num_clusters') \
    .xs('distance', level='correlation_method') \
    .xs('ccf_auc', level='method') \
    .xs('hermitian_rw', level='cluster_method') \
    .droplevel('max_lag')

if dependence in ['linear']:
    results = results.droplevel('sigma_z')
elif dependence in ['heterogeneous']:
    results = results.xs(n_factors, level='n_factors')

color_map = ['b', 'g', 'r', 'c', 'm', 'y']
marker_list = [3, 4, 5, 6, 7]

#%% num cluster hyperparameter comparison

df = results.ari_score.groupby(['sigma_eps', 'rep', 'num_clusters_hyperparam']).mean()\
    .unstack(['num_clusters_hyperparam', 'sigma_eps'])
df = df.aggregate(['mean', 'std']).rename_axis(['num_clusters_hyperparam', 'sigma_eps'], axis=1)
df = df.T.unstack(level=0)

xlim = [0, 4]

fig, ax = plt.subplots()
ax.set_xlim(*xlim)
for id, (name, _) in enumerate(df.groupby(axis=1, level=1)):
    print(id, name, _)
    _ = _.droplevel(axis=1, level=1)
    x = _.index
    y = _.loc[:, 'mean']
    ax.plot(x, y, color=color_map[id], marker=marker_list[id], label=name)
    ci = 1.96 * _.loc[:, 'std']
    ax.fill_between(x, y - ci, y + ci, color=color_map[id], alpha=0.1)

plt.legend()
plt.ylabel('ARI')
plt.xlabel('noise level (sigma)')
plt.tight_layout()
plt.savefig('./figures/synthetic_data/ablation/' + dependence + '.pdf')
plt.show()

