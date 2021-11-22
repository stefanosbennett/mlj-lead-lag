"""

Evaluate the results of the synthetic data grid experiment

"""

import numpy as np
import pickle5 as pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# load in results here
experiment_name = 'experiment_3'
with open('./data/synthetic_data/lead_lag_results/' + experiment_name + '.pkl', 'rb') as fh:
    results = pickle.load(fh, encoding='unicode_escape')

results = results.stack('rep')

dependence = ['linear', 'cosine', 'legendre', 'hermite', 'heterogeneous'][0]
T = 250
num_clusters = [2, 5, 10][2]
n_factors = 2  # 2
num_reps = 48  # 48

results = results.xs(100, level='p').xs(dependence, level='dependence')\
    .xs(T, level='T')\
    .xs(num_clusters, level='num_clusters')\
    .droplevel(['max_lag', 'lag'])

if dependence in ['linear']:
    results = results.droplevel('sigma_z')
elif dependence in ['heterogeneous']:
    results = results.xs(n_factors, level='n_factors')

# dropping the hermitian method
results = results.drop('hermitian', level='cluster_method')
# renaming the clustering methods
results.rename({'co_clustering_left': 'DI-SIM left', 'co_clustering_right': 'DI-SIM right',
                'degree_disc_bib': 'Bibliometric', 'hermitian_rw': 'Hermitian RW'},
               level='cluster_method', axis=0, inplace=True)
results.rename(lambda name: 'Naive' if name == 'naive' else name, level='cluster_method', axis=0, inplace=True)

# renaming the correlation method
results.rename({'ccf_at_lag': 'ccf-lag1', 'ccf_auc': 'ccf-auc'}, level='method', axis=0, inplace=True)
results.rename({'distance': 'Distance', 'kendall': 'Kendall', 'mutual_information': 'Mutual information',
                'pearson': 'Pearson'}, level='correlation_method', axis=0, inplace=True)

color_map = ['b', 'g', 'r', 'c', 'm', 'y']
marker_list = [3, 4, 5, 6, 7]

#%% clustering method comparison

df = results.ari_score.groupby(['sigma_eps', 'cluster_method', 'rep']).mean().unstack(['cluster_method', 'rep'])
df = df.stack('rep').unstack('sigma_eps').aggregate(['mean', 'std']).stack(level=1).unstack(level=0)

if dependence == 'linear':
    xlim = [0, 6] if T!=250 else [0, 4]
elif dependence == 'hermite':
    xlim = [0, 2]
elif dependence == 'heterogeneous':
    xlim = [0, 4]
else:
    xlim = [0, 1]

fig, ax = plt.subplots()
ax.set_xlim(*xlim)
for id, (name, _) in enumerate(df.groupby(level=0, axis=1)):
    _ = _.droplevel(axis=1, level=0)
    x = _.index
    y = _.loc[:, 'mean']
    ax.plot(x, y, color=color_map[id], marker=marker_list[id], label=name)
    ci = 1.96 * _.loc[:, 'std']
    ax.fill_between(x, y - ci, y + ci, color=color_map[id], alpha=0.1)

plt.legend()
plt.ylabel('ARI')
plt.xlabel('noise level (sigma)')
plt.tight_layout()
plt.savefig('./figures/synthetic_data/' + dependence + '.pdf')
plt.show()


#%% correlation method comparison

methods_list = ['ccf-lag1', 'ccf-auc', 'signature']
correlation_method_list = results.index.get_level_values('correlation_method').unique().dropna()
score_type = 'prop_score'

if score_type == 'prop_score':
    ylim = [0, 1]
elif score_type == 'ari_score' or score_type == 'rank_score':
    ylim = [0, 1]
else:
    raise NotImplementedError

if dependence == 'linear':
    xlim = [0, 6] if T!=250 else [0, 4]
elif dependence == 'hermite':
    xlim = [0, 2]
else:
    xlim = [0, 1]

fig, axes = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True)
fig.set_size_inches(9, 3)
get_plot_ax = lambda plot_id: (plot_id // 2, plot_id % 2)

for plot_id, method_name in enumerate(methods_list):
    ax = axes[plot_id]
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel('noise level (sigma)')

    if method_name == 'signature':

        df = results.xs(method_name, level='method', axis=0).loc[:, score_type]
        # averaging over cluster method
        df = df.mean(axis=0, level=['sigma_eps', 'rep'])
        df = df.groupby('sigma_eps').aggregate(['mean', 'std'])
        x = df.index
        y = df.loc[:, 'mean']
        ci = 1.96 * df.loc[:, 'std']
        ax.plot(x, y, color=color_map[len(methods_list) + 1], marker=marker_list[len(methods_list) + 1], label=name)
        ax.fill_between(x, y - ci, y + ci, color=color_map[len(methods_list) + 1], alpha=0.1)

    else:

        df = results.xs(method_name, level='method', axis=0).loc[:, score_type]
        # averaging over cluster method
        df = df.mean(axis=0, level=['sigma_eps', 'rep', 'correlation_method'])
        df = df.groupby(['correlation_method', 'sigma_eps']).aggregate(['mean', 'std'])

        for id, (name, _) in enumerate(df.groupby(level='correlation_method', axis=0)):
            _ = _.droplevel(level='correlation_method', axis=0)
            x = _.index
            y = _.loc[:, 'mean']
            ci = 1.96 * _.loc[:, 'std']
            ax.plot(x, y, color=color_map[id], marker=marker_list[id], label=name)
            ax.fill_between(x, y - ci, y + ci, color=color_map[id], alpha=0.1)

    ax.set_ylabel('accuracy')
    ax.title.set_text(method_name)

# getting plot legend from last ax
lines = []
labels = []
for ax in axes.flatten():
    axLine, axLabel = ax.get_legend_handles_labels()
    lines.extend(axLine)
    labels.extend(axLabel)

labels_lines_series = pd.Series(lines, index=labels)
labels_lines_series = labels_lines_series.loc[correlation_method_list.tolist() + ['signature']]

# dropping duplicates
_ = pd.Series(labels_lines_series.index)
_.drop_duplicates(inplace=True)
labels_lines_series = labels_lines_series.iloc[_.index]
fig.legend(labels_lines_series.values.tolist(), labels_lines_series.index.tolist(), loc='lower right')
plt.tight_layout()
plt.savefig('./figures/synthetic_data/' + dependence + '_corr' + '.pdf')
plt.show()


#%% Visualising the interaction of cluster and correlation method components

sigma_eps_dict = {
    'linear': [0, 1, 2, 3],
    'cosine': [0, 0.2, 0.4, 1],
    'legendre': [0, 0.2, 0.4, 0.6000000000000001],
    'hermite': [0, 0.4, 0.6000000000000001, 1],
    'heterogeneous': [0, 0.4, 1, 3]
}
sigma_eps_list = sigma_eps_dict[dependence]

fig, axes = plt.subplots(1, 4, sharex=True, sharey=True, figsize=(12, 3.5))

for ax_id, sigma_eps_val in enumerate(sigma_eps_list):
    ax = axes[ax_id]
    df = results.loc[:, 'ari_score'].mean(level=['sigma_eps', 'method', 'cluster_method'])
    df = df.xs(sigma_eps_val, level='sigma_eps')
    df = df.unstack()
    # df = df.drop(columns=['Naive'])
    df = df.rename_axis('', axis=0)
    df = df.rename_axis('', axis=1)
    sns.heatmap(df, ax=ax, cmap='BuPu', vmin=0, vmax=1, cbar=(ax_id == len(sigma_eps_list) - 1))  # YlGnBu
    ax.title.set_text('sigma=' + str(round(sigma_eps_val, 1)))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right', va='center_baseline')
    if ax_id == 0:
        ax.set_yticklabels(ax.get_yticklabels(), rotation=45, va="center")


# fig.suptitle('ARI: ' + dependence + ' case')
plt.tight_layout()
plt.savefig('./figures/synthetic_data/interaction/' + 'ari_' + dependence + '.pdf')
plt.show()

