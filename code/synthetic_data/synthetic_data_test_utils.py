'''
Functions for running experiments from synthetic_data_test script

'''

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import adjusted_rand_score
from models.lead_lag_measure import get_lead_lag_matrix
from synthetic_data.synthetic_data_generation import get_clustered_lead_lag_returns
from synthetic_data.lead_lag_eval_metrics import rank_eval, prop_correctly_classified_from_lags_vector
import models.herm_matrix_algo as herm_matrix_algo
import models.co_clustering as co_clustering
import models.symmetrisation as symmetrisation


def run_experiment(returns, lags, method, **method_kwargs):

    # compute lead-lag matrix
    lead_lag_matrix = get_lead_lag_matrix(returns, method=method, **method_kwargs)

    # evaluation
    eval_metrics = {'rank_score': rank_eval(lead_lag_matrix, lags),
                    'prop_score': prop_correctly_classified_from_lags_vector(lead_lag_matrix, lags)}

    return eval_metrics


def compute_lead_lag_metrics(lead_lag_matrix, lags, dependence):
    # evaluate lead-lag matrix
    if dependence == 'heterogeneous':
        rank_score = np.nan
        prop_score = np.nan
    else:
        rank_score = rank_eval(lead_lag_matrix, lags)
        prop_score = prop_correctly_classified_from_lags_vector(lead_lag_matrix, lags)

    eval_metrics = {
        'rank_score': rank_score,
        'prop_score': prop_score
    }

    return eval_metrics


def get_cluster_labels(lead_lag_matrix, num_clusters_hyperparam, cluster_method):
    if cluster_method == 'hermitian':
        cluster_labels = herm_matrix_algo.get_clustering(lead_lag_matrix, num_clusters_hyperparam, None)
    elif cluster_method == 'hermitian_rw':
        cluster_labels = herm_matrix_algo.get_clustering(lead_lag_matrix, num_clusters_hyperparam, 'rw')
    elif cluster_method == 'co_clustering_left':
        cluster_labels = co_clustering.get_clustering(lead_lag_matrix, num_clusters_hyperparam,
                                                      cluster_edge_direction='left')
    elif cluster_method == 'co_clustering_right':
        cluster_labels = co_clustering.get_clustering(lead_lag_matrix, num_clusters_hyperparam,
                                                      cluster_edge_direction='right')
    elif cluster_method == 'naive':
        cluster_labels = symmetrisation.naive(lead_lag_matrix, num_clusters_hyperparam)
    elif cluster_method == 'degree_disc_bib':
        cluster_labels = symmetrisation.degree_disc_bib(lead_lag_matrix, num_clusters_hyperparam)
    else:
        raise NotImplementedError

    return cluster_labels


def compute_clustering_metrics(cluster_labels, true_cluster_labels, dependence):

    if dependence == 'heterogeneous':
        # encode each (factor_id, lag) tuple uniquely
        true_cluster_labels = pd.Categorical(true_cluster_labels).codes

    eval_metrics = {
        'ari_score': adjusted_rand_score(true_cluster_labels, cluster_labels)
    }

    return eval_metrics


def tabulate_to_results_frame(results_list, experiment_grid):

    index = pd.DataFrame(experiment_grid)
    index = pd.MultiIndex.from_frame(index)

    results_frame = pd.DataFrame(results_list, index=index)

    return results_frame


def run_grid_experiment(seed, data_param_grid, method_param_grid, cluster_param_grid):
    results_list = []
    experiment_grid = []

    # need to set a different random seed each for each of the processes when multiprocessing
    np.random.seed(seed)

    with tqdm(total=len(data_param_grid) * len(method_param_grid) * len(cluster_param_grid)) as p_bar:
        for data_param in data_param_grid:
            returns, lags = get_clustered_lead_lag_returns(**data_param)

            for method_param in method_param_grid:
                lead_lag_matrix = get_lead_lag_matrix(data=returns, **method_param)
                lead_lag_results = compute_lead_lag_metrics(lead_lag_matrix, lags,
                                                            dependence=data_param['dependence'])

                for cluster_param in cluster_param_grid:

                    if 'num_clusters_hyperparam' not in cluster_param:
                        # set clustering algorithm hyperparameter to the true number of clusters
                        if data_param['dependence'] == 'heterogeneous':
                            cluster_param['num_clusters_hyperparam'] = data_param['n_factors'] * \
                                                                       data_param['num_clusters']
                        else:
                            cluster_param['num_clusters_hyperparam'] = data_param['num_clusters']

                    try:
                        cluster_labels = get_cluster_labels(lead_lag_matrix, **cluster_param)

                        clustering_results = compute_clustering_metrics(cluster_labels=cluster_labels,
                                                                        true_cluster_labels=lags,
                                                                        dependence=data_param['dependence'])

                        results_list.append(dict(list(lead_lag_results.items()) + list(clustering_results.items())))

                        experiment_grid.append(dict(list(data_param.items()) + list(method_param.items()) +
                                                    list(cluster_param.items())))
                    except:
                        print('exception occurred with ', data_param, method_param, cluster_param)
                    finally:
                        p_bar.update(1)

    results_frame = tabulate_to_results_frame(results_list, experiment_grid)

    return results_frame
