#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compute various metrics for lead-lag relationships in time series data

"""

import numpy as np
import pandas as pd
import itertools
import dcor
import statsmodels.api as sm
import multiprocessing as mp
import matplotlib.pyplot as plt
import iisignature
from tqdm import tqdm
from sklearn.feature_selection import mutual_info_classif


def cross_correlation(data_1, data_2, lag, correlation_method='pearson'):
    # compute cross correlation for a given lag of the first arg vs second arg

    data_1 = data_1.shift(lag)
    ### OLD CODE### data = pd.concat([data_1, data_2], axis=1)
    data = np.column_stack((data_1, data_2))
    data = pd.DataFrame(data)
    # drop rows with na values
    if lag >= 0:
        data = data.iloc[lag:, :]
    else:
        data = data.iloc[:lag, :]

    if correlation_method in ['pearson', 'kendall', 'spearman']:
        cross_corr_val = data.corr(method=correlation_method).iloc[0, 1]

    elif correlation_method == 'distance':
        cross_corr_val = dcor.distance_correlation(data.iloc[:, 0], data.iloc[:, 1])

    elif correlation_method == 'mutual_information':
        # implementation as in Fiedor et al. 2014
        quantile_buckets = pd.qcut(data.values.reshape(-1), q=4, labels=False).reshape(-1, 2)
        cross_corr_val = mutual_info_classif(quantile_buckets[:, [0]], quantile_buckets[:, 1], discrete_features=True)[0]

    elif correlation_method == 'squared_pearson':
        cross_corr_val = (data ** 2).corr(method='pearson').iloc[0, 1]

    else:
        raise NotImplementedError('correlation method not implemented')

    return cross_corr_val


def compute_lead_lag(data_subset, method, **kwargs):
    '''
    :param data_subset: dataframe with 2 cols.
    :param method: which lead-lag method to use
    :param kwargs: kwargs for the lead-lag method
                    - for ccf_at_lag: lag, correlation_method
                    - for ccf_auc: max_lag, correlation_method
                    - signature: none
                    - for ccf_at_max_lag: max_lag, correlation_method
    :return how much col 1 of data_subset leads col 2

    '''
    
    if method == 'ccf_at_lag':
        cross_correlation_measure = cross_correlation(data_subset.iloc[:, 0], data_subset.iloc[:, 1], lag=kwargs['lag'],
                                                      correlation_method=kwargs['correlation_method'])
        cross_correlation_measure -= cross_correlation(data_subset.iloc[:, 1], data_subset.iloc[:, 0], lag=kwargs['lag'],
                                                       correlation_method=kwargs['correlation_method'])
        lead_lag_measure = cross_correlation_measure

    elif method == 'ccf_auc':
        lags = np.arange(1, kwargs['max_lag'] + 1)
        lags = np.r_[-lags, lags]
        cross_correlation_measure = dict()
        for lag in lags:
            cross_correlation_measure[lag] = cross_correlation(data_subset.iloc[:, 0], data_subset.iloc[:, 1], lag=lag,
                                                               correlation_method=kwargs['correlation_method'])
        
        cross_correlation_measure = pd.Series(cross_correlation_measure)
        A = cross_correlation_measure[cross_correlation_measure.index > 0]
        A = np.abs(A).sum()
        B = cross_correlation_measure[cross_correlation_measure.index < 0]
        B = np.abs(B).sum()

        lead_lag_measure = np.array([A, B]).max()/(A + B) * np.sign(A - B)
        # alternative normalisation:
        # lead_lag_measure = (A - B)/(A + B)

    elif method == 'signature':
        path = data_subset.cumsum()
        path /= path.std()
        signature = iisignature.sig(path, 2, 1)
        lead_lag_measure = signature[1][1] - signature[1][2]

    elif method == 'ccf_at_max_lag':
        lags = np.arange(1, kwargs['max_lag'] + 1)
        lags = np.r_[-lags, lags]
        cross_correlation_measure = dict()
        for lag in lags:
            cross_correlation_measure[lag] = cross_correlation(data_subset.iloc[:, 0], data_subset.iloc[:, 1], lag=lag,
                                                               correlation_method=kwargs['correlation_method'])

        cross_correlation_measure = pd.Series(cross_correlation_measure)
        # note that this takes into account positive and negative correlations
        leadingness = cross_correlation_measure[cross_correlation_measure.index > 0].abs().max()
        laggingness = cross_correlation_measure[cross_correlation_measure.index < 0].abs().max()

        lead_lag_measure = leadingness - laggingness

    else:
        raise NotImplementedError
        
    return lead_lag_measure


def lead_lag_given_pair(args):
    pair, data, method, kwargs_compute_lead_lag = args
    
    data_subset = data.loc[:, pair]
    
    return compute_lead_lag(data_subset, method=method, **kwargs_compute_lead_lag)


def get_lead_lag_matrix(data, method, **kwargs_compute_lead_lag):
    pair_list = list(itertools.combinations(data.columns, 2))

    lead_lag_measure = list(map(lead_lag_given_pair, itertools.product(pair_list, [data], [method], [kwargs_compute_lead_lag])))
    lead_lag_measure = {pair: lead_lag_measure[num] for num, pair in enumerate(pair_list)}
    lead_lag_measure = pd.Series(lead_lag_measure).unstack()

    lead_lag_measure = lead_lag_measure.reindex(index=data.columns, columns=data.columns)
    lead_lag_measure = lead_lag_measure.fillna(0)
    lead_lag_measure = lead_lag_measure - lead_lag_measure.T

    return lead_lag_measure
