"""

Generate synthetic datasets with lead-lag structure

"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats
from numpy.polynomial import Legendre, HermiteE


def generate_pearson_7_rv(T, sigma, excess_kurtosis):
    """Generate T samples from a Pearson VII distribution with standard deviation sigma
    and excess kurtosis (= kurtosis - 3) equal to excess_kurtosis"""

    assert excess_kurtosis > 0, 'require positive excess kurtosis'

    m = 5/2 + 3/excess_kurtosis
    alpha = sigma * np.sqrt(2 * m - 3)

    # reparameterising to a scaled student's t distribution
    nu = 2 * m - 1
    student_t_std = alpha / np.sqrt(nu)

    rvs = stats.t.rvs(loc=0, scale=student_t_std, df=nu, size=T)

    return rvs


def get_clustered_lead_lag_returns(T, p, num_clusters, dependence, **distribution_params):

    cluster_lag = np.arange(0, num_clusters)

    lags = np.repeat(cluster_lag, [p // num_clusters] * (num_clusters - 1) + [(p // num_clusters) + (p % num_clusters)])
    lags = pd.Series(lags)

    x = np.zeros((T, p))
    x = pd.DataFrame(x)

    if dependence == 'linear':
        assert 'sigma_z' in distribution_params, 'sigma_z val must be in distribution_params'
        assert 'sigma_eps' in distribution_params, 'sigma_eps val must be in distribution_params'

        z = np.random.normal(loc=0, scale=distribution_params['sigma_z'], size=T)
        z = pd.Series(z)

        for id in range(p):
            lag = lags[id]
            x.loc[:, id] = z.shift(lag).fillna(0) + np.random.normal(loc=0, scale=distribution_params['sigma_eps'], size=T)

    elif dependence == 'multiplicative':
        assert 'excess_kurtosis' in distribution_params, 'excess_kurtosis val must be in distribution_params'
        # correlation at lag l between squared time series i (time t) and j (time t-l) should be:
        # 1/(2 + excess_kurtosis), if l = lag[i] - lag[j],
        # 0,                             otherwise
        # (linear cross-correlation is 0)

        z = np.random.normal(loc=0, scale=1, size=T)
        z = pd.Series(z)

        for id in range(p):
            lag = lags[id]
            eps = generate_pearson_7_rv(T=T, sigma=1, excess_kurtosis=distribution_params['excess_kurtosis'])

            x.loc[:, id] = z.shift(lag).fillna(0) * eps

    elif dependence == 'cosine':
        assert 'sigma_eps' in distribution_params, 'sigma_eps must be in distribution_params'

        z = np.random.uniform(low=-np.pi, high=np.pi, size=T)
        z = pd.Series(z)

        # linear correlation at any lag between two time series is 0,
        # using the fact that cos(mx), m = 1, 2, ... is an orthonormal basis over [-pi, pi].
        # Non-linear correlation is only zero at lag l = lag[i] - lag[j] due to independence
        # of z's across time
        for id in range(p):
            lag = lags[id]
            m = lag + 1

            eps = np.random.normal(loc=0, scale=distribution_params['sigma_eps'], size=T)

            x.loc[:, id] = 1/np.sqrt(np.pi) * np.cos(m * z.shift(m).fillna(0)) + eps

    elif dependence == 'legendre':
        assert 'sigma_eps' in distribution_params, 'sigma_eps val must be in distribution_params'

        z = np.random.uniform(low=-1, high=1, size=T)
        z = pd.Series(z)

        # using the orthonormality of the Legendre polynomials
        for id in range(p):
            lag = lags[id]
            m = lag + 2

            eps = np.random.normal(loc=0, scale=distribution_params['sigma_eps'], size=T)

            polynomial = Legendre([0] * m + [1])
            x.loc[:, id] = polynomial(z.shift(m).fillna(0)) + eps

    elif dependence == 'hermite':
        assert 'sigma_eps' in distribution_params, 'sigma_eps val must be in distribution_params'

        z = np.random.normal(loc=0, scale=1, size=T)
        z = pd.Series(z)

        # using the orthonormality of the hermite polynomials wrt standard normal probability function
        for id in range(p):
            lag = lags[id]
            m = lag + 2

            eps = np.random.normal(loc=0, scale=distribution_params['sigma_eps'], size=T)

            polynomial = HermiteE([0] * m + [1])
            x.loc[:, id] = 1/np.sqrt(np.math.factorial(m)) * polynomial(z.shift(m).fillna(0)) + eps

    elif dependence == 'heterogeneous':
        assert 'sigma_eps' in distribution_params, 'sigma_eps val must be in distribution_params'
        assert 'n_factors' in distribution_params, 'n_factors val must be in distribution_params'

        n_time_series = p // distribution_params['n_factors']

        for factor_id in range(distribution_params['n_factors']):
            x_, lags_ = get_clustered_lead_lag_returns(T=T, p=n_time_series,
                                                       num_clusters=num_clusters, dependence='linear',
                                                       sigma_z=1, sigma_eps=distribution_params['sigma_eps'])

            x.iloc[:, (factor_id * n_time_series):((factor_id + 1) * n_time_series)] = x_.values
            # lags becomes a series of tuples (factor_id, lag)
            lags.iloc[(factor_id * n_time_series):((factor_id + 1) * n_time_series)] = lags_.map(
                lambda lag_val: (factor_id, lag_val)).values

    else:
        raise NotImplementedError('dependence is not implemented')

    return x, lags

