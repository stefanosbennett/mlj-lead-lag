"""
Functions for strat

"""

import os
import numpy as np
import pandas as pd
import itertools
import multiprocessing as mp
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import GridSearchCV
from real_data.rolling_utils import get_flow_graph_rolling
from real_data.sharpe_ratio_test import sharpe_ratio_test


class LinearARModel:

    def __init__(self, train_log_ret_x, train_log_ret_y, reversion_days, smooth_days):
        self.log_ret_y = train_log_ret_y
        self.log_ret_x = train_log_ret_x
        self.reversion_days = reversion_days
        self.model = None
        self.smooth_days = smooth_days

    def get_design_matrix(self, log_ret_x):
        log_ret_x = log_ret_x.ewm(span=self.smooth_days, min_periods=0).mean()
        X = {lag: log_ret_x.shift(lag).fillna(0) for lag in range(1, self.reversion_days + 1)}
        X = pd.concat(X, axis=1)

        return X

    def fit(self):
        X = self.get_design_matrix(self.log_ret_x)
        y = self.log_ret_y

        X = X.values
        y = y.values

        self.model = LinearRegression(fit_intercept=False)
        self.model.fit(X, y)

        return self.model

    def predict(self, log_ret_x):
        log_ret_x = log_ret_x.ewm(span=self.smooth_days, min_periods=0).mean()
        x = {lag: log_ret_x.iloc[-lag] for lag in range(1, self.reversion_days + 1)}
        x = pd.Series(x)
        x = x.values.reshape(1, self.reversion_days)

        return np.float(self.model.predict(x).squeeze())


class VAR_model:

    def __init__(self, log_ret, n_lags=5):
        self.log_ret = log_ret
        self.n_lags = n_lags
        self.model = None

    def get_design_matrix(self, log_ret_x):
        X = {lag: log_ret_x.shift(lag).fillna(0) for lag in range(1, self.n_lags + 1)}
        X = pd.concat(X, axis=1)
        X = X.values

        return X

    def fit(self):
        # getting features and labels
        X = self.get_design_matrix(self.log_ret)
        y = self.log_ret.values

        self.model = Lasso(max_iter=100)
        self.model = GridSearchCV(self.model, param_grid={'alpha': np.logspace(-4, 0, 5)}, n_jobs=5, cv=5)
        self.model.fit(X, y)

        return self.model

    def predict(self, log_ret_x):
        """
        :param log_ret_x: dataframe of historical values for all time series
        :return: vector of predictions
        """
        x = {lag: log_ret_x.iloc[-lag] for lag in range(1, self.n_lags + 1)}
        x = pd.concat(x, axis=0)
        x = x.values.reshape(1, -1)

        predictions = self.model.predict(x).squeeze()
        predictions = pd.Series(predictions)
        predictions = predictions.values

        return predictions


def VAR_signal(log_ret_subset, VAR_model):
    predictions = VAR_model.predict(log_ret_subset)
    predictions = pd.Series(predictions, index=log_ret_subset.columns)

    return predictions


def flow_graph_process(flow_graph, cut_off_quantile, flow_graph_sign):
    """process flow_graph for signal construction"""

    flow_graph = np.maximum(flow_graph, 0)
    flow_graph = pd.DataFrame(flow_graph)

    cut_off = flow_graph.stack()[flow_graph.stack() > 0].quantile(cut_off_quantile)
    flow_graph = flow_graph.where(flow_graph > cut_off, 0)

    flow_graph = flow_graph.values

    if flow_graph_sign:
        flow_graph = np.sign(flow_graph)

    return flow_graph


def signal_aggregate(flow_graph, clusters, cluster_signals, signal_sign):
    """aggregate signals from different clusters"""
    signal = (flow_graph * cluster_signals).mean(0)

    if signal_sign:
        signal = np.sign(signal)

    signal = pd.Series(signal)
    signal = clusters.map(lambda cluster: signal[cluster])

    return signal


def weighted_flow_signal(log_ret_subset, clusters, flow_graph, reversion_days, cut_off_quantile, flow_graph_sign,
                         signal_sign, cluster_models, smooth_days):

    # compute signal for each cluster-cluster pair
    cluster_returns = log_ret_subset.iloc[-(smooth_days + reversion_days):].groupby(clusters, axis=1).mean()
    cluster_signals = cluster_models.apply(lambda row_models: [model.predict(cluster_returns.loc[:, row_models.name]) for model in row_models], axis=0)

    # process flow graph
    flow_graph = flow_graph_process(flow_graph, cut_off_quantile, flow_graph_sign)
    # aggregate signals
    signal = signal_aggregate(flow_graph, clusters, cluster_signals, signal_sign)

    return signal


def run_strat(log_ret, clusters_rolling, flow_graph_rolling, reversion_days, signal_fn, **signal_kwargs):
    position = dict()
    update_date_index = list(flow_graph_rolling.index)
    # reverse so that we pop dates in increasing chronological order
    update_date_index = update_date_index[::-1]
    update_date = update_date_index.pop()

    day_index = log_ret.index
    day_index = day_index[day_index > update_date]

    for date in tqdm(day_index):

        log_ret_subset = log_ret.loc[:date, :]

        if date > update_date:
            # updating clusters and flow_graph
            clusters = clusters_rolling.loc[update_date]
            flow_graph = flow_graph_rolling.loc[update_date]

            if signal_fn.__name__ == 'weighted_flow_signal':
                # updating cluster-cluster model
                cluster_returns = log_ret_subset.groupby(clusters, axis=1).mean()
                cluster_models = pd.DataFrame()

                for leading, lagging in itertools.product(cluster_returns.columns, repeat=2):
                    leading_log_ret = cluster_returns.loc[:, leading]
                    lagging_log_ret = cluster_returns.loc[:, lagging]
                    # using past year of data
                    leading_log_ret = leading_log_ret.iloc[-250:]
                    lagging_log_ret = lagging_log_ret.iloc[-250:]

                    model = LinearARModel(leading_log_ret, lagging_log_ret, reversion_days, signal_kwargs['smooth_days'])
                    model.fit()
                    cluster_models.loc[leading, lagging] = model

                signal_kwargs['cluster_models'] = cluster_models

            elif signal_fn.__name__ == 'VAR_signal':
                # LASSO-VAR model with 5 lag
                # using past year of data
                log_ret_subset = log_ret_subset.iloc[-250:]
                # reversion_days is the number of lags to use
                model = VAR_model(log_ret_subset, n_lags=reversion_days)
                model.fit()

                signal_kwargs['VAR_model'] = model

            if len(update_date_index) > 0:
                update_date = update_date_index.pop()

        if signal_fn.__name__ == 'weighted_flow_signal':

            position[date] = signal_fn(log_ret_subset=log_ret_subset,
                                       clusters=clusters,
                                       flow_graph=flow_graph,
                                       reversion_days=reversion_days,
                                       **signal_kwargs)

        elif signal_fn.__name__ == 'VAR_signal':

            position[date] = signal_fn(log_ret_subset=log_ret_subset,
                                       VAR_model=signal_kwargs['VAR_model'])

    position = pd.concat(position).unstack(-1)
    position = position.shift(1)
    position = position.ffill().fillna(0)
    # setting holding period to value of reversion_days
    # position = position.asfreq(str(reversion_days) + 'B').resample('B').ffill()

    pnl = position * log_ret

    return position, pnl


def sound_notification(duration=1, freq=300):
    """Make sound after executing code"""
    duration = 2
    os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq))


def perm_test(seed, log_ret, lead_lag_matrix_rolling, clusters_rolling):

    np.random.seed(seed)
    clusters_rolling_perm = clusters_rolling.apply(lambda row: pd.Series(row.iloc[np.random.permutation(range(len(row)))].\
                                                                         values, index=row.index), axis=1)
    flow_graph_rolling_perm = get_flow_graph_rolling(lead_lag_matrix_rolling, clusters_rolling_perm, normalise=True)

    position, pnl = run_strat(log_ret, clusters_rolling_perm, flow_graph_rolling_perm,
                              reversion_days=1,
                              cut_off_quantile=0.9,
                              signal_fn=weighted_flow_signal,
                              flow_graph_sign=True,
                              signal_sign=True,
                              smooth_days=4)
    position_normalised = position.apply(lambda col: col / pnl.sum(1).rolling(21, min_periods=5).std().shift(1).ffill().\
                                         bfill())
    pnl_linear = position_normalised * (np.exp(log_ret) - 1)

    return sharpe_ratio_test(pnl_linear.sum(1).loc['2002':])[0]
