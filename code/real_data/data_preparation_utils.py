import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.linear_model import TheilSenRegressor
from sklearn.linear_model import LinearRegression


def get_date_from_filename(name):
    return name[:4] + '-' + name[4:6] + '-' + name[6:8]


def process_single_day_data(data_frame):
    """
    close: closing price (unadj)
    pvCLCL: RET (linear return) in Wharton DB
    prevAdjClose: close price, adjusted for splits and dividends
    volume: daily traded volume
    SICCD: most recent SIC code for share

    """
    col_names = ['close', 'pvCLCL', 'prevAdjClose', 'volume', 'SICCD', 'sharesOut']
    data_subset = data_frame.set_index('ticker')
    data_subset = data_subset.loc[:, col_names]
    data_subset.loc[:, 'dollar_volume'] = data_subset.loc[:, 'close'] * data_subset.loc[:, 'volume']
    data_subset.loc[:, 'market_cap'] = data_subset.loc[:, 'close'] * data_subset.loc[:, 'sharesOut']

    data_subset = data_subset.drop(columns=['close', 'volume', 'sharesOut'])
    data_subset = data_subset.rename_axis(columns=['feature'])

    return data_subset


def get_data(directory):
    data_store = dict()

    for entry in tqdm(list(os.walk(directory))):
        if len(entry[1]) == 0:
            for filename in entry[2]:
                if filename[-3:] == 'csv':
                    data_frame = pd.read_csv(entry[0] + '/' + filename)
                    data_frame = process_single_day_data(data_frame)
                    data_store[get_date_from_filename(filename)] = data_frame

    data = pd.concat(data_store).unstack()
    data.index = pd.DatetimeIndex(data.index)
    data = data.sort_index()

    return data


def transform_sic(sic_series):
    sic_transformed = {}

    for ticker, val in sic_series.items():
        try:
            sic_transformed[ticker] = int(val)
        except ValueError:
            sic_transformed[ticker] = np.nan

    sic_transformed = pd.Series(sic_transformed, dtype='Int64')

    return sic_transformed


def filter_by_na_and_volume(close_price, dollar_volume, lookback, min_number_non_na):
    """
    For each time row, checks if a stock:
    - is among the top 500 highest mean daily dollar volume stocks in the lookback window
    - has more than a max number of non-na values

    """

    av_volume = dollar_volume.rolling(window=lookback, min_periods=1).mean()
    av_volume_rank = av_volume.rank(axis=1, ascending=False, method='first', na_option='bottom')
    av_volume_indicator = av_volume_rank <= 500

    non_na_price = (~close_price.isna()).rolling(window=lookback, min_periods=1).sum()
    non_na_price = non_na_price.fillna(0)
    non_na_price_indicator = non_na_price > min_number_non_na

    return av_volume_indicator & non_na_price_indicator


def calculate_market_beta(log_ret_col, log_ret_spy, linear_fit):

    na_indicator = log_ret_spy.isna() | log_ret_col.isna()

    if sum(~na_indicator) < 10:
        return np.nan
    else:
        y = log_ret_col[~na_indicator].values
        X = log_ret_spy[~na_indicator].values.reshape(-1, 1)
        if linear_fit == 'LS':
            model = LinearRegression(fit_intercept=False)
            model.fit(X, y)
        elif linear_fit == 'TheilSen':
            model = TheilSenRegressor(fit_intercept=False, max_subpopulation=int(1e4), max_iter=10)
            model.fit(X, y)

        return model.coef_


def remove_market_return(log_ret_col, log_ret_spy, market_beta):

    na_indicator = log_ret_spy.isna() | log_ret_col.isna()

    log_ret_res = log_ret_col.copy()
    log_ret_res[~na_indicator] = log_ret_res[~na_indicator] - market_beta.loc[log_ret_col.name] \
                                 * log_ret_spy[~na_indicator]
    log_ret_res = log_ret_res.fillna(0)

    return log_ret_res
