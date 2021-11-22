#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
load and prepare NYSE CRSP equity data

"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functools import partial
from real_data.data_preparation_utils import get_data, transform_sic, filter_by_na_and_volume, calculate_market_beta, \
    remove_market_return

#%% LOADING DATA

directory = ''
data = get_data(directory)

#%% PREPARING DATA

# computing log-returns
# row t: gives the log return going from the close at day t - 1 to close at day t
log_ret = np.log(data.prevAdjClose).ffill().diff().shift(-1).fillna(0)

# extracting SIC code
sic = data.SICCD.ffill().iloc[-1]
sic = transform_sic(sic)
# subsetting to first layer of sic
sic = sic // 1000

# First layer SIC names
sic_names = {
    0: 'Agri., Forest. & Fish.',
    1: 'Mining',
    2: 'Construction',
    3: 'Manufacturing',
    4: 'Trans., Util. & other',
    5: 'Wholesale',
    6: 'Retail',
    7: 'Fin., Ins. & RE',
    8: 'Services',
    9: 'Public Adm.'
}
sic_names = pd.Series(sic_names)

# close_price gives the close at day t
close_price = data.prevAdjClose.shift(-1)
dollar_volume = data.dollar_volume
mean_dollar_volume = dollar_volume.mean()

# rolling filtering
ticker_indicator = filter_by_na_and_volume(close_price, dollar_volume,
                                           lookback=int(40 * 250),
                                           min_number_non_na=int(0.5 * 5 * 250))
ticker_indicator = ticker_indicator.ffill().bfill()

# calculating market beta
log_ret_market_beta_calc = np.log(data.prevAdjClose).ffill().diff().shift(-1) # no fillna(0) for calc
market_beta = log_ret_market_beta_calc.apply(partial(calculate_market_beta, log_ret_spy=log_ret_market_beta_calc.SPY,
                                                     linear_fit='LS'))
market_beta = market_beta.squeeze()
market_beta = market_beta.fillna(market_beta.mean())

# calculating market residual returns
log_ret_res = log_ret_market_beta_calc.apply(partial(remove_market_return, log_ret_spy=log_ret_market_beta_calc.SPY,
                                                     market_beta=market_beta))
log_ret_res = log_ret_res.drop(columns=['SPY'])

# processing market cap
market_cap = data.market_cap * 1000
market_cap = market_cap.mean()

# ETF instruments
etf = ['SPY', 'IWM', 'EEM', 'TLT', 'USO', 'GLD', 'XLF', 'XLB', 'XLK', 'XLV', 'XLI', 'XLU', 'XLY', 'XLP', 'XLE']
etf = pd.Series(etf)

#%% loading and processing gics sector

gics = pd.read_csv('./data/real_data/gics.csv', index_col='datadate')
gics.index = pd.DatetimeIndex(gics.index.map(lambda val: str(val)))

def get_gics_sector(series):

    mode = pd.Series.mode(series)

    if len(mode) == 1:
        return int(mode.iloc[0])
    elif len(mode) == 0:
        return np.nan
    else:
        raise ValueError('Series has multiple modes')

gics_sector = gics.groupby('tic').gsector.agg(get_gics_sector)
# doesn't fully overlap with crsp data
gics_sector = gics_sector.reindex(index=log_ret.columns)

gics_name = {
    10: 'Energy',
    15: 'Materials',
    20: 'Industrials',
    25: 'Consumer Discretionary',
    30: 'Consumer Staples',
    35: 'Health Care',
    40: 'Financials',
    45: 'Information Technology',
    50: 'Communications Services',
    55: 'Utilities',
    60: 'Real Estate'
}
gics_name = pd.Series(gics_name)

#%% saving pickled objects
log_ret.to_pickle('./data/real_data/log_returns.pkl')
log_ret_res.to_pickle('./data/real_data/log_returns_residuals.pkl')
sic.to_pickle('./data/real_data/sic.pkl')
sic_names.to_pickle('./data/real_data/sic_names.pkl')
ticker_indicator.to_pickle('./data/real_data/ticker_indicator.pkl')
# ticker_indicator.to_pickle('./data/real_data/ticker_indicator_full.pkl')
etf.to_pickle('./data/real_data/etf.pkl')
market_beta.to_pickle('./data/real_data/market_beta.pkl')
mean_dollar_volume.to_pickle('./data/real_data/mean_dollar_volume.pkl')
market_cap.to_pickle('./data/real_data/market_cap.pkl')
gics_sector.to_pickle('./data/real_data/gics_sector.pkl')
gics_name.to_pickle('./data/real_data/gics_name.pkl')
