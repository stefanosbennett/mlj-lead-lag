'''

Implements the sharpe ratio test from Opdyke 2007

'''

import itertools
import numpy as np
import pandas as pd
from scipy.stats import norm

def sharpe_ratio_test(log_ret):
    '''
    Implements the sharpe ratio test from Opdyke 2007
    :param log_ret: daily log-returns
    :return: annualised sharpe ratio and p-value
    '''
    sharpe = log_ret.mean()/log_ret.std()
    sharpe_annualised = sharpe * np.sqrt(252)

    T = len(log_ret)
    std = log_ret.std()
    skew = (((log_ret - log_ret.mean()) / std) ** 3).mean()
    kurtosis = (((log_ret - log_ret.mean()) / std) ** 4).mean()

    sharpe_se = np.sqrt((1 + sharpe ** 2 / 4 * (kurtosis - 1) - sharpe * skew) / T)

    p_value = 1 - norm.cdf(sharpe/sharpe_se)

    return [sharpe_annualised.round(2), p_value]
