"""
Functions for generating synthetic time series data and loading and processing real data

"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functools import partial
from scipy.optimize import bisect


def generate_mean_shift_changepoint(T, means=[0, 1], std_eps=0.25):
    """
    Generate data from a simple mean-shift (equally spaced changepoints) model

    """

    # number of regimes
    N = len(means)
    # include remainder in the final regime
    regime_lengths = (N - 1) * [T // N] + [(T // N) + (T % N)]
    means_sequence = np.repeat(means, regime_lengths)
    means_sequence = pd.Series(means_sequence)

    y = means_sequence + np.random.normal(loc=0, scale=std_eps, size=T)

    return y


#%% AR models from Kuznetsov et al. (2018)

# T = 3000 used in paper.


def generate_ar_model(T, generate_coefs_fn, std_eps=0.05, **generate_coefs_kwargs):

    coefs = generate_coefs_fn(T, **generate_coefs_kwargs)

    eps = np.random.normal(loc=0, scale=std_eps, size=T)
    y = np.zeros(shape=T)

    for t in range(T):
        y[t] = coefs[t] * y[t - 1] + eps[t]

    y = pd.Series(y)

    return y


def generate_ads1_coefs(T, n_regimes=3):
    """
    There are 2 different regimes in this data gen setting.
    (n_regimes - 1) gives the number of times we switch between each of the 2 regimes.
    """

    rep_len = T // n_regimes
    regime = np.repeat(range(n_regimes), [rep_len] * (n_regimes - 1) + [rep_len + (T % rep_len)])
    regime = regime % 2

    coefs = 2 * regime - 1
    coefs = -0.9 * coefs

    return coefs


def generate_ads2_coefs(T):

    coefs = np.arange(1, T + 1)
    coefs = 1 - (coefs / (T/2))

    return coefs


def generate_ads3_coefs(T, remain_prob=0.99995):

    i = np.zeros(T, dtype=int)
    i[0] = np.random.binomial(1, 0.5, 1)

    for t in range(1, T):
        curr_state = i[t - 1]
        tau = np.flatnonzero(i[:t] - curr_state)

        if len(tau) < 1:
            tau = -1
        else:
            tau = tau[-1]

        tau = (t - 1) - tau

        p = (1 - curr_state) * (1 - (remain_prob) ** tau) + curr_state * ((remain_prob) ** tau)

        i[t] = np.random.binomial(1, p, 1)

    alphas = np.array([-0.5, 0.9])
    coefs = alphas[i]

    return coefs


def generate_ads4_coefs(T):

    coefs = np.repeat(-0.5, T)

    return coefs


generate_ads1 = partial(generate_ar_model, generate_coefs_fn=generate_ads1_coefs)
generate_ads2 = partial(generate_ar_model, generate_coefs_fn=generate_ads2_coefs)
generate_ads3 = partial(generate_ar_model, generate_coefs_fn=generate_ads3_coefs)
generate_ads4 = partial(generate_ar_model, generate_coefs_fn=generate_ads4_coefs)


#%% Real data

def generate_real_data(problem, ticker, data_set):

    directory = './data/real_data/' + data_set + '/'

    if problem in ['log_ret', 'daily_vol']:
        data_directory = directory + problem
        data_directory += '.pkl'
    else:
        raise NotImplementedError('problem type not implemented')

    data = pd.read_pickle(data_directory)
    date_range = pd.read_pickle(directory + 'date_range.pkl')

    if data_set in ['crsp/etf', 'crsp/vol']:
        data = data.loc[:, ticker]
        data = data.loc[date_range.loc[ticker].loc['start']:date_range.loc[ticker].loc['end']]

    elif data_set == 'crsp/factor':
        data = data.loc[:, [ticker, 'mktrf', 'smb', 'hml']]
        data = data.loc[date_range.loc[ticker].loc['start']:date_range.loc[ticker].loc['end']]
    else:
        raise NotImplementedError('problem type not implemented')
    return data


#%% functions to calculate and reverse engineer expected time spent in a state for the ads3 example

def expected_time(remain_prob, max_t=1e7):

    t = np.arange(1, max_t)
    summands = t * (1 - remain_prob ** t) * remain_prob ** ((t - 1) * t/2)

    return summands.sum()


def solve_for_remain_prob(time_required, max_t=1e8):  # 1e7

    remain_prob = bisect(lambda remain_prob: expected_time(remain_prob, max_t) - time_required,
                         1e-15, 1 - 1e-15)

    return remain_prob


if __name__ == 'main':
    solve_for_remain_prob = np.vectorize(solve_for_remain_prob, max_t=1e8)
    solve_for_remain_prob([2000])  # [1000, 2000, 300, 30]
    # 0.99999843, 0.9999996073001958, 0.99998255, 0.99825543
    np.vectorize(expected_time)([0.99999843, 0.9999996073001958, 0.99998255, 0.99825543], max_t=1e8)
