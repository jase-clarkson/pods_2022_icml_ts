"""
"""

import numpy as np
from functools import partial
from scipy.optimize import bisect


def compute_alpha_scheme(t, alpha):
    return alpha ** np.arange(t)[::-1]


def compute_rolling_scheme(t, lookback):
    return np.repeat([0, 1], repeats=[max(0, t - lookback), min(lookback, t)])


#%% functions for converting between alpha (or power) and span parameters (as defined in pd.ewma)


@np.vectorize
def span_to_alpha(span):
    return 1 - 2/(span + 1)


@np.vectorize
def alpha_to_span(alpha):
    return int((2/(1 - alpha) - 1))

