"""
Script to pre-process CRSP equity return data for experiments

"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from functools import partial
from real_data_preparation.data_preparation_utils import get_data, filter_by_na_and_volume, compute_date_range

#%% LOADING DATA

directory = ''
data = get_data(directory)

#%% PREPARING DATA

# computing log-returns
# row t: gives the log return going from the close at day t - 1 to close at day t
log_ret = np.log(data.prevAdjClose).ffill().diff().shift(-1).fillna(0)

# close_price gives the close at day t
close_price = data.prevAdjClose.shift(-1)
dollar_volume = data.dollar_volume
mean_dollar_volume = dollar_volume.mean()

# rolling filtering
ticker_indicator = filter_by_na_and_volume(close_price, dollar_volume,
                                           lookback=int(40 * 250),
                                           min_number_non_na=int(10 * 250),  # int(0.5 * 5 * 250)
                                           mean_daily_dollar_vol_rank=50)
ticker_indicator = ticker_indicator.ffill().bfill()

# ticker_indicator.iloc[-1].sum()
# ticker_indicator.sum(1).plot(); plt.show()

largest_stocks = ticker_indicator.iloc[-1].index[ticker_indicator.iloc[-1]]
log_ret.loc[:, largest_stocks].cumsum().plot(); plt.show()

log_ret = log_ret.loc[:, largest_stocks]

log_ret = log_ret.where(log_ret.abs() < 1, 0)

# np.exp(log_ret).cumprod().plot(); plt.show()
# log_ret.cumsum().plot(); plt.show()

date_range = compute_date_range(close_price)

# creating ticker_list
ticker_list = log_ret.columns.tolist()

#%% saving pickled objects
data_directory = './data/real_data/crsp/equity/'
log_ret.to_pickle(data_directory + 'log_returns.pkl')
ticker_indicator.to_pickle(data_directory + 'ticker_indicator.pkl')
date_range.to_pickle(data_directory + 'date_range.pkl')

with open(data_directory + 'ticker_list.pkl', 'wb') as f:
    pickle.dump(ticker_list, f)
