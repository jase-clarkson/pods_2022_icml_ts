"""
Script to pre-process CRSP ETF return data for experiments

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from real_data_preparation.data_preparation_utils import get_data, compute_date_range

#%% LOADING NYSE CRSP equity data

# loading ETF instruments
etf = ['SPY', 'IWM', 'EEM', 'TLT', 'USO', 'GLD', 'XLF', 'XLB', 'XLK', 'XLV', 'XLI', 'XLU', 'XLY', 'XLP', 'XLE']
etf = pd.Series(etf)

etf_description = pd.Series({
    'SPY': 'SPDR S&P 500 Trust ETF',
    'IWM': 'iShares Russell 2000 ETF',
    'EEM': 'iShares MSCI Emerging Markets ETF',
    'TLT': 'iShares 20 Plus Year Treasury Bond ETF',
    'USO': 'United States Oil Fund LP',
    'GLD': 'SPDR Gold Trust',
    'XLF': 'Financial Select Sector SPDR Fund',
    'XLB': 'Materials Select Sector SPDR Fund',
    'XLK': 'Technology Select Sector SPDR Fund',
    'XLV': 'Health Care Select Sector SPDR',
    'XLI': 'Industrial Select Sector SPDR Fund',
    'XLU': 'Utilities Select Sector SPDR Fund',
    'XLY': 'Consumer Discretionary Select Sector SPDR Fund',
    'XLP': 'Consumer Staples Select Sector SPDR Fund',
    'XLE': 'Energy Select Sector SPDR Fund'
})

directory = ''
data = get_data(directory, etf.to_list())

#%% PREPARING DATA

# computing log-returns
# row t: gives the log return going from the close at day t - 1 to close at day t
log_ret = np.log(data.prevAdjClose).ffill().diff().shift(-1).fillna(0)
log_ret.cumsum().plot(); plt.show()
log_ret.plot(); plt.show()
# EEM has corrupted data on two different dates
data.prevAdjClose.EEM.plot(); plt.show()
log_ret.loc[:, 'EEM'][log_ret.abs().EEM > 1] = 0
# IWM has corrupted data on 1 date
log_ret.loc[log_ret.loc[:, 'IWM'].idxmin(), 'IWM'] = 0

# squared daily log-return
squared_log_ret = log_ret ** 2
squared_log_ret.plot(); plt.show()

# absolute daily log_return
abs_log_ret = log_ret.abs()
abs_log_ret.plot(); plt.show()

# daily price range
price_range = data.high.ffill().bfill() - data.low.ffill().bfill()
price_range.plot(); plt.show()
# SPY has corrupted data on 1 date
price_range.loc[data.low.SPY.idxmin(), 'SPY'] = np.nan
# IWM has corrupted data on 1 date
price_range.loc[data.low.IWM.idxmin(), 'IWM'] = np.nan
price_range = price_range.ffill()

price_range.plot(); plt.show()

# computing start date for each instrument
close_price = data.prevAdjClose.shift(-1)
date_range = compute_date_range(close_price)

# checking for missing values
# print(pd.Series({ticker: data.prevAdjClose.loc[date_range.loc[ticker].iloc['start']:date_range.loc[ticker].iloc['end'],
#                          ticker].isna().sum() for ticker in etf}))
# print(pd.Series({ticker: price_range.loc[date_range.loc[ticker].iloc['start']:date_range.loc[ticker].iloc['end'],
#                          ticker].isna().sum() for ticker in etf}))

ticker_list = log_ret.columns.tolist()

#%% saving pickled objects

data_directory = './data/real_data/crsp/etf/'
log_ret.to_pickle(data_directory + 'log_ret.pkl')
squared_log_ret.to_pickle(data_directory + 'squared_log_ret.pkl')
abs_log_ret.to_pickle(data_directory + 'abs_log_ret.pkl')
price_range.to_pickle(data_directory + 'price_range.pkl')
date_range.to_pickle(data_directory + 'date_range.pkl')
etf.to_pickle(data_directory + 'etf.pkl')
etf_description.to_pickle(data_directory + 'etf_description.pkl')

with open(data_directory + 'ticker_list.pkl', 'wb') as f:
    pickle.dump(ticker_list, f)
