"""
Create dataset of volatilities from CRSP ETF data
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle


#%% loading crsp etf data

etf_log_ret = pd.read_pickle('./data/real_data/crsp/etf/log_ret.pkl')
etf = pd.read_pickle('./data/real_data/crsp/etf/etf.pkl')
date_range = pd.read_pickle('./data/real_data/crsp/etf/date_range.pkl')
#%% daily vol
# volatility processing
daily_vol = etf_log_ret.abs().iloc[:, :12].drop(['GLD', 'TLT', 'USO', 'EEM'], axis=1)
ticker_list = etf.tolist()[:12]
for t in ['GLD', 'TLT', 'USO', 'EEM']:
    ticker_list.remove(t)
date_range_daily = date_range.copy()



#%% saving pickled objects
data_directory = './data/real_data/crsp/vol/'
daily_vol.to_pickle(data_directory + 'daily_vol.pkl')
date_range.to_pickle(data_directory + 'date_range.pkl')
# date_range.to_pickle(data_directory + 'date_range.pkl')

with open(data_directory + 'ticker_list.pkl', 'wb') as f:
    pickle.dump(ticker_list, f)
