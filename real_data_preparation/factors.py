"""
Download wrds factor data and combine with saved crsp data

https://wrds-www.wharton.upenn.edu/pages/support/manuals-and-overviews/fama-french/fama-french-research-portfolios-and-
factors/

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wrds
import pickle

monthly = True

#%% load or download factor data

# db = wrds.Connection(wrds_username='')
#
# db.list_tables('ff')

# download daily factor data
# db.describe_table('ff', 'factors_daily')
# factors = db.get_table('ff', 'factors_daily', columns=['date', 'mktrf', 'smb', 'hml', 'rf', 'umd'], obs=10)
# factors = factors.set_index('date')
# factors.index = pd.DatetimeIndex(factors.index)

# load factor data from csv (downloaded from https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
# in April 2022)
factors = pd.read_csv('./data/real_data/crsp/factor/F-F_Research_Data_Factors_daily.CSV', index_col=0, parse_dates=True)
factors.index = pd.DatetimeIndex(factors.index)
factors.rename({'Mkt-RF': 'mktrf', 'SMB': 'smb', 'HML': 'hml', 'RF': 'rf'}, axis=1, inplace=True)

#%% load in crsp equity data and combine with factor data
crsp_log_ret = pd.read_pickle('./data/real_data/crsp/equity/log_returns.pkl')

factors = np.log(factors/100 + 1)

# substracting rf return from equity returns
crsp_log_ret = pd.concat([crsp_log_ret, factors.loc[:, ['rf']]], axis=1, join='inner')
crsp_log_ret = crsp_log_ret.sub(crsp_log_ret.loc[:, 'rf'], axis=0)
crsp_log_ret = crsp_log_ret.drop('rf', axis=1)
# concatenating to a single df
data = pd.concat([factors, crsp_log_ret], axis=1, join='inner')
# data.cumsum().plot(); plt.show()

# load in date_range for crsp equity data
date_range = pd.read_pickle('./data/real_data/crsp/equity/date_range.pkl')


date_range = date_range.reindex(index=crsp_log_ret.columns)

# creating list of the equity tickers that will be used as cross-sectional regression targets
ticker_list = crsp_log_ret.columns.tolist()

#%% saving factor data
data_directory = './data/real_data/crsp/factor/'
data.to_pickle(data_directory + 'log_ret.pkl')
date_range.to_pickle(data_directory + 'date_range.pkl')

with open(data_directory + 'ticker_list.pkl', 'wb') as f:
    pickle.dump(ticker_list, f)