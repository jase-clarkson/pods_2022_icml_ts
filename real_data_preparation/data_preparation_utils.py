import os
import numpy as np
import pandas as pd
from tqdm import tqdm


def get_date_from_filename(name):
    return name[:4] + '-' + name[4:6] + '-' + name[6:8]


def process_single_day_data(data_frame, tickers=None):
    """
    close: closing price (unadj)
    pvCLCL: RET (linear return) in Wharton DB
    prevAdjClose: close price, adjusted for splits and dividends
    volume: daily traded volume
    high: daily high
    low: daily low

    """
    col_names = ['close', 'pvCLCL', 'prevAdjClose', 'volume', 'high', 'low', 'sharesOut']
    data_subset = data_frame.set_index('ticker')
    data_subset = data_subset.loc[:, col_names]
    data_subset.loc[:, 'dollar_volume'] = data_subset.loc[:, 'close'] * data_subset.loc[:, 'volume']
    data_subset.loc[:, 'market_cap'] = data_subset.loc[:, 'close'] * data_subset.loc[:, 'sharesOut']

    data_subset = data_subset.drop(columns=['close', 'volume', 'sharesOut'])
    data_subset = data_subset.rename_axis(columns=['feature'])

    if tickers is not None:
        data_subset = data_subset.reindex(index=tickers)

    return data_subset


def get_data(directory, tickers=None):
    data_store = dict()

    for entry in tqdm(list(os.walk(directory))):
        if len(entry[1]) == 0:
            for filename in entry[2]:
                if filename[-3:] == 'csv':
                    data_frame = pd.read_csv(entry[0] + '/' + filename)
                    data_frame = process_single_day_data(data_frame, tickers)
                    data_store[get_date_from_filename(filename)] = data_frame

    data = pd.concat(data_store).unstack()
    data.index = pd.DatetimeIndex(data.index)
    data = data.sort_index()

    return data


def col_name_process(col_name):
    contract_name, feature = col_name.split(' - ')

    # drop 'CHRIS/CME_' and '1'
    contract_name = contract_name[10:][:-1]

    return contract_name, feature


def compute_start_date(df):
    """
    Computing start date (first non-na value) for a given feature each instrument
    """
    start_date = ~df.isna()
    start_date = start_date.idxmax()

    # start_date['TY'] = pd.Timestamp('2014-01-01')

    return start_date


def compute_date_range(df):

    start_date = compute_start_date(df)
    end_date = compute_start_date(df.iloc[::-1])

    date_range = pd.concat([start_date, end_date], axis=1, keys=['start', 'end'])

    return date_range


def dollar_volume_rank_indicator(row, non_na_price_indicator, mean_daily_dollar_vol_rank):
    # if it is not a non-na price, then fill with np.nan
    av_volume_rank = row.where(non_na_price_indicator.loc[row.name], np.nan)
    # rank by dollar volume
    av_volume_rank = av_volume_rank.rank(ascending=False, method='first', na_option='bottom')
    av_volume_rank_indicator = av_volume_rank <= mean_daily_dollar_vol_rank

    return av_volume_rank_indicator


def filter_by_na_and_volume(close_price, dollar_volume, lookback, min_number_non_na, mean_daily_dollar_vol_rank=500):
    """
    For each time row, checks if a stock:
    - is among the top "mean_daily_dollar_vol_rank" highest mean daily dollar volume stocks in the lookback window
    - has more than a max number of non-na values

    """
    non_na_price = (~close_price.isna()).rolling(window=lookback, min_periods=1).sum()
    non_na_price = non_na_price.fillna(0)
    non_na_price_indicator = non_na_price > min_number_non_na

    av_volume = dollar_volume.rolling(window=lookback, min_periods=1).mean()

    av_volume_indicator = av_volume.apply(dollar_volume_rank_indicator, non_na_price_indicator=non_na_price_indicator,
                                          mean_daily_dollar_vol_rank=mean_daily_dollar_vol_rank, axis=1)

    return av_volume_indicator

