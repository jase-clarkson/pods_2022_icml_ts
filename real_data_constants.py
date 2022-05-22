""" A file containing constants used for real data experiments"""

# Map different frequencies to observation periods
train_valid_test_params_dict = {
    'daily': {
        'test_start': 6 * 250,
        'validation_len': 150,  #63,
        'update_period': 150
    },
}


lookback_list_dict = {
    'daily': [5, 10, 21, 63, 126, 252, 252 * 2, 252 * 5, 252 * 10, 252 * 20],
}


